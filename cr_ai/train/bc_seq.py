"""cr_ai/train/bc_seq.py — 带记忆的模仿学习：在单步 BC 的基础上，用连续时间线微调，让 LSTM 学会用前面发生过的事。

    python -m cr_ai.train.bc_seq --config cr_ai/configs/bc.yaml cr_ai/configs/bc_seq.yaml --run runs/seq_b --init runs/cap_b/bc_final.pt \\
        model.channels=192 model.res_blocks=8 model.hidden=768 [--resume]

单步 BC（train/bc.py）每个样本都从零状态算，LSTM 的循环部分（W_hh、遗忘门）从没被训练过。这里：
  1. 从单步 BC 的存档出发，先 init_recurrence（W_hh 置 0、遗忘门偏置 −4）：起点的输出和单步 BC 几乎一样；
  2. 数据是 replay_resim --seq 的分片（每步都有样本），按（局, 玩家）的时间线切成片段（data/bc_dataset.py: SeqShards）：
     前 burn_in 步只推进记忆不算损失，后 seq_len 步算损失；LSTM 整段展开（与 PPO 的 seq_forward 相同）；
  3. 损失与单步 BC 相同（牌头 / 位置头 / 价值头），只在片段的损失区、且样本权重 w > 0 的行上算。
验证：同一批验证局，第 0 步先用单步方式（每步零状态 = 原来的 BC）评一次作为基线，之后都用带记忆的方式评；
指标只算有标签的步（去掉强制不出、标签缺失的占位），口径与单步 BC 一致。
存档：bc_latest.pt / bc_step<N>.pt / bc_final.pt，带 recurrent=True —— PPO 用 ppo.init_ckpt 从它开始时不再 init_recurrence。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import dump, limit_torch_threads, load, repo_path
from ..data.bc_dataset import TorchSeqShards, idle_worker_init
from ..model.net import NCELL, Net, init_recurrence, n_params
from .bc import INPUT_KEYS, amp_dtype, gauss_kernel, lr_at, metrics_acc, summarize, to_dev
from .ppo import seq_forward


def seq_losses(net, b, kern, bc, K, mode='seq', train=True):
    """b：[S·K] 行（按片段排列）。mode='seq'：片段整段展开 LSTM；'single'：每行零状态（= 单步 BC），用来在同一批数据上做基线。"""
    sel = b['sel'].long()
    B = sel.shape[0]
    ar = torch.arange(B, device=sel.device)
    cm = b['card_mask'].clone()
    cm[ar, sel] = True                                      # 强制合法标记的样本：标签选项放进 mask
    o = {k: b[k] for k in getattr(net, 'input_keys', INPUT_KEYS)}
    o['card_mask'] = cm
    if mode == 'seq':
        S = B // K
        z = torch.zeros(S, net.H, device=sel.device)
        out, card, hs = seq_forward(net, o, z, z.clone(), K)
        out['card'] = card
        out['value'] = torch.tanh(net.value_head(hs).squeeze(-1))
        out['place'] = net.place_logits(out, sel)
    else:
        out, _ = net(o, place_card=sel)
    lm = b['lmask'].float()
    w = b['w'].float() * lm
    cw = torch.where(sel == 0, torch.full_like(w, bc.noop_w), torch.ones_like(w)) * w
    l_card = (F.cross_entropy(out['card'], sel, reduction='none') * cw).sum() / cw.sum().clamp(min=1e-6)
    is_play = (sel >= 1) & (sel <= 4)
    cell = b['cell'].long().clamp(0, NCELL - 1)
    pm = b['pmask'].clone()
    pm[ar, cell] |= is_play
    tgt = kern[cell] * pm
    tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-6)
    lp = F.log_softmax(out['place'].masked_fill(~pm, -1e9), 1)
    pw = is_play.float() * w
    l_place = (-(tgt * lp).sum(1) * pw).sum() / pw.sum().clamp(min=1)
    vw = w
    if train and bc.get('value_frac', 1.0) < 1.0:
        vw = w * (torch.rand_like(w) < bc.value_frac).float()
    l_val = (((out['value'] - b['ret'].float()) ** 2) * vw).sum() / vw.sum().clamp(min=1e-6)
    loss = l_card + bc.place_w * l_place + bc.value_w * l_val
    # 指标只算有标签的步：损失区、w > 0、不是强制不出（单步 BC 的数据里没有这些步，口径一致）
    mm = b['lmask'] & (b['w'] > 0) & ((b['flags'] & 4) == 0)
    return loss, dict(card=l_card.detach(), place=l_place.detach(), value=l_val.detach()), out, pm, is_play, cell, mm


def masked_metrics(net, out, b, pm, is_play, cell, mm):
    i = torch.nonzero(mm).squeeze(1)
    sub_out = {k: out[k][i] for k in ('card', 'place', 'value')}
    sub_b = {k: b[k][i] for k in ('sel', 'ret')}
    return metrics_acc(net, sub_out, sub_b, pm[i], is_play[i], cell[i])


def make_loader(files, split, cfg, seed, epochs, workers, mirror):
    bc = cfg.bc
    ds = TorchSeqShards(files, split, bc.val_frac, bc.seq_batch, bc.seq_len, bc.burn_in, mirror, seed, bc.shards_in_buffer, epochs,
                        drop_last=split == 'train', entities=cfg.model.get('entity', False))
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, prefetch_factor=4 if workers else None,
                                       worker_init_fn=idle_worker_init if workers else None, persistent_workers=False)


def validate(net, cfg, files, kern, dev, mode='seq'):
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False             # 验证的最后一批尺寸零碎，不为它搜算法
    try:
        return _validate(net, cfg, files, kern, dev, mode)
    finally:
        torch.backends.cudnn.benchmark = bench


@torch.no_grad()
def _validate(net, cfg, files, kern, dev, mode):
    net.eval()
    K = cfg.bc.burn_in + cfg.bc.seq_len
    acc, ls, n = {}, {}, 0
    for b in make_loader(files, 'val', cfg, 0, 1, min(4, cfg.bc.loader_workers), 0.0):
        b = to_dev(b, dev)
        with torch.autocast('cuda', dtype=amp_dtype(cfg.bc)):
            _loss, parts, out, pm, is_play, cell, mm = seq_losses(net, b, kern, cfg.bc, K, mode=mode, train=False)
        for k, v in masked_metrics(net, out, b, pm, is_play, cell, mm).items():
            acc[k] = acc.get(k, 0) + v
        for k, v in parts.items():
            ls[k] = ls.get(k, 0) + v
        n += 1
    net.train()
    r = summarize(acc)
    r.update({'loss_' + k: float(v) / max(1, n) for k, v in ls.items()})
    r['n_samples'] = int(acc.get('n_act', 0) + acc.get('n_noop', 0))
    r['mode'] = mode
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['cr_ai/configs/bc.yaml', 'cr_ai/configs/bc_seq.yaml'])
    ap.add_argument('--run', default='runs/seq_b')
    ap.add_argument('--init', default=None, help='单步 BC 存档（新训练从它开始；--resume 且有 bc_latest.pt 时忽略）')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--bench', type=int, default=0, help='只测吞吐：跑 N 步后退出')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    limit_torch_threads()
    cfg = load(*a.config, overrides=a.overrides)
    bc = cfg.bc
    run = repo_path(a.run)
    os.makedirs(run, exist_ok=True)
    dump(cfg, os.path.join(run, 'config.yaml'))
    torch.manual_seed(cfg.seed)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    files = sorted(sum([glob.glob(os.path.join(repo_path(d), 'shard_*.npz')) for d in bc.data_dirs], []))
    val_files = files[:bc.val_shards]
    K = bc.burn_in + bc.seq_len
    accum = int(bc.get('accum', 1))                         # 梯度累积份数（arch_v2 一次 32 段显存不够）
    assert bc.seq_batch % accum == 0, 'seq_batch 要能被 accum 整除'
    net = Net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=bc.lr, weight_decay=bc.weight_decay)
    dt_amp = amp_dtype(bc)
    scaler = torch.amp.GradScaler('cuda', enabled=dt_amp == torch.float16)
    total = bc.steps
    step = 0
    ck = os.path.join(run, 'bc_latest.pt')
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        scaler.load_state_dict(st['scaler'])
        step = st['step']
        print('续训：step %d' % step, flush=True)
    elif a.init:
        st = torch.load(repo_path(a.init), map_location=dev, weights_only=False)
        net.load_state_dict(st['model'])
        if not st.get('recurrent'):
            init_recurrence(net, bc.forget_bias)            # 单步 BC 的循环部分没训过：先置成“几乎不用记忆”，起点与单步 BC 一致
        print('从 %s 开始（单步 BC step %s）' % (a.init, st.get('step')), flush=True)
    else:
        raise SystemExit('需要 --init <单步 BC 存档>（或 --resume）')
    print('参数量 %.2fM，总步数 %d，每批 %d 段 × (%d burn-in + %d) 步，分片 %d' % (n_params(net) / 1e6, total, bc.seq_batch, bc.burn_in, bc.seq_len, len(files)),
          flush=True)
    with open(os.path.join(run, 'run_info.json'), 'w', encoding='utf-8') as f:          # 给训练看板用
        json.dump(dict(kind='bc', total_steps=total, batch_size=bc.seq_batch * bc.seq_len, n_params=n_params(net)), f)
    kern = gauss_kernel(bc.place_sigma, dev)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run, 'tb'))
    flog = open(os.path.join(run, 'train_log.jsonl'), 'a', encoding='utf-8')
    vlog = open(os.path.join(run, 'val_log.jsonl'), 'a', encoding='utf-8')

    def save(path):
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, cfg=dict(cfg), recurrent=True),
                   path + '.tmp')
        os.replace(path + '.tmp', path)

    def log_val(r):
        vlog.write(json.dumps(r) + '\n')
        vlog.flush()
        for k, v in r.items():
            if isinstance(v, (int, float)) and k != 'step':
                tb.add_scalar('val/' + k, v, r['step'])
        print('验证 @%d（%s）：%s' % (r['step'], r['mode'], json.dumps({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})), flush=True)

    if step == 0 and not a.bench:
        for mode in ('single', 'seq'):                     # 基线：同一批验证局上，单步方式（= 原 BC）与刚初始化的记忆方式
            r = validate(net, cfg, val_files, kern, dev, mode)
            r['step'] = 0
            log_val(r)
    loader = make_loader(files, 'train', cfg, cfg.seed + step, 1000, bc.loader_workers, bc.mirror_prob)
    t0, ns, acc, agg, nagg = time.time(), 0, {}, {}, 0
    tw = time.time()
    wait = 0.0
    r = None
    net.train()
    for b in loader:
        if step >= total:
            break
        wait += time.time() - tw
        b = to_dev(b, dev)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, total, bc)
        opt.zero_grad(set_to_none=True)
        rows = bc.seq_batch // accum * K                    # 每段的行连续存放：按段切成 accum 份，逐份前向/反向（省显存）
        for j in range(accum):
            bj = {k: v[j * rows:(j + 1) * rows] for k, v in b.items()}
            with torch.autocast('cuda', dtype=dt_amp):
                loss, parts, out, pm, is_play, cell, mm = seq_losses(net, bj, kern, bc, K)
            scaler.scale(loss / accum).backward()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / accum
            if (step + 1) % 10 == 0:
                for k, v in masked_metrics(net, out, bj, pm, is_play, cell, mm).items():
                    acc[k] = acc.get(k, 0) + v
            del loss, out
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), bc.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        ns += int(bc.seq_batch * bc.seq_len)
        nagg += 1
        if step % bc.log_every == 0 or (a.bench and step == a.bench):
            dt = time.time() - t0
            rec = dict(step=step, samples_per_s=ns / dt, loader_wait=wait / dt, lr=opt.param_groups[0]['lr'], grad_norm=float(gn),
                       **{'loss_' + k: float(v) / nagg for k, v in agg.items()}, **summarize(acc), t=time.time())
            flog.write(json.dumps(rec) + '\n')
            flog.flush()
            for k, v in rec.items():
                if k not in ('step', 't'):
                    tb.add_scalar('train/' + k, v, step)
            print('[%d/%d] %.0f 样本/s 等数据 %.0f%% loss 牌 %.3f 位 %.3f 值 %.3f | 牌top1 %.3f 位≤1 %.3f 不出召回 %.3f' % (
                step, total, rec['samples_per_s'], 100 * rec['loader_wait'], rec['loss_card'], rec['loss_place'], rec['loss_value'],
                rec['card_top1_play'], rec['place_le1'], rec['noop_recall']), flush=True)
            t0, ns, acc, agg, nagg, wait = time.time(), 0, {}, {}, 0, 0.0
        if a.bench and step >= a.bench:
            print('显存峰值 %.2f GB' % (torch.cuda.max_memory_allocated() / 2 ** 30))
            return
        if step % bc.ckpt_every == 0:
            save(ck)
            save(os.path.join(run, 'bc_step%d.pt' % step))
        if step % bc.eval_every == 0:
            r = validate(net, cfg, val_files, kern, dev)
            r['step'] = step
            log_val(r)
        tw = time.time()
    save(ck)
    save(os.path.join(run, 'bc_final.pt'))
    if r is None or step % bc.eval_every:
        r = validate(net, cfg, val_files, kern, dev)
        r['step'] = step
        log_val(r)
    print('最终验证：%s' % json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    main()
