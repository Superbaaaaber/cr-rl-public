"""cr_ai/train/bc.py — M2 行为克隆。

    python -m cr_ai.train.bc --config cr_ai/configs/bc.yaml [更多叠加的配置…] --run runs/bc1 [--resume] [--max-steps N] [--bench 200]

数据：data_bc/il_replay 的分片（replay_resim 产出），loader 多进程现场栅格化 + 左右镜像。
损失：牌头交叉熵（“不出”类权重 noop_w）+ 位置头交叉熵（高斯平滑 σ 格，只在出牌步）+ 价值头 MSE（该方视角终局 ±1/0）。
样本权重 w：“不出”样本只留了 20%，w = 5 还原真实步频。
价值损失只在随机 value_frac 的样本上算：同一局约 330 个样本共用一个胜负标签，loader 缓冲区里同一局会反复出现，
全量算会让价值头记住“正在看的这几局”的结果（训练 MSE 0.47、验证 1.16，见 runs/bc1 前 2 万步）。
LSTM：BC 里每个样本单步、零初始状态（样本是抽稀过的，不是连续步）；循环状态在 PPO 里学。
输出：runs/<run>/bc_latest.pt（续训用）、bc_step<N>.pt、train_log.jsonl、val_log.jsonl、tb/（TensorBoard）、config.yaml。
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import dump, limit_torch_threads, load, repo_path
from ..data.bc_dataset import TorchShards, idle_worker_init
from ..model.net import GH, GW, INPUT_KEYS, NCELL, Net, load_weights, n_params          # noqa: F401（INPUT_KEYS 在这里再导出，别的模块按老路径 import）


def gauss_kernel(sigma, device):
    idx = torch.arange(NCELL, device=device)
    y, x = (idx // GW).float(), (idx % GW).float()
    d2 = (y[:, None] - y[None]) ** 2 + (x[:, None] - x[None]) ** 2
    return torch.exp(-d2 / (2 * sigma * sigma))


def to_dev(b, dev):
    o = {}
    for k, v in b.items():
        v = v.to(dev, non_blocking=True)
        o[k[5:] if k.startswith('feat_') else k] = v
    return o


def losses(net, b, kern, bc, train=True):
    B = b['sel'].shape[0]
    sel = b['sel'].long()
    ar = torch.arange(B, device=sel.device)
    cm = b['card_mask'].clone()
    cm[ar, sel] = True                                      # 强制合法标记的样本：标签选项放进 mask
    o = {k: b[k] for k in getattr(net, 'input_keys', INPUT_KEYS)}
    o['card_mask'] = cm
    out, _ = net(o, place_card=sel)
    w = b['w'].float()
    cw = torch.where(sel == 0, torch.full_like(w, bc.noop_w), torch.ones_like(w)) * w
    ce = F.cross_entropy(out['card'], sel, reduction='none')
    l_card = (ce * cw).sum() / cw.sum()
    is_play = (sel >= 1) & (sel <= 4)
    cell = b['cell'].long().clamp(0, NCELL - 1)
    pm = b['pmask'].clone()
    pm[ar, cell] |= is_play
    tgt = kern[cell] * pm
    tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1e-6)
    lp = F.log_softmax(out['place'].masked_fill(~pm, -1e9), 1)
    pl = -(tgt * lp).sum(1)
    pw = is_play.float()
    l_place = (pl * pw).sum() / pw.sum().clamp(min=1)
    vw = w
    if train and bc.get('value_frac', 1.0) < 1.0:
        # 同一局的几百个样本共用一个胜负标签：只抽一部分算价值损失，减轻“记住缓冲区里这几局的结果”
        vw = w * (torch.rand_like(w) < bc.value_frac).float()
    l_val = (((out['value'] - b['ret'].float()) ** 2) * vw).sum() / vw.sum().clamp(min=1e-6)
    loss = l_card + bc.place_w * l_place + bc.value_w * l_val
    # 分项损失留在显卡上（不 .item()）：每步取值会强制 CPU 等显卡算完，打断流水；记日志时再取
    return loss, dict(card=l_card.detach(), place=l_place.detach(), value=l_val.detach()), out, cm, pm, is_play, cell


@torch.no_grad()
def metrics_acc(net, out, b, pm, is_play, cell):
    """累加用的计数：出牌步牌头 top-1（在出牌选项里 / 整体含不出）、位置误差 ≤1 格、“不出”召回、价值胜负预测。"""
    sel = b['sel'].long()
    pol = net.policy_logits(out['card'])
    top_all = pol.argmax(1)
    play_logits = out['card'].clone()
    play_logits[:, 0] = -1e9
    top_play = play_logits.argmax(1)
    act = sel > 0
    pa = out['place'].masked_fill(~pm, -1e9).argmax(1)
    dy = (pa // GW - cell // GW).abs()
    dx = (pa % GW - cell % GW).abs()
    near = (torch.maximum(dx, dy) <= 1) & is_play
    noop = sel == 0
    dec = b['ret'] != 0
    vok = (torch.sign(out['value']) == torch.sign(b['ret'].float())) & dec
    # 出牌时机的校准（“不出”占 95%，noop_recall 恒为 1、没信息量）：按样本权重 w（“不出”样本抽稀后 w = 5）还原真实步频，
    # 比较模型给的“这一步出牌”概率（推理口径，已加回 noop 先验）和实际出牌频率，另算 Brier 分数
    w = b['w'].float() if 'w' in b else torch.ones_like(sel, dtype=torch.float32)
    p_play = 1.0 - torch.softmax(pol.float(), 1)[:, 0]
    y = act.float()
    return dict(n_act=act.sum(), card_top1_play=((top_play == sel) & act).sum(), card_top1_all=((top_all == sel) & act).sum(),
                n_play=is_play.sum(), place_le1=near.sum(), place_exact=((pa == cell) & is_play).sum(),
                n_noop=noop.sum(), noop_recall=((top_all == 0) & noop).sum(),
                n_dec=dec.sum(), value_acc=vok.sum(),
                w_sum=w.sum(), play_pred=(w * p_play).sum(), play_true=(w * y).sum(), play_brier=(w * (p_play - y) ** 2).sum())   # 显卡上的计数张量，累加后在 summarize 里一次取回


def summarize(acc):
    acc = {k: float(v) for k, v in acc.items()}
    return dict(card_top1_play=acc['card_top1_play'] / max(1, acc['n_act']), card_top1_all=acc['card_top1_all'] / max(1, acc['n_act']),
                place_le1=acc['place_le1'] / max(1, acc['n_play']), place_exact=acc['place_exact'] / max(1, acc['n_play']),
                noop_recall=acc['noop_recall'] / max(1, acc['n_noop']), value_acc=acc['value_acc'] / max(1, acc['n_dec']),
                play_rate_pred=acc['play_pred'] / max(1e-9, acc['w_sum']), play_rate_true=acc['play_true'] / max(1e-9, acc['w_sum']),
                play_brier=acc['play_brier'] / max(1e-9, acc['w_sum']))


def make_loader(files, split, cfg, bs, mirror, seed, epochs, workers):
    ds = TorchShards(files, split, cfg.bc.val_frac, bs, mirror, seed, cfg.bc.shards_in_buffer, epochs, drop_last=split == 'train',   # 训练只出整 batch（见 BCShards.batches）
                     entities=cfg.model.get('entity', False))
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers, pin_memory=True, prefetch_factor=4 if workers else None,
                                       worker_init_fn=idle_worker_init if workers else None, persistent_workers=False)


def validate(net, cfg, files, kern, dev):
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False             # 验证 batch 的尺寸零碎，每种新尺寸都搜一遍算法反而慢得多（第一次验证 80 多秒）
    try:
        return _validate(net, cfg, files, kern, dev)
    finally:
        torch.backends.cudnn.benchmark = bench


@torch.no_grad()
def _validate(net, cfg, files, kern, dev):
    net.eval()
    acc, ls, n = {}, {}, 0
    for b in make_loader(files, 'val', cfg, 1024, 0.0, 0, 1, min(4, cfg.bc.loader_workers)):
        b = to_dev(b, dev)
        with torch.autocast('cuda', dtype=amp_dtype(cfg.bc)):
            _loss, parts, out, _cm, pm, is_play, cell = losses(net, b, kern, cfg.bc, train=False)
        for k, v in metrics_acc(net, out, b, pm, is_play, cell).items():
            acc[k] = acc.get(k, 0) + v
        for k, v in parts.items():
            ls[k] = ls.get(k, 0) + v
        n += 1
    net.train()
    r = summarize(acc)
    r.update({'loss_' + k: float(v) / max(1, n) for k, v in ls.items()})
    r['n_samples'] = int(acc.get('n_act', 0) + acc.get('n_noop', 0))
    return r


def amp_dtype(bc):
    """混合精度类型：bc.amp = fp16（默认，要 GradScaler，每步检查溢出会同步一次）| bf16（指数范围同 fp32，不用 GradScaler）。"""
    return torch.bfloat16 if bc.get('amp', 'fp16') == 'bf16' else torch.float16


def lr_at(step, total, bc):
    if step < bc.warmup_steps:
        return bc.lr * (step + 1) / bc.warmup_steps
    t = min(1.0, (step - bc.warmup_steps) / max(1, total - bc.warmup_steps))
    return bc.lr_end + 0.5 * (bc.lr - bc.lr_end) * (1 + math.cos(math.pi * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['cr_ai/configs/bc.yaml'], help='可给多个，依次叠加（如 bc.yaml bc_mix.yaml）')
    ap.add_argument('--run', default='runs/bc1')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init', default='', help='从这个存档的模型权重开始（微调用；优化器从头建，--resume 有存档时以存档为准）')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--bench', type=int, default=0, help='只测吞吐：跑 N 步后退出')
    ap.add_argument('--stop-at', type=int, default=0, help='练到第 N 步就停（存档并验证），学习率仍按完整训练的总步数走；容量测试和 bc1 同步数对比用')
    ap.add_argument('--samples-est', type=float, default=82.9e6, help='样本总数（算总步数用）')
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
    net = Net(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=bc.lr, weight_decay=bc.weight_decay)
    dt_amp = amp_dtype(bc)
    scaler = torch.amp.GradScaler('cuda', enabled=dt_amp == torch.float16)          # bf16 不需要（关掉后 step 不再每步同步检查溢出）
    # bc.compile：'' 不编译 | 'default' | 'reduce-overhead'（CUDA Graph）| 'max-autotune'。只编译训练用的前向；验证、指标、存档都用原模型
    fwd = torch.compile(net, mode=None if bc.get('compile') == 'default' else bc.get('compile'), dynamic=False) if bc.get('compile') else net
    total = a.max_steps or int(bc.epochs * a.samples_est * (1 - bc.val_frac) / bc.batch_size)
    step = 0
    ck = os.path.join(run, 'bc_latest.pt')
    if a.init and not (a.resume and os.path.exists(ck)):
        load_weights(net, torch.load(repo_path(a.init), map_location=dev, weights_only=False)['model'])
        print('从 %s 初始化' % a.init, flush=True)
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        scaler.load_state_dict(st['scaler'])
        step = st['step']
        print('续训：step %d' % step, flush=True)
    print('参数量 %.2fM，总步数 %d，分片 %d' % (n_params(net) / 1e6, total, len(files)), flush=True)
    with open(os.path.join(run, 'run_info.json'), 'w', encoding='utf-8') as f:          # 给训练看板用
        json.dump(dict(kind='bc', total_steps=min(total, a.stop_at) if a.stop_at else total, batch_size=bc.batch_size, n_params=n_params(net)), f)   # 看板按这个算进度和剩余时间
    kern = gauss_kernel(bc.place_sigma, dev)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run, 'tb'))
    flog = open(os.path.join(run, 'train_log.jsonl'), 'a', encoding='utf-8')
    vlog = open(os.path.join(run, 'val_log.jsonl'), 'a', encoding='utf-8')

    def save(path):
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, cfg=dict(cfg)), path + '.tmp')
        os.replace(path + '.tmp', path)

    # 续训时换个种子重新打乱（不精确跳过已训练的样本）
    loader = make_loader(files, 'train', cfg, bc.batch_size, bc.mirror_prob, cfg.seed + step, max(1, math.ceil(bc.epochs)) + 1, bc.loader_workers)
    t0, ns, acc, agg, nagg = time.time(), 0, {}, {}, 0
    tw = time.time()
    wait = 0.0
    r = None                                               # 最近一次验证结果
    net.train()
    for b in loader:
        if step >= total:
            break
        wait += time.time() - tw
        b = to_dev(b, dev)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, total, bc)
        with torch.autocast('cuda', dtype=dt_amp):
            loss, parts, out, _cm, pm, is_play, cell = losses(fwd, b, kern, bc)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), bc.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        ns += b['sel'].shape[0]
        for k, v in parts.items():
            agg[k] = agg.get(k, 0) + v
        nagg += 1
        if step % 10 == 0:
            for k, v in metrics_acc(net, out, b, pm, is_play, cell).items():
                acc[k] = acc.get(k, 0) + v
        if step % bc.log_every == 0 or (a.bench and step == a.bench):
            dt = time.time() - t0
            rec = dict(step=step, samples_per_s=ns / dt, loader_wait=wait / dt, lr=opt.param_groups[0]['lr'], grad_norm=float(gn),
                       **{'loss_' + k: float(v) / nagg for k, v in agg.items()}, **summarize(acc), t=time.time())
            flog.write(json.dumps(rec) + '\n')
            flog.flush()
            for k, v in rec.items():
                if k not in ('step', 't'):
                    tb.add_scalar('train/' + k, v, step)
            print('[%d/%d] %.0f 样本/s 等数据 %.0f%% loss 牌 %.3f 位 %.3f 值 %.3f | 牌top1 %.3f 位≤1 %.3f 出牌率 预测 %.4f / 实际 %.4f' % (
                step, total, rec['samples_per_s'], 100 * rec['loader_wait'], rec['loss_card'], rec['loss_place'], rec['loss_value'],
                rec['card_top1_play'], rec['place_le1'], rec['play_rate_pred'], rec['play_rate_true']), flush=True)
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
            vlog.write(json.dumps(r) + '\n')
            vlog.flush()
            for k, v in r.items():
                if k != 'step':
                    tb.add_scalar('val/' + k, v, step)
            print('验证 @%d：%s' % (step, json.dumps({k: round(v, 4) for k, v in r.items()})), flush=True)
        if a.stop_at and step >= a.stop_at:                 # 提前停：学习率仍按 total 的余弦走（和完整训练同一时刻可比）
            break
        tw = time.time()
    save(ck)
    save(os.path.join(run, 'bc_final.pt'))
    if r is None or step % bc.eval_every:                  # 最后一步刚验证过就不重复验证
        r = validate(net, cfg, val_files, kern, dev)
        r['step'] = step
        vlog.write(json.dumps(r) + '\n')
    print('最终验证：%s' % json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    main()
