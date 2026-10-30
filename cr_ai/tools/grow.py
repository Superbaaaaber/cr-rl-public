"""cr_ai/tools/grow.py — 给训练中的模型扩容（「手术」，扩容后和原来是同一个函数；做法见 model/grow.py）。

    python -m cr_ai.tools.grow --src runs/ppo_v2_m4/ppo_latest.pt --dst runs/grow/ppo_latest.pt --obs obs.pkl \
        model.hidden=1536 model.head_hidden=256 model.exit_channels=128 model.grid_dim=512 model.place_res_blocks=4 model.entity_layers=4

步骤：按新结构建网（--seed 定随机部分）→ grow_state_dict →（可选 --redo-margin：重置融合层的死单元）
→ 新 LSTM 单元的遗忘门偏置设 1 → 校准新 BN 通道的统计量 → 新 ReLU 单元的偏置按样本居中 → AdamW 状态（opt、opt_v）按同样的映射搬过去
→ 在留出的观测流上逐步（带 LSTM 状态）对比新旧网络：出牌 logits、价值、特权价值、位置 logits、LSTM 旧单元的 h/c 的最大差 → 存档。
存档的 cfg 换成新结构；续训时训练配置（如 runs/m4_main.txt）要加上同样的 model.* 覆盖，联赛里的旧快照和 BC 参考网络由
load_weights(..., grow=True) 在加载时自动补成新结构。
观测样本：pickle 的 {流: [逐步观测 dict]}（arena 被测方的逐步观测；没有特权信息时特权价值头用随机生成的特权输入对比）。
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import yaml

from ..config import _merge, _unwrap, _wrap, limit_torch_threads, load
from ..model import grow as G
from ..model.net import Net, rl_keys


def merged_cfg(ck_cfg, overrides=()):
    """默认配置 ← 存档里的配置 ← overrides（'a.b=值'，值按 YAML 解析）。"""
    d = _unwrap(load())
    _merge(d, _unwrap(ck_cfg))
    for o in overrides:
        key, val = o.split('=', 1)
        x = d
        ks = key.split('.')
        for k in ks[:-1]:
            x = x.setdefault(k, {})
        x[ks[-1]] = yaml.safe_load(val)
    return _wrap(d)


def stack(rows, keys, dev):
    return {k: torch.as_tensor(np.stack([r[k] for r in rows])).to(dev) for k in keys}


class FuseProbe:
    """记融合层（ReLU 之前）每个单元的最大值和取正值的次数。"""

    def __init__(self, net):
        self.mx, self.pos, self.n = None, None, 0
        self.h = net.fuse[0].register_forward_hook(self)

    def __call__(self, m, inp, out):
        x = out.detach().float()
        mx, pos = x.amax(0), (x > 0).sum(0)
        self.mx = mx if self.mx is None else torch.maximum(self.mx, mx)
        self.pos = pos if self.pos is None else self.pos + pos
        self.n += x.shape[0]

    def close(self):
        self.h.remove()


@torch.no_grad()
def verify(old, new, streams, keys, a0, T, dev, seed=0):
    """逐步（带各自的 LSTM 状态）对比：→ 各项最大绝对差、量级。"""
    S, H0 = len(streams), a0['H']
    gen = torch.Generator().manual_seed(seed)
    st0 = st1 = None
    d = dict(card=0.0, value=0.0, priv_value=0.0, place=0.0, h=0.0, c=0.0)
    scale = dict(card=0.0, place=0.0, value=0.0)
    for t in range(T):
        o = stack([s[t] for s in streams], keys, dev)
        op = o if 'priv_id' in o else dict(o, **G.synth_priv(S, gen, dev))
        pc = torch.full((S,), 1 + t % 4, dtype=torch.long, device=dev)
        out0, st0 = old(op, st0, place_card=pc)
        out1, st1 = new(op, st1, place_card=pc)
        m = o['card_mask'].bool()
        d['card'] = max(d['card'], float((out0['card'] - out1['card'])[m].abs().max()))
        d['priv_value'] = max(d['priv_value'], float((out0['value'] - out1['value']).abs().max()))
        plain = {k: v for k, v in o.items() if not k.startswith('priv_')}
        d['value'] = max(d['value'], float((old.value_from(out0['h'], plain) - new.value_from(out1['h'], plain)).abs().max()))
        d['place'] = max(d['place'], float((out0['place'] - out1['place']).abs().max()))
        d['h'] = max(d['h'], float((st0[0] - st1[0][:, :H0]).abs().max()))
        d['c'] = max(d['c'], float((st0[1] - st1[1][:, :H0]).abs().max()))
        scale['card'] = max(scale['card'], float(out0['card'][m].abs().max()))
        scale['place'] = max(scale['place'], float(out0['place'].abs().max()))
        scale['value'] = max(scale['value'], float(out0['value'].abs().max()))
    return d, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--obs', required=True, help='观测样本 pickle：{流: [逐步观测 dict]}')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # 10-28 实测（6.839 亿，12 条流 9,772 步对半分）：样本上恒 ≤ 0 的 343–347 个单元里，换另一半流有 5–9 个会取正——「死」只是在这几局里死，
    # 训练里约 600 套卡组，重置它们就不再是同一个函数。默认不重置（新加的单元是活的）；给了 margin 才重置样本上最大值 < −margin 的
    ap.add_argument('--redo-margin', type=float, default=None, help='重置融合层里样本上预激活最大值 < −margin 的单元（默认不重置）')
    ap.add_argument('--verify-streams', type=int, default=32)
    ap.add_argument('--verify-steps', type=int, default=96)
    ap.add_argument('--probe-rows', type=int, default=6000, help='找死单元、校准 BN 用的观测行数（不含留出验证的流）')
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('overrides', nargs='*', help='新结构：model.hidden=1536 …')
    a = ap.parse_args()
    limit_torch_threads()                                             # 容器里 torch 默认按宿主机开 104 个线程（无卡模式只有 0.5 核）
    dev = torch.device(a.device)
    torch.backends.cudnn.allow_tf32 = False                           # 严格 fp32 比较（Ampere 上卷积默认走 TF32，误差约 1e-3）
    torch.backends.cuda.matmul.allow_tf32 = False
    st = torch.load(a.src, map_location='cpu', weights_only=False)
    c0, c1 = merged_cfg(st['cfg']), merged_cfg(st['cfg'], a.overrides)
    old = Net(c0)
    old.load_state_dict(st['model'])
    torch.manual_seed(a.seed)
    new = Net(c1)
    tpl = {k: v.clone() for k, v in new.state_dict().items()}
    a0, a1 = G.arch_of(st['model']), G.arch_of(tpl)
    print('结构 %s → %s；参数 %.2fM → %.2fM' % (a0, a1, sum(p.numel() for p in old.parameters()) / 1e6,
                                               sum(p.numel() for p in new.parameters()) / 1e6), flush=True)
    sd = G.grow_state_dict(st['model'], tpl)

    streams = pickle.load(open(a.obs, 'rb'))
    names = sorted(streams, key=str)
    first = streams[names[0]][0]
    keys = [k for k in rl_keys(c0) if k in first]                     # 样本里有特权信息就用真的，没有时验证用随机生成的
    ok = [streams[k] for k in names if len(streams[k]) >= a.verify_steps]
    ver = (ok if len(ok) >= 2 * a.verify_streams else ok[::2])[:a.verify_streams]     # 流不多时按流对半分：一半验证、一半找死单元
    ver_ids = {id(s) for s in ver}
    pool = [r for k in names if id(streams[k]) not in ver_ids for r in streams[k]]
    rng = np.random.RandomState(a.seed)
    probe = [pool[i] for i in rng.choice(len(pool), min(a.probe_rows, len(pool)), replace=False)]
    del streams, pool
    batches = [stack(probe[i:i + 256], keys, dev) for i in range(0, len(probe), 256)]
    print('观测：验证 %d 条流 × %d 步，另取 %d 行找死单元 / 校准 BN' % (len(ver), a.verify_steps, len(probe)), flush=True)

    old.to(dev).eval()
    fp = FuseProbe(old)
    with torch.no_grad():
        for o in batches:
            old(o)
    fp.close()
    dead = [] if a.redo_margin is None else (fp.mx < -a.redo_margin).nonzero().flatten().cpu().numpy()
    print('融合层死单元（%d 行上恒 ≤ 0）：%d / %d；重置 %d 个' % (fp.n, int((fp.mx <= 0).sum()), a0['H'], len(dead)), flush=True)
    G.redo_dead(sd, tpl, dead, a0['H'], a1['H'])
    G.set_forget_bias(sd, a0['H'], a1['H'], 1.0)
    new.load_state_dict(sd)
    new.to(dev).eval()
    bn = G.calibrate_bn(new, a0, batches[:8])
    print('新 BN 通道校准（|均值|、方差）：%s' % json.dumps({k: [round(x, 4) for x in v] for k, v in bn.items()}, ensure_ascii=False), flush=True)
    cen = G.center_new_units(new, a0, dead, batches[:8], a.seed)
    print('新单元偏置居中（居中前取正的比例中位数）：%s' % json.dumps({k: round(v, 3) for k, v in cen.items()}), flush=True)
    sd = {k: v.detach().cpu() for k, v in new.state_dict().items()}

    p_old, p_new = FuseProbe(old), FuseProbe(new)
    diff, scale = verify(old, new, ver, keys, a0, a.verify_steps, dev, a.seed)
    p_old.close()
    p_new.close()
    d_idx = torch.as_tensor(np.asarray(dead, np.int64)).to(dev)
    woke = int((p_old.mx[d_idx] > 0).sum()) if len(dead) else 0
    act = (p_new.pos.float() / p_new.n)
    print('验证（留出的流）最大差：%s；量级 %s' % (json.dumps({k: float('%.3g' % v) for k, v in diff.items()}),
                                               json.dumps({k: round(v, 3) for k, v in scale.items()})), flush=True)
    print('重置的死单元在验证流上：旧网络里其实取正的 %d 个；新网络里取正的比例中位数 %.3f（新加的单元 %.3f）'
          % (woke, float(act[d_idx].median()) if len(dead) else float('nan'), float(act[a0['H']:].median())), flush=True)
    bad = {k: v for k, v in diff.items() if v > a.tol}
    if bad:
        raise SystemExit('新旧网络输出对不上（> %g）：%s' % (a.tol, bad))

    out = dict(model=sd, cfg=dict(c1), step=st.get('step', 0), update=st.get('update', 0), kind=st.get('kind', 'ppo'),
               grow=dict(src=a.src, arch_old=a0, arch_new=a1, overrides=list(a.overrides), dead=int(len(dead)), seed=a.seed,
                         verify=diff, woke=woke))
    if 'opt' in st:
        fz = bool(c1.ppo.get('freeze_unit_emb', True)) if c1.get('ppo') else True
        out['opt'] = G.grow_adam(st['opt'], G.opt_names(old, freeze_unit_emb=fz), G.opt_names(new, freeze_unit_emb=fz),
                                 a0, a1, st['model'], dead)
        out['opt_v'] = G.grow_adam(st['opt_v'], G.opt_names(old, value_only=True), G.opt_names(new, value_only=True),
                                   a0, a1, st['model'])
        out['scaler'] = st.get('scaler')
        print('优化器状态已按新结构搬过去（opt %d → %d 个参数）' % (len(st['opt']['param_groups'][0]['params']),
                                                       len(out['opt']['param_groups'][0]['params'])), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    torch.save(out, a.dst + '.tmp')
    os.replace(a.dst + '.tmp', a.dst)
    print('已存：%s' % a.dst, flush=True)


if __name__ == '__main__':
    main()
