"""M2 汇报：在验证局上算 BC 模型的指标。

    python -m cr_ai.tools.m2_report runs/bc1/bc_final.pt [--shards 300] [--json runs/bc1/m2_val.json]

规格书要求的四项（argmax 口径）：出牌步牌头 top-1、位置误差 ≤1 格（切比雪夫距离）、“不出”召回、价值头胜负预测准确率。
补充（采样口径，评测/RL 实际用的是采样）：
  - 不出步上 P(不出) 的均值、出牌步上 P(正确选项) 的均值；
  - 出牌频率校准：模型每步期望出牌数 Σw·(1−P(不出)) 与真人 Σw·[出牌] 之比（1 = 节奏与真人一致）；
  - 价值头按对局时段（0–60 s / 60–120 s / 120–180 s / 加时）的胜负预测准确率。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from ..config import ROOT, load
from ..model.net import GW, Net
from ..train.bc import gauss_kernel, losses, make_loader, to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--shards', type=int, default=300)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    dev = 'cuda'
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = load(overrides=())
    cfg.update(st['cfg'])
    net = Net(cfg).to(dev)
    net.load_state_dict(st['model'])
    net.eval()
    kern = gauss_kernel(cfg.bc.place_sigma, dev)
    files = sorted(glob.glob(os.path.join(ROOT, cfg.bc.data_dirs[0], 'shard_*.npz')))[:a.shards]
    S = {k: 0.0 for k in ('n_act', 'top1_play', 'top1_all', 'n_play', 'le1', 'exact', 'n_noop', 'noop_argmax', 'p_noop_sum', 'p_correct_act_sum',
                          'w_model_plays', 'w_human_plays')}
    vb = {i: [0, 0] for i in range(4)}
    vb_all = [0, 0]
    with torch.no_grad():
        for b in make_loader(files, 'val', cfg, 1024, 0.0, 0, 1, 4):
            b = to_dev(b, dev)
            with torch.autocast('cuda', dtype=torch.float16):
                _l, _p, out, _cm, pm, is_play, cell = losses(net, b, kern, cfg.bc, train=False)
            sel = b['sel'].long()
            w = b['w'].float()
            pol = net.policy_logits(out['card'].float())
            prob = torch.softmax(pol, 1)
            act = sel > 0
            noop = ~act
            raw = out['card'].float().clone()
            raw[:, 0] = -1e9
            S['n_act'] += act.sum().item()
            S['top1_play'] += ((raw.argmax(1) == sel) & act).sum().item()
            S['top1_all'] += ((pol.argmax(1) == sel) & act).sum().item()
            pa = out['place'].float().masked_fill(~pm, -1e9).argmax(1)
            d = torch.maximum((pa // GW - cell // GW).abs(), (pa % GW - cell % GW).abs())
            S['n_play'] += is_play.sum().item()
            S['le1'] += ((d <= 1) & is_play).sum().item()
            S['exact'] += ((d == 0) & is_play).sum().item()
            S['n_noop'] += noop.sum().item()
            S['noop_argmax'] += ((pol.argmax(1) == 0) & noop).sum().item()
            S['p_noop_sum'] += prob[:, 0][noop].sum().item()
            S['p_correct_act_sum'] += prob.gather(1, sel[:, None])[:, 0][act].sum().item()
            S['w_model_plays'] += (w * (1 - prob[:, 0])).sum().item()
            S['w_human_plays'] += (w * act.float()).sum().item()
            r = b['ret'].float()
            ok = (torch.sign(out['value'].float()) == r)
            dec = r != 0
            sec = b['tick'].float() / 20.0
            ph = torch.clamp((sec // 60).long(), max=3)
            for i in range(4):
                m = dec & (ph == i)
                vb[i][0] += (ok & m).sum().item()
                vb[i][1] += m.sum().item()
            vb_all[0] += (ok & dec).sum().item()
            vb_all[1] += dec.sum().item()
    rep = dict(
        ckpt=a.ckpt, step=st.get('step'), val_samples=int(S['n_act'] + S['n_noop']),
        card_top1_play=S['top1_play'] / S['n_act'], card_top1_all_argmax=S['top1_all'] / S['n_act'],
        place_le1=S['le1'] / S['n_play'], place_exact=S['exact'] / S['n_play'],
        noop_recall_argmax=S['noop_argmax'] / S['n_noop'], value_acc=vb_all[0] / max(1, vb_all[1]),
        sampled=dict(mean_p_noop_on_noop_steps=S['p_noop_sum'] / S['n_noop'], mean_p_correct_on_action_steps=S['p_correct_act_sum'] / S['n_act'],
                     play_rate_ratio_model_over_human=S['w_model_plays'] / S['w_human_plays']),
        value_acc_by_phase={k: vb[i][0] / max(1, vb[i][1]) for i, k in enumerate(['0-60s', '60-120s', '120-180s', 'overtime'])},
    )
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
