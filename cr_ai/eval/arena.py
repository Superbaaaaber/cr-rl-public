"""cr_ai/eval/arena.py — 在引擎里评测策略（按分布采样出动作，不用贪心）。

    python -m cr_ai.eval.arena --ckpt runs/bc1/bc_final.pt --opp rule --games 400 [--workers 11] [--deck main|pool] [--stateful]
    python -m cr_ai.eval.arena --ckpt runs/ppo1/ppo_step2000000.pt --stateful --suite rule,bc_main,bc_pool --bc-ckpt runs/bc1/bc_final.pt --step 2000000 --json runs/ppo1/arena.jsonl

对手：rule（规则 AI）、random、bc_main（BC 策略用主角卡组 = “BC 野猪”，镜像）、bc_pool（BC 策略、卡组从对手池按使用量抽）。

agent 一方：--deck main 用主角卡组（对手从对手池按使用量抽），--deck pool 双方都从对手池抽（测陪练）。
两边轮换先后手（玩家 0/1 各一半）。结果：胜/负/平、胜率（平局算半场）、平均冠数、作废率；写到 --json。
--stateful：LSTM 状态逐步传递（PPO 之后的模型）；默认每步零状态（与 BC 训练一致）。
--paired（只用于 --deck pool，2026-10-27）：配对换边——每两局用同一对卡组，第二局双方交换卡组（连形态、塔兵），种子和先后手不变；
  卡组强弱对两边的影响互相抵消，同样局数下胜率的误差更小（期望值不变）。结果多一项 pairs：两局都赢 / 一胜一负 / 两局都输的对数，
  一胜一负 = 这对卡组谁拿到谁赢（卡组运气决定了胜负）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ..config import ROOT, _wrap, cpu_count, load
from ..env import masks as MK
from ..env.vec_env import CTL, VecEnv
from ..model.infer import InferEngine
from ..model.net import Net


class Policy:
    """批量推理：obs（numpy dict，[N, ...]）→ 动作 [N, 2]（sel, cell）。推理走 CUDA Graph（见 model/infer.py）。"""

    def __init__(self, ckpt, device='cuda', stateful=False, temperature=1.0):
        st = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = load(overrides=())
        self.cfg.update(_wrap(st['cfg']))                 # 存档里的 cfg 可能是普通 dict（发布的权重）：转成支持属性访问的配置
        self.net = Net(self.cfg).to(device)
        self.net.load_state_dict(st['model'])
        self.net.eval()
        flat, card_cls = MK.all_masks()
        self.eng = InferEngine(self.net, flat, card_cls, device)
        self.stateful = stateful
        self.T = temperature
        self.state = {}

    def reset(self, key):
        self.state.pop(key, None)

    def act(self, obs, keys, rng=None):
        h = c = None
        if self.stateful:
            H = self.net.H
            h = np.zeros((len(keys), H), np.float32)
            c = np.zeros((len(keys), H), np.float32)
            for i, k in enumerate(keys):
                if k in self.state:
                    h[i], c[i] = self.state[k]
        r = self.eng.run(obs, h, c, self.T)
        if self.stateful:
            for i, k in enumerate(keys):
                self.state[k] = (r['h'][i], r['c'][i])
        return np.stack([r['sel'], r['cell']], 1).astype(np.int32)


def set_cfg(cfg, key, val):
    """cfg['a']['b']['c'] = val（key = 'a.b.c'）。"""
    ks = key.split('.')
    d = cfg
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = val


def paired_episodes(cfg, spec, n, seed):
    """配对评测的对局设定（n 局，补成偶数）：每对第一局照常抽，第二局双方交换卡组、形态和塔兵，种子和控制方（先后手）不变。"""
    from ..env.episodes import EpisodeSampler
    smp = EpisodeSampler(cfg, spec, seed=seed)
    out = []
    for _ in range((n + 1) // 2):
        e = smp.sample()
        sw = dict(e)
        for k in ('decks', 'forms', 'deck_names', 'tower_troop'):
            sw[k] = [e[k][1], e[k][0]]
        out += [e, sw]
    return out


def pair_stats(outcome, n_pairs):
    """outcome：局号 → 被测方得分（1 / 0.5 / 0）。两局都打完的对：→ {n, both_win, split, both_loss}。"""
    s = [outcome[2 * k] + outcome[2 * k + 1] for k in range(n_pairs) if 2 * k in outcome and 2 * k + 1 in outcome]
    return dict(n=len(s), both_win=sum(x == 2 for x in s), split=sum(0 < x < 2 for x in s), both_loss=sum(x == 0 for x in s))


def run(ckpt, opp='rule', games=200, workers=8, deck='main', stateful=False, seed=12345, verbose=True, bc_ckpt=None, temperature=1.0,
        opp_stateful=False, env=None, paired=False):
    """env：{'paths.engine_base': ..., 'paths.deck_pool': ...}——对局环境的配置覆盖（默认按被测存档自己的配置：
    哪个引擎快照、哪个卡组池）。固定考官要在固定环境里比，才能跨时间比较。"""
    pol = Policy(ckpt, stateful=stateful, temperature=temperature)
    # --bc-ckpt 也可以是 PPO 存档（新旧模型直接对打），这时对手要 opp_stateful（带 LSTM 状态，和训练时一样）
    opp_pol = Policy(bc_ckpt, stateful=opp_stateful) if opp.startswith('bc') else None
    cfg = pol.cfg
    for k, v in (env or {}).items():
        set_cfg(cfg, k, v)
    cfg.vec_env.workers = workers
    if opp_pol is not None and 'ent_id' in opp_pol.eng.keys:
        cfg.obs.entities = True                                       # 对手的网络要单位列表：环境也要出（被测网络不用就忽略）
    if opp_pol is not None and 'ab_ext' in opp_pol.eng.keys:
        cfg.obs.ext = True                                            # 对手的网络用扩展观测（model.obs_ext）：同上
    if opp_pol is not None:                                           # 对手存档开过的观测修补，环境也照开（被测网络没开时只多看到一些法术）
        for k in ('show_effects', 'king_shots'):
            if opp_pol.cfg.obs.get(k):
                cfg.obs[k] = True
    octl = 'bc' if opp.startswith('bc') else opp
    spec = dict(controllers=[['agent', octl, 1], [octl, 'agent', 1]], main_for='agent' if deck == 'main' else None, main_deck_prob=0.0,
                opp_main_prob=1.0 if opp == 'bc_main' else 0.0)
    eps = None
    if paired:
        if deck != 'pool':
            raise ValueError('配对换边只用于 deck=pool（双方都从卡组池抽）')
        eps = paired_episodes(cfg, spec, max(games, workers), seed + 1)
    venv = VecEnv(cfg, n_workers=workers, episode_spec=spec, seed=seed)
    rng = np.random.RandomState(seed)
    obs = venv.reset()
    gi = np.arange(workers)                               # 各环境正在打第几局（配对统计用）
    outcome = {}
    if eps is not None:
        for w in range(workers):
            venv.set_episode(w, eps[w])
    res = dict(win=0, loss=0, draw=0, error=0, crowns_for=0, crowns_against=0, void=0, plays=0, by_side=[[0, 0], [0, 0]], games=0)
    t0 = time.time()
    epi = np.zeros(workers, np.int64)
    started = workers                                     # 已开局数（含正在进行的）
    active = np.ones(workers, bool)
    # 流水线：worker 分两组，一组在 CPU 上推进引擎时，另一组在 GPU 上推理（CPU 和 GPU 不互相等待）
    groups = [list(range(0, workers, 2)), list(range(1, workers, 2))] if workers > 1 else [[0]]

    def launch(ids):
        ids = [w for w in ids if active[w]]
        if not ids:
            return ids
        am = venv.agent_mask
        idx = [(w, p) for w in ids for p in (0, 1) if am[w, p]]
        acts = np.zeros((workers, 2, 2), np.int32)
        if idx:
            sub = {k: np.stack([obs[k][w, p] for w, p in idx]) for k in pol.eng.keys if k in obs}
            a = pol.act(sub, [(w, int(epi[w]), p) for w, p in idx], rng)
            for i, (w, p) in enumerate(idx):
                acts[w, p] = a[i]
        if opp_pol is not None:
            ctl = venv.ctl
            bidx = [(w, p) for w in ids for p in (0, 1) if ctl[w, p] == CTL['bc']]
            if bidx:
                sub = {k: np.stack([obs[k][w, p] for w, p in bidx]) for k in opp_pol.eng.keys if k in obs}
                a = opp_pol.act(sub, [(w, int(epi[w]), p) for w, p in bidx])
                for i, (w, p) in enumerate(bidx):
                    acts[w, p] = a[i]
        venv.step_async(ids, acts[ids])
        return ids

    def collect(ids, infos):
        nonlocal started
        for w, inf in zip(ids, infos):
            if inf is None:
                continue
            for p in (0, 1):
                pol.reset((w, int(epi[w]), p))
                if opp_pol is not None:
                    opp_pol.reset((w, int(epi[w]), p))
            epi[w] += 1
            if inf.get('error'):
                res['error'] += 1
            else:
                ag = [p for p in (0, 1) if inf['controllers'][p] == 'agent'][0]
                win = inf['winner']
                key = 'draw' if win in (None, -1) else ('win' if win == ag else 'loss')
                res[key] += 1
                outcome[int(gi[w])] = {'win': 1.0, 'loss': 0.0, 'draw': 0.5}[key]
                res['by_side'][ag][0 if key == 'win' else 1] += 1 if key != 'draw' else 0
                cr = inf.get('crowns') or [0, 0]
                res['crowns_for'] += cr[ag]
                res['crowns_against'] += cr[1 - ag]
                st = inf.get('stats') or {}
                res['plays'] += (st.get('plays') or [0, 0])[ag]
                res['void'] += (st.get('void') or [0, 0])[ag]
                res['games'] += 1
            if started >= games:
                active[w] = False                         # 这个 worker 已自动开了新局，不再推进（不偏向短局）
            else:
                if eps is not None:
                    venv.set_episode(w, eps[started])     # 换成配对表里的下一局（worker 自动开的那局作废）
                gi[w] = started
                started += 1
            if verbose and res['games'] % 20 == 0 and res['games']:
                print('  %d 局 胜 %d 负 %d 平 %d（%.0fs）' % (res['games'], res['win'], res['loss'], res['draw'], time.time() - t0), flush=True)

    inflight = [launch(g) for g in groups]
    while any(inflight):
        for gi_, g in enumerate(groups):
            ids = inflight[gi_]
            if ids:
                collect(ids, venv.step_wait(ids))
            inflight[gi_] = launch(g)
    venv.close()
    n = max(1, res['games'])
    res.update(win_rate=(res['win'] + 0.5 * res['draw']) / n, avg_crowns_for=res['crowns_for'] / n, avg_crowns_against=res['crowns_against'] / n,
               plays_per_game=res['plays'] / n, void_rate=res['void'] / max(1, res['plays'] + res['void']), secs=time.time() - t0,
               ckpt=ckpt, opp=opp, deck=deck, stateful=stateful, temperature=temperature)
    if opp_stateful:
        res.update(opp_ckpt=bc_ckpt, opp_stateful=True)
    if eps is not None:
        res.update(paired=True, pairs=pair_stats(outcome, len(eps) // 2))
    del pol, opp_pol
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--opp', default='rule', choices=['rule', 'random', 'bc_main', 'bc_pool'])
    ap.add_argument('--suite', default=None, help='逗号分隔的一组对手，依次各打 --games 局（覆盖 --opp）')
    ap.add_argument('--bc-ckpt', default=os.path.join(ROOT, 'runs', 'bc1', 'bc_final.pt'))
    ap.add_argument('--step', type=int, default=None, help='训练步数（写进结果，供看板画曲线）')
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    ap.add_argument('--deck', default='main', choices=['main', 'pool'])
    ap.add_argument('--stateful', action='store_true')
    ap.add_argument('--opp-stateful', action='store_true', help='对手（--bc-ckpt）也带 LSTM 状态：--bc-ckpt 是 PPO 存档时用')
    ap.add_argument('--temperature', type=float, default=1.0, help='被测策略的采样温度（对手 BC 固定 1.0）；<1 更接近贪心，但仍是采样')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--paired', action='store_true', help='配对换边（只用于 --deck pool）：同一对卡组换边各打一局')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    for item in (a.suite.split(',') if a.suite else [a.opp]):
        deck, opp = item.split(':') if ':' in item else (a.deck, item)          # 'pool:bc_pool'：agent 也从对手池抽卡组（测其他卡组）
        r = run(a.ckpt, opp, a.games, a.workers, deck, a.stateful, a.seed, bc_ckpt=a.bc_ckpt, temperature=a.temperature,
                opp_stateful=a.opp_stateful, paired=a.paired)
        if deck != 'main':
            r['opp'] = '%s:%s' % (deck, opp)
        if a.step is not None:
            r['step'] = a.step
        print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)
        if a.json:
            with open(a.json, 'a', encoding='utf-8') as f:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, ROOT)
    sys.setrecursionlimit(20000)
    main()
