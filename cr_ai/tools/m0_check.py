"""M0 验收：单进程速度、随机策略双方 N 局无报错（多进程）、规则 AI 对随机策略的胜率。

    .venv\\Scripts\\python.exe -m cr_ai.tools.m0_check --workers 6 --games 1000 --rule-games 200
结果写到 runs/m0/m0_report.json。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from cr_ai.config import ROOT, load


def bench_single(cfg, games):
    from cr_ai.bots.random_bot import RandomBot
    from cr_ai.env.episodes import EpisodeSampler
    from cr_ai.env.game_env import GameEnv
    env = GameEnv(cfg)
    sp = EpisodeSampler(cfg, {}, seed=123)
    env.reset(**{k: v for k, v in sp.sample().items() if k in ('seed', 'decks', 'forms', 'tower_troop')})   # 预热（装载引擎）
    steps, t0 = 0, time.time()
    for g in range(games):
        ep = sp.sample()
        obs = env.reset(ep['seed'], ep['decks'], ep['forms'], tower_troop=ep['tower_troop'])
        bots = [RandomBot(np.random.RandomState(g * 2 + p), 0.1) for p in (0, 1)]
        done = False
        while not done:
            obs, rew, done, info = env.step([bots[p].act_obs(obs[p]) for p in (0, 1)])
            steps += 1
    dt = time.time() - t0
    return dict(games=games, steps=steps, secs=dt, games_per_s=games / dt, steps_per_s=steps / dt)


def run_vec(cfg, workers, games, spec, label):
    from cr_ai.env.vec_env import VecEnv
    venv = VecEnv(cfg, n_workers=workers, episode_spec=spec)
    venv.reset()
    W = venv.W
    acts = np.zeros((W, 2, 2), np.int32)
    eps, steps, t0 = [], 0, time.time()
    last = t0
    while len(eps) < games:
        _o, _r, _d, infos = venv.step(acts)
        steps += W
        eps += [i for i in infos if i]
        if time.time() - last > 30:
            last = time.time()
            print('  [%s] %d/%d 局，%.0f 步/秒' % (label, len(eps), games, steps / (time.time() - t0)), flush=True)
    dt = time.time() - t0
    venv.close()
    return eps[:games], dict(secs=dt, env_steps=steps, env_steps_per_s=steps / dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--rule-games', type=int, default=200)
    ap.add_argument('--bench-games', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'm0'))
    a = ap.parse_args()
    sys.setrecursionlimit(20000)
    cfg = load()
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    # 1. 单进程速度（含双方观测编码）
    b = bench_single(cfg, a.bench_games)
    rep['single_process'] = b
    print('单进程：%d 局 %.1fs → %.2f 局/秒、%.0f 步/秒（含双方观测编码）' % (b['games'], b['secs'], b['games_per_s'], b['steps_per_s']), flush=True)
    # 2. 随机策略双方 N 局
    eps, sp = run_vec(cfg, a.workers, a.games, dict(controllers=[['random', 'random', 1]], main_deck_prob=0.5), 'random')
    errs = [e for e in eps if e.get('error')]
    ok = [e for e in eps if not e.get('error')]
    void = sum(sum(e['stats']['void']) for e in ok)
    plays = sum(sum(e['stats']['plays']) for e in ok)
    rep['random_vs_random'] = dict(games=len(eps), errors=len(errs), error_samples=[e['error'] for e in errs[:5]],
                                   avg_steps=float(np.mean([e['steps'] for e in ok])), avg_plays_per_side=plays / max(1, 2 * len(ok)),
                                   void=void, void_rate=void / max(1, void + plays), winner_dist={str(k): sum(e['winner'] == k for e in ok) for k in (-1, 0, 1)}, **sp)
    print('随机 vs 随机：%d 局，报错 %d，平均 %.0f 步/局，每方平均出牌 %.1f，作废 %d（%.2f%%），%.0f 环境步/秒（%d 进程）' % (
        len(eps), len(errs), rep['random_vs_random']['avg_steps'], rep['random_vs_random']['avg_plays_per_side'], void,
        100 * rep['random_vs_random']['void_rate'], sp['env_steps_per_s'], a.workers), flush=True)
    # 3. 规则 AI 对随机策略
    eps, sp = run_vec(cfg, a.workers, a.rule_games, dict(controllers=[['rule', 'random', 1], ['random', 'rule', 1]], main_deck_prob=0.5), 'rule')
    ok = [e for e in eps if not e.get('error')]
    side = [e['controllers'].index('rule') for e in ok]
    win = sum(e['winner'] == s for e, s in zip(ok, side))
    draw = sum(e['winner'] == -1 for e in ok)
    crowns = np.mean([e['crowns'][s] for e, s in zip(ok, side)]), np.mean([e['crowns'][1 - s] for e, s in zip(ok, side)])
    rep['rule_vs_random'] = dict(games=len(ok), errors=len(eps) - len(ok), win=win, draw=draw, winrate=win / max(1, len(ok)),
                                 crowns=[float(crowns[0]), float(crowns[1])], **sp)
    print('规则 AI vs 随机：%d 局，胜 %d 平 %d（胜率 %.1f%%），平均冠 %.2f : %.2f' % (len(ok), win, draw, 100 * win / max(1, len(ok)), crowns[0], crowns[1]), flush=True)
    json.dump(rep, open(os.path.join(a.out, 'm0_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
