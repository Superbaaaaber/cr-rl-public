"""cr_ai/tools/gauntlet.py — 固定“考官”评测 + 离线 Bradley–Terry Elo（带置信区间）。

BC 评测已经饱和（墓园镜像 97–98%，200 局的误差 ±4–5%），看不出进步。改成：一组固定的冻结存档当考官，
被测存档对每个考官打 N 局（arena 两边轮换先后手、考官带 LSTM 状态），结果累积到一个 jsonl；
再把所有对局一起用 Bradley–Terry 拟合，每个存档一个分数（以第一个考官为 0），给出 95% 置信区间。

    # 打：结果追加到 --out，并把拟合结果写到同目录的 gauntlet_elo.json
    python -m cr_ai.tools.gauntlet play --ckpt runs/ppo_b/ppo_step8000000.pt --examiners cr_ai/configs/examiners.yaml --games 200 --out runs/ppo_b/gauntlet.jsonl --step 8000000
    # 只拟合（可给多个文件；--anchor 指定定为 0 分的存档，默认第一个考官）
    python -m cr_ai.tools.gauntlet fit runs/ppo_b/gauntlet.jsonl

考官文件（YAML）：examiners: [{name, ckpt, opp, pool?, deck?, env?, elo?, games?}]；opp = bc_main（考官用主卡组，镜像）| bc_pool（考官从对手池按使用量抽卡组，剥削者就是这么训的）。
  pool：考官改用这个卡组池（opp 要是 bc_pool）——专项评测：同一个考官拿主流卡组 / 亡灵巨人卡组 / 随机卡组分别打，
        和主流卡组比胜率高出多少，就是被测模型在这类卡组上的缺口；
  env：对局环境的配置覆盖（如 paths.engine_base），文件顶层的 env 对所有考官生效、条目里的覆盖顶层；
  deck：被测方用什么卡组——main（默认，主卡组）| pool（被测方也从卡组池抽：和 pool 一起用就是“双方都用随机卡组”，测通用性，2026-10-24）；
  elo：是否参加 Bradley–Terry 拟合（默认：没给 pool、deck 是 main 的参加；专项评测不参加，只看胜率）；
  games：这个考官打多少局（不给 = 命令行的 --games）。2026-10-21：100 局的 95% 区间约 ±10 个点，主要考官改成 400 局。
  paired：配对换边（只能和 deck: pool 一起用，2026-10-27）：同一对卡组换边各打一局，抵消卡组运气；结果行带 paired 和 pairs（见 arena）。
  专项评测（给了 pool 或 deck 不是 main）的对手身份 = 存档 + '@' + 考官名（和同一个存档的普通考官、别的专项分开记）。
缺失的考官存档跳过并提示（不中断）。不同架构的存档可以直接对打（arena 按各自存档里的配置建网）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from ..config import ROOT, read_yaml, repo_path

ELO = 400.0 / math.log(10.0)


def player_name(path):
    """存档的身份：相对仓库根目录的路径（不同文件里同一个存档对得上）。"""
    p = os.path.abspath(repo_path(path))
    try:
        return os.path.relpath(p, ROOT).replace('\\', '/')
    except ValueError:
        return p.replace('\\', '/')


def load_examiners(path):
    y = read_yaml(path)
    top_env = dict(y.get('env') or {})
    out = []
    for e in y.get('examiners') or []:
        env = dict(top_env, **(e.get('env') or {}))
        if e.get('pool'):
            env['paths.deck_pool'] = e['pool']
        deck = e.get('deck') or 'main'
        if deck not in ('main', 'pool'):
            raise ValueError('考官 %s：deck 只能是 main / pool（%s）' % (e.get('name'), deck))
        paired = bool(e.get('paired', False))
        if paired and deck != 'pool':
            raise ValueError('考官 %s：paired 只能和 deck: pool 一起用' % e.get('name'))
        special = bool(e.get('pool')) or deck != 'main'
        out.append(dict(name=e.get('name') or player_name(e['ckpt']), ckpt=e['ckpt'], opp=e.get('opp', 'bc_main'), pool=e.get('pool'), deck=deck,
                        env=env, elo=bool(e.get('elo', not special)), games=e.get('games'), paired=paired))
    return out


def play(ckpt, examiners, games, workers, out, step=None, seed=12345, temperature=1.0):
    from ..eval.arena import run
    me = player_name(ckpt)
    rows = []
    for ex in examiners:
        if not os.path.exists(repo_path(ex['ckpt'])):
            print('考官存档不存在，跳过：%s' % ex['ckpt'], flush=True)
            continue
        deck = ex.get('deck') or 'main'
        paired = bool(ex.get('paired'))
        r = run(repo_path(ckpt), ex['opp'], int(ex.get('games') or games), workers, deck, True, seed, verbose=False, bc_ckpt=repo_path(ex['ckpt']),
                temperature=temperature, opp_stateful=True, env=ex.get('env'), paired=paired)
        special = bool(ex.get('pool')) or deck != 'main'
        b = player_name(ex['ckpt']) + ('' if not special else '@' + ex['name'])
        row = dict(a=me, b=b, b_name=ex['name'], opp=ex['opp'], win=r['win'], loss=r['loss'], draw=r['draw'],
                   games=r['games'], error=r['error'], win_rate=r['win_rate'], step=step, secs=r['secs'])
        if special or not ex.get('elo', True):
            row.update(pool=ex.get('pool'), elo=bool(ex.get('elo', True)))
        if deck != 'main':
            row['deck'] = deck
        if paired:
            row.update(paired=True, pairs=r.get('pairs'))
        if ex.get('env'):
            row['env'] = ex['env']
        rows.append(row)
        pr = r.get('pairs') or {}
        print('%s 对 %s（%s%s%s）：%d 胜 %d 负 %d 平，胜率 %.3f（%.0f 秒）%s' % (
            me, ex['name'], ex['opp'], '' if deck == 'main' else '，被测方也抽卡组', '，配对换边' if paired else '',
            r['win'], r['loss'], r['draw'], r['win_rate'], r['secs'],
            '；%d 对：两局都赢 %d、一胜一负 %d、两局都输 %d' % (pr['n'], pr['both_win'], pr['split'], pr['both_loss']) if pr else ''), flush=True)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows


def read_rows(files):
    rows = []
    for fn in files:
        with open(fn, encoding='utf-8') as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def fit(rows, anchor=None, prior=0.01, iters=100):
    """Bradley–Terry：P(i 胜 j) = σ(r_i − r_j)，平局算半胜。带很弱的 L2 先验（prior，单位：logit⁻²），
    免得全胜/全负的存档分数发散。→ {存档: (Elo, 95% 半宽, 局数)}，anchor 定为 0（默认第一行的考官）。"""
    names = []
    for r in rows:
        for k in ('b', 'a'):
            if r[k] not in names:
                names.append(r[k])
    if not names:
        return {}
    anchor = anchor or rows[0]['b']
    idx = {n: i for i, n in enumerate(names)}
    P = len(names)
    I = np.array([idx[r['a']] for r in rows])
    J = np.array([idx[r['b']] for r in rows])
    N = np.array([r['win'] + r['loss'] + r['draw'] for r in rows], np.float64)
    S = np.array([r['win'] + 0.5 * r['draw'] for r in rows], np.float64)
    a = idx[anchor]
    free = np.array([i for i in range(P) if i != a])
    x = np.zeros(P)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x[I] - x[J])))
        g = np.zeros(P)
        np.add.at(g, I, S - N * p)
        np.add.at(g, J, -(S - N * p))
        g -= prior * x
        w = N * p * (1 - p)
        H = np.zeros((P, P))
        np.add.at(H, (I, I), w)
        np.add.at(H, (J, J), w)
        np.add.at(H, (I, J), -w)
        np.add.at(H, (J, I), -w)
        H += prior * np.eye(P)
        step = np.linalg.solve(H[np.ix_(free, free)], g[free])
        x[free] += step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(H[np.ix_(free, free)])
    se = np.zeros(P)
    se[free] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    games = np.zeros(P)
    np.add.at(games, I, N)
    np.add.at(games, J, N)
    return {n: (float(ELO * (x[i] - x[a])), float(1.96 * ELO * se[i]), int(games[i])) for n, i in idx.items()}


def report(res, anchor):
    print('Bradley–Terry Elo（%s = 0，±95%%）：' % anchor)
    for n, (e, ci, g) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print('  %+7.0f ± %4.0f  %5d 局  %s' % (e, ci, g, n))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('play')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--examiners', default='cr_ai/configs/examiners.yaml')
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--step', type=int, default=None)
    p.add_argument('--seed', type=int, default=12345)
    f = sub.add_parser('fit')
    f.add_argument('files', nargs='+')
    f.add_argument('--anchor', default=None)
    a = ap.parse_args()
    if a.cmd == 'play':
        play(a.ckpt, load_examiners(a.examiners), a.games, a.workers, a.out, a.step, a.seed)
        files = [a.out]
    else:
        files = a.files
    rows = [r for r in read_rows(files) if r.get('elo', True)]          # 专项评测（elo: false）不参加拟合，只看胜率
    if not rows:
        print('没有对局记录')
        return
    anchor = getattr(a, 'anchor', None) or rows[0]['b']
    res = fit(rows, anchor)
    report(res, anchor)
    out = os.path.join(os.path.dirname(os.path.abspath(files[0])), 'gauntlet_elo.json')
    with open(out, 'w', encoding='utf-8') as fo:
        json.dump(dict(anchor=anchor, elo={n: dict(elo=e, ci95=ci, games=g) for n, (e, ci, g) in res.items()}), fo, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.path.insert(0, ROOT)
    sys.setrecursionlimit(20000)
    main()
