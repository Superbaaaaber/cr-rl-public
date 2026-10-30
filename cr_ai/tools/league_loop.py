"""cr_ai/tools/league_loop.py — 主模型和剥削者分时训练（同一台机器、同一组环境进程轮流用）。

    python tools/lowprio.py <py> -u -m cr_ai.tools.league_loop [--start exploit]

一轮 = 主模型训 --main-steps 步 → 剥削者训 --exp-steps 步：
  - 剥削者每一代从主模型当时的 ppo_latest 出发（模型 + 优化器状态），对手是主模型的冻结副本（每轮刷新成主模型的最新版本）；
  - 每轮结束看剥削者最后一段的胜率：≥ --add-min 就把它的模型放进主模型联赛的 exploiter_dir（主模型的 PFSP 会多选它）；
  - 胜率 ≥ --success，或这一代已经训了 --gen-max 步，就开新的一代（重新从主模型出发，找新的漏洞）。
每一段都套 cr_ai.tools.supervise（崩溃自动续训）。状态存在 runs/<exp-prefix>_state.json，中断后重跑接着来。

大模型（2026-10-20）：主模型的配置和覆盖从外面给——--main-config（配置文件，依次叠加）、--overrides-file（每行一个 a.b=值，
主模型和剥削者共用：模型尺寸、引擎快照、卡组池、BC……）、--main-overrides-file（只给主模型：联赛的剥削者目录、卡组 PFSP、考官……）；
剥削者在主模型配置上再叠 --exp-config（默认 exploit.yaml）。

2026-10-20 修正（第一轮实测：考官评测一次没跑成；剥削者对主模型只有 23% 胜率，一个都没入选）：
  - 固定考官评测改由这里在每段主模型结束后启动（--gauntlet-examiners，后台跑、和剥削者那段并行，上一次没跑完就跳过）；
    ppo 自己的考官评测要关掉（gauntlet_every_steps=0）——它要等普通评测跑完才开始，而一段主模型只有两小时，总轮不上；
  - 剥削者入选看“进步了多少”：起点胜率 = 这一代训练日志最早几批的胜率（剥削者用对手池卡组、主模型用主卡组，起点本来就只有 20–40%），
    这一段后半段比起点高 --add-gain 就入选（或绝对胜率 ≥ --add-min），高 --success-gain 就开新的一代。
2026-10-20 专项剥削者（--exp-variants，按代轮换）：比如一代只用亡灵巨人卡组、一代只用新版本卡组，专练用这些卡打主模型；
  入选时 .json 记下它的卡组池（deck_pool），主模型联赛里轮到它当对手时，这局它一方换成这个池里的卡组（rollout 的 redeck）。
2026-10-21 --no-exploit：用户决定关掉剥削者（两代都没入选），改 OpenAI Five 式 80% 自我对弈 + 20% 历史快照；
  这里只剩“每段主模型 → 后台考官评测”的循环。
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch

from ..config import ROOT, repo_path

MAIN_CFG = ['cr_ai/configs/ppo.yaml', 'cr_ai/configs/m4.yaml', 'cr_ai/configs/m4_gy.yaml']


def log(msg, f):
    line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    f.write(line + '\n')
    f.flush()


def ckpt_step(path):
    if not os.path.exists(path):
        return None
    return int(torch.load(path, map_location='cpu', weights_only=False)['step'])


def run_phase(cmd, logfile, flog):
    log('启动：%s' % ' '.join(cmd[cmd.index('cr_ai.train.ppo'):]), flog)
    with open(logfile, 'a', encoding='utf-8') as out:
        out.write('==== league_loop %s ====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
        out.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    log('结束，返回码 %d' % rc, flog)
    return rc


def phase_winrate(train_log, since_step, tail_frac=0.5):
    """这一段（step > since_step）后半段的对主模型胜率（按局数加权）。"""
    rows = [json.loads(l) for l in open(train_log, encoding='utf-8')] if os.path.exists(train_log) else []
    rows = [r for r in rows if r['step'] > since_step and r.get('n_hist')]
    rows = rows[int(len(rows) * (1 - tail_frac)):]
    n = sum(r['n_hist'] for r in rows)
    if n == 0:
        return None, 0
    return sum(r['win_hist'] * r['n_hist'] for r in rows if r.get('win_hist') is not None) / n, n


def gen_base_winrate(train_log, frac=0.05, min_rows=3):
    """这一代的起点胜率：训练日志最早 frac（至少 min_rows 批）的对主模型胜率（按局数加权）。"""
    rows = [json.loads(l) for l in open(train_log, encoding='utf-8')] if os.path.exists(train_log) else []
    rows = sorted((r for r in rows if r.get('n_hist')), key=lambda r: r['step'])
    rows = rows[:max(min_rows, int(len(rows) * frac))]
    n = sum(r['n_hist'] for r in rows)
    if n == 0:
        return None
    return sum(r['win_hist'] * r['n_hist'] for r in rows if r.get('win_hist') is not None) / n


def launch_gauntlet(py, main_run, examiners, games, workers, flog):
    """主模型这一段的终点存一个只含模型的快照，后台跑固定考官评测（结果追加到 <main_run>/gauntlet.jsonl）。"""
    st = torch.load(os.path.join(main_run, 'ppo_latest.pt'), map_location='cpu', weights_only=False)
    step = int(st['step'])
    snap = os.path.join(main_run, 'ppo_step%d.pt' % step)
    if not os.path.exists(snap):
        torch.save(dict(model=st['model'], cfg=st['cfg'], step=st['step'], update=st['update'], kind='ppo'), snap + '.tmp')
        os.replace(snap + '.tmp', snap)
    del st
    cmd = [py, os.path.join(ROOT, 'tools', 'lowprio.py'), py, '-u', '-m', 'cr_ai.tools.gauntlet', 'play', '--ckpt', snap,
           '--examiners', examiners, '--games', str(games), '--workers', str(workers), '--out', os.path.join(main_run, 'gauntlet.jsonl'),
           '--step', str(step)]
    out = open(os.path.join(main_run, 'gauntlet_%d.log' % step), 'w', encoding='utf-8')
    log('固定考官评测（后台）：%s 步' % step, flog)
    return subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))


def read_overrides(path):
    if not path:
        return []
    return [l.strip() for l in open(repo_path(path), encoding='utf-8') if l.strip() and not l.strip().startswith('#')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main-run', default='runs/ppo3')
    ap.add_argument('--main-init', default='runs/ppo2/ppo_latest.pt')
    ap.add_argument('--main-config', nargs='+', default=MAIN_CFG)
    ap.add_argument('--exp-config', nargs='+', default=['cr_ai/configs/exploit.yaml'])
    ap.add_argument('--overrides-file', default=None, help='主模型和剥削者共用的配置覆盖（每行一个 a.b=值）')
    ap.add_argument('--main-overrides-file', default=None, help='只给主模型的配置覆盖')
    ap.add_argument('--main-log', default='logs/ppo3.log')
    ap.add_argument('--exp-log', default='logs/exp.log')
    ap.add_argument('--exp-prefix', default='runs/exp', help='每一代的目录是 <prefix>_g<代号>')
    ap.add_argument('--main-steps', type=float, default=12e6)
    ap.add_argument('--exp-steps', type=float, default=4e6)
    ap.add_argument('--gen-max', type=float, default=16e6, help='一代最多训多少步（没达到 --success 也换新一代）')
    ap.add_argument('--success', type=float, default=0.70)
    ap.add_argument('--add-min', type=float, default=0.55)
    ap.add_argument('--add-gain', type=float, default=None, help='后半段胜率比这一代起点高这么多就入选（不给 = 只看 --add-min）')
    ap.add_argument('--success-gain', type=float, default=None, help='比起点高这么多就开新的一代')
    ap.add_argument('--gauntlet-examiners', default=None, help='每段主模型结束后跑固定考官评测（考官文件）；不给 = 不跑')
    ap.add_argument('--gauntlet-games', type=int, default=100)
    ap.add_argument('--gauntlet-workers', type=int, default=6)
    ap.add_argument('--exp-variants', nargs='+', default=None,
                    help='剥削者按代轮换的配置覆盖文件：第 g 代用第 g %% n 个。文件里有 paths.deck_pool= 的是专项剥削者——'
                         '入选时把这个池记进 .json，主模型联赛里轮到它当对手时只用这个池的卡组')
    ap.add_argument('--no-exploit', action='store_true',
                    help='不训剥削者：每段主模型结束后（跑完考官评测的启动）直接接下一段主模型（2026-10-21 起用这个）')
    ap.add_argument('--start', choices=['main', 'exploit'], default='main')
    ap.add_argument('--rounds', type=int, default=0, help='跑几轮就停（0 = 一直跑到主模型的 total_steps）')
    a = ap.parse_args()
    py = sys.executable
    sup =[py, '-u', '-m', 'cr_ai.tools.supervise', '--']
    main_run, exp_dir = repo_path(a.main_run), None
    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    flog = open(os.path.join(ROOT, 'logs', 'league_loop.log'), 'a', encoding='utf-8')
    stf = repo_path(a.exp_prefix + '_state.json')
    state = json.load(open(stf, encoding='utf-8')) if os.path.exists(stf) else dict(gen=0, gen_steps=0)
    inbox = os.path.join(main_run, 'exploiters')
    os.makedirs(inbox, exist_ok=True)
    total = None
    gaunt = None
    phase = a.start
    rounds = 0
    while True:
        # 每段开始时重读配置覆盖文件：改 runs/m4_*.txt 不用重启 league_loop，下一段生效
        common = read_overrides(a.overrides_file)
        main_ov = common + read_overrides(a.main_overrides_file)
        if phase == 'main':
            s = ckpt_step(os.path.join(main_run, 'ppo_latest.pt'))
            from ..config import load
            total = int(load(*a.main_config, overrides=main_ov).ppo.total_steps)
            if s is not None and s >= total:
                log('主模型已到 total_steps（%d），结束' % total, flog)
                break
            stop = min(total, int((s or 0) + a.main_steps))
            rc = run_phase(sup + [py, '-u', '-m', 'cr_ai.train.ppo', '--config', *a.main_config,
                                  '--run', a.main_run, '--init', a.main_init, '--resume', *main_ov, 'ppo.total_steps=%d' % stop],
                           repo_path(a.main_log), flog)
            if rc != 0:
                log('主模型这一段异常结束，停下等人看', flog)
                break
            if a.gauntlet_examiners:
                if gaunt is not None and gaunt.poll() is None:
                    log('上一次固定考官评测还没跑完，这次跳过', flog)
                else:
                    gaunt = launch_gauntlet(py, main_run, a.gauntlet_examiners, a.gauntlet_games, a.gauntlet_workers, flog)
            rounds += 1
            if a.rounds and rounds >= a.rounds:
                break
            phase = 'main' if a.no_exploit else 'exploit'
            continue

        # ---- 剥削者 ----
        g = state['gen']
        var = a.exp_variants[g % len(a.exp_variants)] if a.exp_variants else None
        var_ov = read_overrides(var)
        var_pool = next((o.split('=', 1)[1] for o in var_ov if o.startswith('paths.deck_pool=')), None)
        gdir = '%s_g%d' % (a.exp_prefix, g)
        gpath = repo_path(gdir)
        os.makedirs(gpath, exist_ok=True)
        main_latest = os.path.join(main_run, 'ppo_latest.pt')
        ms = ckpt_step(main_latest)
        # 对手：主模型当前版本的冻结副本（只存模型，采样进程每次换对手都要读它）
        st = torch.load(main_latest, map_location='cpu', weights_only=False)
        tgt = os.path.join(gpath, 'target_step%d.pt' % ms)
        if not os.path.exists(tgt):
            torch.save(dict(model=st['model'], cfg=st['cfg'], step=st['step'], update=st['update'], kind='ppo'), tgt)
        del st
        es = ckpt_step(os.path.join(gpath, 'ppo_latest.pt'))
        start = es if es is not None else ms
        stop = int(start + a.exp_steps)
        log('剥削者第 %d 代（%s；已训 %.1fM 步），对手 = 主模型 %.1fM 步' % (
            g, os.path.basename(var) if var else '默认', state['gen_steps'] / 1e6, ms / 1e6), flog)
        rc = run_phase(sup + [py, '-u', '-m', 'cr_ai.train.ppo', '--config', *a.main_config, *a.exp_config,
                              '--run', gdir, '--init', a.main_run + '/ppo_latest.pt', '--resume', '--no-eval', *common, *var_ov,
                              'ppo.total_steps=%d' % stop, 'league.target=%s' % os.path.relpath(tgt, ROOT).replace('\\', '/')],
                       repo_path(a.exp_log), flog)
        if rc != 0:
            log('剥削者这一段异常结束，停下等人看', flog)
            break
        wr, n = phase_winrate(os.path.join(gpath, 'train_log.jsonl'), start)
        base = gen_base_winrate(os.path.join(gpath, 'train_log.jsonl'))
        gain = None if wr is None or base is None else wr - base
        state['gen_steps'] += stop - start
        log('剥削者第 %d 代这一段：对主模型胜率 %s（后半段 %d 局；这一代起点 %s，进步 %s）' % (
            g, 'None' if wr is None else '%.3f' % wr, n, 'None' if base is None else '%.3f' % base, 'None' if gain is None else '%+.3f' % gain), flog)
        added = wr is not None and (wr >= a.add_min or (a.add_gain is not None and gain is not None and gain >= a.add_gain))
        if added:
            es2 = ckpt_step(os.path.join(gpath, 'ppo_latest.pt'))
            name = 'exp_g%d_step%d' % (g, es2)
            sd = torch.load(os.path.join(gpath, 'ppo_latest.pt'), map_location='cpu', weights_only=False)
            torch.save(dict(model=sd['model'], cfg=sd['cfg'], step=sd['step'], update=sd['update'], kind='ppo'), os.path.join(inbox, name + '.pt.tmp'))
            os.replace(os.path.join(inbox, name + '.pt.tmp'), os.path.join(inbox, name + '.pt'))
            json.dump(dict(gen=g, step=es2, winrate=wr, base_winrate=base, games=n, target_step=ms, created=time.time(),
                           variant=os.path.basename(var) if var else None, deck_pool=var_pool),
                      open(os.path.join(inbox, name + '.json'), 'w', encoding='utf-8'))
            log('  → 加入主模型联赛：%s' % name, flog)
        succeeded = wr is not None and (wr >= a.success or (a.success_gain is not None and gain is not None and gain >= a.success_gain))
        if succeeded or state['gen_steps'] >= a.gen_max:
            state = dict(gen=g + 1, gen_steps=0)
            log('  → 开新的一代（第 %d 代）' % state['gen'], flog)
        json.dump(state, open(stf, 'w', encoding='utf-8'))
        phase = 'main'


if __name__ == '__main__':
    sys.path.insert(0, ROOT)
    main()
