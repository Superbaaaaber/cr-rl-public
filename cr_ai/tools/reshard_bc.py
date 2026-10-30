"""cr_ai/tools/reshard_bc.py — 把 BC 分片按样本打散重写：每个输出分片混几千局、每局在一个分片里至多一两个样本。

为什么：replay_resim 的分片是按局写的（每片 50 局），loader 一个缓冲区只有约 600 局，同一局的约 330 个样本会在缓冲区里反复出现，
价值头会记住“正在看的这几局”的结果——bc.yaml 只好用 value_frac 0.1（只在 10% 的样本上算价值损失）绕开。打散之后，
一个缓冲区（12 个分片）里同一局最多只有十来个样本，可以全量算价值损失（configs/bc_mix.yaml：value_frac 1.0），策略头的样本相关性也更小。

    python -m cr_ai.tools.reshard_bc --src data_bc/il_replay --dst data_bc/il_replay_mix [--groups 64] [--chunk 64] [--workers 8]

做法（两遍、不需要把 8 GB 全读进内存）：
  1. 输入分片按 --chunk 个一块，每块里每一帧（连同共用这帧的双方样本）随机分到 --groups 组之一，每块每组写一个临时文件（dst/_parts/）；
  2. 每组读回所有临时文件、按帧整体打乱，切成“块数”个输出分片（总分片数与输入相同，大小也差不多）；删掉临时文件。
  按帧而不是按样本分：同一步双方的两个样本共用一帧单位列表，拆开会把帧复制一份（实测输出膨胀到 1.8 倍）。
输出格式与 replay_resim 完全相同（units + frame_off + 每样本一行）；局号不变，所以训练/验证按局号哈希的切分不变。
"""
import argparse
import glob
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

FRAME_KEYS = ('units', 'frame_off')


def take(d, rows):
    """分片 d 里 rows 这些样本 → 新的分片 dict（只带这些样本用到的帧，帧号重排）。"""
    fr = d['frame'][rows]
    uf, inv = np.unique(fr, return_inverse=True)
    off = d['frame_off']
    st, ln = off[uf], off[uf + 1] - off[uf]
    idx = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())
    out = {k: v[rows] for k, v in d.items() if k not in FRAME_KEYS and k != 'frame'}
    out['units'] = d['units'][idx]
    out['frame_off'] = np.concatenate([[0], np.cumsum(ln)]).astype(np.int64)
    out['frame'] = inv.astype(np.int32)
    return out


def concat(parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in FRAME_KEYS and k != 'frame'}
    out['units'] = np.concatenate([p['units'] for p in parts])
    nf = np.cumsum([0] + [len(p['frame_off']) - 1 for p in parts])
    nu = np.cumsum([0] + [len(p['units']) for p in parts])
    out['frame'] = np.concatenate([p['frame'] + nf[i] for i, p in enumerate(parts)]).astype(np.int32)
    out['frame_off'] = np.concatenate([[0]] + [p['frame_off'][1:] + nu[i] for i, p in enumerate(parts)]).astype(np.int64)
    return out


def load(fn):
    z = np.load(fn)
    return {k: z[k] for k in z.files}


def _pass1(args):
    ci, files, parts_dir, groups, seed = args
    rng = np.random.RandomState(seed + ci)
    buf = [[] for _ in range(groups)]
    for fn in files:
        d = load(fn)
        g = rng.randint(groups, size=len(d['frame_off']) - 1)[d['frame']]     # 按帧分组：共用一帧的样本在同一组
        for k in range(groups):
            rows = np.flatnonzero(g == k)
            if len(rows):
                buf[k].append(take(d, rows))
    for k in range(groups):
        if buf[k]:
            np.savez_compressed(os.path.join(parts_dir, 'g%03d_c%05d.npz' % (k, ci)), **concat(buf[k]))   # 必须压缩：不压缩时全量临时文件约 80 GB（2026-10-17 在 AutoDL 上把盘写到只剩 8 GB）
    return len(files)


def _pass2(args):
    k, parts, dst, n_out, seed = args
    if not parts:
        return 0
    d = concat([load(p) for p in parts])
    rng = np.random.RandomState(seed * 7919 + k)
    nf = len(d['frame_off']) - 1
    out_of = np.empty(nf, np.int64)                                     # 每一帧去哪个输出分片（按帧打乱，帧不拆开）
    for j, fs in enumerate(np.array_split(rng.permutation(nf), n_out)):
        out_of[fs] = j
    dest = out_of[d['frame']]
    order = np.argsort(dest, kind='stable')
    bounds = np.searchsorted(dest[order], np.arange(n_out + 1))
    for j in range(n_out):
        rows = np.sort(order[bounds[j]:bounds[j + 1]])
        if len(rows):
            np.savez_compressed(os.path.join(dst, 'shard_m%03d_%05d.npz' % (k, j)), **take(d, rows))
    for p in parts:
        os.remove(p)
    return len(d['frame'])


def reshard(src, dst, groups=64, chunk=64, workers=4, seed=0):
    files = sorted(glob.glob(os.path.join(src, 'shard_*.npz')))
    if not files:
        raise SystemExit('没有输入分片：%s' % src)
    os.makedirs(dst, exist_ok=True)
    if glob.glob(os.path.join(dst, 'shard_*.npz')):
        raise SystemExit('输出目录里已经有分片：%s（换个目录或先清空）' % dst)
    parts_dir = os.path.join(dst, '_parts')
    os.makedirs(parts_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    order = list(files)
    rng.shuffle(order)
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    print('输入 %d 个分片 → %d 块 × %d 组' % (len(files), len(chunks), groups), flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        done = 0
        for n in pool.imap_unordered(_pass1, [(ci, c, parts_dir, groups, seed) for ci, c in enumerate(chunks)]):
            done += n
            print('  第 1 遍 %d/%d' % (done, len(files)), flush=True)
        t1 = time.time()
        jobs = [(k, sorted(glob.glob(os.path.join(parts_dir, 'g%03d_c*.npz' % k))), dst, len(chunks), seed) for k in range(groups)]
        n_rows = 0
        for i, n in enumerate(pool.imap_unordered(_pass2, jobs)):
            n_rows += n
            print('  第 2 遍 %d/%d 组' % (i + 1, groups), flush=True)
    print('耗时：第 1 遍 %.0f 秒，第 2 遍 %.0f 秒' % (t1 - t0, time.time() - t1), flush=True)
    shutil.rmtree(parts_dir, ignore_errors=True)
    for f in ('config.yaml', 'm1_report.json'):                    # 生成参数/报告原样带过去（说明数据来源）
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print('完成：%d 个样本，%d 个输出分片 → %s' % (n_rows, len(glob.glob(os.path.join(dst, 'shard_*.npz'))), dst), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data_bc/il_replay')
    ap.add_argument('--dst', default='data_bc/il_replay_mix')
    ap.add_argument('--groups', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=64, help='第 1 遍每块的输入分片数（每个进程同时在内存里放这么多片，约 10 MB/片）')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from ..config import repo_path
    reshard(repo_path(a.src), repo_path(a.dst), a.groups, a.chunk, a.workers, a.seed)


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    main()
