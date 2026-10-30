"""cr_ai/config.py — 读 YAML 配置（默认配置 + 叠加文件 + 命令行覆盖），路径解析到仓库根目录。

    from cr_ai.config import load, repo_path
    cfg = load()                                   # cr_ai/configs/default.yaml
    cfg = load('runs/x/config.yaml', overrides=['env.frame_skip=5'])
"""
import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 仓库根目录（cr-rl/）
CONFIG_DIR = os.path.join(ROOT, 'cr_ai', 'configs')


class Cfg(dict):
    """dict，同时支持属性访问（cfg.env.frame_skip）。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    def __deepcopy__(self, memo):
        return _wrap(copy.deepcopy(_unwrap(self), memo))


def _wrap(o):
    if isinstance(o, dict):
        return Cfg({k: _wrap(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_wrap(v) for v in o]
    return o


def _unwrap(o):
    if isinstance(o, dict):
        return {k: _unwrap(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_unwrap(v) for v in o]
    return o


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def cpu_count():
    """实际可用的核数：cgroup 配额 > CPU 亲和性 > os.cpu_count()。
    云容器（AutoDL）里 os.cpu_count() 报的是整台宿主机（例如 208），实际配额只有 20 核；Windows 上就等于 os.cpu_count()。"""
    n = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        n = len(os.sched_getaffinity(0)) or n
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split()
        if q != 'max':
            n = min(n, max(1, int(q) // int(per)))
    except (OSError, ValueError):
        try:
            q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
            per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
            if q > 0:
                n = min(n, max(1, q // per))
        except (OSError, ValueError):
            pass
    return n


def limit_torch_threads():
    """torch 默认按 os.cpu_count() 开线程；在配额小于宿主机的容器里压到配额以内（本机不变）。"""
    import torch
    n = cpu_count()
    if torch.get_num_threads() > n:
        torch.set_num_threads(n)


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def read_yaml(path):
    with open(repo_path(path), encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load(*files, overrides=()):
    """默认配置 ← 依次叠加 files ← overrides（'a.b.c=值'，值按 YAML 解析）。"""
    cfg = read_yaml(os.path.join(CONFIG_DIR, 'default.yaml'))
    for f in files:
        if f in (None, '', 'default'):
            continue
        _merge(cfg, read_yaml(f if os.path.exists(repo_path(f)) else os.path.join(CONFIG_DIR, f)))
    for o in overrides:
        key, val = o.split('=', 1)
        d = cfg
        ks = key.split('.')
        for k in ks[:-1]:
            d = d.setdefault(k, {})
        d[ks[-1]] = yaml.safe_load(val)
    return _wrap(cfg)


def dump(cfg, path):
    """把配置存到 path（随 checkpoint 一起保存，便于复现）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(_unwrap(cfg), f, allow_unicode=True, sort_keys=False)


def load_deck(path=None, cfg=None):
    """deck.yaml → (牌名列表, 形态 {牌名: 'evo'|'hero'})。"""
    path = path or (cfg.paths.deck if cfg is not None else os.path.join(CONFIG_DIR, 'deck.yaml'))
    d = read_yaml(path)
    names = [c['engine'] for c in d['cards']]
    forms = {c['engine']: c['form'] for c in d['cards'] if c.get('form')}
    return names, forms
