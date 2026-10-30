import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.setrecursionlimit(20000)


@pytest.fixture(scope='session')
def cfg():
    from cr_ai.config import load
    return load()


@pytest.fixture(scope='session')
def env(cfg):
    """整个测试会话共用一个引擎（一个进程只能有一个）。"""
    from cr_ai.env.game_env import GameEnv
    return GameEnv(cfg)


def make_units(rows):
    """[(uid, player, cls, x, y, hp, maxhp, air?, vx?, vy?, deploy?, radius?)] → UNIT_DTYPE 数组。"""
    from cr_ai.env.engine_adapter import UNIT_DTYPE
    u = np.zeros(len(rows), UNIT_DTYPE)
    for i, r in enumerate(rows):
        r = list(r) + [0] * (12 - len(r))
        uid, pl, cls, x, y, hp, mx, air, vx, vy, dep, rad = r
        u[i] = (i + 1, uid, pl, cls, air, x, y, vx, vy, hp, mx, dep, rad, 0, 0)
    return u


def towers(alive=((True, True, True), (True, True, True))):
    """标准塔位：[(玩家, 国王, 左公主, 右公主)]。"""
    from cr_ai.env.engine_adapter import CLS_KING, CLS_TOWER
    rows = []
    for p, (k, l, r) in enumerate(alive):
        ky, py = (2500, 6500) if p == 0 else (29500, 25500)
        rows.append((3, p, CLS_KING, 9000, ky, 4824 if k else 0, 4824))
        rows.append((4, p, CLS_TOWER, 3500, py, 3052 if l else 0, 3052))
        rows.append((4, p, CLS_TOWER, 14500, py, 3052 if r else 0, 3052))
    return rows
