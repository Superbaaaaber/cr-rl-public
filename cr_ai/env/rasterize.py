"""cr_ai/env/rasterize.py — 单位列表 → 网格特征图。训练、BC 重模拟、部署推理共用这一份（唯一入口：rasterize）。

    gi, gf = rasterize(units, p)       # units：engine_adapter.UNIT_DTYPE 数组（世界坐标）；p：视角玩家（自己永远在下方）
    gi：int16 [5, 32, 18]   单位/法术 ID（嵌入在 GPU 上做）
    gf：float16 [18, 32, 18] 数值通道

通道（“每方”= 我方、敌方各一份，我方在前）：
  gi  0-1 我方单位 ID 槽 ×2、2-3 敌方单位 ID 槽 ×2（每格按当前血量取前 2 个）、4 法术 ID（正在飞行/生效的法术区域）
  gf  每方 7 个：血量比例（格内最大）、单位数/4、空中、建筑（含塔）、部署中（剩余部署时间/1 s）、dx、dy（速度，世界单位/tick/100）
      14 法术归属（我方 +1、敌方 −1）、15 河道/桥（河 1、桥 0.5）、16 己方当前可下兵区域、17 塔占位
视角：玩家 1 = 世界网格上下翻转（行 cy → 31 − cy）、两方互换、dy 取反；x 不变。
"""
import numpy as np

from . import masks as M
from .engine_adapter import CLS_BUILDING, CLS_EFFECT, CLS_KING, CLS_SPELL, CLS_TOWER, CLS_TROOP

GW, GH = 18, 32
N_INT, N_FLOAT = 5, 18
PER_SIDE = 7
F_SPELL_OWN, F_RIVER, F_DEPLOY_AREA, F_TOWER_OCC = 14, 15, 16, 17
SPEED_NORM = 100.0
DEPLOY_NORM_MS = 1000.0
COUNT_NORM = 4.0
UNIT_CLS = (CLS_TROOP, CLS_BUILDING, CLS_TOWER, CLS_KING)


def _river():
    r = np.zeros((GH, GW), np.float32)
    r[15:17, :] = 1.0
    r[15:17, 2:5] = 0.5                     # 左桥（x 2000–5000）
    r[15:17, 13:16] = 0.5                   # 右桥（x 13000–16000）
    return r


RIVER = _river()
_CX = (np.arange(GW) * 1000 + 500).astype(np.float32)
_CY = (np.arange(GH) * 1000 + 500).astype(np.float32)


def to_view(units, p):
    """世界坐标 → 玩家 p 的视角坐标（返回副本）：y、vy 翻转；player 换成 team（0 = 自己，1 = 对手）。"""
    u = units.copy()
    if p == 1:
        u['y'] = 32000 - u['y']
        u['vy'] = -u['vy']
    u['player'] = (u['player'] != p).astype(np.int8)
    return u


def _world(units):
    """按世界坐标（玩家 0 在下）算一次：单位 ID 槽与每名玩家的数值通道、法术、塔占位。视角由 rasterize 从这里派生。"""
    gi = np.zeros((N_INT, GH * GW), np.int16)         # 0-1 玩家0 槽、2-3 玩家1 槽、4 法术
    gf = np.zeros((2 * PER_SIDE + 2, GH * GW), np.float32)   # 0-6 玩家0、7-13 玩家1、14 法术归属（玩家0 +1）、15 塔占位
    cls = units['cls']
    uu = units[cls <= CLS_KING]
    if len(uu):
        cell = np.clip(uu['y'] // 1000, 0, GH - 1) * GW + np.clip(uu['x'] // 1000, 0, GW - 1)
        key = uu['player'].astype(np.int64) * (GH * GW) + cell
        order = np.lexsort((-uu['hp'], key))
        k = key[order]
        v = uu[order]
        start = np.ones(len(k), bool)
        start[1:] = k[1:] != k[:-1]
        si = np.flatnonzero(start)
        rank = np.arange(len(k)) - si[np.cumsum(start) - 1]
        team = k // (GH * GW)
        c = k % (GH * GW)
        for slot in (0, 1):
            m = rank == slot
            gi[2 * team[m] + slot, c[m]] = v['uid'][m]
        gc = c[si]
        base = team[si] * PER_SIDE
        hpr = np.where(v['maxhp'] > 0, v['hp'] / np.maximum(v['maxhp'], 1), 0.0).astype(np.float32)
        n = np.diff(np.append(si, len(k))).astype(np.float32)
        gf[base + 0, gc] = np.maximum.reduceat(hpr, si)
        gf[base + 1, gc] = np.minimum(n / COUNT_NORM, 1.0)
        gf[base + 2, gc] = np.maximum.reduceat(v['air'].astype(np.float32), si)
        gf[base + 3, gc] = np.maximum.reduceat((v['cls'] != CLS_TROOP).astype(np.float32), si)
        gf[base + 4, gc] = np.minimum(np.maximum.reduceat(v['deploy'], si) / DEPLOY_NORM_MS, 1.0)
        gf[base + 5, gc] = np.add.reduceat(v['vx'], si) / n / SPEED_NORM
        gf[base + 6, gc] = np.add.reduceat(v['vy'], si) / n / SPEED_NORM
    gi = gi.reshape(N_INT, GH, GW)
    gf = gf.reshape(-1, GH, GW)
    sp = units[cls == CLS_SPELL]
    if len(sp):
        rad = np.maximum(sp['radius'], 500.0)
        d2 = (_CX[None, None, :] - sp['x'][:, None, None]) ** 2 + (_CY[None, :, None] - sp['y'][:, None, None]) ** 2
        m = d2 <= (rad * rad)[:, None, None]
        m[np.arange(len(sp)), np.clip(sp['y'] // 1000, 0, GH - 1), np.clip(sp['x'] // 1000, 0, GW - 1)] = True
        for i in range(len(sp)):
            gi[4][m[i]] = sp['uid'][i]
        sign = np.where(sp['player'] == 0, 1.0, -1.0).astype(np.float32)
        gf[14] = np.clip((m * sign[:, None, None]).sum(0), -1, 1)
    tw = units[((cls == CLS_TOWER) | (cls == CLS_KING)) & (units['hp'] > 0)]
    if len(tw):
        half = np.where(tw['cls'] == CLS_KING, 2000, 1500)[:, None, None]
        gf[15] = ((np.abs(_CX[None, None, :] - tw['x'][:, None, None]) < half) & (np.abs(_CY[None, :, None] - tw['y'][:, None, None]) < half)).any(0)
    return gi, gf


_CACHE = [None, None]


def rasterize(units, p):
    """玩家 p 视角的网格（唯一入口）。同一份 units 连续给两个视角调用时只算一次世界网格。
    视角：玩家 1 = 世界网格上下翻转（行 cy → 31 − cy，与放置 mask 的视角一致）、两方通道互换、dy 与法术归属取反。"""
    if _CACHE[0] is not units:
        _CACHE[0], _CACHE[1] = units, _world(units)
    wi, wf = _CACHE[1]
    gi = np.empty((N_INT, GH, GW), np.int16)
    gf = np.empty((N_FLOAT, GH, GW), np.float32)
    o, e = (0, 1) if p == 0 else (1, 0)
    gi[0:2], gi[2:4], gi[4] = wi[2 * o:2 * o + 2], wi[2 * e:2 * e + 2], wi[4]
    gf[0:PER_SIDE] = wf[o * PER_SIDE:(o + 1) * PER_SIDE]
    gf[PER_SIDE:2 * PER_SIDE] = wf[e * PER_SIDE:(e + 1) * PER_SIDE]
    gf[F_SPELL_OWN] = wf[14] if p == 0 else -wf[14]
    gf[F_TOWER_OCC] = wf[15]
    if p == 1:
        gi = gi[:, ::-1].copy()
        gf[:F_RIVER] = gf[:F_RIVER, ::-1]
        gf[6] *= -1
        gf[PER_SIDE + 6] *= -1
        gf[F_TOWER_OCC] = gf[F_TOWER_OCC, ::-1]
    gf[F_RIVER] = RIVER
    gf[F_DEPLOY_AREA] = M.troop_area(p, M.tower_state(units, p))
    return gi, gf.astype(np.float16)


# ---------------- 单位列表（实体分支，model.entity） ----------------
MAX_ENT = 128                    # 与 obs.max_units 一致；超出时塔优先、其余按血量从高到低截断
N_ENT_F = 18
ENT_X, ENT_VX = 1, 3             # 镜像要改的列


def entities(units, p, n=MAX_ENT):
    """玩家 p 视角的单位列表（兵、建筑、塔、法术、投射物；不含特效）→ (ent_id int16 [n, 2]：单位词表编号、所在格子（与网格同一套视角/格子）,
    ent_num float16 [n, N_ENT_F])。补齐的行全 0。
    ent_num 列：0 有效、1 x/18000、2 y/32000（精确坐标，视角坐标）、3-4 速度（世界单位/tick/100，视角）、5 血量比例、6 log1p(血量)/10、
    7 部署剩余/1 s、8 半径/5000、9 空中、10 敌方、11-16 类别 one-hot（兵、建筑、公主塔、国王塔、法术、投射物）、17 技能就绪。"""
    u = units[entity_rows(units, n)]
    k = len(u)
    eid = np.zeros((n, 2), np.int16)
    num = np.zeros((n, N_ENT_F), np.float32)
    if k:
        v = to_view(u, p)
        cy = np.clip(u['y'] // 1000, 0, GH - 1)
        if p == 1:
            cy = GH - 1 - cy                                    # 与 rasterize 的行翻转一致（不用 32000 − y 再取整，整千坐标会差一格）
        eid[:k, 0] = v['uid']
        eid[:k, 1] = cy * GW + np.clip(u['x'] // 1000, 0, GW - 1)
        num[:k, 0] = 1.0
        num[:k, 1] = v['x'] / 18000.0
        num[:k, 2] = v['y'] / 32000.0
        num[:k, 3] = v['vx'] / SPEED_NORM
        num[:k, 4] = v['vy'] / SPEED_NORM
        num[:k, 5] = np.where(v['maxhp'] > 0, v['hp'] / np.maximum(v['maxhp'], 1), 0.0)
        num[:k, 6] = np.log1p(np.maximum(v['hp'], 0)) / 10.0
        num[:k, 7] = np.minimum(v['deploy'] / DEPLOY_NORM_MS, 1.0)
        num[:k, 8] = v['radius'] / 5000.0
        num[:k, 9] = v['air']
        num[:k, 10] = v['player']                               # to_view 之后：0 = 自己，1 = 对手
        num[np.arange(k), 11 + v['cls'].astype(np.int64)] = 1.0
        num[:k, 17] = v['ab_ready']
    return eid, num.astype(np.float16)


def entity_rows(units, n=MAX_ENT):
    """单位列表选中的行（units 里的下标，按 entities 输出的顺序）：兵、建筑、塔、法术、投射物；超出 n 个时塔优先、其余按血量从高到低。"""
    idx = np.flatnonzero(units['cls'] < CLS_EFFECT)
    if len(idx) > n:
        u = units[idx]
        tower = (u['cls'] == CLS_TOWER) | (u['cls'] == CLS_KING)
        idx = idx[np.lexsort((-u['hp'], ~tower))[:n]]
    return idx


def entity_status(units, statuses, n=MAX_ENT):
    """和 entities 同顺序的单位状态特征 [n, N_STATUS]（features.status_features；statuses = {对象编号: 状态名}，见 engine_adapter）。"""
    from .features import N_STATUS, status_features
    out = np.zeros((n, N_STATUS), np.float32)
    if statuses:
        for i, j in enumerate(entity_rows(units, n)):
            s = statuses.get(int(units['oid'][j]))
            if s:
                out[i] = status_features(s)
    return out.astype(np.float16)


# ---------------- 左右镜像增强 ----------------
def mirror_grids(gi, gf):
    """观测左右翻转（x → 17 − x）：dx 通道取反。返回新数组；支持任意前导维度 [..., C, 32, 18]。"""
    gi2 = gi[..., ::-1].copy()
    gf2 = gf[..., ::-1].copy()
    for t in (0, 1):
        gf2[..., t * PER_SIDE + 5, :, :] *= -1
    return gi2, gf2


def mirror_cell(cell):
    """位置标签左右翻转：cell = cy*18 + cx → cy*18 + (17 − cx)。"""
    cell = np.asarray(cell)
    return np.where(cell >= 0, (cell // GW) * GW + (GW - 1 - cell % GW), cell)
