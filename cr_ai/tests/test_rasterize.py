"""栅格化、视角翻转、镜像增强。"""
import numpy as np

from cr_ai.env.engine_adapter import CLS_BUILDING, CLS_SPELL, CLS_TROOP
from cr_ai.env.rasterize import (F_DEPLOY_AREA, F_RIVER, F_SPELL_OWN, F_TOWER_OCC, N_FLOAT, N_INT, PER_SIDE, mirror_cell, mirror_grids,
                                 rasterize)

from .conftest import make_units, towers


def test_shapes_and_dtypes(cfg):
    gi, gf = rasterize(make_units(towers()), 0)
    assert gi.shape == (N_INT, 32, 18) and gi.dtype == np.int16
    assert gf.shape == (N_FLOAT, 32, 18) and gf.dtype == np.float16


def test_id_slots_top2_by_hp_and_count():
    # 同一格 (cx=5, cy=10) 三个我方单位：血量 100 / 300 / 200 → 槽 0 = 300 的、槽 1 = 200 的；数量 3/4
    u = make_units(towers() + [(10, 0, CLS_TROOP, 5500, 10500, 100, 100), (11, 0, CLS_TROOP, 5400, 10400, 300, 600),
                               (12, 0, CLS_TROOP, 5600, 10600, 200, 200)])
    gi, gf = rasterize(u, 0)
    assert gi[0, 10, 5] == 11 and gi[1, 10, 5] == 12
    assert abs(float(gf[1, 10, 5]) - 0.75) < 1e-3
    assert abs(float(gf[0, 10, 5]) - 1.0) < 1e-3                 # 格内最大血量比例（100/100）
    assert gi[2, 10, 5] == 0                                       # 敌方槽为空


def test_flags_velocity_deploy():
    u = make_units(towers() + [(20, 1, CLS_TROOP, 9500, 20500, 50, 100, 1, 30, -60, 500),
                               (21, 1, CLS_BUILDING, 2500, 22500, 10, 10)])
    gi, gf = rasterize(u, 0)
    e = PER_SIDE
    assert gi[2, 20, 9] == 20
    assert float(gf[e + 2, 20, 9]) == 1.0                          # 空中
    assert float(gf[e + 3, 22, 2]) == 1.0                          # 建筑
    assert abs(float(gf[e + 4, 20, 9]) - 0.5) < 1e-3               # 部署剩余 500 ms / 1 s
    assert abs(float(gf[e + 5, 20, 9]) - 0.3) < 1e-3 and abs(float(gf[e + 6, 20, 9]) + 0.6) < 1e-3


def test_perspective_flip_is_row_flip_and_team_swap():
    rng = np.random.RandomState(0)
    rows = towers()
    for i in range(40):
        rows.append((30 + i % 5, i % 2, CLS_TROOP, rng.randint(0, 18000), rng.randint(0, 32000), rng.randint(1, 500), 500, i % 3 == 0,
                     rng.randint(-50, 50), rng.randint(-50, 50), rng.randint(0, 1000)))
    rows.append((90, 1, CLS_SPELL, 9000, 20000, 0, 0, 0, 0, 0, 0, 2500))
    u = make_units(rows)
    gi0, gf0 = rasterize(u, 0)
    gi1, gf1 = rasterize(u, 1)
    # 玩家 1 视角 = 玩家 0 视角上下翻转 + 两方互换
    assert (gi1[0:2] == gi0[2:4, ::-1]).all() and (gi1[2:4] == gi0[0:2, ::-1]).all() and (gi1[4] == gi0[4, ::-1]).all()
    for ch in range(PER_SIDE):
        a, b = gf1[ch].astype(np.float32), gf0[PER_SIDE + ch, ::-1].astype(np.float32)
        assert np.allclose(a, -b if ch == 6 else b), ch           # dy 取反
    assert np.allclose(gf1[F_SPELL_OWN].astype(np.float32), -gf0[F_SPELL_OWN, ::-1].astype(np.float32))
    assert np.allclose(gf1[F_TOWER_OCC].astype(np.float32), gf0[F_TOWER_OCC, ::-1].astype(np.float32))
    assert np.allclose(gf1[F_RIVER].astype(np.float32), gf0[F_RIVER].astype(np.float32))       # 河道上下对称


def test_spell_area_and_owner():
    u = make_units(towers() + [(90, 1, CLS_SPELL, 9500, 20500, 0, 0, 0, 0, 0, 0, 2500)])
    gi, gf = rasterize(u, 0)
    assert gi[4, 20, 9] == 90 and float(gf[F_SPELL_OWN, 20, 9]) == -1.0
    assert gi[4, 20, 11] == 90 and gi[4, 20, 13] == 0               # 半径 2.5 格
    gi1, gf1 = rasterize(u, 1)
    assert float(gf1[F_SPELL_OWN, 11, 9]) == 1.0                   # 对玩家 1 是自己的法术


def test_deploy_area_follows_tower_state():
    full = rasterize(make_units(towers()), 0)[1][F_DEPLOY_AREA].astype(np.float32)
    pocket = rasterize(make_units(towers(((True, True, True), (True, False, True)))), 0)[1][F_DEPLOY_AREA].astype(np.float32)
    assert full[16:].sum() == 0 and pocket[17:21, :9].sum() > 0      # 推掉敌方左公主塔 → 左侧口袋可下兵


def test_mirror_roundtrip_and_labels():
    rng = np.random.RandomState(1)
    rows = towers() + [(40, 0, CLS_TROOP, rng.randint(0, 18000), rng.randint(0, 32000), 10, 10, 0, rng.randint(-50, 50), 0) for _ in range(20)]
    gi, gf = rasterize(make_units(rows), 0)
    gi2, gf2 = mirror_grids(gi, gf)
    assert (gi2[0, :, ::-1] == gi[0]).all()
    assert np.allclose(gf2[5, :, ::-1].astype(np.float32), -gf[5].astype(np.float32))      # dx 取反
    gi3, gf3 = mirror_grids(gi2, gf2)
    assert (gi3 == gi).all() and (gf3 == gf).all()
    assert mirror_cell(10 * 18 + 3) == 10 * 18 + 14 and mirror_cell(-1) == -1
    cells = np.arange(576)
    assert (mirror_cell(mirror_cell(cells)) == cells).all()
