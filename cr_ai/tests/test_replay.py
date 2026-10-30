"""回放管线：解析、起手反推、标签对齐、存储往返（单位列表 → 栅格与环境一致）、镜像增强的一致性。"""
import json

import numpy as np

from cr_ai.data import replay_parser as RP


def _rows(cfg, n=3):
    return list(RP.iter_rows(RP.replay_files(cfg), limit=n))


def test_parse_orientation_and_levels(cfg):
    gi, tag, bt, gm, pj = _rows(cfg, 1)[0]
    rec, why = RP.parse_payload(tag, bt, gm, pj)
    assert why is None and len(rec['decks'][0]) == 8 and len(rec['decks'][1]) == 8
    p = json.loads(pj)
    e0 = [e for e in p['events'] if e['kind'] == 'play_card'][0]
    team_pl = 0 if e0['source_fields']['data_i'] == 1 else 1
    assert rec['team_pl'] == team_pl
    pl = team_pl if e0['side'] == 'team' else 1 - team_pl
    ev = [e for e in rec['events'] if not e['ab']][0]
    assert ev['pl'] == pl and (ev['x'], ev['y']) == (e0['source_fields']['data_x'], e0['source_fields']['data_y'])
    assert rec['levels']['king'][0] == max(rec['levels']['card'][0].values())
    assert rec['real']['crowns'][team_pl] == p['battle']['team']['crowns']


def test_skip_showdown():
    rec, why = RP.parse_payload('x', 'trail', '1v1 Showdown', '{}')
    assert rec is None and why == 'mode'


def test_opening_constraints():
    from crx.rl.replay_sl import opening_candidates
    deck = list('ABCDEFGH')
    plays = ['E', 'A', 'B', 'C']
    ticks = [100, 200, 300, 400]
    c = opening_candidates(deck, plays, ticks)
    assert c and all('E' in h or q[0] == 'E' for h, q in c)
    # 第 1 次出的牌只能在起手里（排队第 0 张要等第 1 次出牌生效后才进手牌）
    assert all('E' in h for h, q in c)
    # 补牌延迟：第 2 次出牌紧跟第 1 次（< 21 tick）时，它不可能是刚补进来的那张
    c2 = opening_candidates(deck, ['E', 'F'], [100, 110])
    assert all('F' in h for h, q in c2)


def test_resim_roundtrip_and_labels(cfg, env):
    """重放一局：样本的标签在 mask 内（或带强制合法标记）、出牌样本对齐到 ⌊t/5⌋、由存下的单位列表重算的网格与环境给的一致。"""
    from cr_ai.data.replay_resim import Resim
    rs = Resim.__new__(Resim)
    rs.cfg, rs.env, rs.noop_keep = cfg, env, 0.2
    rs.feat_keys = [k for k in env.spec if k not in ('gi', 'gf')]
    gi, tag, bt, gm, pj = _rows(cfg, 2)[1]
    rec, _ = RP.parse_payload(tag, bt, gm, pj)
    frames, samples, st = rs.run(rec, check_raster=True)
    assert st['raster_bad'] == 0 and st['err'] is None and samples
    play_steps = {(e['t'] // 5, e['pl']) for e in rec['events']}
    for s in samples:
        f = s['feat']
        if s['sel'] > 0:
            assert f['card_mask'][s['sel']]
            assert (f['tick'] // 5, s['player']) in play_steps
            if s['sel'] <= 4:
                assert 0 <= s['cell'] < 576
        else:
            assert f['card_mask'][1:].any()                                 # 强制不出的步不进样本
            assert abs(s['w'] - 5.0) < 1e-6


def test_mirror_batch_consistency(cfg):
    """镜像 = 先镜像单位再栅格化（网格部分）；标签/塔状态/标量的变换是对合（做两次还原）。"""
    from cr_ai.data.bc_dataset import mirror_batch, mirror_place_mask, mirror_tstate
    from cr_ai.env.engine_adapter import CLS_TROOP
    from cr_ai.env.rasterize import rasterize
    from .conftest import make_units, towers
    rng = np.random.RandomState(0)
    rows = towers(((True, True, False), (True, False, True))) + [(50, rng.randint(2), CLS_TROOP, rng.randint(0, 18) * 1000 + 500, rng.randint(0, 32) * 1000 + 500, 10, 10, 0, 7, 3) for _ in range(30)]
    u = make_units(rows)
    gi, gf = rasterize(u, 0)
    um = u.copy()
    um['x'] = (17 - um['x'] // 1000) * 1000 + 500                   # 格中心镜像：cx → 17 − cx
    um['vx'] = -um['vx']
    gim, gfm = rasterize(um, 0)
    b = dict(gi=gi[None].copy(), gf=gf[None].copy(), cell=np.array([3 * 18 + 2]), feat_scal=np.arange(15, dtype=np.float32)[None].copy(),
             feat_tstate=np.array([5], np.int16), feat_pend_num=np.array([[[0.25, 0.5, 0.1], [0, 0, 0]]], np.float32),
             feat_last_num=np.array([[0.75, 0.2]], np.float32), mirrored=np.zeros(1, bool))
    mirror_batch(b, np.array([0]))
    keep = [c for c in range(18) if c not in (16, 17)]                # 静态层“可下兵区域/塔占位”依塔位而定，不参与比较
    assert (b['gi'][0][:4] == gim[:4]).all()
    assert np.allclose(b['gf'][0][keep[:14]].astype(np.float32), gfm[keep[:14]].astype(np.float32))
    assert b['cell'][0] == 3 * 18 + 15 and b['mirrored'][0]
    assert b['feat_scal'][0, 7] == 8 and b['feat_scal'][0, 10] == 11
    assert b['feat_tstate'][0] == mirror_tstate(5) == 10
    assert abs(b['feat_pend_num'][0, 0, 0] - 0.75) < 1e-6 and b['feat_pend_num'][0, 1, 0] == 0
    assert (mirror_tstate(mirror_tstate(np.arange(16))) == np.arange(16)).all()
    m = np.zeros(576, bool)
    m[2] = True
    assert mirror_place_mask(m)[15]
