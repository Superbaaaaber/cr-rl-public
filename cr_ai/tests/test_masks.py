"""放置 mask、塔状态、牌头 mask（含引擎集成）。"""
import numpy as np

from cr_ai.env import masks as MK
from cr_ai.env import tables

from .conftest import make_units, towers


def grid(card, p, st):
    return MK.place_mask(tables.card_id(card), p, st).reshape(32, 18)


def test_tower_state_bits():
    assert MK.tower_state(make_units(towers()), 0) == 0
    u = make_units(towers(((True, False, True), (True, True, False))))   # 玩家0 左塔拆、玩家1 右塔拆
    assert MK.tower_state(u, 0) == 2 + 4                                  # 敌方右(2) + 己方左(4)
    assert MK.tower_state(u, 1) == 1 + 8                                  # 敌方左(1) + 己方右(8)


def test_troop_own_half_only():
    for p in (0, 1):
        m = grid('Knight', p, 0)
        assert m[:15].sum() > 200 and m[15:].sum() == 0
        assert not m[5:8, 2:5].any() and not m[5:8, 13:16].any()          # 己方公主塔位不能下


def test_pocket_after_enemy_tower_destroyed():
    for p in (0, 1):
        m1 = grid('Knight', p, 1)                                         # 敌方左公主塔已拆
        assert m1[17:21, :8].sum() > 20 and m1[17:21, 10:].sum() == 0
        b1 = grid('Cannon', p, 1)
        assert b1[17:21].sum() < m1[17:21].sum()                          # 建筑的口袋更小


def test_own_tower_destroyed_frees_footprint():
    assert not grid('Knight', 0, 0)[6, 3] and grid('Knight', 0, 4)[6, 3]


def test_spells_and_special_cards():
    assert grid('Fireball', 0, 0).all() and grid('Log', 1, 0).all()
    rd = grid('RoyalDelivery', 0, 0)
    assert rd[:15].sum() > 200 and rd[16:].sum() == 0
    mi = grid('Miner', 0, 0)
    assert mi[20:].sum() > 100                                            # Miner 可以下到对方半场
    assert grid('Heal', 0, 0)[16:].sum() == 0                             # 治疗精灵按兵的区域


def test_card_mask_in_engine(env):
    from cr_ai.config import load_deck
    D, F = load_deck(cfg=env.cfg)
    obs = env.reset(seed=11, decks=[D, D], forms=[F, F])
    assert not obs[0]['card_mask'][1:].any() and obs[0]['forced_noop']      # 开局 4.5 s 内只能不出
    for _ in range(19):
        obs, *_ = env.step([(0, -1), (0, -1)])
    cm, pm, abil, hand, nxt, st = env.legal(0)
    el = env.ad.avail_elixir(0)
    for s, c in enumerate(hand):
        assert cm[1 + s] == (tables.card_cost(c) <= el), (c, el)
    # 打出一张牌后，它在引擎生效前仍在手牌槽里，但 mask 不允许再打
    s = int(np.flatnonzero(cm[1:5])[0])
    card = hand[s]
    cell = int(np.flatnonzero(pm[s])[0])
    obs, *_ = env.step([(1 + s, cell), (0, -1)])
    cm2, _pm, _a, hand2, _n, _s = env.legal(0)
    if card in hand2:
        assert not cm2[1 + hand2.index(card)]
    assert obs[0]['pend_id'][0] == tables.card_id(card)
