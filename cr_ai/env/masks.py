"""cr_ai/env/masks.py — 动作 mask。合法区域以引擎的判断为准（cr_ai/assets/place_masks.npz 由 cr_ai/tools/build_assets.py 探测生成）。

牌头（card head）选项：0 = 不出；1..4 = 打第 k 张手牌；5.. = 发动第 k 个技能（英雄/冠军，env.max_abilities 个）。
  - 手牌：槽里有牌、已过开局 4.5 s、引擎可用圣水 ≥ 费用、该牌在当前塔状态下有合法落点、没有同一张牌还在延迟队列里等执行；
  - 技能：场上有该技能单位、技能就绪、圣水够。
位置头：576 格（玩家视角，cell = cy*18 + cx），按牌的放置类别 × 玩家 × 塔状态查表：
  - 兵：己方半场（扣掉塔位）；推掉敌方公主塔后该侧向对方半场扩展（口袋）；己方公主塔被拆后它的占位格可下；
  - 建筑（如加农炮）：比兵小一圈，口袋也更小；不能与其他建筑重叠——这条是动态的，由执行时引擎校验兜底（延迟队列统计作废率）；
  - 法术（火球、滚木……）：全场；Miner / GoblinDrill：全场除塔位；RoyalDelivery：己方半场。
塔状态（玩家视角，16 种）= 敌方左公主塔已拆(1) + 敌方右(2) + 己方左(4) + 己方右(8)。
"""
import os

import numpy as np

from . import tables
from .engine_adapter import CLS_TOWER

GW, GH = 18, 32
NCELL = GW * GH
_PM = {}


def _load():
    if not _PM:
        z = np.load(os.path.join(tables.ASSETS, 'place_masks.npz'))
        _PM['masks'] = z['masks']                       # [类别, 玩家, 16, 32, 18]
        _PM['card_cls'] = z['card_cls']                 # [卡牌编号] → 类别
        _PM['flat'] = z['masks'].reshape(z['masks'].shape[0], 2, 16, NCELL)
        _PM['troop_cls'] = int(z['card_cls'][tables.card_id('Knight')])
    return _PM


def tower_state(units, p):
    """玩家 p 视角的塔状态（16 种）。units：UNIT_DTYPE 数组（世界坐标）。"""
    t = units[(units['cls'] == CLS_TOWER) & (units['hp'] > 0)]
    left = t['x'] < 9000
    own = t['player'] == p
    el, er = (~own & left).any(), (~own & ~left).any()
    ol, orr = (own & left).any(), (own & ~left).any()
    return int(not el) + 2 * int(not er) + 4 * int(not ol) + 8 * int(not orr)


def place_class(card_ids):
    return _load()['card_cls'][np.asarray(card_ids)]


def place_mask(card_id, p, state):
    """[576] 布尔（玩家视角）。"""
    pm = _load()
    return pm['flat'][pm['card_cls'][card_id], p, state]


def hand_place_masks(hand_ids, p, state):
    """手牌 4 张 → [4, 576]。"""
    pm = _load()
    return pm['flat'][pm['card_cls'][np.asarray(hand_ids)], p, state]


def troop_area(p, state):
    """己方当前可下兵区域（[32, 18] float，栅格化的静态层）。"""
    pm = _load()
    return pm['masks'][pm['troop_cls'], p, state].astype(np.float32)


def all_masks():
    """(masks [类别, 2, 16, 576], card_cls [n_cards])：给 GPU 端查表用。"""
    pm = _load()
    return pm['flat'], pm['card_cls']
