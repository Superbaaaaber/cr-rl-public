"""cr_ai/bots/rule_bot.py — 简单的规则 AI（基准和冒烟测试用），对任意卡组都能打。

规则（按优先级，玩家视角：自己在下方，y < 16000 是己方半场）：
  1. 防守：敌方地面/空中部队进入己方半场 →
       有建筑牌就在中路（x≈9000、第 9 行附近）下建筑；
       否则敌人扎堆（2.5 格内 ≥ 3 个）且有伤害法术 → 法术砸在扎堆中心；
       否则用最便宜的部队拦在敌人前方 2 格。
  2. 进攻：圣水 ≥ 推塔牌费用 + 1 → 在敌方公主塔血量较低的一路桥头下推塔牌（没有推塔牌就用最贵的部队，圣水需 ≥ 8）。
  3. 圣水快满（≥ 9.5）→ 在后场角落过一张最便宜的牌。
每次出牌后冷却 1 s；以 noise 的概率改做随机合法动作，让对局更多样。
只用自己能看到的信息（单位列表、自己的手牌和圣水），不看对手手牌/圣水。
"""
import numpy as np

from ..env import masks as MK
from ..env import tables
from ..env.engine_adapter import CLS_BUILDING, CLS_TROOP

DAMAGE_SPELLS = {'Fireball', 'Arrows', 'Zap', 'Log', 'BarbLog', 'Rocket', 'Lightning', 'Poison', 'Snowball', 'Earthquake', 'GoblinCurse',
                 'Vines', 'DarkMagic'}
GW, GH = 18, 32


class RuleBot:
    def __init__(self, rng=None, noise=0.03, cooldown_steps=4):
        self.rng = rng or np.random.RandomState()
        self.noise = noise
        self.cooldown_steps = cooldown_steps
        self.cool = 0

    @staticmethod
    def _cell(cx, cy):
        return int(cy) * GW + int(cx)

    def _snap(self, mask, cx, cy):
        """(cx, cy) 不合法就找最近的合法格。"""
        cx, cy = int(np.clip(round(cx), 0, GW - 1)), int(np.clip(round(cy), 0, GH - 1))
        if mask[self._cell(cx, cy)]:
            return self._cell(cx, cy)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return -1
        d = np.abs(idx % GW - cx) + np.abs(idx // GW - cy)
        return int(idx[np.argmin(d)])

    def act(self, env, p, obs):
        cm = obs['card_mask']
        if self.cool > 0:
            self.cool -= 1
            return 0, -1
        if not cm[1:].any():
            return 0, -1
        flat, _ = MK.all_masks()
        pl, st = int(obs['pl']), int(obs['tstate'])
        pmask = lambda s: flat[int(obs['place_cls'][s]), pl, st]
        if self.rng.rand() < self.noise:
            legal = np.flatnonzero(cm[1:5]) + 1
            if len(legal):
                sel = int(self.rng.choice(legal))
                self.cool = self.cooldown_steps
                return sel, int(self.rng.choice(np.flatnonzero(pmask(sel - 1))))
        # 技能：就绪就放
        for k in range(len(cm) - 5):
            if cm[5 + k]:
                return 5 + k, -1
        _cm, _pm, _ab, hand, _nxt, _st = env.legal(p)
        elixir = float(obs['scal'][0]) * 10.0
        opts = [(s, hand[s]) for s in range(4) if cm[1 + s]]
        if not opts:
            return 0, -1
        kind = lambda c: tables.card_info().get(c, {}).get('kind', 'troop')
        cost = tables.card_cost
        u = env.view_units(p)
        enemy = u[(u['player'] == 1) & ((u['cls'] == CLS_TROOP) | (u['cls'] == CLS_BUILDING))]
        threats = enemy[(enemy['y'] < 16000) & (enemy['cls'] == CLS_TROOP)]
        choice = None
        if len(threats):
            t = threats[np.argmin(threats['y'])]
            tx, ty = t['x'] / 1000.0, t['y'] / 1000.0
            blds = [(s, c) for s, c in opts if kind(c) == 'building']
            if blds:
                s, c = min(blds, key=lambda e: cost(e[1]))
                choice = (s, 9 if tx >= 9 else 8, 9)
            if choice is None:
                d = np.hypot(threats['x'] / 1000.0 - tx, threats['y'] / 1000.0 - ty)
                if (d < 2.5).sum() >= 3:
                    sp = [(s, c) for s, c in opts if c in DAMAGE_SPELLS]
                    if sp:
                        s, c = min(sp, key=lambda e: cost(e[1]))
                        choice = (s, tx, ty)
            if choice is None:
                tr = [(s, c) for s, c in opts if kind(c) == 'troop' and not tables.targets_buildings(c)]
                if tr:
                    s, c = min(tr, key=lambda e: cost(e[1]))
                    choice = (s, tx, max(1.0, ty - 2.0))
        if choice is None:
            wins = [(s, c) for s, c in opts if tables.targets_buildings(c)]
            if not wins:
                big = [(s, c) for s, c in opts if kind(c) == 'troop' and elixir >= 8.0]
                wins = [max(big, key=lambda e: cost(e[1]))] if big else []
            if wins:
                s, c = max(wins, key=lambda e: cost(e[1]))
                if elixir >= cost(c) + 1 or elixir >= 9.5:
                    tw = obs['scal'][9:12]                   # 敌方 [国王, 左, 右]
                    lane_x = 3 if (tw[1] > 0 and (tw[1] <= tw[2] or tw[2] <= 0)) else 14
                    choice = (s, lane_x, 14)
        if choice is None and elixir >= 9.5:
            s, c = min(opts, key=lambda e: cost(e[1]))
            choice = (s, 1 if self.rng.rand() < 0.5 else 16, 2) if kind(c) != 'spell' else None
        if choice is None:
            return 0, -1
        s, cx, cy = choice
        cell = self._snap(pmask(s), cx, cy)
        if cell < 0:
            return 0, -1
        self.cool = self.cooldown_steps
        return 1 + s, cell
