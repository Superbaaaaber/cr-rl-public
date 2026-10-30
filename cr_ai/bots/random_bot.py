"""cr_ai/bots/random_bot.py — 随机策略：每步以 p_play 的概率在合法的牌/技能里均匀挑一个，落点在该牌的合法格里均匀挑。"""
import numpy as np


class RandomBot:
    def __init__(self, rng=None, p_play=0.1):
        self.rng = rng or np.random.RandomState()
        self.p_play = p_play

    def act_obs(self, obs):
        """只看观测（与网络同样的输入）。"""
        from ..env import masks as MK
        cm = obs['card_mask']
        legal = np.flatnonzero(cm[1:]) + 1
        if len(legal) == 0 or self.rng.rand() > self.p_play:
            return 0, -1
        sel = int(self.rng.choice(legal))
        if sel >= 5:
            return sel, -1
        flat, _cls = MK.all_masks()
        m = flat[int(obs['place_cls'][sel - 1]), int(obs['pl']), int(obs['tstate'])]
        return sel, int(self.rng.choice(np.flatnonzero(m)))

    def act(self, env, p, obs):
        return self.act_obs(obs)
