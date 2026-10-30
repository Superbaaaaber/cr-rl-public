"""cr_ai/env/episodes.py — 每局的设定抽样：双方卡组/形态、塔兵、控制方（策略 agent / 规则 AI / 随机）。

spec（dict，可从 YAML 来）：
  controllers: [[side0, side1, 权重], …]   side ∈ {'agent', 'rule', 'random'}；默认 [['agent', 'agent', 1]]
  main_deck_prob: 0.5                      各方用主角卡组（deck.yaml）的概率，否则从对手池（deck_pool.yaml）按使用量加权抽
  main_side: null                           非空时（0/1）这一方固定用主角卡组
  main_for: null                            'agent'：agent 控制的一方固定用主角卡组、另一方从对手池抽（评测/RL 用）
  opp_main_prob: 0.0                        main_for='agent' 时，对手也用主角卡组（镜像）的概率
  agent_main_prob: 1.0                      main_for='agent' 时，agent 一方用主角卡组的概率（其余从对手池按使用量抽：M4 多卡组训练）
  tower_troops: {tower-princess: 0.93, royal-chef: 0.03, dagger-duchess: 0.03, cannoneer: 0.01}
"""
import numpy as np

from ..config import load_deck, read_yaml

DEFAULT_TOWERS = {'tower-princess': 0.93, 'royal-chef': 0.03, 'dagger-duchess': 0.03, 'cannoneer': 0.01}     # HF 数据里的分布


class EpisodeSampler:
    def __init__(self, cfg, spec=None, seed=0):
        self.spec = dict(spec or {})
        self.rng = np.random.RandomState(seed)
        self.main = load_deck(cfg=cfg)
        pool = read_yaml(cfg.paths.deck_pool)['decks']
        self.pool = [(d['cards'], d.get('forms') or {}) for d in pool]
        w = np.array([d['weight'] for d in pool], np.float64)
        self.pool_p = w / w.sum()
        self.opp_p = self.pool_p                 # 对手一方抽卡组的概率：默认同 pool_p；联赛按卡组 PFSP 时由学习进程下发（set_opp_p）
        ctl = self.spec.get('controllers') or [['agent', 'agent', 1]]
        self.ctl = [(c[0], c[1]) for c in ctl]
        cw = np.array([c[2] for c in ctl], np.float64)
        self.ctl_p = cw / cw.sum()
        tt = self.spec.get('tower_troops') or DEFAULT_TOWERS
        self.towers = list(tt.keys())
        self.tower_p = np.array(list(tt.values()), np.float64) / sum(tt.values())

    def set_opp_p(self, p):
        """对手一方的卡组抽样概率（长度 = 卡组池套数；学习进程按卡组 PFSP 算好下发）。长度对不上（池子换了）就忽略。"""
        p = np.asarray(p, np.float64)
        if p.shape == self.pool_p.shape and p.sum() > 0:
            self.opp_p = p / p.sum()

    def _deck(self, main, opp=False):
        if main:
            return list(self.main[0]), dict(self.main[1]), 'main'
        i = self.rng.choice(len(self.pool), p=self.opp_p if opp else self.pool_p)
        return list(self.pool[i][0]), dict(self.pool[i][1]), 'pool%d' % i

    def sample(self):
        sp = self.spec
        mside = sp.get('main_side')
        ctl = self.ctl[self.rng.choice(len(self.ctl), p=self.ctl_p)]
        decks, forms, names = [], [], []
        for p in (0, 1):
            opp = False
            if sp.get('main_for') == 'agent':
                if ctl[p] == 'agent':
                    use_main = self.rng.rand() < sp.get('agent_main_prob', 1.0)
                else:
                    use_main = self.rng.rand() < sp.get('opp_main_prob', 0.0)
                    opp = True
            else:
                use_main = (p == mside) if mside is not None else (self.rng.rand() < sp.get('main_deck_prob', 0.5))
            d, f, n = self._deck(use_main, opp)
            decks.append(d)
            forms.append(f)
            names.append(n)
        towers = [str(self.rng.choice(self.towers, p=self.tower_p)) for _ in (0, 1)]
        return dict(seed=int(self.rng.randint(1, 2 ** 30)), decks=decks, forms=forms, deck_names=names, controllers=list(ctl), tower_troop=towers)
