"""cr_ai/env/reward.py — 奖励。

- 胜负：胜 +1、负 −1、平 0（只在终局给）。
- 塔血势能塑形：Φ(s) = 0.3 × (Σ我方 wᵢ·血量比ᵢ − Σ敌方 wᵢ·血量比ᵢ)，公主塔 w = 0.25、国王塔 w = 0.5；r = c·(γΦ(s′) − Φ(s))。
  势能塑形不改变最优策略；系数 c 由训练器按步数从 1 退火到 0.2（shaping_coef）。终局 Φ(s′) 记 0（标准做法，保证不改变最优策略）。
- 圣水溢出：圣水满 10 时每步 −0.0005。
禁止加入伤害量、击杀数、出牌次数之类的奖励。
"""


def potential(tw_own, tw_enemy, rc):
    """tw_*：[国王, 左公主, 右公主] 的血量比例。"""
    w = (rc.king_w, rc.princess_w, rc.princess_w)
    return rc.shaping_scale * (sum(a * b for a, b in zip(w, tw_own)) - sum(a * b for a, b in zip(w, tw_enemy)))


def shaping_coef(step, rc):
    f = min(1.0, step / max(1, rc.shaping_anneal_steps))
    return rc.shaping_c_start + (rc.shaping_c_end - rc.shaping_c_start) * f


def step_reward(phi_prev, phi_next, done, outcome, elixir_full, coef, rc):
    """一个玩家一步的奖励。outcome：终局时 +1/−1/0（否则忽略）。返回 (总奖励, 分项)。"""
    gamma = rc.gamma
    shaped = coef * ((0.0 if done else gamma * phi_next) - phi_prev)
    win = rc.win * outcome if done else 0.0
    over = -rc.overflow_penalty if elixir_full else 0.0
    return win + shaped + over, dict(win=win, shaping=shaped, overflow=over)
