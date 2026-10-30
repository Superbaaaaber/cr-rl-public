"""cr_ai/env/latency.py — 动作/观测延迟。

- 动作延迟 d_act（步）：第 t 步的决策在第 t + d_act 步执行。执行时重新检查合法性（牌还在手里、圣水够、技能还能放），不合法则作废并计数。
  引擎自身的出牌命令 → 单位出生（约 1.05 s）由引擎负责，这里不重复；当前配置 d_act = 0（用户实测的下牌延迟就是引擎的这 1.05 s）。
- 观测延迟 d_obs（步）：决策看到的是第 t − d_obs 步的观测（环形缓冲）。
每局从配置的候选里均匀抽取。
"""
from collections import deque


class ActionQueue:
    def __init__(self, d_act=0):
        self.d_act = int(d_act)
        self.q = deque()                 # (执行步, 动作)
        self.n_pushed = 0
        self.n_void = 0
        self.void_reasons = {}

    def reset(self, d_act=None):
        if d_act is not None:
            self.d_act = int(d_act)
        self.q.clear()

    def push(self, step, action):
        """action：任意（None/不出牌不入队）。"""
        if action is None:
            return
        self.q.append((step + self.d_act, action))
        self.n_pushed += 1

    def pop_due(self, step):
        """取出所有到期（执行步 ≤ step）的动作，按决策顺序。"""
        out = []
        while self.q and self.q[0][0] <= step:
            out.append(self.q.popleft()[1])
        return out

    def pending(self):
        """已决策、尚未执行的动作 [(执行步, 动作)]（观测里的“待执行动作”，防止延迟期间重复下牌）。"""
        return list(self.q)

    def void(self, reason):
        self.n_void += 1
        self.void_reasons[reason] = self.void_reasons.get(reason, 0) + 1

    @property
    def void_rate(self):
        return self.n_void / max(1, self.n_pushed)


class ObsDelay:
    def __init__(self, d_obs=0):
        self.d_obs = int(d_obs)
        self.buf = deque(maxlen=self.d_obs + 1)

    def reset(self, d_obs=None):
        if d_obs is not None:
            self.d_obs = int(d_obs)
        self.buf = deque(maxlen=self.d_obs + 1)

    def push(self, obs):
        """存入本步的观测，返回 d_obs 步之前的观测（开局不足时返回最早的那个）。"""
        self.buf.append(obs)
        return self.buf[0]
