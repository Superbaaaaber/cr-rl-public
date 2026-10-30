"""cr_ai/env/opponent_tracker.py — 对手牌序追踪与圣水估计（只用部署时真实可见的信息：对手打出了什么牌、什么时候）。

对手出牌在命令生效时（单位出生/法术出现）才可见，调用方在那一刻调用 on_play。
循环规则：打出的牌进入队尾，再经过 4 次出牌才可能回到手上 → 最近 4 次打出的牌一定不在对手手里。
圣水估计：开局圣水 + 按时间段回复（单/双/三倍）− 每次可见出牌的费用（含技能）+ 对手圣水采集器的产出，截断在 [0, 10]。
采集器（场上可见）：对象年龄每过 interval 产 1 点，消失（被拆 / 寿命到）再给 1 点。interval：离线引擎按数据表 13 s
（ManaGenerateTimeMs，实测逐次对上），真机游戏实测 15 s（cr-live 传 15000）。
"""
import numpy as np

MAX_SEEN = 8


class OpponentTracker:
    def __init__(self, env_cfg):
        self.cfg = env_cfg
        self.reset()

    def reset(self):
        self.seen = []                     # [(牌名, 形态)]，按第一次出现顺序，最多 8 张
        self.last_tick = {}                # 牌名 → 最近一次打出的 tick
        self.history = []                  # 最近的出牌 [牌名]（技能不算）
        self.counts = {}                   # 牌名 → 看到打出的次数（算对手进化牌的充能）
        self.elixir = float(self.cfg.start_elixir)
        self.tick = 0

    def rate(self, tick):
        c = self.cfg
        r = c.elixir_per_tick
        return r[0] if tick < c.double_elixir_tick else (r[1] if tick < c.triple_elixir_tick else r[2])

    def advance(self, tick):
        """把估计推进到 tick（逐段按回复速度累加，截断在上限）。"""
        c = self.cfg
        while self.tick < tick:
            nxt = tick
            for b in (c.double_elixir_tick, c.triple_elixir_tick):
                if self.tick < b < nxt:
                    nxt = b
            self.elixir = min(c.elixir_max, self.elixir + self.rate(self.tick) * (nxt - self.tick))
            self.tick = nxt

    def on_play(self, card, form, tick, cost):
        """对手打出了 card（tick = 可见时刻）。"""
        self.advance(tick)
        self.elixir = max(0.0, self.elixir - cost)
        if card not in [c for c, _ in self.seen] and len(self.seen) < MAX_SEEN:
            self.seen.append((card, form))
        self.last_tick[card] = tick
        self.history.append(card)
        self.counts[card] = self.counts.get(card, 0) + 1

    def on_ability(self, tick, cost):
        self.advance(tick)
        self.elixir = max(0.0, self.elixir - cost)

    def on_collector(self, tick, n):
        """对手的圣水采集器产出了 n 点（见 CollectorWatch）。"""
        self.advance(tick)
        self.elixir = min(float(self.cfg.elixir_max), self.elixir + n)

    def recent4(self):
        return set(self.history[-4:])

    def features(self, tick, card_id_fn, form_id_fn, since_norm_s=30.0):
        """→ (ids [8] int, forms [8] int, num [8, 3] float：是否已出现、是否在最近 4 次、距上次打出秒数/归一化)。"""
        ids = np.zeros(MAX_SEEN, np.int16)
        forms = np.zeros(MAX_SEEN, np.int8)
        num = np.zeros((MAX_SEEN, 3), np.float32)
        r4 = self.recent4()
        for i, (c, f) in enumerate(self.seen):
            ids[i] = card_id_fn(c)
            forms[i] = form_id_fn(f)
            num[i, 0] = 1.0
            num[i, 1] = 1.0 if c in r4 else 0.0
            num[i, 2] = min(1.0, (tick - self.last_tick[c]) / 20.0 / since_norm_s)
        return ids, forms, num

    def ext_features(self, charge_fn):
        """→ [8, 2]（和 features 同顺序）：每张见过的牌 [下一次打出是不是特殊形态, 充能进度]；charge_fn = features.form_charge。"""
        out = np.zeros((MAX_SEEN, 2), np.float32)
        for i, (c, f) in enumerate(self.seen):
            out[i] = charge_fn(c, f, self.counts.get(c, 0))
        return out

    def elixir_estimate(self, tick):
        self.advance(tick)
        return self.elixir


class CollectorWatch:
    """对手圣水采集器的产出（只用场上可见的信息：采集器对象和它的年龄）。update 每次传当前全部对手采集器 {对象编号: 年龄 ms}。"""

    def __init__(self, interval_ms):
        self.interval = int(interval_ms)
        self.reset()

    def reset(self):
        self.prev = None                    # 对象编号 → 已产出次数；None = 还没看过（第一次只记基线）

    def update(self, ages):
        cur = {k: int(a) // self.interval for k, a in ages.items()}
        gain = 0
        if self.prev is not None:
            gain = sum(max(0, n - self.prev.get(k, 0)) for k, n in cur.items())
            gain += sum(1 for k in self.prev if k not in cur)      # 消失：ManaOnDeath 1 点
        self.prev = cur
        return gain
