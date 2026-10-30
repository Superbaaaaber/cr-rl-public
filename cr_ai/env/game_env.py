"""cr_ai/env/game_env.py — 单局双人环境（一个进程一个，因为引擎内存是进程级全局的；并行见 vec_env.py）。

    env = GameEnv(cfg)
    obs = env.reset(seed, decks=[D0, D1], forms=[F0, F1], tower_troop=['tower-princess', 'cannoneer'])
    obs, rew, done, info = env.step([(sel0, cell0), (sel1, cell1)])

动作（每名玩家）：(sel, cell)。sel：0 不出、1..4 打第 k 张手牌、5..4+K 发动第 k 个技能；cell：0..575（玩家视角，cy*18+cx），不出/技能时忽略。
一步 = frame_skip(5) 个 tick = 0.25 s。决策进动作延迟队列，到期后在本步开头逐 tick 执行（同一玩家每 tick 至多一条出牌命令——
引擎会吞掉同 tick 的第二条），执行前重新检查合法性，不合法则作废并计数。出牌命令约 20 tick 后由引擎生效（单位出生、扣费、手牌补位）；
在那之前这张牌仍在引擎的手牌槽里，环境把它记为“待执行”，mask 里不允许重复打出。
对手出牌在命令生效时对另一方可见（对手追踪器、圣水估计在那一刻更新）。
观测（每名玩家一份 dict，形状见 OBS_SPEC）：网格 gi/gf、标量、手牌、卡组、对手追踪、技能、待执行、上一个动作、牌头 mask、放置类别与塔状态。
2026-10-28 的观测修补（默认都关 = 旧观测）：obs.show_effects（狂暴、技能区域等以前被丢掉的对象画进法术层和单位列表）、
obs.king_shots（国王塔开过火就算已激活）、model.obs_ext / obs.ext（扩展字段 ab_ext、hand_ext、opp_ext、ent_ext，见 features.py）。
"""
import numpy as np

from . import features as FT
from . import masks as MK
from . import tables
from .engine_adapter import CLS_KING, CLS_TOWER, EngineAdapter
from .latency import ActionQueue, ObsDelay
from .opponent_tracker import MAX_SEEN, CollectorWatch, OpponentTracker
from .rasterize import GH, GW, MAX_ENT, N_ENT_F, N_FLOAT, N_INT, entities, entity_status, rasterize
from .reward import potential, step_reward

NCELL = GW * GH


def wants_entities(cfg):
    """观测里要不要带单位列表：网络用实体分支（model.entity），或者显式要求（obs.entities，例如 arena 里对手的网络要）。"""
    return bool(cfg.model.get('entity', False) or cfg.obs.get('entities', False))


def wants_ext(cfg):
    """观测里要不要带扩展字段：网络用（model.obs_ext），或者显式要求（obs.ext，例如 arena 里对手的网络要）。"""
    return bool(cfg.model.get('obs_ext', False) or cfg.obs.get('ext', False))


def obs_spec(cfg):
    K = cfg.env.max_abilities
    spec = {
        'gi': ((N_INT, GH, GW), np.int16), 'gf': ((N_FLOAT, GH, GW), np.float16), 'scal': ((FT.N_SCAL,), np.float32),
        'hand_id': ((FT.N_HAND,), np.int16), 'hand_form': ((FT.N_HAND,), np.int8), 'hand_num': ((FT.N_HAND, 3), np.float32),
        'deck_id': ((8,), np.int16), 'deck_form': ((8,), np.int8),
        'opp_id': ((MAX_SEEN,), np.int16), 'opp_form': ((MAX_SEEN,), np.int8), 'opp_num': ((MAX_SEEN, 3), np.float32),
        'ab_id': ((K,), np.int16), 'ab_num': ((K, 2), np.float32),
        'pend_id': ((FT.MAX_PEND,), np.int16), 'pend_num': ((FT.MAX_PEND, 3), np.float32),
        'last_id': ((1,), np.int16), 'last_num': ((2,), np.float32),
        'card_mask': ((5 + K,), np.bool_), 'place_cls': ((4,), np.int16), 'tstate': ((), np.int16), 'pl': ((), np.int8),
        'forced_noop': ((), np.bool_), 'tick': ((), np.int32),
    }
    if wants_entities(cfg):
        spec['ent_id'] = ((MAX_ENT, 2), np.int16)
        spec['ent_num'] = ((MAX_ENT, N_ENT_F), np.float16)
    if wants_ext(cfg):
        spec['ab_ext'] = ((K, FT.AB_EXT), np.float32)
        spec['hand_ext'] = ((FT.N_HAND, FT.FORM_EXT), np.float32)
        spec['opp_ext'] = ((MAX_SEEN, FT.FORM_EXT), np.float32)
        if wants_entities(cfg):
            spec['ent_ext'] = ((MAX_ENT, FT.N_STATUS), np.float16)
    if cfg.model.get('priv_value', False):                                # 特权信息（只给价值头）：对手真实手牌 4 张 + 下一张、真实圣水
        spec['priv_id'] = ((5,), np.int16)
        spec['priv_form'] = ((5,), np.int8)
        spec['priv_num'] = ((1,), np.float32)
    return spec


class GameEnv:
    def __init__(self, cfg, adapter=None):
        self.cfg = cfg
        self.ec = cfg.env
        self.oc = cfg.obs
        self.K = cfg.env.max_abilities
        self.ad = adapter or EngineAdapter(cfg)
        self.trk = [OpponentTracker(self.ec), OpponentTracker(self.ec)]       # trk[p]：p 眼里的对手
        self.colw = [CollectorWatch(tables.collector_interval_ms()) for _ in (0, 1)]   # colw[p]：p 眼里对手的圣水采集器
        self.aq = [ActionQueue(), ActionQueue()]
        self.od = [ObsDelay(), ObsDelay()]
        self.reward_coef = 1.0                      # 塑形系数 c（训练器按步数退火后写进来）
        self.configure(cfg)

    def configure(self, cfg):
        """按配置定观测内容（构造时调用；测试里换配置也用它——一个进程只能有一个引擎）。"""
        self.cfg = cfg
        self.oc = cfg.obs
        self.spec = obs_spec(cfg)
        self.ent = wants_entities(cfg)
        self.ext = wants_ext(cfg)
        self.priv = bool(cfg.model.get('priv_value', False))
        self.king_shots = bool(cfg.obs.get('king_shots', False))
        self.ad.show_effects = bool(cfg.obs.get('show_effects', False))
        self.ad.track_status = self.ext and self.ent
        self._king_proj = tables.unit_id('KingProjectile')

    # ---------------- 回合 ----------------
    def reset(self, seed, decks, forms=None, tower_troop=None, d_act=None, d_obs=None, levels=None):
        forms = forms or [{}, {}]
        self.ad.reset(seed, decks, forms, tower_troop=tower_troop, levels=levels)
        self.decks = [self.ad.deck(0), self.ad.deck(1)]
        self.forms = [dict(forms[0] or {}), dict(forms[1] or {})]
        rng = np.random.RandomState(seed % (2 ** 31))
        la = self.cfg.latency
        for p in (0, 1):
            self.trk[p].reset()
            self.colw[p].reset()
            self.aq[p].reset(d_act if d_act is not None else rng.choice(la.d_act_steps))
            self.od[p].reset(d_obs if d_obs is not None else rng.choice(la.d_obs_steps))
        self.inflight = {0: [], 1: []}                  # 已下达、引擎尚未生效的命令：dict(exec, card, cell, cost, kind)
        self.last = {0: (None, -1), 1: (None, -1)}
        self.t = 0
        self.done = False
        self.winner = None
        self.king_fired = [False, False]                # 国王塔开过火（obs.king_shots）
        # ab_ready：有技能可以发动的决策步数（和 abilities 一起看技能使用率）；ab_by / ab_ready_by：按技能单位名分开的发动次数 / 可以按的步数
        self.stats = dict(plays=[0, 0], abilities=[0, 0], void=[0, 0], forced_noop=[0, 0], rejected=[0, 0], ab_ready=[0, 0],
                          ab_by=[{}, {}], ab_ready_by=[{}, {}])
        self._refresh()
        self.phi = [self._phi(0), self._phi(1)]
        return [self._obs(0), self._obs(1)]

    def _refresh(self):
        self.units = self.ad.units()
        self.tick = self.ad.now()
        self.tw = self.ad.towers(self.units)
        self._legal_cache = {}
        if self.king_shots:
            u = self.units
            kp = u['player'][u['uid'] == self._king_proj]
            for p in (0, 1):
                if (kp == p).any():
                    self.king_fired[p] = True

    def _phi(self, p):
        return potential(self.tw[p], self.tw[1 - p], self.cfg.reward)

    # ---------------- 合法性 ----------------
    def legal(self, p):
        """→ (牌头 mask [5+K], 手牌放置 mask [4, 576], 技能单位 [(对象, 单位名, 牌名, 可发动, 费用)] (≤K), 手牌, 下一张, 塔状态)。"""
        c = self._legal_cache.get(p)
        if c is not None:
            return c
        ad, ec = self.ad, self.ec
        hand, queue = ad.hand(p)
        hand = list(hand[:4]) + [None] * (4 - len(hand[:4]))
        st = MK.tower_state(self.units, p)
        pm = MK.hand_place_masks([tables.card_id(h) for h in hand], p, st)
        cm = np.zeros(5 + self.K, bool)
        cm[0] = True
        busy = {f['card'] for f in self.inflight[p]} | {a[1][1] for a in self.aq[p].pending() if a[1][0] == 'card'}
        avail = ad.avail_elixir(p)
        if self.tick >= ec.min_deploy_tick:
            for s, card in enumerate(hand):
                if card and card != 'Mirror' and card not in busy and tables.card_cost(card) <= avail and pm[s].any():
                    cm[1 + s] = True
        abil = []
        deck = self.decks[p]
        for obj, nm, ok, cost in ad.abilities(p)[:self.K]:
            abil.append((obj, nm, tables.ability_card(nm, deck), ok, cost))
        for k, a in enumerate(abil):
            cm[5 + k] = a[3] and not any(x[1][0] == 'ab' for x in self.aq[p].pending())
        c = (cm, pm, abil, hand, queue[0] if queue else None, st)
        self._legal_cache[p] = c
        return c

    # ---------------- 步进 ----------------
    def _cell_world(self, p, cell):
        cy, cx = divmod(int(cell), GW)
        wy = (GH - 1 - cy) if p == 1 else cy
        return cx * 1000 + 500, wy * 1000 + 500

    def _decide(self, p, act):
        """决策 → 延迟队列里的执行项。"""
        sel, cell = (act if act is not None else (0, -1))
        sel = int(sel)
        if sel == 0:
            return
        cm, pm, abil, hand, _n, _st = self.legal(p)
        if not cm[sel]:
            self.aq[p].void('illegal_at_decision')
            self.stats['void'][p] += 1
            return
        if sel <= 4:
            self.aq[p].push(self.t, ('card', hand[sel - 1], int(cell)))
        else:
            obj, nm, card, ok, cost = abil[sel - 5]
            self.aq[p].push(self.t, ('ab', nm, card))

    def _execute(self, p, item):
        """在当前 tick 执行一条到期的动作；不合法则作废。返回是否下达了出牌命令。"""
        ad = self.ad
        if item[0] == 'card':
            _, card, cell = item
            hand, _q = ad.hand(p)
            busy = {f['card'] for f in self.inflight[p]}
            reason = None
            if card not in hand[:4]:
                reason = 'not_in_hand'
            elif card in busy:
                reason = 'in_flight'
            elif tables.card_cost(card) > ad.avail_elixir(p):
                reason = 'elixir'
            elif not MK.place_mask(tables.card_id(card), p, MK.tower_state(self.units, p))[cell]:
                reason = 'place'
            if reason is None:
                x, y = self._cell_world(p, cell)
                ok, code = ad.play(p, card, x, y)
                if ok:
                    self.inflight[p].append(dict(exec=ad.now() + self.ec.deploy_delay_ticks, card=card, cell=cell, cost=tables.card_cost(card), kind='card'))
                    self.last[p] = (card, cell)
                    self.stats['plays'][p] += 1
                    return True
                reason = 'engine_%s' % code
                self.stats['rejected'][p] += 1
            self.aq[p].void(reason)
            self.stats['void'][p] += 1
            return False
        _, nm, card = item
        for obj, unm, ok, cost in ad.abilities(p):
            if unm == nm and ok:
                okk, code = ad.use_ability(p, obj)
                if okk:
                    self.inflight[p].append(dict(exec=ad.now() + self.ec.deploy_delay_ticks, card=card, cell=-1, cost=cost, kind='ab'))
                    self.stats['abilities'][p] += 1
                    ab = self.stats['ab_by'][p]
                    ab[nm] = ab.get(nm, 0) + 1
                    return False
                break
        self.aq[p].void('ability')
        self.stats['void'][p] += 1
        return False

    def _reveal(self):
        now = self.ad.now()
        for p in (0, 1):
            keep = []
            for f in self.inflight[p]:
                if f['exec'] <= now:
                    if f['kind'] == 'card':
                        self.trk[1 - p].on_play(f['card'], self.forms[p].get(f['card']), f['exec'], f['cost'])
                    else:
                        self.trk[1 - p].on_ability(f['exec'], f['cost'])
                else:
                    keep.append(f)
            self.inflight[p] = keep

    def _collectors(self):
        """对手圣水采集器的产出计入 p 眼里的对手圣水估计（场上可见：采集器对象和它的年龄）。"""
        u, ad = self.units, self.ad
        sel = np.flatnonzero(u['uid'] == tables.unit_id('ElixirCollector')) if len(u) else []
        addr = getattr(ad, 'last_addr', None)
        for p in (0, 1):
            ages = {int(u['oid'][i]): int(ad.i32[(int(addr[i]) + 0x128) // 4]) for i in sel if int(u['player'][i]) == 1 - p}
            g = self.colw[p].update(ages)
            if g:
                self.trk[p].on_collector(self.tick, g)

    def step(self, actions):
        assert not self.done, '先 reset'
        for p in (0, 1):
            if self._legal_cache.get(p) is not None:
                cm = self.legal(p)[0]
                if not cm[1:].any():
                    self.stats['forced_noop'][p] += 1
                if cm[5:].any():
                    self.stats['ab_ready'][p] += 1
                    rb = self.stats['ab_ready_by'][p]
                    for k, a in enumerate(self.legal(p)[2]):
                        if cm[5 + k]:
                            rb[a[1]] = rb.get(a[1], 0) + 1
            self._decide(p, actions[p] if actions is not None else None)
        due = {p: self.aq[p].pop_due(self.t) for p in (0, 1)}
        ad = self.ad
        for k in range(self.ec.frame_skip):
            for p in (0, 1):
                # 每 tick 每名玩家至多一条出牌命令；技能命令不受限
                while due[p]:
                    item = due[p].pop(0)
                    if self._execute(p, item):
                        break
            ad.tick(1)
            self._reveal()
            if ad.ended or ad.now() >= self.ec.max_ticks:
                break
        for p in (0, 1):                                 # 本步没轮到执行的（同步多条）留到下一步开头
            for item in reversed(due[p]):
                self.aq[p].q.appendleft((self.t + 1, item))
        self.t += 1
        self._refresh()
        self._collectors()
        done = bool(ad.ended or self.tick >= self.ec.max_ticks or self.tw[0][0] <= 0 or self.tw[1][0] <= 0)
        crowns = self.crowns()
        if done:
            self.done = True
            self.winner = self._winner(crowns)
        rew, parts = [0.0, 0.0], [None, None]
        for p in (0, 1):
            phi = self._phi(p)
            outcome = 0 if not done else (0 if self.winner == -1 else (1 if self.winner == p else -1))
            rew[p], parts[p] = step_reward(self.phi[p], phi, done, outcome, ad.elixir(p) >= self.ec.elixir_max - 1e-6, self.reward_coef, self.cfg.reward)
            self.phi[p] = phi
        info = dict(tick=self.tick, crowns=crowns, winner=self.winner, reward_parts=parts)
        return [self._obs(0), self._obs(1)], rew, done, info

    def crowns(self):
        out = [0, 0]
        for p in (0, 1):
            e = self.tw[1 - p]
            out[p] = 3 if e[0] <= 0 else int(e[1] <= 0) + int(e[2] <= 0)
        return out

    def _winner(self, crowns):
        """冠数多的赢。冠数相同（加时打满）按引擎的规则：加时结束后所有塔以同样速度掉血，先倒的那座塔的一方输
        ——等价于比双方存活塔里最低的绝对血量，低的一方输，相等为平局（2026-10-16 在引擎里实测；以前按三塔血量比例之和判，
        会把“双方最低塔一样、但一方国王塔掉了血”这种局判成胜负，引擎判平）。环境在 max_ticks 截停，引擎来不及自己掉血判胜负，所以在这里算。"""
        if crowns[0] != crowns[1]:
            return 0 if crowns[0] > crowns[1] else 1
        u = self.units
        t = u[((u['cls'] == CLS_TOWER) | (u['cls'] == CLS_KING)) & (u['hp'] > 0)]
        m = [int(t['hp'][t['player'] == p].min()) if (t['player'] == p).any() else 0 for p in (0, 1)]
        if m[0] != m[1]:
            return 0 if m[0] > m[1] else 1
        return -1

    # ---------------- 观测 ----------------
    def _obs(self, p):
        cm, pm, abil, hand, nxt, st = self.legal(p)
        ad, ec, oc = self.ad, self.ec, self.oc
        gi, gf = rasterize(self.units, p)
        trk = self.trk[p]
        o = {'gi': gi, 'gf': gf}
        kf = (self.king_fired[p], self.king_fired[1 - p]) if self.king_shots else None
        o['scal'] = FT.scalars(self.tick, ad.elixir(p), self.tw[p], self.tw[1 - p], trk.elixir_estimate(self.tick), ec, king_fired=kf)
        o['hand_id'], o['hand_form'], o['hand_num'] = FT.hand_features(hand, nxt, self.forms[p], ad.avail_elixir(p), ad.elixir(p), self.tick, ec, oc.wait_norm_s, cm[1:5])
        o['deck_id'], o['deck_form'] = FT.deck_features(self.decks[p], self.forms[p])
        o['opp_id'], o['opp_form'], o['opp_num'] = trk.features(self.tick, tables.card_id, FT.form_id, oc.since_norm_s)
        ab_id = np.zeros(self.K, np.int16)
        ab_num = np.zeros((self.K, 2), np.float32)
        for k, (obj, nm, card, ok, cost) in enumerate(abil):
            ab_id[k] = tables.card_id(card) if card else tables.UNK
            ab_num[k] = (1.0 if cm[5 + k] else 0.0, cost / 10.0)
        o['ab_id'], o['ab_num'] = ab_id, ab_num
        pid = np.zeros(FT.MAX_PEND, np.int16)
        pnum = np.zeros((FT.MAX_PEND, 3), np.float32)
        pend = [(f['card'], f['cell'], f['exec'] - self.tick) for f in self.inflight[p]]
        pend += [(a[1][1] if a[1][0] == 'card' else a[1][2], a[1][2] if a[1][0] == 'card' else -1, (a[0] - self.t) * ec.frame_skip + ec.deploy_delay_ticks)
                 for a in self.aq[p].pending()]
        for i, (card, cell, dt) in enumerate(pend[:FT.MAX_PEND]):
            pid[i] = tables.card_id(card)
            x, y = FT.cell_xy(cell)
            pnum[i] = (x, y, max(0.0, dt) / 20.0)
        o['pend_id'], o['pend_num'] = pid, pnum
        lc, lcell = self.last[p]
        o['last_id'] = np.array([tables.card_id(lc)], np.int16)
        o['last_num'] = np.array(FT.cell_xy(lcell), np.float32)
        o['card_mask'] = cm
        o['place_cls'] = MK.place_class([tables.card_id(h) for h in hand]).astype(np.int16)
        o['tstate'] = np.int16(st)
        o['pl'] = np.int8(p)
        o['forced_noop'] = np.bool_(not cm[1:].any())
        o['tick'] = np.int32(self.tick)
        if self.ent:
            o['ent_id'], o['ent_num'] = entities(self.units, p)
        if self.ext:
            ab_ext = np.zeros((self.K, FT.AB_EXT), np.float32)
            for k, (obj, nm, card, ok, cost) in enumerate(abil):
                ready, uses, last = ad.ability_state(obj)
                ab_ext[k] = FT.ability_ext(ready, uses, last, self.tick)
            o['ab_ext'] = ab_ext
            o['hand_ext'] = FT.hand_ext(hand, nxt, self.forms[p], ad.plays[p])
            o['opp_ext'] = trk.ext_features(FT.form_charge)
            if self.ent:
                o['ent_ext'] = entity_status(self.units, ad.statuses)
        if self.priv:
            q = 1 - p
            ohand, oqueue = ad.hand(q)
            cards = (list(ohand[:4]) + [None] * 4)[:4] + [oqueue[0] if oqueue else None]
            o['priv_id'] = np.array([tables.card_id(c) if c else 0 for c in cards], np.int16)
            o['priv_form'] = np.array([FT.form_id(self.forms[q].get(c)) if c else 0 for c in cards], np.int8)
            o['priv_num'] = np.array([ad.elixir(q) / ec.elixir_max], np.float32)
        return self.od[p].push(o)

    # ---------------- 给机器人/日志用（真实状态） ----------------
    def view_units(self, p):
        from .rasterize import to_view
        return to_view(self.units, p)
