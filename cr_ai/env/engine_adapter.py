"""cr_ai/env/engine_adapter.py — 引擎适配层：cr_ai 和引擎（crx）之间的唯一接口。不修改引擎，只调用它的公开 API 和读它的内存。

    ad = EngineAdapter(cfg)                       # 一个进程只能有一个（引擎内存是进程级全局的）
    ad.reset(seed, decks=[D0, D1], forms=[F0, F1], tower_troop=['tower-princess', 'cannoneer'])
    ad.tick(5); u = ad.units()                    # 单位列表（UNIT_DTYPE 结构化数组，世界坐标）
    ok, code = ad.play(p, 'Wizard', x, y)          # 严格出牌：牌必须在手里；命令约 20 tick 后生效

单位列表字段（世界坐标、玩家 0 在下半场）：
  oid 对象编号（整局唯一）、uid 单位词表编号、player、cls（见 CLS_*）、air、x/y、vx/vy（世界单位/tick）、hp/maxhp、
  deploy（部署剩余毫秒，0 = 已落地）、radius（法术对象的作用半径）、ab_ready/ab_cost（英雄/冠军技能）。
另记（2026-10-28，扩展观测用）：plays（每名玩家每张牌本局成功打出的次数，进化充能按它算）、ab_uses（每个技能单位发动过几次、
最近一次的 tick）、statuses（track_status 时：单位身上的状态名，按对象编号）。show_effects：区域效果和狂暴瓶子也归成法术（见 tables.visible_effects）。
读取方式：引擎内存在 Python 里是一整块 memoryview，这里在上面套 numpy 视图按对象地址批量取字段（对象布局见 docs/ENGINE_MAP.md §5）。
"""
import numpy as np

from ..config import repo_path
from . import tables

CLS_TROOP, CLS_BUILDING, CLS_TOWER, CLS_KING, CLS_SPELL, CLS_PROJ, CLS_EFFECT = range(7)
UNIT_DTYPE = np.dtype([('oid', 'i4'), ('uid', 'i2'), ('player', 'i1'), ('cls', 'i1'), ('air', 'i1'), ('x', 'i4'), ('y', 'i4'),
                       ('vx', 'f4'), ('vy', 'f4'), ('hp', 'i4'), ('maxhp', 'i4'), ('deploy', 'f4'), ('radius', 'f4'),
                       ('ab_ready', 'i1'), ('ab_cost', 'i1')])
VT_KING, VT_PROJ = 0x19f7ef8, 0x19f7a30
M64 = (1 << 64) - 1
_INSTANCE = [None]


class EngineError(RuntimeError):
    """引擎陷阱（没翻译的函数等）：之后引擎状态不可信，适配层会在下次 reset 时重新装载。"""


class EngineAdapter:
    def __init__(self, cfg):
        if _INSTANCE[0] is not None:
            raise RuntimeError('一个进程只能有一个 EngineAdapter（引擎内存是进程级全局的）')
        _INSTANCE[0] = self
        self.cfg = cfg
        tables.use_cfg(cfg)                 # 数据表跟着引擎快照的版本走
        self.eng = None
        self.b = None
        self._row = {}                      # 数据行地址 → (名字, uid, cls 基础类别, air, deploy_ms, radius, 有技能, 法术对象, 可见的特效)
        self._spell = tables.spell_objects()
        self._fx = tables.visible_effects()
        self._utab = tables.unit_table()
        self._buff_names = set(tables.buff_table())
        self._buff_row = {}                 # 状态数据行地址 → 名字（不是已知状态 → None）
        self.show_effects = bool(cfg.obs.get('show_effects', False))
        self.track_status = False           # GameEnv 按配置打开（扩展观测 + 单位列表时）
        self.plays = ({}, {})
        self.ab_uses = {}
        self.statuses = {}

    # ---------------- 生命周期 ----------------
    def _load(self):
        from crx.engine import Engine
        self.eng = Engine(repo_path(self.cfg.paths.engine_base), backend=self.cfg.engine.backend)
        meta = self.eng.meta                # 驱动只用 world / sim / deck_index；其余（分配器表等，约 120 MB 的 Python 对象）装载后就没用了
        for k in ('sizes', 'free_lists', 'imports', 'stub_names'):
            meta.pop(k, None)
        mv = self.eng.mem.mv
        self.m = self.eng.mem
        self.u8 = np.frombuffer(mv, dtype=np.uint8)
        self.i32 = self.u8.view(np.int32)
        self.u64 = self.u8.view(np.uint64)
        self._row.clear()
        self._buff_row.clear()

    def reset(self, seed, decks, forms=None, tower_troop=None, levels=None):
        """新开一局。decks：[玩家0 的 8 张牌名, 玩家1 的 8 张]；forms：[{牌名: 'evo'|'hero'}, …]；tower_troop：str 或 [玩家0, 玩家1]。"""
        if self.eng is None:
            self._load()
        e = self.cfg.engine
        lv = levels or {}
        tt = tower_troop if tower_troop is not None else e.tower_troop
        try:
            self.b = self.eng.new_battle(int(seed), 'Normal', decks=decks, forms=forms,
                                         king_level=lv.get('king', e.king_level), card_level=lv.get('card', e.card_level),
                                         tower_level=lv.get('tower', e.tower_level), tower_troop=tt)
        except Exception as ex:                                                     # noqa：陷阱后丢掉引擎，下次重新装载
            self.eng = None
            raise EngineError(str(ex)[:300])
        self.decks = [self.b.deck(0), self.b.deck(1)]
        self.kings = [int(self.b.king(0)), int(self.b.king(1))]          # 国王塔（玩家）对象地址，整局不变
        self._stall = 0
        self.ended = False
        self._ab = None
        self._avail = {}
        self.plays = ({}, {})
        self.ab_uses = {}                   # 技能单位的对象编号 → [发动次数, 最近一次的 tick]
        self.statuses = {}

    def drop(self):
        """出错后丢掉引擎（下次 reset 重新装载）。"""
        self.eng = None
        self.b = None

    # ---------------- 推进 ----------------
    def now(self):
        return self.b.now()

    def tick(self, n=1):
        """推进 n 个 tick；引擎判定对局结束后 tick 不再前进，置 ended。"""
        b = self.b
        try:
            for _ in range(n):
                t = b.now()
                b.tick()
                if b.now() == t:
                    self._stall += 1
                    if self._stall >= 3:
                        self.ended = True
                        return
                else:
                    self._stall = 0
        except Exception as ex:                                                     # noqa
            self.drop()
            raise EngineError(str(ex)[:300])

    # ---------------- 玩家状态（真实值：只给适配层/机器人/日志用；观测里对手的这些值不可见） ----------------
    def elixir(self, p):
        return int(self.i32[(self.kings[p] + 0x2f8) // 4]) / 10000.0

    def avail_elixir(self, p):
        """引擎出牌校验口径：floor(圣水) − 已下达未结算的预留（同一 tick 内缓存；出牌后失效）。"""
        key = (p, self.b.now())
        v = self._avail.get(key)
        if v is None:
            if len(self._avail) > 8:
                self._avail.clear()
            v = self._avail[key] = self.b.avail_elixir(p)
        return v

    def hand(self, p):
        """(手牌 4 张牌名, 排队 4 张)：直接读国王塔对象的手牌槽 [+0x210]/计数 [+0x21c] 与排队 [+0x220]/[+0x22c]。"""
        k, i32, u64, dk = self.kings[p], self.i32, self.u64, self.decks[p]
        out = []
        for off in (0x210, 0x220):
            arr = int(u64[(k + off) // 8])
            cnt = int(i32[(k + off + 0xc) // 4])
            idx = i32[arr // 4: arr // 4 + cnt] if cnt > 0 else ()
            out.append([dk[i] if 0 <= i < 8 else None for i in idx])
        return out[0], out[1]

    def deck(self, p):
        return self.decks[p]

    # ---------------- 动作 ----------------
    def play(self, p, card, x, y):
        """严格出牌（牌必须在手里，引擎自己做手牌循环）。返回 (是否入队, 引擎错误码 | 'not_in_hand')。
        注意：同一玩家同一 tick 的第二条出牌命令会被引擎吞掉（入队成功但不生效），调用方要保证每 tick 至多一张。"""
        try:
            self._avail.clear()
            r = self.b.play(p, card, int(x), int(y), strict=True, report=True)
        except Exception as ex:                                                     # noqa
            self.drop()
            raise EngineError(str(ex)[:300])
        if r[0]:
            self.plays[p][card] = self.plays[p].get(card, 0) + 1
        return r

    def abilities(self, p):
        """场上玩家 p 的技能单位 [(对象地址, 单位名, 是否可发动, 技能费用)]（对象表顺序）。"""
        if self._ab is None or self._ab[0] != self.b.now():
            self.units()
        out = []
        for obj, nm, pl, ready, cost in self._ab[1]:
            if pl == p:
                out.append((obj, nm, bool(ready and self.avail_elixir(p) >= cost), cost))
        return out

    def use_ability(self, p, obj):
        try:
            self._avail.clear()
            self._ab = None
            r = self.b.use_ability(p, obj, report=True)
        except Exception as ex:                                                     # noqa
            self.drop()
            raise EngineError(str(ex)[:300])
        if r[0]:
            u = self.ab_uses.setdefault(int(self.i32[(int(obj) + 8) // 4]), [0, -1])
            u[0] += 1
            u[1] = self.b.now()
        return r

    def ability_state(self, obj):
        """技能单位 obj 的 (技能本身就绪（不管圣水）, 发动过几次, 最近一次发动的 tick | -1)。"""
        if self._ab is None or self._ab[0] != self.b.now():
            self.units()
        ready = next((r for o, _n, _p, r, _c in self._ab[1] if o == obj), False)
        uses, last = self.ab_uses.get(int(self.i32[(int(obj) + 8) // 4]), (0, -1))
        return bool(ready), uses, last

    def validate_play(self, p, card, x, y, fill_elixir=True):
        """只做引擎的出牌校验、不入队（放置 mask 探测用）：返回校验码（0 = 合法，19 = 落点非法，13 = 圣水不足……）。
        复刻 driver.play → 0x121c930 入队前的步骤：牌放进 0 号槽、构造命令、cmd->[0x40](tick)、0xdc0bf0(cmd, 玩家对)、cmd->[0x18](cmd, world, 1, 0)。
        与真正 play() 的结果逐格对拍过（192/192 一致）。会改 0 号槽和圣水，只在一次性的探测局里用。"""
        from crx.flat import rt as R
        b, m, F, host = self.b, self.b.m, R.F, self.b.host
        di = b.deck_index[p][card]
        mgr = m.mu64(b.world + 0xa8)
        king = F[0xd85cb0](mgr, p) & M64
        if fill_elixir:
            m.w32(king + 0x2f8, 100000)
        m.w32(m.mu64(king + 0x210), di)
        c = F[0xfe71c0](king, 0) & M64
        tmp = host.malloc(0x40)
        m.w(tmp, bytes(0x40))
        F[0xeca8a0](tmp, c, king, 1)
        cmd = host.malloc(0x60)
        m.w(cmd, bytes(0x60))
        F[0xdc4ce0](cmd, m.mu64(b.sim))
        F[0xdc5520](cmd, tmp)
        F[0xdc52b0](cmd, int(x), int(y))
        vt = m.mu64(cmd)
        R.callp(m.mu64(vt + 0x40), cmd, b.now())
        F[0xdc0bf0](cmd, F[0xd8c520](mgr, p) & M64)
        return R.callp(m.mu64(vt + 0x18), cmd, b.world, 1, 0) & 0xffffffff

    def set_tower_hp(self, p, which, hp):
        """（探测/测试用）把玩家 p 的塔血量改成 hp：which = 'king' | 'left' | 'right'。"""
        for o, x, y, pl in self.b.objects():
            if pl != p:
                continue
            nm = self.b.obj_name(o)
            is_king = self.u64[o // 8] == VT_KING
            if (which == 'king') != bool(is_king) or (not is_king and nm not in tables.TOWER_NAMES):
                continue
            if which in ('left', 'right') and (x < 9000) != (which == 'left'):
                continue
            c = self.m.mu64(self.m.mu64(o + 0x18) + 16)
            self.m.w32(c + 16, int(hp))

    def kill(self, addrs):
        """（探测/测试用）把这些对象的血量改成 0，引擎下一 tick 自己处理死亡。"""
        for o in addrs:
            o = int(o)
            if self.m.mi32(o + 0x24) >= 3:
                c = self.m.mu64(self.m.mu64(o + 0x18) + 16)
                if c:
                    self.m.w32(c + 16, 0)

    # ---------------- 状态读取 ----------------
    def _row_info(self, row, o):
        r = self._row.get(row)
        if r is None:
            nm = self.b.obj_name(o)
            t = self._utab.get(nm, {})
            ab = self.m.mu64(row + 0x710) if row else 0
            has_ab = bool(self.b._heap_ptr(ab) and 0 <= self.m.mi32(ab + 0xb4) <= 10)
            r = (nm, tables.unit_id(nm), t.get('building', False), t.get('air', False), t.get('deploy_ms', 0),
                 t.get('radius', 0.0), has_ab, nm in self._spell, nm in self._fx)
            self._row[row] = r
        return r

    def units(self):
        """当前全部场上对象 → UNIT_DTYPE 数组（世界坐标）。"""
        b, u64, i32 = self.b, self.u64, self.i32
        lst = u64[(b.world + 0x40) // 8]
        n = int(i32[(int(lst) + 0x14) // 4])
        out = np.zeros(n, UNIT_DTYPE)
        if n == 0:
            return out
        arr = int(u64[(int(lst) + 8) // 8])
        objs = u64[arr // 8: arr // 8 + n].astype(np.int64)
        self.last_addr = objs                       # 与返回的单位列表同序的对象地址（探测/测试用）
        out['oid'] = i32[(objs + 8) // 4]
        out['player'] = i32[(objs + 0x78) // 4]
        x = i32[(objs + 0x7c) // 4]
        y = i32[(objs + 0x80) // 4]
        out['x'], out['y'] = x, y
        out['vx'] = x - i32[(objs + 0x84) // 4]
        out['vy'] = y - i32[(objs + 0x88) // 4]
        age = i32[(objs + 0x128) // 4]
        vt = u64[objs // 8]
        rows = u64[(objs + 0x48) // 8]
        ncomp = i32[(objs + 0x24) // 4]
        # 血量：组件槽 2 的 [+16]/[+20]（driver.hp_of）
        has = ncomp >= 3
        hp = np.zeros(n, np.int32)
        mx = np.zeros(n, np.int32)
        if has.any():
            comps = u64[(objs[has] + 0x18) // 8].astype(np.int64)
            c = u64[(comps + 16) // 8].astype(np.int64)
            ok = c != 0
            hh = np.zeros(has.sum(), np.int32)
            mm = np.zeros(has.sum(), np.int32)
            hh[ok] = i32[(c[ok] + 16) // 4]
            mm[ok] = i32[(c[ok] + 20) // 4]
            hp[has], mx[has] = hh, mm
            has = has.copy()
            has[np.flatnonzero(has)[~ok]] = False
        out['hp'], out['maxhp'] = hp, mx
        cls = np.zeros(n, np.int8)
        ab_list = []
        fx_on, st_on = self.show_effects, self.track_status
        if st_on:
            self.statuses = {}
        for i in range(n):
            nm, uid, bld, air, dep_ms, rad, has_ab, spell, fx = self._row_info(int(rows[i]), int(objs[i]))
            spell = spell or (fx_on and fx)
            out['uid'][i] = uid
            out['air'][i] = air
            out['radius'][i] = rad
            if dep_ms:
                out['deploy'][i] = max(0, dep_ms - int(age[i]))
            if vt[i] == VT_KING:
                cls[i] = CLS_KING
            elif nm in tables.TOWER_NAMES:
                cls[i] = CLS_TOWER
            elif vt[i] == VT_PROJ:
                cls[i] = CLS_SPELL if spell else CLS_PROJ
            elif not has[i]:
                cls[i] = CLS_SPELL if spell else CLS_EFFECT
            else:
                cls[i] = CLS_BUILDING if bld else CLS_TROOP
            if st_on and has[i] and cls[i] <= CLS_KING:
                sn = self._statuses(int(objs[i]))
                if sn:
                    self.statuses[int(out['oid'][i])] = sn
            if has_ab and cls[i] == CLS_TROOP:
                code, cost = b.ability_status(int(out['player'][i]), int(objs[i]))
                if code != -1:
                    out['ab_ready'][i] = 1 if code == 0 else 0
                    out['ab_cost'][i] = cost
                    ab_list.append((int(objs[i]), nm, int(out['player'][i]), code == 0, cost))
        out['cls'] = cls
        self._ab = (b.now(), ab_list)               # 技能单位（对象表顺序），abilities() 直接用
        return out

    def _statuses(self, o):
        """单位 o 身上的状态名（只留数据表里有的）。布局（2026-10-28 在引擎里找到，冰法减速/冰冻/电击/毒药都对上）：
        组件 0 = [[o+0x18]]；它的 +0x108 是状态实例指针数组、+0x110 容量、+0x114 个数；实例 +0x18 是状态的数据行（行名在 +0x28）。"""
        hp, u64, i32 = self.b._heap_ptr, self.u64, self.i32
        comps = int(u64[(o + 0x18) // 8])
        if not hp(comps):
            return ()
        c0 = int(u64[comps // 8])
        if not hp(c0):
            return ()
        cnt = int(i32[(c0 + 0x114) // 4])
        if cnt <= 0:
            return ()
        arr, cap = int(u64[(c0 + 0x108) // 8]), int(i32[(c0 + 0x110) // 4])
        if not (hp(arr) and cnt <= cap <= 64):
            return ()
        out = []
        for a in range(arr, arr + 8 * cnt, 8):
            inst = int(u64[a // 8])
            if not hp(inst):
                continue
            row = int(u64[(inst + 0x18) // 8])
            nm = self._buff_row.get(row, 0)
            if nm == 0:
                nm = self._row_name(row) if hp(row) else None
                nm = nm if nm in self._buff_names else None
                self._buff_row[row] = nm
            if nm:
                out.append(nm)
        return tuple(out)

    def _row_name(self, row):
        """数据行的名字（行 +0x28 是短字符串：+4 长度，<8 字节内联在 +8，否则 +8 是指针）；读不出 → None。"""
        a, hp = row + 0x28, self.b._heap_ptr
        n = int(self.i32[(a + 4) // 4])
        if not 1 <= n <= 64:
            return None
        if n >= 8:
            q = int(self.u64[(a + 8) // 8])
            if not hp(q):
                return None
            b = bytes(self.u8[q:q + n])
        else:
            b = bytes(self.u8[a + 8:a + 8 + n])
        try:
            return b.decode('ascii')
        except UnicodeDecodeError:
            return None

    def towers(self, units=None):
        """塔血量比例 {玩家: [国王, 左公主, 右公主]}（被拆 = 0）与国王塔是否被打过。"""
        u = self.units() if units is None else units
        tw = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
        for r in u[(u['cls'] == CLS_KING) | (u['cls'] == CLS_TOWER)]:
            i = 0 if r['cls'] == CLS_KING else (1 if r['x'] < 9000 else 2)
            tw[int(r['player'])][i] = max(0.0, r['hp'] / max(1, r['maxhp']))
        return tw
