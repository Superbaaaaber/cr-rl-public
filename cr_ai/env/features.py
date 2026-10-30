"""cr_ai/env/features.py — 标量与卡牌特征（训练、BC 重模拟、部署推理共用）。

观测里只放部署时真实可得的信息：对手手牌、对手真实圣水、对手牌序都不可见（只用 opponent_tracker 的估计/追踪）。

标量 scal [15]：自己圣水/10、圣水阶段 one-hot（单/双/三倍）、加时、剩余时间（归一化）、我方三塔血量比 [国王, 左, 右]、敌方三塔血量比、
              我方国王塔已激活、敌方国王塔已激活、对手圣水估计/10
手牌 hand [5]（4 张 + 下一张）：牌编号、形态、[费用/10, 当前可出, 还差几秒能出/10]
卡组 deck [8]：牌编号、形态（网络里嵌入后取平均作卡组条件）
对手 opp [8]：opponent_tracker.features
技能 ab [K]：对应英雄/冠军牌的编号、[可发动, 费用/10]
待执行 pend [2]：牌编号、[x/18, y/32, 距执行的 tick/20]；上一个动作 last：牌编号、[x/18, y/32]

扩展观测（model.obs_ext / obs.ext，2026-10-28；以前看不到、人在游戏里看得到的信息）：
技能 ab_ext [K, 3]：[技能本身就绪（不管圣水）, 这个单位发动过技能, 距上次发动秒数/20（没发动过 = 1）]
  ——以前“按过了”和“圣水不够”在观测里一样；英雄冰法按下后要等打中人才落冰块，模型看不到自己已经按过。
手牌 hand_ext [5, 2]、对手 opp_ext [8, 2]：[下一次打出是不是特殊形态, 充能进度]——进化牌要先充能（见 tables.evo_cycles），
  以前观测里形态永远标“进化”；英雄形态每次都是英雄（实测），记 [1, 1]。对手的按看得到的出牌次数算。
单位状态 ent_ext [单位, N_STATUS]（见 status_features）：减速/冻结/狂暴/中毒/隐身……以前一概看不到。
"""
import numpy as np

from . import tables

N_SCAL = 15
N_HAND = 5
MAX_PEND = 2


def form_id(f):
    return tables.FORMS.get(f, 0)


def phase(tick, ec):
    return 0 if tick < ec.double_elixir_tick else (1 if tick < ec.triple_elixir_tick else 2)


def time_left(tick, ec):
    if tick < ec.regulation_ticks:
        return (ec.regulation_ticks - tick) / ec.regulation_ticks
    ot = ec.max_ticks - ec.regulation_ticks
    return max(0.0, (ec.max_ticks - tick) / max(1, ot))


def king_active(tw):
    """国王塔已激活：被打过，或己方任一公主塔被拆（规则近似；引擎里没有直接读激活标志）。"""
    return 1.0 if (tw[0] < 0.999 or tw[1] <= 0 or tw[2] <= 0) else 0.0


def scalars(tick, own_elixir, tw_own, tw_enemy, opp_elixir_est, ec, king_fired=None):
    """king_fired：(我方, 敌方) 国王塔是否已经开过火（obs.king_shots，见 GameEnv）；开过火的国王塔一定已激活
    （龙卷风把敌兵拉进国王射程激活、规则近似 king_active 算不到的情况）。None = 只用规则近似（旧行为）。"""
    s = np.zeros(N_SCAL, np.float32)
    s[0] = own_elixir / ec.elixir_max
    s[1 + phase(tick, ec)] = 1.0
    s[4] = 1.0 if tick >= ec.regulation_ticks else 0.0
    s[5] = time_left(tick, ec)
    s[6:9] = tw_own
    s[9:12] = tw_enemy
    s[12] = king_active(tw_own)
    s[13] = king_active(tw_enemy)
    if king_fired is not None:
        s[12] = max(s[12], float(bool(king_fired[0])))
        s[13] = max(s[13], float(bool(king_fired[1])))
    s[14] = opp_elixir_est / ec.elixir_max
    return s


def elixir_rate(tick, ec):
    return ec.elixir_per_tick[phase(tick, ec)]


def hand_features(hand, nxt, forms, avail_elixir, elixir, tick, ec, wait_norm_s, playable):
    """hand：4 张牌名（可能为 None）；nxt：下一张；playable：[4] bool（mask 里该牌是否可出）。"""
    ids = np.zeros(N_HAND, np.int16)
    fm = np.zeros(N_HAND, np.int8)
    num = np.zeros((N_HAND, 3), np.float32)
    rate = elixir_rate(tick, ec) * 20.0                          # 每秒
    for i, c in enumerate(list(hand[:4]) + [nxt]):
        if not c:
            continue
        ids[i] = tables.card_id(c)
        fm[i] = form_id(forms.get(c))
        cost = tables.card_cost(c)
        num[i, 0] = cost / 10.0
        num[i, 1] = 1.0 if (i < 4 and playable[i]) else 0.0
        need = max(0.0, cost - elixir)
        num[i, 2] = min(1.0, need / max(rate, 1e-6) / wait_norm_s)
    return ids, fm, num


def deck_features(deck, forms):
    ids = np.array([tables.card_id(c) for c in deck], np.int16)
    fm = np.array([form_id(forms.get(c)) for c in deck], np.int8)
    return ids, fm


AB_EXT = 3
FORM_EXT = 2
N_STATUS = 11


def form_charge(card, form, n_played):
    """(下一次打出是不是特殊形态, 充能进度)。n_played = 这张牌本局已打出的次数。"""
    if form == 'hero':
        return 1.0, 1.0
    if form != 'evo':
        return 0.0, 0.0
    c = tables.evo_cycles(card) or 2
    k = n_played % (c + 1)
    return (1.0 if k == c else 0.0), k / c


def hand_ext(hand, nxt, forms, plays):
    """手牌 4 张 + 下一张的 [下一次是特殊形态, 充能进度]；plays = {牌名: 本局已打出次数}。"""
    out = np.zeros((N_HAND, FORM_EXT), np.float32)
    for i, c in enumerate(list(hand[:4]) + [nxt]):
        if c:
            out[i] = form_charge(c, forms.get(c), plays.get(c, 0))
    return out


def ability_ext(ready, uses, last_tick, tick):
    """[技能本身就绪, 发动过, 距上次发动秒数/20（没发动过 = 1）]。"""
    since = 1.0 if last_tick is None or last_tick < 0 else min(1.0, (tick - last_tick) / 20.0 / 20.0)
    return (1.0 if ready else 0.0, 1.0 if uses else 0.0, since)


def _mult_effect(v):
    """数据表里的倍率 → 相对正常的变化：负数是“减百分之几”（减速 −30 → −0.3、冻结 −100 → −1），≥ 100 是“变成百分之几”（狂暴 130 → +0.3）。"""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return 0.0
    return v / 100.0 - 1.0 if v >= 100 else v / 100.0


_STATUS_CACHE = {}


def status_features(names):
    """单位身上的状态名（引擎读出来的，见 engine_adapter.statuses）→ [N_STATUS]：
    0 有状态、1 移速变化、2 攻速变化、3 出兵/装填速度变化（各项相加后截到 [−1, 1]）、4 持续掉血（中毒、龙卷风……）、5 回血、
    6 隐身、7 减伤/不死、8 被拉扯（龙卷风）、9 伤害倍率变化、10 状态个数/4。"""
    key = (tables.LOGIC, tuple(names))                  # 数据表跟着引擎快照换版本时数值会变
    v = _STATUS_CACHE.get(key)
    if v is not None:
        return v
    bt = tables.buff_table()
    v = np.zeros(N_STATUS, np.float32)
    if names:
        v[0] = 1.0
        v[10] = min(1.0, len(names) / 4.0)
    for n in names:
        d = bt.get(n)
        if d is None:
            continue
        v[1] += _mult_effect(d.get('SpeedMultiplier', 0))
        v[2] += _mult_effect(d.get('HitSpeedMultiplier', 0))
        v[3] += _mult_effect(d.get('SpawnSpeedMultiplier', 0))
        dps = d.get('DamagePerSecond', 0)
        v[4] = max(v[4], 1.0 if isinstance(dps, (int, float)) and dps > 0 else 0.0)
        hps = d.get('HealPerSecond', 0)
        v[5] = max(v[5], 1.0 if (isinstance(hps, (int, float)) and hps > 0) or isinstance(hps, str) else 0.0)
        v[6] = max(v[6], 1.0 if d.get('Invisible') is True else 0.0)
        dr = d.get('DamageReduction', 0)
        v[7] = max(v[7], 1.0 if (isinstance(dr, (int, float)) and dr > 0) or 'UNKILLABLE' in str(d.get('GameTagsToSet', '')) else 0.0)
        ap = d.get('AttractPercentage', 0)
        v[8] = max(v[8], 1.0 if isinstance(ap, (int, float)) and ap > 0 else 0.0)
        v[9] += _mult_effect(d.get('DamageMultiplier', 0))
    for i in (1, 2, 3, 9):
        v[i] = min(1.0, max(-1.0, v[i]))
    _STATUS_CACHE[key] = v
    return v


def cell_xy(cell):
    """cell → (x/18, y/32)，cell < 0 → (0, 0)。"""
    if cell is None or cell < 0:
        return 0.0, 0.0
    return (cell % 18 + 0.5) / 18.0, (cell // 18 + 0.5) / 32.0
