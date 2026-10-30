"""生成 cr_ai/assets 下的静态资源（生成一次、入库；词表决定嵌入编号，生成后不要再改）。

    .venv\\Scripts\\python.exe -m cr_ai.tools.build_assets [--vocab] [--masks]

- unit_vocab.json / card_vocab.json：单位词表（含衍生单位、塔、弹道、区域效果）/ 卡牌词表（引擎牌池）。已存在则不覆盖（--force 才重建）。
- place_masks.npz：每张牌的合法落点，用引擎自己的出牌校验逐格探测（EngineAdapter.validate_play，不入队）。
    masks[类别, 玩家, 塔状态, 32, 18]（玩家视角：自己在下方）；card_cls[卡牌编号] = 类别。
    塔状态（玩家视角，16 种）= 敌方左公主塔已拆(1) + 敌方右(2) + 己方左(4) + 己方右(8)（左右 = 世界 x）。
    推掉敌方公主塔后兵/建筑可放区域向对方半场扩展；己方公主塔被拆后它的占位格可以下。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from cr_ai.config import load
from cr_ai.env import tables

GW, GH = 18, 32
N_STATE = 16


def config_bits(c):
    """对局配置 c → 每名玩家被拆的塔：{玩家: (左, 右)}。"""
    return {0: (bool(c & 1), bool(c & 2)), 1: (bool(c & 4), bool(c & 8))}


def view_state(c, p):
    """对局配置 c 下玩家 p 视角的塔状态编号。"""
    d = config_bits(c)
    (el, er), (ol, orr) = d[1 - p], d[p]
    return int(el) + 2 * int(er) + 4 * int(ol) + 8 * int(orr)


def setup_config(ad, deck, c):
    ad.reset(seed=1, decks=[deck, deck], tower_troop='tower-princess')
    ad.tick(100)
    for pl, (l, r) in config_bits(c).items():
        if l:
            ad.set_tower_hp(pl, 'left', 0)
        if r:
            ad.set_tower_hp(pl, 'right', 0)
    ad.tick(10)
    tw = ad.towers()
    for pl, (l, r) in config_bits(c).items():
        if (tw[pl][1] > 0) == l or (tw[pl][2] > 0) == r:
            raise RuntimeError('拆塔没生效：配置 %d 玩家 %d 塔 %s' % (c, pl, tw))


def scan_names(cfg, games=80, seed=7):
    """用引擎跑随机牌组/形态的随机对局，收集实际出现过的场上对象名（数据表里可能漏掉的衍生对象）。"""
    import json as _json
    from crx.env import card_pool
    from cr_ai.env.engine_adapter import EngineAdapter, EngineError
    cf = _json.load(open(os.path.join(tables.ROOT, 're', 'status', 'card_forms.json'), encoding='utf-8'))
    pool = [c for c in card_pool() if c != 'Mirror']
    rng = np.random.RandomState(seed)
    ad = EngineAdapter(cfg)
    names = set()
    troops = ['tower-princess', 'cannoneer', 'dagger-duchess', 'royal-chef']
    for g in range(games):
        decks = [list(rng.choice(pool, 8, replace=False)) for _ in range(2)]
        forms = [{c: ('hero' if cf[c]['hero'] and rng.rand() < 0.5 else 'evo') for c in d if cf[c]['evo'] or cf[c]['hero']} for d in decks]
        forms = [{c: f for c, f in fm.items() if cf[c][f]} for fm in forms]
        try:
            ad.reset(int(rng.randint(1, 1 << 30)), decks, forms, tower_troop=[troops[g % 4], troops[(g + 1) % 4]])
            while ad.now() < 4000 and not ad.ended:
                for p in (0, 1):
                    if rng.rand() < 0.15:
                        hand, _q = ad.hand(p)
                        c = hand[rng.randint(4)]
                        if c and c != 'Mirror':
                            ad.play(p, c, rng.randint(18) * 1000 + 500, (rng.randint(15) if p == 0 else 31 - rng.randint(15)) * 1000 + 500)
                    for obj, nm, ok, cost in ad.abilities(p):
                        if ok and rng.rand() < 0.3:
                            ad.use_ability(p, obj)
                            break
                ad.tick(5)
                ad.units()
        except EngineError as e:
            print('  第 %d 局引擎出错（跳过）：%s' % (g, str(e)[:100]))
            continue
        names.update(r[0] for r in ad._row.values())
    return names


def build_vocab(cfg, force=False, scan_games=80):
    os.makedirs(tables.ASSETS, exist_ok=True)
    for name, fn in (('unit_vocab', tables.build_unit_vocab), ('card_vocab', tables.build_card_vocab)):
        path = os.path.join(tables.ASSETS, name + '.json')
        if os.path.exists(path) and not force:
            print('已存在，跳过：', path)
            continue
        v = fn()
        if name == 'unit_vocab' and scan_games:
            seen = scan_names(cfg, scan_games)
            extra = sorted(n for n in seen if n not in v and n != '?')
            print('引擎实际出现 %d 种对象，数据表里没有的 %d 种：%s' % (len(seen), len(extra), ', '.join(extra[:40])))
            names = sorted(set(v) | set(extra))
            v = {n: i + 2 for i, n in enumerate(names)}
        json.dump(v, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0, sort_keys=True)
        print('写入 %s（%d 项）' % (path, len(v)))


SPAWN_DELAY = 21
LANDING_CARDS = ('Knight', 'Cannon', 'RoyalDelivery')
SUMMON_TROOP = ('Heal', 'MergeMaiden')              # 数据表里算法术、实际召唤的是兵（治疗精灵、Spirit Empress）：按兵的区域
ANYWHERE = ('Miner', 'GoblinDrill')                 # 钻地出现，可以下全场（只受校验码限制）


def landing_rep(card):
    """这张牌的合法区域还要与哪张代表牌的落点探测结果取交集（None = 只看校验码）。"""
    if card in LANDING_CARDS:
        return card
    if card in ANYWHERE:
        return None
    if card in SUMMON_TROOP:
        return 'Knight'
    k = tables.card_info().get(card, {}).get('kind', 'troop')
    return {'troop': 'Knight', 'building': 'Cannon'}.get(k)


def probe_landing(ad, card):
    """真实出牌探测落点：每 tick 双方各下一张（圣水灌到 100，预留不会卡住），出生后按下达顺序对上格子，立刻清掉单位。
    返回 [玩家, 塔状态, 32, 18]（玩家视角）：落点与目标格中心相差 ≤ 600 的格为合法。"""
    from cr_ai.env.engine_adapter import CLS_BUILDING, CLS_KING, CLS_SPELL, CLS_TOWER, CLS_TROOP
    deck = [card] + [c for c in ('Knight', 'Cannon', 'Fireball', 'Zap', 'Archer', 'Giant', 'Musketeer', 'Log', 'Bomber') if c != card][:7]
    out = np.zeros((2, N_STATE, GH, GW), bool)
    for c, probe_p in [(c, pp) for c in range(N_STATE) for pp in (0, 1)]:   # 两名玩家分开探测：同 tick 往同一格下建筑会互相挤掉
        setup_config(ad, deck, c)
        issued = {0: {}, 1: {}}
        seen = set(ad.units()['oid'].tolist())
        cells = [(cy, cx) for cy in range(GH) for cx in range(GW)]
        i = 0
        t_last = ad.now()
        t_start = time.time()
        for _it in range(len(cells) + 60):
            if ad.ended:
                print('    %s 配置 %d：对局在 tick %d 结束（已探测 %d 格）' % (card, c, ad.now(), i), flush=True)
                break
            t = ad.now()
            if i < len(cells):
                cy, cx = cells[i]
                for p in (probe_p,):
                    ad.m.w32(ad.b.king(p) + 0x2f8, 1000000)
                    wy = cy * 1000 + 500
                    ok, code = ad.b.play(p, card, cx * 1000 + 500, wy, strict=False, slot=0, report=True)
                    if ok:
                        issued[p][t] = (cx * 1000 + 500, wy)
                t_last = t
                i += 1
            ad.tick(1)
            u = ad.units()
            new = ~np.isin(u['oid'], list(seen))
            seen.update(u['oid'][new].tolist())
            for p in (0, 1):
                nm = new & (u['player'] == p) & np.isin(u['cls'], (CLS_TROOP, CLS_BUILDING, CLS_SPELL))
                if nm.any():
                    tt = ad.now() - SPAWN_DELAY                 # 命令下达后正好 21 tick 出生：按 tick 精确对回命令（不按顺序，免得某条命令没生成单位时整体错位）
                    if tt in issued[p]:
                        tx, ty = issued[p].pop(tt)
                        r = u[nm][0]
                        if abs(int(r['x']) - tx) <= 600 and abs(int(r['y']) - ty) <= 600:
                            vy = (GH - 1 - ty // 1000) if p == 1 else ty // 1000
                            out[p, view_state(c, p), vy, tx // 1000] = True
            kill = np.isin(u['cls'], (CLS_TROOP, CLS_BUILDING)) & ~np.isin(u['cls'], (CLS_TOWER, CLS_KING))
            ad.kill(ad.last_addr[kill])
        print('    落点 %s 配置 %d 玩家 %d：%.1fs，没生成单位的命令 %d' % (card, c, probe_p, time.time() - t_start, len(issued[probe_p])), flush=True)
    return out


def probe_masks(cfg):
    from cr_ai.env.engine_adapter import EngineAdapter
    ad = EngineAdapter(cfg)
    cards = sorted(tables.card_vocab(), key=lambda n: tables.card_vocab()[n])
    groups = [cards[i:i + 8] for i in range(0, len(cards), 8)]
    if len(groups[-1]) < 8:                                       # 最后一组凑满 8 张
        groups[-1] = groups[-1] + [c for c in cards if c not in groups[-1]][:8 - len(groups[-1])]
    per_card = {}
    failed = {}
    t0 = time.time()

    for gi, grp in enumerate(groups):
        for c in range(N_STATE):
            setup_config(ad, grp, c)
            for p in (0, 1):
                s = view_state(c, p)
                for card in grp:
                    if card in failed:
                        continue
                    m = np.zeros((GH, GW), bool)
                    try:
                        for cy in range(GH):
                            wy = cy * 1000 + 500
                            vy = (GH - 1 - cy) if p == 1 else cy                 # 视角：玩家 1 沿 y 翻转
                            for cx in range(GW):
                                m[vy, cx] = ad.validate_play(p, card, cx * 1000 + 500, wy) == 0
                    except Exception as ex:                                       # noqa：校验路径撞到没翻译的函数
                        failed[card] = str(ex)[:120]
                        print('  %s 探测失败：%s' % (card, failed[card]), flush=True)
                        ad.drop()
                        setup_config(ad, grp, c)
                        continue
                    per_card.setdefault(card, np.zeros((2, N_STATE, GH, GW), bool))[p, s] = m
        print('组 %d/%d  %.0fs' % (gi + 1, len(groups), time.time() - t0), flush=True)
    # 落点探测：引擎不拒绝放到对方半场的兵/建筑，而是执行时夹回合法区域 → 用真实出牌看落点，没被挪动的格才合法
    landing = {c: probe_landing(ad, c) for c in LANDING_CARDS}
    for c, m in landing.items():
        print('  落点 %-14s 塔状态 0/1/2/3 下合法格（玩家 0 / 1）：%s / %s' % (c, [int(m[0, s].sum()) for s in range(4)], [int(m[1, s].sum()) for s in range(4)]), flush=True)
    # 探测失败的牌：用同种类（兵/建筑/法术）代表牌的 mask
    rep = {'troop': 'Knight', 'building': 'Cannon', 'spell': 'Fireball'}
    for card in failed:
        per_card.pop(card, None)
        per_card[card] = per_card[rep[tables.card_info().get(card, {}).get('kind', 'troop')]]
    json.dump(failed, open(os.path.join(tables.ASSETS, 'place_masks_failed.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for card in list(per_card):
        rep_l = landing_rep(card)
        if rep_l is not None:
            per_card[card] = per_card[card] & landing[rep_l]
    # 合并成类别
    keys, masks, card_cls = {}, [], np.zeros(tables.n_cards(), np.int16)
    for card, m in per_card.items():
        k = m.tobytes()
        if k not in keys:
            keys[k] = len(masks)
            masks.append(m)
        card_cls[tables.card_id(card)] = keys[k]
    masks = np.stack(masks)
    # 填充/UNK 卡用“全图都不合法”的类别
    empty = np.zeros_like(masks[0])
    masks = np.concatenate([masks, empty[None]])
    card_cls[tables.PAD] = card_cls[tables.UNK] = len(masks) - 1
    path = os.path.join(tables.ASSETS, 'place_masks.npz')
    np.savez_compressed(path, masks=masks, card_cls=card_cls)
    sym = all((m[0] == m[1]).all() for m in masks)
    print('写入 %s：%d 张牌 → %d 个放置类别；玩家 0/1 视角下完全相同：%s；%.0fs' % (path, len(per_card), len(masks), sym, time.time() - t0))
    for i, m in enumerate(masks[:-1]):
        names = [c for c in per_card if card_cls[tables.card_id(c)] == i]
        print('  类别 %d：%d 张（%s…），塔状态 0 下合法格 %d' % (i, len(names), ', '.join(names[:5]), m[0, 0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab', action='store_true')
    ap.add_argument('--masks', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    sys.setrecursionlimit(20000)
    cfg = load()
    if a.vocab or not (a.vocab or a.masks):
        build_vocab(cfg, a.force)
    if a.masks or not (a.vocab or a.masks):
        probe_masks(cfg)


if __name__ == '__main__':
    main()
