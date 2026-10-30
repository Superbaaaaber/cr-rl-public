"""cr_ai/env/tables.py — 游戏数据表（data/game/csv_logic）里训练需要的静态信息，以及固定的单位/卡牌词表。

- 单位表：场上对象名 → 类别（character / building / projectile / area_effect）、空中、建筑、作用半径、部署时间……
- 法术对象：法术牌打出后在场上的弹道/区域对象名（spells_other.csv / spells_evolved.csv 的 Projectile、AreaEffectObject 列）。
- 词表（cr_ai/assets/*.json，由 cr_ai/tools/build_assets.py 生成后入库，之后不再变）：
    unit_vocab：0 = 填充，1 = UNK，其余按名字排序；card_vocab：0 = 填充（空槽），1 = UNK，其余为引擎牌池按名字排序。
"""
import csv
import glob
import json
import os
import re

from ..config import ROOT, repo_path

LOGIC = os.path.join(ROOT, 'data', 'game', 'csv_logic')
ASSETS = os.path.join(ROOT, 'cr_ai', 'assets')
PAD, UNK = 0, 1
TOWER_NAMES = ('KingTower', 'PrincessTower', 'Cannoneer', 'DaggerDuchess', 'ChefTower')   # 与 crx/flat/driver.py 一致
FORMS = {None: 0, '': 0, 'evo': 1, 'hero': 2}

_SEC = re.compile(r'^\[([A-Z_]+)\.([A-Za-z0-9_\-]+)\]\s*$')
_KV = re.compile(r'^([A-Za-z0-9_.]+)\s*=\s*(.+?)\s*$')
_TYPE = {'CHARACTER': 'character', 'BUILDING': 'building', 'PROJECTILE': 'projectile', 'AREA_EFFECT_OBJECT': 'area_effect', 'AEO': 'area_effect'}


def logic_dir_for(cfg):
    """配置 → 数据表目录。paths.logic_dir = auto 时跟着引擎快照：base_<版本> → data/game_<版本>/csv_logic，其余 → data/game/csv_logic。"""
    p = cfg.paths.get('logic_dir', 'auto')
    if p != 'auto':
        return repo_path(p)
    m = re.match(r'^base_(\w+)$', os.path.basename(os.path.normpath(cfg.paths.engine_base)))
    return os.path.join(ROOT, 'data', 'game_' + m.group(1) if m else 'game', 'csv_logic')


def use_logic(path):
    """切换数据表目录（进程级；之后的查询都按新表，缓存清空）。引擎快照换版本时数值变了，观测这边要跟着换。"""
    global LOGIC, _UNITS, _SPELL_OBJ, _CARD_INFO, _SUMMON, _FX_VIS, _EVO_CYC, _BUFFS
    path = os.path.abspath(path)
    if path == os.path.abspath(LOGIC):
        return
    if not os.path.isdir(path):
        raise FileNotFoundError('数据表目录不存在：%s（新版本的数据表用 python tools/balance/apply.py <版本> 生成）' % path)
    LOGIC = path
    _UNITS = _SPELL_OBJ = _CARD_INFO = _SUMMON = _FX_VIS = _EVO_CYC = _BUFFS = None


def use_cfg(cfg):
    use_logic(logic_dir_for(cfg))


def _val(s):
    s = s.strip()
    if s in ('true', 'false'):
        return s == 'true'
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _parse_toml_sections():
    """csv_logic 根目录与 characters/（含 hero_form/）下所有 TOML 的对象段 → {名字: {'_type': .., 字段..}}。
    段类型：CHARACTER / BUILDING / PROJECTILE / AREA_EFFECT_OBJECT / AEO（区域效果），以及 EXT（扩展：Base = "类型.基础名"，继承后覆盖）。
    同名段出现多次时后出现的字段覆盖前面的。"""
    files = sorted(glob.glob(os.path.join(LOGIC, '*.toml'))) + sorted(glob.glob(os.path.join(LOGIC, 'characters', '**', '*.toml'), recursive=True))
    raw, ext = {}, {}
    for f in files:
        cur = None
        for line in open(f, encoding='utf-8', errors='replace'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                m = _SEC.match(line)
                cur = None
                if m and m.group(1) in _TYPE:
                    cur = raw.setdefault(m.group(2), {'_type': _TYPE[m.group(1)]})
                elif m and m.group(1) == 'EXT':
                    cur = ext.setdefault(m.group(2), {})
                continue
            if cur is None:
                continue
            m = _KV.match(line)
            if m:
                cur[m.group(1)] = _val(m.group(2))

    def resolve(name, depth=0):
        if name in raw:
            return raw[name]
        d = ext.get(name)
        if d is None or depth > 8:
            return None
        base = str(d.get('Base', ''))
        bname = base.split('.', 1)[1] if '.' in base else base
        bd = resolve(bname, depth + 1)
        if bd is None:
            return None
        out = dict(bd)
        out.update(d)
        raw[name] = out
        return out
    for n in list(ext):
        resolve(n)
    return raw


_UNITS = None


def collector_interval_ms():
    """圣水采集器的产出间隔（ManaGenerateTimeMs；离线引擎按它跑，实测逐次对上）。"""
    d = _parse_toml_sections().get('ElixirCollector', {})
    return int(d.get('ManaGenerateTimeMs', 13000))


def unit_table():
    """{对象名: dict(type, air, building, radius, deploy_ms, lifetime_ms, hp)}。"""
    global _UNITS
    if _UNITS is None:
        raw = _parse_toml_sections()
        out = {}
        for n, d in raw.items():
            t = d['_type']
            fh = d.get('FlyingHeight', 0)
            out[n] = dict(type=t, air=bool(isinstance(fh, (int, float)) and fh > 0),
                          building=(t == 'building') or bool(d.get('IsBuilding')),
                          radius=float(d.get('Radius', 0) or 0) if isinstance(d.get('Radius', 0), (int, float)) else 0.0,
                          deploy_ms=int(d.get('DeployTime', 0) or 0) if isinstance(d.get('DeployTime', 0), int) else 0,
                          lifetime_ms=int(d.get('LifeTime', d.get('LifeDuration', 0)) or 0) if isinstance(d.get('LifeTime', d.get('LifeDuration', 0)), int) else 0,
                          hp=int(d.get('Hitpoints', 0) or 0) if isinstance(d.get('Hitpoints', 0), int) else 0,
                          target_buildings=bool(d.get('TargetOnlyBuildings') is True))
        for n in TOWER_NAMES:
            out.setdefault(n, dict(type='building', air=False, building=True, radius=0.0, deploy_ms=0, lifetime_ms=0, hp=0))
            out[n]['tower'] = True
        _UNITS = out
    return _UNITS


def _csv_rows(fn):
    path = os.path.join(LOGIC, fn)
    if not os.path.exists(path):
        return [], {}
    rows = list(csv.reader(open(path, encoding='utf-8')))
    return rows[2:], {h: i for i, h in enumerate(rows[0])}


_SPELL_OBJ = None


def spell_objects():
    """法术牌在场上的对象名 → 法术牌名（Fireball 的 FireballSpell、Zap 的 Zap、Vines 的 Vines_AeO……）。"""
    global _SPELL_OBJ
    if _SPELL_OBJ is None:
        out = {}
        for fn in ('spells_other.csv', 'spells_evolved.csv'):
            rows, ix = _csv_rows(fn)
            for r in rows:
                if not r or not r[0]:
                    continue
                for col in ('Projectile', 'AreaEffectObject', 'CustomFirstProjectile'):
                    if col in ix and r[ix[col]]:
                        out.setdefault(r[ix[col]], r[0])
        _SPELL_OBJ = out
    return _SPELL_OBJ


_FX_VIS = None


def visible_effects():
    """obs.show_effects 时也当法术画出来的无血量对象（2026-10-28）：数据表里的全部区域效果（AEO），加上法术牌召唤出来的
    无血量建筑（狂暴的 RageBottle，落地后才变成 Rage 区域）。不开时它们都被归成“特效”，网格和单位列表里都没有——
    实测看不见的有狂暴（瓶子和区域都看不见）、英雄技能的区域（ValkyrieHero_AEO、FreezeIceGolemite、IceWizardHero_FreezeAeo、
    WizardHero_MiniTornadoAEO）、IceGolemiteDeathArea、EarthquakeHiddenDamage、GoblinCurseBase 等。"""
    global _FX_VIS
    if _FX_VIS is None:
        ut = unit_table()
        out = {n for n, d in ut.items() if d['type'] == 'area_effect'}
        for fn in ('spells_other.csv', 'spells_evolved.csv'):
            rows, ix = _csv_rows(fn)
            if 'SummonCharacter' not in ix:
                continue
            for r in rows:
                sc = r[ix['SummonCharacter']] if r else ''
                if sc and ut.get(sc, {}).get('type') == 'building' and not ut[sc].get('hp'):
                    out.add(sc)
        _FX_VIS = out
    return _FX_VIS


_EVO_CYC = None


def evo_cycles(card):
    """进化牌要先充几次能：spells_evolved.csv 里 <牌>_EV1 行的 DarkElixirCost（值 1 或 2）。没有进化形态 → None。
    引擎实测（base_1019，2026-10-28）：充能 c 次的牌，第 c+1、2(c+1)、… 次打出是进化版（骑士 c=2：普通、普通、进化；野蛮人 c=1：普通、进化）。"""
    global _EVO_CYC
    if _EVO_CYC is None:
        rows, ix = _csv_rows('spells_evolved.csv')
        out = {}
        if 'DarkElixirCost' in ix:
            for r in rows:
                if r and r[0].endswith('_EV1') and r[ix['DarkElixirCost']].strip().isdigit():
                    out[r[0][:-4]] = int(r[ix['DarkElixirCost']])
        _EVO_CYC = out
    return _EVO_CYC.get(card)


_BUFFS = None


def buff_table():
    """状态（TOML 的 [BUFF.名字] 段，含 Base = "BUFF.X" 的 EXT 继承）→ {名字: {字段: 值}}。"""
    global _BUFFS
    if _BUFFS is None:
        files = sorted(glob.glob(os.path.join(LOGIC, '**', '*.toml'), recursive=True))
        raw, ext = {}, {}
        for f in files:
            cur = None
            for line in open(f, encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('['):
                    m = _SEC.match(line)
                    cur = None
                    if m and m.group(1) == 'BUFF':
                        cur = raw.setdefault(m.group(2), {})
                    elif m and m.group(1) == 'EXT':
                        cur = ext.setdefault(m.group(2), {})
                    continue
                if cur is not None:
                    m = _KV.match(line)
                    if m:
                        cur[m.group(1)] = _val(m.group(2))
        for _ in range(4):                                  # EXT 也可能继承另一个 EXT 出来的状态
            for name, d in ext.items():
                base = str(d.get('Base', ''))
                if base.startswith('BUFF.') and base[5:] in raw and name not in raw:
                    raw[name] = dict(raw[base[5:]], **d)
        _BUFFS = raw
    return _BUFFS


# ---------------- 词表 ----------------
def build_unit_vocab():
    names = set(unit_table().keys()) | set(TOWER_NAMES)
    return {n: i + 2 for i, n in enumerate(sorted(names))}


def build_card_vocab():
    from crx.env import card_pool
    return {n: i + 2 for i, n in enumerate(sorted(card_pool()))}


_VOC = {}


def _load_vocab(name):
    if name not in _VOC:
        path = os.path.join(ASSETS, name + '.json')
        if not os.path.exists(path):
            raise FileNotFoundError('缺少 %s：先运行 python -m cr_ai.tools.build_assets' % path)
        _VOC[name] = json.load(open(path, encoding='utf-8'))
    return _VOC[name]


def unit_vocab():
    return _load_vocab('unit_vocab')


def card_vocab():
    return _load_vocab('card_vocab')


def n_units():
    return max(unit_vocab().values()) + 1


def n_cards():
    return max(card_vocab().values()) + 1


def unit_id(name):
    return unit_vocab().get(name, UNK)


def card_id(name):
    if not name:
        return PAD
    return card_vocab().get(name, UNK)


_CARD_INFO = None


def card_info():
    """{牌名: dict(cost, kind: troop|building|spell)}（crx.cards.catalog）。"""
    global _CARD_INFO
    if _CARD_INFO is None:
        from crx import cards
        _CARD_INFO = {n: dict(cost=v.get('cost', 0), kind=v.get('kind', 'troop')) for n, v in cards.catalog().items()}
    return _CARD_INFO


def card_cost(name):
    return card_info().get(name, {}).get('cost', 0)


def ability_card(unit_name, deck):
    """英雄/冠军单位名 → 牌组里对应的牌名（英雄单位名 = 牌名 + Hero 等后缀，如 WizardHero、BarbLogBarbarianHero）。"""
    best = None
    for c in deck:
        if unit_name == c or unit_name.startswith(c):
            if best is None or len(c) > len(best):
                best = c
    return best


_SUMMON = None


def card_summon():
    """{牌名: 召唤的角色/建筑名}（spells_characters.csv / spells_buildings.csv / spells_other.csv 的 SummonCharacter，建筑牌即其本身）。"""
    global _SUMMON
    if _SUMMON is None:
        out = {}
        for fn in ('spells_characters.csv', 'spells_buildings.csv', 'spells_other.csv'):
            rows, ix = _csv_rows(fn)
            for r in rows:
                if r and r[0]:
                    sc = r[ix['SummonCharacter']] if 'SummonCharacter' in ix else ''
                    out[r[0]] = sc or r[0]
        _SUMMON = out
    return _SUMMON


def targets_buildings(card):
    """这张牌召唤的单位是否只打建筑（推塔牌：野猪、攻城锤、巨人……）。"""
    u = unit_table().get(card_summon().get(card, card), {})
    return bool(u.get('target_buildings'))
