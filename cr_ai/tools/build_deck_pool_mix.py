"""新旧混合的对手卡组池：老池（HF 旧版本，deck_pool.yaml）+ cr_scrape 抓的新版本对战记录（battles.jsonl）里的热门卡组。

    .venv\\Scripts\\python.exe -m cr_ai.tools.build_deck_pool_mix [--old_share 0.5] [--top 300]
老池各卡组按原使用量、合计占 old_share；新卡组合计占 1 - old_share，套与套之间按 --weight 分：
  players（默认，2026-10-20 起）= 用过这套的不同玩家数（同一人打再多场也只算一次）。数据是跟着 32 个高手翻对战历史抓的，
          按场数算会把少数高手的招牌卡组放大（第 1 套 493 场里一个人就打了 437 场）；
  games = 出现场数（10-19 版本的做法）。
两边都有的同一套卡组权重相加（形态用新版本的）。
新数据只取标准 1v1 八张牌的模式（终极挑战/天梯/经典挑战），同一场对战从双方历史各抓到一次的去重；
每套取最常见的形态组合，引擎没有的形态丢掉，引擎牌池里没有的牌、Mirror 所在的卡组跳过。
用法：训练时 paths.deck_pool=cr_ai/configs/deck_pool_mix.yaml（deck_pool.yaml 本身不动，旧评测可比）。
"""
import argparse
import collections
import json
import os

import yaml

from cr_ai.config import ROOT
from crx.env import card_pool
from crx.royaleapi import SLUG2NAME, load_card_forms, split_key

MODES = ('Grand Challenge', 'Ranked', 'Ladder', 'Classic Challenge')
SCALE = 1_000_000          # 权重换算成整数：两边合计约 SCALE


def new_decks(path, pool, card_forms):
    """battles.jsonl → {卡组（排序后的牌名元组）: dict(n, win, forms Counter, slugs Counter)}，以及有效方数、对战数。"""
    seen, out = set(), {}
    sides = games = 0
    for line in open(path, encoding='utf-8'):
        b = json.loads(line)
        if b.get('game_mode') not in MODES:
            continue
        t, o = b.get('team') or {}, b.get('opponent') or {}
        key = (b.get('timestamp'), frozenset((t.get('tag'), o.get('tag'))))
        if key in seen:
            continue
        seen.add(key)
        games += 1
        res = b.get('result')
        for side, won in ((t, res == 'Victory'), (o, res == 'Defeat')):
            keys = side.get('deck_keys') or []
            if len(keys) != 8:
                continue
            names, forms, ok = [], {}, True
            for k in keys:
                slug, form = split_key(k)
                nm = SLUG2NAME.get(slug)
                if nm is None or nm not in pool or nm == 'Mirror':
                    ok = False
                    break
                names.append(nm)
                if form and card_forms.get(nm, {}).get(form):
                    forms[nm] = form
            if not ok or len(set(names)) != 8:
                continue
            d = out.setdefault(tuple(sorted(names)), dict(n=0, win=0, forms=collections.Counter(), slugs=collections.Counter(), players=set()))
            d['n'] += 1
            d['players'].add(side.get('tag') or side.get('name'))
            d['win'] += int(won)
            d['forms'][tuple(sorted(forms.items()))] += 1
            d['slugs'][tuple(sorted(split_key(k)[0] for k in keys))] += 1
            sides += 1
    return out, sides, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default=os.path.join(ROOT, 'cr_ai', 'configs', 'deck_pool.yaml'))
    ap.add_argument('--battles', default=os.path.join(os.path.dirname(ROOT), 'cr_scrape', 'data', 'battles.jsonl'))
    ap.add_argument('--old_share', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=300)
    ap.add_argument('--weight', default='players', choices=['players', 'games'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'cr_ai', 'configs', 'deck_pool_mix.yaml'))
    a = ap.parse_args()
    pool = set(card_pool())
    old = yaml.safe_load(open(a.old, encoding='utf-8'))['decks']
    nd, sides, games = new_decks(a.battles, pool, load_card_forms())
    cnt = (lambda d: len(d['players'])) if a.weight == 'players' else (lambda d: d['n'])
    top = sorted(nd.items(), key=lambda kv: (-cnt(kv[1]), -kv[1]['n']))[:a.top]
    used = sum(d['n'] for _, d in top)
    used_w = sum(cnt(d) for _, d in top)

    merged = collections.OrderedDict()
    w_old = sum(d['weight'] for d in old)
    for d in old:
        merged[tuple(sorted(d['cards']))] = dict(cards=list(d['cards']), forms=dict(d.get('forms') or {}),
                                                 weight=a.old_share * d['weight'] / w_old, winrate=d.get('winrate'),
                                                 slugs=list(d.get('slugs') or []), src='old')
    both = 0
    for names, d in top:
        w = (1 - a.old_share) * cnt(d) / used_w
        forms = dict(d['forms'].most_common(1)[0][0])
        wr = round(d['win'] / d['n'], 3)
        slugs = list(d['slugs'].most_common(1)[0][0])
        if names in merged:
            m = merged[names]
            m.update(weight=m['weight'] + w, forms=forms, winrate=wr, src='old+new')
            both += 1
        else:
            merged[names] = dict(cards=list(names), forms=forms, weight=w, winrate=wr, slugs=slugs, src='new')
    decks = []
    for d in merged.values():
        d['weight'] = max(1, int(round(d['weight'] * SCALE)))
        decks.append(d)
    doc = dict(source='老池：%s ＋ 新版本：cr_scrape/data/battles.jsonl（%d 场 %d 方，%s）' % (os.path.basename(a.old), games, sides, '/'.join(MODES)),
               note='新旧混合对手卡组池（新卡组按%s加权）：老池 %d 套占 %.0f%%；新版本前 %d 套（覆盖新数据 %.1f%% 的方）占 %.0f%%，其中 %d 套两边都有、权重相加；共 %d 套，按 weight 加权抽样'
                    % ({'players': '不同玩家数', 'games': '场数'}[a.weight], len(old), 100 * a.old_share, len(top), 100.0 * used / max(1, sides), 100 * (1 - a.old_share), both, len(decks)),
               decks=decks)
    with open(a.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=200)
    print('写入 %s：%d 套（老 %d、新 %d、重合 %d）；新数据 %d 场 %d 方、%d 套不同卡组，前 %d 套覆盖 %.1f%%'
          % (a.out, len(decks), len(old), len(top), both, games, sides, len(nd), len(top), 100.0 * used / max(1, sides)))


if __name__ == '__main__':
    main()
