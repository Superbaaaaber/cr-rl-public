"""从 HF IL_Replay 的卡组统计（data/il_replay/deck_stats.json，tools/il_deck_stats.py 生成）生成对手卡组池 cr_ai/configs/deck_pool.yaml。

    .venv\\Scripts\\python.exe -m cr_ai.tools.build_deck_pool [--top 300]
每套卡组取它最常见的形态组合，按使用次数加权；引擎牌池里没有的牌、Mirror（首出会让原生崩溃）所在的卡组跳过。
"""
import argparse
import json
import os

import yaml

from cr_ai.config import ROOT
from crx.env import card_pool
from crx.royaleapi import SLUG2NAME, split_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default=os.path.join(ROOT, 'data', 'il_replay', 'deck_stats.json'))
    ap.add_argument('--top', type=int, default=300)
    ap.add_argument('--out', default=os.path.join(ROOT, 'cr_ai', 'configs', 'deck_pool.yaml'))
    a = ap.parse_args()
    st = json.load(open(a.stats, encoding='utf-8'))
    pool = set(card_pool())
    decks, used = [], 0
    for t in st['top'][:a.top]:
        keys = t['forms'][0][0]
        names, forms, ok = [], {}, True
        for k in keys:
            slug, form = split_key(k)
            nm = SLUG2NAME.get(slug)
            if nm is None or nm not in pool or nm == 'Mirror':
                ok = False
                break
            names.append(nm)
            if form:
                forms[nm] = form
        if not ok:
            continue
        decks.append(dict(cards=names, forms=forms, weight=int(t['n']), winrate=t['winrate'], slugs=list(t['deck'])))
        used += t['n']
    doc = dict(source='HF VanguardX101/IL_Replay replays/（%d 局 %d 方）' % (st['replays'], st['sides']),
               note='对手卡组池：数据里使用量前 %d 的卡组（每套取最常见形态），覆盖 %.1f%% 的方；按 weight 加权抽样' % (len(decks), 100.0 * used / st['sides']),
               decks=decks)
    with open(a.out, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=200)
    print('写入 %s：%d 套，覆盖 %.1f%% 的方' % (a.out, len(decks), 100.0 * used / st['sides']))


if __name__ == '__main__':
    main()
