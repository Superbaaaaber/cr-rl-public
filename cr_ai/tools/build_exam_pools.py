"""固定考官专项评测用的卡组池（生成一次后入库、不再改，否则以前的分数不可比）：

    .venv\\Scripts\\python.exe -m cr_ai.tools.build_exam_pools

- deck_pool_exam_meta.yaml：主流卡组 = 生成时的 deck_pool_mix.yaml 原样冻结（新旧混合，新卡组按用过的玩家数加权）；
- deck_pool_exam_mg.yaml：其中含亡灵巨人（MinionGiant，10-06 之后的新牌）的卡组，权重照搬；
- deck_pool_exam_rand.yaml：随机卡组 200 套（固定种子）：从正常对战牌（老池或新版本卡组里出现过的牌，不含活动模式专用牌）
  里随机抽 8 张，再像真实卡组一样随机套上至多 2 个进化、至多 1 个英雄（只套引擎有该形态的牌）；等权重。
考官拿这三种卡组分别和被测模型打：对随机卡组的胜率比对主流卡组高出多少（被测模型赢得更多 = 考官拿随机卡组更弱，正常；
被测模型赢得更少 = 它不会对付奇怪卡组），用来看泛化缺口。
"""
import argparse
import os

import numpy as np
import yaml

from cr_ai.config import ROOT
from crx.royaleapi import load_card_forms

CFG = os.path.join(ROOT, 'cr_ai', 'configs')


def dump(path, doc):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-rand', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20261020)
    a = ap.parse_args()
    mix = yaml.safe_load(open(os.path.join(CFG, 'deck_pool_mix.yaml'), encoding='utf-8'))
    old = yaml.safe_load(open(os.path.join(CFG, 'deck_pool.yaml'), encoding='utf-8'))['decks']
    dump(os.path.join(CFG, 'deck_pool_exam_meta.yaml'),
         dict(source='deck_pool_mix.yaml 于 2026-10-20 冻结', note='固定考官专项评测：主流卡组（不要改）', decks=mix['decks']))
    mg = [d for d in mix['decks'] if 'MinionGiant' in d['cards']]
    dump(os.path.join(CFG, 'deck_pool_exam_mg.yaml'),
         dict(source='deck_pool_mix.yaml（2026-10-20）里含亡灵巨人的卡组', note='固定考官专项评测：新卡亡灵巨人（不要改）', decks=mg))
    cards = sorted({c for d in old + mix['decks'] for c in d['cards']} - {'Mirror'})
    cf = load_card_forms()
    rng = np.random.RandomState(a.seed)
    rand = []
    for _ in range(a.n_rand):
        dk = [cards[i] for i in rng.choice(len(cards), 8, replace=False)]
        forms = {}
        evo = [c for c in dk if cf.get(c, {}).get('evo')]
        for c in rng.permutation(evo)[:rng.randint(0, 3)]:
            forms[str(c)] = 'evo'
        hero = [c for c in dk if cf.get(c, {}).get('hero') and c not in forms]
        if hero and rng.rand() < 0.5:
            forms[str(hero[rng.randint(len(hero))])] = 'hero'
        rand.append(dict(cards=dk, forms=forms, weight=1))
    dump(os.path.join(CFG, 'deck_pool_exam_rand.yaml'),
         dict(source='随机生成（种子 %d，%d 张正常对战牌）' % (a.seed, len(cards)), note='固定考官专项评测：随机卡组（不要改）', decks=rand))
    print('主流 %d 套；亡灵巨人 %d 套；随机 %d 套（从 %d 张牌里抽）' % (len(mix['decks']), len(mg), len(rand), len(cards)))


if __name__ == '__main__':
    main()
