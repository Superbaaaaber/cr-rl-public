"""专项剥削者的训练卡组池（league_loop --exp-variants，2026-10-20）：从当前 deck_pool_mix.yaml 里挑。

    .venv\Scripts\python.exe -m cr_ai.tools.build_spec_pools

- deck_pool_spec_mg.yaml：含亡灵巨人（MinionGiant，10-06 之后的新牌）的卡组；
- deck_pool_spec_new.yaml：新版本卡组（cr_scrape 抓的 10-06 之后的对局里的热门卡组，src=new 或 old+new）。
权重照搬（按用过的玩家数）。和考官专项评测的池（deck_pool_exam_*，冻结不改）分开：这里随卡组池更新可以重新生成。
"""
import os

import yaml

from cr_ai.config import ROOT

CFG = os.path.join(ROOT, 'cr_ai', 'configs')


def main():
    mix = yaml.safe_load(open(os.path.join(CFG, 'deck_pool_mix.yaml'), encoding='utf-8'))['decks']
    out = dict(mg=('含亡灵巨人的卡组', [d for d in mix if 'MinionGiant' in d['cards']]),
               new=('新版本卡组（src=new 或 old+new）', [d for d in mix if d.get('src') in ('new', 'old+new')]))
    for k, (note, decks) in out.items():
        with open(os.path.join(CFG, 'deck_pool_spec_%s.yaml' % k), 'w', encoding='utf-8') as f:
            yaml.safe_dump(dict(source='deck_pool_mix.yaml', note='专项剥削者训练用：' + note, decks=decks), f, allow_unicode=True, sort_keys=False, width=200)
        print(k, len(decks))


if __name__ == '__main__':
    main()
