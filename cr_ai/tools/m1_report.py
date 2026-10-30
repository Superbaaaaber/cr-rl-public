"""M1 回放管线统计：python -m cr_ai.tools.m1_report data_bc/pilot
可用对局数、跳过/出错原因、各卡组局数分布、每局平均出牌数、“不出”步占比、重模拟一致率、标签修正情况、存储量。"""
import collections
import glob
import json
import os
import sys

import numpy as np
import yaml

from cr_ai.config import ROOT, read_yaml


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data_bc', 'pilot')
    S = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, 'meta_w*.jsonl'))) for l in open(f, encoding='utf-8')]
    skips = collections.Counter(s['skip'] for s in S if s.get('skip'))
    errs = collections.Counter((s.get('err') or '')[:40] for s in S if s.get('err'))
    ok = [s for s in S if not s.get('skip') and not s.get('err') and s.get('samples')]
    pool = read_yaml(os.path.join(ROOT, 'cr_ai', 'configs', 'deck_pool.yaml'))['decks']
    names = {tuple(sorted(p['cards'])): i for i, p in enumerate(pool)}
    main_deck = tuple(sorted(c['engine'] for c in read_yaml(os.path.join(ROOT, 'cr_ai', 'configs', 'deck.yaml'))['cards']))
    deck_cnt = collections.Counter()
    for s in ok:
        for dk in s['decks']:
            k = tuple(sorted(dk))
            deck_cnt['主卡组' if k == main_deck else ('池#%d' % names[k] if k in names else '池外')] += 1
    n = len(ok)
    plays = sum(s['play_samples'] for s in ok)
    abs_ = sum(s['ab_samples'] for s in ok)
    noop = sum(s['noop'] for s in ok)
    noop_steps = noop * 5.0                                   # 不出样本只留 20%（权重 5）
    rep = dict(
        rows=len(S), usable=n, skipped=dict(skips), errors=dict(errs),
        main_deck_sides=deck_cnt['主卡组'], decks_top=deck_cnt.most_common(12),
        avg_plays_per_game=float(np.mean([s['plays'] for s in ok])), avg_abilities_per_game=float(np.mean([s['abilities'] for s in ok])),
        noop_step_ratio=noop_steps / (noop_steps + plays + abs_), samples_per_game=float(np.mean([s['samples'] for s in ok])),
        consistency=dict(winner=float(np.mean([s['ok_winner'] for s in ok])), crowns=float(np.mean([s['ok_crowns'] for s in ok])),
                         hp_exact=float(np.mean([bool(s['ok_hp']) for s in ok])),
                         hp_within_500=float(np.mean([s['hp_err'] is not None and max(abs(x) for x in s['hp_err']) <= 500 for s in ok]))),
        label=dict(forced_legal=sum(s['forced_legal'] for s in ok) / max(1, plays + abs_), clamped=sum(s['clamped'] for s in ok) / max(1, plays),
                   missing=sum(s['label_missing'] for s in ok) / max(1, plays + abs_ + sum(s['label_missing'] for s in ok)),
                   unplayed_events=sum(s['unplayed'] for s in ok) / max(1, sum(s['n_events'] for s in ok)),
                   delayed_plays=sum(s['delayed'] for s in ok) / max(1, sum(s['plays'] for s in ok))),
        raster_bad=sum(s.get('raster_bad', 0) for s in ok),
        by_type={k: dict(n=len(v), winner=float(np.mean([s['ok_winner'] for s in v])), crowns=float(np.mean([s['ok_crowns'] for s in v])))
                 for k, v in collections.defaultdict(list, {t: [s for s in ok if s.get('battle_type') == t] for t in {s.get('battle_type') for s in ok}}).items()},
        disk_mb=sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, 'shard_*.npz'))) / 2 ** 20,
    )
    rep['disk_kb_per_game'] = rep['disk_mb'] * 1024 / max(1, n)
    json.dump(rep, open(os.path.join(d, 'm1_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
