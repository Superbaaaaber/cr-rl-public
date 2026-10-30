"""联赛簿记：历史快照槽（每局绑定一个快照、有对局时不换）、快照池保留规则、Bradley–Terry 拟合。不需要引擎和显卡。"""
import json
import os

import numpy as np
import pytest

from cr_ai.config import Cfg
from cr_ai.train.league import HistSlots, League
from cr_ai.tools.gauntlet import fit


def test_hist_slots_never_swap_under_live_games():
    """随机模拟：28 个环境、对局长短不一、每批来一组新候选。断言：
    - 一局从开局到结束，所在槽的快照编号不变（结果记在真正打这局的快照上）；
    - 任何时候至少有一个槽在接新局；候选确实会被换上。"""
    rng = np.random.RandomState(0)
    hs = HistSlots(4, 4)
    W, nxt_id = 28, 0
    live = {}                                   # 环境 → (槽, 开局时的快照编号, 剩余步数)
    installed = []

    def install(todo):
        for i, sid, path in todo:
            assert not hs.active[i], '换快照的槽手上不能有对局'
            installed.append(sid)
            hs.installed(i, sid, path)

    for t in range(20000):
        if t % 384 == 0:                        # 每批开头：新候选 + 补槽
            hs.set_picks([(nxt_id + j, 'p%d' % (nxt_id + j)) for j in range(4)])
            nxt_id += 4
            install(hs.refills())
        for w in range(W):
            if w in live:
                i, sid, left = live[w]
                assert hs.ids[i] == sid, '对局进行中快照被换了'
                if left <= 1:
                    del live[w]
                    if hs.release(w) is not None:
                        install(hs.refills())
                else:
                    live[w] = (i, sid, left - 1)
            elif rng.rand() < 0.6:              # 新开一局历史快照对局
                i = hs.assign(w)
                live[w] = (i, hs.ids[i], int(rng.randint(500, 1200)))
        assert any(hs.accepting)
    assert len(set(installed)) > 20             # 候选在持续换上


def test_hist_slots_placeholder_filled_first():
    hs = HistSlots(3, 2)
    assert hs.refills() == []                   # 没有候选：保持 BC 占位
    hs.set_picks([(5, 'a'), (6, 'b'), (7, 'c'), (8, 'd')])
    todo = hs.refills()
    assert [t[1] for t in todo] == [5, 6, 7]


def _league(tmp_path, **kw):
    lc = Cfg(dict(max_snapshots=50, pfsp_floor=0.01, **kw))
    cfg = Cfg(dict(league=lc))
    return League(cfg, str(tmp_path), seed=0)


def test_prune_recent_plus_geometric_old(tmp_path):
    lg = _league(tmp_path, recent_snapshots=30, old_snapshots=20)
    for k in range(1, 301):                     # 3 亿步，每 100 万步一个快照
        lg.add('s%d.pt' % k, k * 1000000, save=False)
    steps = sorted(s['step'] for s in lg.snaps)
    assert len(steps) == 50
    assert steps[-30:] == [k * 1000000 for k in range(271, 301)]       # 最近 30 个全留
    old = steps[:20]
    assert old[0] == 1000000                                          # 最老的总留着
    gaps = np.diff(old + [steps[20]])
    assert gaps[-1] < gaps[len(gaps) // 2] < gaps[0] or gaps[0] >= gaps[-1] * 3   # 越老越稀


def test_prune_legacy_drop_oldest(tmp_path):
    lg = _league(tmp_path)
    for k in range(1, 61):
        lg.add('s%d.pt' % k, k, save=False)
    assert sorted(s['step'] for s in lg.snaps) == list(range(11, 61))


def test_bradley_terry_recovers_ratings():
    rng = np.random.RandomState(1)
    true = {'e0': 0.0, 'e1': -150.0, 'e2': 100.0, 'c0': 50.0, 'c1': 250.0}
    rows = []
    for c in ('c0', 'c1'):
        for e in ('e0', 'e1', 'e2'):
            p = 1 / (1 + 10 ** ((true[e] - true[c]) / 400))
            w = rng.binomial(2000, p)
            rows.append(dict(a=c, b=e, win=w, loss=2000 - w, draw=0))
    res = fit(rows, anchor='e0')
    for n, r in true.items():
        assert abs(res[n][0] - r) < 3 * max(res[n][1], 10) / 1.96 + 15, (n, res[n], r)
    assert res['e0'] == (0.0, 0.0, res['e0'][2])


def _deck_league(tmp_path, lam=0.5, **kw):
    import yaml
    pool = tmp_path / 'pool.yaml'
    decks = [dict(cards=['C%d_%d' % (i, j) for j in range(8)], forms={}, weight=w) for i, w in enumerate([50, 30, 20])]
    pool.write_text(yaml.safe_dump(dict(decks=decks)), encoding='utf-8')
    lc = Cfg(dict(max_snapshots=50, pfsp_floor=0.01, deck_pfsp=lam, **kw))
    cfg = Cfg(dict(league=lc, paths=Cfg(dict(deck_pool=str(pool)))))
    (tmp_path / "run").mkdir()
    return League(cfg, str(tmp_path / "run"), seed=0)


def test_deck_pfsp_tilts_toward_hard_decks(tmp_path):
    """对第 2 套（使用量 30%）一直输、其余一直赢：它的抽样概率要明显高于使用量，但不超过 (1−λ)·u + λ；
    另外两套仍保留至少 (1−λ)·u（不会被冷落）；没开（deck_pfsp = 0）时没有概率。"""
    lg = _deck_league(tmp_path)
    u = np.array([0.5, 0.3, 0.2])
    assert np.allclose(lg.deck_probs(), u)                             # 还没有对局：所有卡组胜率相同 → 等于使用量
    for _ in range(50):
        eps = [dict(opp='hist', hist=-1, result=1.0, opp_deck='pool0'), dict(opp='hist', hist=-1, result=0.0, opp_deck='pool1'),
               dict(opp='bc_pool', result=1.0, opp_deck='pool2'), dict(opp='hist', hist=-1, result=1.0, opp_deck='main')]
        lg.update(eps)
    p = lg.deck_probs()
    assert abs(p.sum() - 1) < 1e-9
    assert p[1] > 0.6 and p[1] <= 0.5 * 0.3 + 0.5 + 1e-9
    assert p[0] >= 0.5 * 0.5 - 1e-9 and p[2] >= 0.5 * 0.2 - 1e-9
    s = lg.summary()
    assert s['deck_min_winrate'] < 0.2 and s['deck_tv'] > 0.2
    lg.save()                                                         # 重新载入：统计按卡组牌名接上
    lg2 = League(lg_cfg(tmp_path), str(tmp_path / 'run'), seed=0)
    assert np.allclose(lg2.deck_probs(), p)
    assert lg2.decks.hardest(1)[0]['cards'].startswith('C1_')


def test_deck_pfsp_source_filters_episodes(tmp_path):
    """deck_pfsp_source：hist 只用对历史快照的局、main 只用 agent 一方是主卡组的局；all（默认）全用。"""
    eps = [dict(opp='self', result=0.0, deck='pool5', opp_deck='pool0'),       # 自我对弈 + 随机卡组：hist、main 都不算
           dict(opp='hist', hist=-1, result=1.0, deck='pool5', opp_deck='pool1'),   # 只有 hist 算
           dict(opp='self', result=0.0, deck='main', opp_deck='pool2')]        # 只有 main 算
    for src, want in (('all', [1, 1, 1]), ('hist', [0, 1, 0]), ('main', [0, 0, 1])):
        d = tmp_path / src
        d.mkdir()
        lg = _deck_league(d, deck_pfsp_source=src)
        lg.update(eps)
        assert np.allclose(lg.decks.n, want), (src, lg.decks.n)
    with pytest.raises(ValueError):
        (tmp_path / 'bad').mkdir()
        _deck_league(tmp_path / 'bad', deck_pfsp_source='nope')


def lg_cfg(tmp_path):
    lc = Cfg(dict(max_snapshots=50, pfsp_floor=0.01, deck_pfsp=0.5))
    return Cfg(dict(league=lc, paths=Cfg(dict(deck_pool=str(tmp_path / 'pool.yaml')))))


def test_episode_sampler_opp_probs_only_affect_opponent():
    """set_opp_p 只改对手一方的卡组抽样；agent 一方照旧按使用量。"""
    from cr_ai.config import load
    from cr_ai.env.episodes import EpisodeSampler
    cfg = load()
    spec = dict(controllers=[['agent', 'hist', 1], ['hist', 'agent', 1]], main_for='agent', agent_main_prob=0.0, opp_main_prob=0.0)
    s = EpisodeSampler(cfg, spec, seed=3)
    n = len(s.pool)
    p = np.zeros(n)
    p[7] = 1.0
    s.set_opp_p(p)
    s.set_opp_p(np.ones(n + 1))                                       # 长度对不上（换了池子）：忽略
    opp_decks, ag_decks = set(), set()
    for _ in range(200):
        ep = s.sample()
        for k in (0, 1):
            (ag_decks if ep['controllers'][k] == 'agent' else opp_decks).add(ep['deck_names'][k])
    assert opp_decks == {'pool7'}
    assert len(ag_decks) > 20


def test_gauntlet_examiners_v2_and_special_rows_excluded():
    from cr_ai.tools.gauntlet import load_examiners
    ex = load_examiners('cr_ai/configs/examiners_v2.yaml')
    by = {e['name']: e for e in ex}
    assert set(by) == {'metagen_248M'}                                 # 10-28 起每段只打 metagen（其余停用）
    m = by['metagen_248M']                                             # 主指标：双方主流卡组、配对换边 400 局
    assert m['deck'] == 'pool' and m['paired'] and m['games'] == 400 and not m['elo'] and m['env']['paths.deck_pool'].endswith('deck_pool_exam_meta.yaml')
    assert all(e['env']['paths.engine_base'] == 're/snap/base_1019' for e in ex)


def test_gauntlet_deck_pool_examiner(tmp_path, monkeypatch):
    """考官条目 deck: pool：被测方也从卡组池抽（arena 的 deck='pool'）；默认不进 Elo、对手身份和同一个存档的其他专项分开；
    不写 deck 时照旧用主卡组；deck 写错报错。"""
    import pytest
    import yaml
    import cr_ai.eval.arena as arena
    from cr_ai.tools.gauntlet import load_examiners, play
    ck = tmp_path / 'ex.pt'
    ck.write_bytes(b'x')
    exf = tmp_path / 'ex.yaml'
    exf.write_text(yaml.safe_dump(dict(examiners=[
        dict(name='rand', ckpt=str(ck), opp='bc_pool', pool='P.yaml'),
        dict(name='gen', ckpt=str(ck), opp='bc_pool', pool='P.yaml', deck='pool', games=4),
        dict(name='mirror', ckpt=str(ck), opp='bc_main'),
        dict(name='pgen', ckpt=str(ck), opp='bc_pool', pool='P.yaml', deck='pool', paired=True, games=4)])), encoding='utf-8')
    ex = load_examiners(str(exf))
    by = {e['name']: e for e in ex}
    assert by['gen']['deck'] == 'pool' and not by['gen']['elo'] and by['rand']['deck'] == 'main' and by['mirror']['elo']
    assert by['pgen']['paired'] and not by['gen']['paired']
    calls = []

    def fake_run(ckpt, opp, games, workers, deck, *a, **k):
        calls.append((opp, games, deck, (k.get('env') or {}).get('paths.deck_pool'), k.get('paired')))
        r = dict(win=3, loss=1, draw=0, games=4, error=0, win_rate=0.75, secs=1.0)
        if k.get('paired'):
            r['pairs'] = dict(n=2, both_win=1, split=1, both_loss=0)
        return r
    monkeypatch.setattr(arena, 'run', fake_run)
    rows = play(str(ck), ex, 10, 1, str(tmp_path / 'out.jsonl'))
    assert calls == [('bc_pool', 10, 'main', 'P.yaml', False), ('bc_pool', 4, 'pool', 'P.yaml', False), ('bc_main', 10, 'main', None, False),
                     ('bc_pool', 4, 'pool', 'P.yaml', True)]
    r = {x['b_name']: x for x in rows}
    assert r['gen']['deck'] == 'pool' and r['gen']['elo'] is False and r['gen']['b'].endswith('@gen') and r['rand']['b'].endswith('@rand')
    assert 'deck' not in r['rand'] and 'elo' not in r['mirror'] and 'paired' not in r['gen']
    assert r['pgen']['paired'] is True and r['pgen']['pairs']['split'] == 1
    for bad in (dict(name='bad', ckpt=str(ck), deck='nope'), dict(name='bad', ckpt=str(ck), paired=True)):   # paired 只能配 deck: pool
        exf.write_text(yaml.safe_dump(dict(examiners=[bad])), encoding='utf-8')
        with pytest.raises(ValueError):
            load_examiners(str(exf))


def test_arena_paired_episodes(tmp_path):
    """配对换边：每对两局种子、控制方相同，第二局双方交换卡组、形态和塔兵；局数补成偶数；pair_stats 按两局得分之和分类。"""
    import yaml
    from cr_ai.config import load
    from cr_ai.eval.arena import pair_stats, paired_episodes
    pool = tmp_path / 'pool.yaml'
    decks = [dict(cards=['C%d_%d' % (i, j) for j in range(8)], forms={'C%d_0' % i: 'evo'}, weight=1) for i in range(6)]
    pool.write_text(yaml.safe_dump(dict(decks=decks)), encoding='utf-8')
    cfg = load(overrides=['paths.deck_pool=%s' % pool])
    spec = dict(controllers=[['agent', 'bc', 1], ['bc', 'agent', 1]], main_for=None, main_deck_prob=0.0, opp_main_prob=0.0)
    eps = paired_episodes(cfg, spec, 7, seed=5)
    assert len(eps) == 8
    for a, b in zip(eps[0::2], eps[1::2]):
        assert a['seed'] == b['seed'] and a['controllers'] == b['controllers']
        for k in ('decks', 'forms', 'deck_names', 'tower_troop'):
            assert b[k] == [a[k][1], a[k][0]]
    assert len({e['seed'] for e in eps[0::2]}) == 4                     # 不同对之间是独立抽的
    assert eps == paired_episodes(cfg, spec, 7, seed=5)                 # 同一个种子 → 同一张配对表
    assert pair_stats({0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0, 5: 0.5, 6: 0.0, 7: 0.0, 8: 1.0}, 5) == dict(n=4, both_win=1, split=2, both_loss=1)


def test_league_loop_gen_base_winrate(tmp_path):
    """剥削者这一代的起点胜率 = 训练日志最早几批（按局数加权）；没有对局记录时为 None。"""
    import json
    from cr_ai.tools.league_loop import gen_base_winrate, phase_winrate
    f = tmp_path / 'train_log.jsonl'
    rows = [dict(step=100 + k, n_hist=10, win_hist=0.2 if k < 3 else 0.6) for k in range(60)]
    f.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    assert abs(gen_base_winrate(str(f)) - 0.2) < 1e-9
    wr, n = phase_winrate(str(f), 100)
    assert abs(wr - 0.6) < 1e-9 and n > 0
    assert gen_base_winrate(str(tmp_path / 'none.jsonl')) is None


def test_spec_pool(tmp_path):
    """专项剥削者：快照旁 .json 里有 deck_pool → 卡组池和概率；没有 → None（普通快照）。"""
    from cr_ai.train.league import spec_pool
    pool = tmp_path / 'p.yaml'
    pool.write_text('decks:\n- {cards: [A, B], forms: {A: evo}, weight: 3}\n- {cards: [C, D], weight: 1}\n', encoding='utf-8')
    pt = tmp_path / 'exp_g1_step5.pt'
    pt.write_bytes(b'')
    (tmp_path / 'exp_g1_step5.json').write_text(json.dumps(dict(gen=1, deck_pool=str(pool))), encoding='utf-8')
    dk, p = spec_pool(str(pt))
    assert dk == [(['A', 'B'], {'A': 'evo'}), (['C', 'D'], {})] and abs(p[0] - 0.75) < 1e-9
    pt2 = tmp_path / 'exp_g2_step5.pt'
    (tmp_path / 'exp_g2_step5.json').write_text(json.dumps(dict(gen=2, deck_pool=None)), encoding='utf-8')
    assert spec_pool(str(pt2)) is None and spec_pool(None) is None


def test_exp_variant_files():
    """--exp-variants 的三个文件：专项的两个指向存在的卡组池、agent 一方不用主卡组。"""
    from cr_ai.config import ROOT, read_yaml
    from cr_ai.tools.league_loop import read_overrides
    for v in ('mg', 'new'):
        ov = read_overrides('cr_ai/configs/exp_variants/%s.txt' % v)
        pool = [o.split('=', 1)[1] for o in ov if o.startswith('paths.deck_pool=')][0]
        assert len(read_yaml(os.path.join(ROOT, pool))['decks']) > 10
        assert 'league.agent_main_prob=0.0' in ov
    assert read_overrides('cr_ai/configs/exp_variants/normal.txt') == []
