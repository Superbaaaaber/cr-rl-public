"""对手牌序追踪与圣水估计。"""
import numpy as np

from cr_ai.env import tables
from cr_ai.env.opponent_tracker import OpponentTracker


def test_recent4_and_seen(cfg):
    t = OpponentTracker(cfg.env)
    for i, c in enumerate(['Knight', 'Archer', 'Fireball', 'Zap', 'Giant', 'Knight']):
        t.on_play(c, None, 200 + 20 * i, 1)
    assert t.recent4() == {'Fireball', 'Zap', 'Giant', 'Knight'}
    assert [c for c, _ in t.seen] == ['Knight', 'Archer', 'Fireball', 'Zap', 'Giant']
    ids, forms, num = t.features(400, tables.card_id, lambda f: 0)
    assert ids[0] == tables.card_id('Knight') and num[0, 1] == 1.0 and num[1, 1] == 0.0      # Archer 不在最近 4 次
    assert num[:5, 0].all() and not num[5:, 0].any()


def test_elixir_estimate_math(cfg):
    ec = cfg.env
    t = OpponentTracker(ec)
    assert abs(t.elixir_estimate(100) - min(10, 6 + 100 * ec.elixir_per_tick[0])) < 1e-6
    t.on_play('Giant', None, 100, 5)
    assert abs(t.elixir - (6 + 100 * ec.elixir_per_tick[0] - 5)) < 1e-6
    t2 = OpponentTracker(ec)
    assert t2.elixir_estimate(3000) == 10.0                            # 截断在上限
    t2.on_play('Golem', None, 3000, 8)
    assert abs(t2.elixir_estimate(3100) - (2 + 100 * ec.elixir_per_tick[1])) < 1e-6   # 双倍圣水段


def test_estimate_tracks_engine(env):
    """整局里对手圣水估计与引擎里的真实值（只用于检验，不进观测）保持接近。"""
    from cr_ai.bots.random_bot import RandomBot
    from cr_ai.config import load_deck
    D, F = load_deck(cfg=env.cfg)
    obs = env.reset(seed=21, decks=[D, D], forms=[F, F])
    bots = [RandomBot(np.random.RandomState(1), 0.2), RandomBot(np.random.RandomState(2), 0.2)]
    errs = []
    for _ in range(600):
        obs, rew, done, info = env.step([bots[p].act_obs(obs[p]) for p in (0, 1)])
        if done:
            break
        est = env.trk[0].elixir_estimate(env.tick)
        errs.append(abs(est - env.ad.elixir(1)))
    assert np.mean(errs) < 0.6 and np.percentile(errs, 95) < 1.6, (np.mean(errs), np.max(errs))
    # 观测里没有对手真实圣水：scal[14] 是估计值
    assert abs(obs[0]['scal'][14] * 10 - env.trk[0].elixir_estimate(env.tick)) < 1e-4


def test_collector_watch():
    """采集器：年龄每过 interval 产 1 点，消失再给 1 点；第一次只记基线。"""
    from cr_ai.env.opponent_tracker import CollectorWatch
    w = CollectorWatch(13000)
    assert w.update({7: 20000}) == 0
    assert w.update({7: 25950}) == 0
    assert w.update({7: 26000, 8: 0}) == 1
    assert w.update({8: 13000}) == 2                                  # 7 号消失 + 8 号第一次产出


def test_collector_counted_in_estimate(env):
    """对手（玩家 1）放圣水采集器：玩家 0 眼里的对手圣水估计按引擎的真实产出加上（整个寿命 7 次 + 消失 1 次 = 8 点）。"""
    from cr_ai.config import load_deck
    D, F = load_deck(cfg=env.cfg)
    D1 = ['Elixir Collector'] + [c for c in D if c != 'Elixir Collector'][:7]
    env.reset(seed=5, decks=[D, D1], forms=[F, {}])
    ad = env.ad
    for _ in range(40):                                                # 过开局部署锁、攒够圣水
        env.step([(0, -1), (0, -1)])
    m, b = ad.b.m, ad.b
    m.w32(m.mu64(int(ad.kings[1]) + 0x210), b.deck_index[1]['Elixir Collector'])    # 直接放进对手 0 号槽
    assert ad.play(1, 'Elixir Collector', 9500, 23500)[0]
    gains = []
    orig = env.trk[0].on_collector
    env.trk[0].on_collector = lambda tick, n: (gains.append((tick, n)), orig(tick, n))
    try:
        for _ in range(420):
            env.step([(0, -1), (0, -1)])
            if env.done:
                break
    finally:
        env.trk[0].on_collector = orig
    assert sum(n for _t, n in gains) == 8, gains
