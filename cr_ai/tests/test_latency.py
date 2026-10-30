"""延迟队列。"""
from cr_ai.env.latency import ActionQueue, ObsDelay


def test_action_delay_and_order():
    q = ActionQueue(d_act=2)
    q.push(0, 'a')
    q.push(1, 'b')
    q.push(1, None)                                # 不出牌不入队
    assert q.pop_due(0) == [] and q.pop_due(1) == []
    assert q.pop_due(2) == ['a']
    assert [x[1] for x in q.pending()] == ['b']
    assert q.pop_due(5) == ['b'] and q.pending() == []
    assert q.n_pushed == 2


def test_zero_delay_executes_same_step():
    q = ActionQueue(d_act=0)
    q.push(7, 'x')
    assert q.pop_due(7) == ['x']


def test_void_rate():
    q = ActionQueue(1)
    for i in range(4):
        q.push(i, i)
    q.void('elixir')
    q.void('elixir')
    q.void('not_in_hand')
    assert abs(q.void_rate - 0.75) < 1e-9 and q.void_reasons == {'elixir': 2, 'not_in_hand': 1}
    q.reset(d_act=3)
    assert q.d_act == 3 and q.pending() == []


def test_obs_delay():
    d = ObsDelay(2)
    assert d.push(0) == 0 and d.push(1) == 0 and d.push(2) == 0 and d.push(3) == 1 and d.push(4) == 2
    d0 = ObsDelay(0)
    assert d0.push('a') == 'a' and d0.push('b') == 'b'


def test_env_latency_in_engine(env):
    """d_act = 2：决策两步后才下达命令。"""
    from cr_ai.config import load_deck
    D, F = load_deck(cfg=env.cfg)
    env.reset(seed=12, decks=[D, D], forms=[F, F], d_act=2, d_obs=0)
    for _ in range(20):
        env.step([(0, -1), (0, -1)])
    cm, pm, abil, hand, nxt, st = env.legal(0)
    s = int([i for i in range(4) if cm[1 + i]][0])
    env.step([(1 + s, int(pm[s].nonzero()[0][0])), (0, -1)])
    assert env.stats['plays'][0] == 0 and len(env.aq[0].pending()) == 1
    env.step([(0, -1), (0, -1)])
    assert env.stats['plays'][0] == 0
    env.step([(0, -1), (0, -1)])
    assert env.stats['plays'][0] == 1
