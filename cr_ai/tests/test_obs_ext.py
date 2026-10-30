"""2026-10-28 观测修补：狂暴等对象可见（obs.show_effects）、国王塔开火即激活（obs.king_shots）、扩展观测（model.obs_ext：
技能状态、进化充能、单位状态）。数据表辅助函数、和引擎对照（进化第几次出、冰法减速读得到、技能状态）、扩展观测的网络打开后函数不变、
续训时优化器状态接得上。"""
import numpy as np
import pytest
import torch

from cr_ai.config import load
from cr_ai.env import features as FT
from cr_ai.env import tables
from cr_ai.env.engine_adapter import CLS_EFFECT, CLS_SPELL, CLS_TROOP

SMALL = ['model.channels=32', 'model.skip_channels=16', 'model.res_blocks=2', 'model.hidden=48', 'model.head_hidden=16',
         'model.entity=true', 'model.entity_dim=32', 'model.entity_layers=1', 'model.entity_grid_channels=8',
         'model.deck_pool=mlp', 'model.hand_slot=next', 'model.priv_value=true']


def test_tables_and_status_features():
    assert tables.evo_cycles('Knight') == 2 and tables.evo_cycles('Barbarians') == 1 and tables.evo_cycles('Graveyard') == 1
    assert tables.evo_cycles('NoSuchCard') is None
    vis = tables.visible_effects()
    assert {'Rage', 'RageBottle', 'ValkyrieHero_AEO', 'IceWizardHero_FreezeAeo'} <= vis
    assert 'Knight' not in vis and 'FireballSpell' not in vis
    s = FT.status_features(('IceWizardSlowDown',))
    assert s[0] == 1 and abs(s[1] + 0.3) < 1e-6 and abs(s[2] + 0.3) < 1e-6 and s[4] == 0
    assert FT.status_features(('Freeze',))[1] == -1 and abs(FT.status_features(('Rage',))[1] - 0.3) < 1e-6
    p = FT.status_features(('Poison',))
    assert p[4] == 1 and p[1] < 0
    assert FT.status_features(('Tornado',))[8] == 1 and FT.status_features(('Invisibility',))[6] == 1
    assert abs(FT.status_features(('Freeze', 'Rage'))[1] + 0.7) < 1e-6
    assert not FT.status_features(()).any() and FT.status_features(('NotABuff',))[0] == 1


def test_form_charge():
    k = [FT.form_charge('Knight', 'evo', n) for n in range(4)]
    assert [x[0] for x in k] == [0, 0, 1, 0] and k[1][1] == 0.5 and k[2][1] == 1.0
    assert [FT.form_charge('Barbarians', 'evo', n)[0] for n in range(4)] == [0, 1, 0, 1]
    assert FT.form_charge('Tombstone', 'hero', 5) == (1.0, 1.0) and FT.form_charge('Knight', None, 2) == (0.0, 0.0)
    h = FT.hand_ext(['Knight', 'Barbarians', None, 'Zap'], 'Knight', {'Knight': 'evo', 'Barbarians': 'evo'}, {'Knight': 2, 'Barbarians': 1})
    assert h.shape == (5, 2) and h[0, 0] == 1 and h[1, 0] == 1 and h[2, 0] == 0 and h[3, 0] == 0 and h[4, 0] == 1
    assert FT.ability_ext(True, 0, -1, 500) == (1.0, 0.0, 1.0) and FT.ability_ext(False, 1, 400, 500) == (0.0, 1.0, 0.25)


def test_scalars_king_fired(cfg):
    ec = cfg.env
    s = FT.scalars(100, 5, [1, 1, 1], [1, 1, 1], 5, ec)
    assert s[12] == 0 and s[13] == 0
    s = FT.scalars(100, 5, [1, 1, 1], [1, 1, 1], 5, ec, king_fired=(True, False))
    assert s[12] == 1 and s[13] == 0
    assert (FT.scalars(100, 5, [1, 1, 1], [1, 1, 1], 5, ec, king_fired=(False, False)) == FT.scalars(100, 5, [1, 1, 1], [1, 1, 1], 5, ec)).all()


def _rand_obs(cfg, B, seed=0):
    from cr_ai.tests.test_grow import _rand_obs as ro
    o = ro(cfg, B, torch.Generator().manual_seed(seed))
    for k in ('ab_ext', 'hand_ext', 'opp_ext', 'ent_ext'):
        if k in o:
            o[k] = torch.rand(o[k].shape) * 2 - 1                     # 有正有负，确认零权重真的把它们挡住了
    return o


def test_net_obs_ext_same_function_and_trainable():
    """打开 model.obs_ext、装旧权重：逐步（带 LSTM 状态）出牌/价值/位置/h 和旧网络逐位相同；新输入层拿得到梯度。"""
    from cr_ai.model.net import Net, load_weights
    c0 = load('cr_ai/configs/bc.yaml', overrides=SMALL)
    c1 = load('cr_ai/configs/bc.yaml', overrides=SMALL + ['model.obs_ext=true'])
    torch.manual_seed(0)
    old = Net(c0).eval()
    new = Net(c1).eval()
    miss = load_weights(new, old.state_dict())
    assert miss and all(k.startswith('ext_') for k in miss)
    names = [n for n, _ in new.named_parameters()]
    assert all(n.startswith('ext_') for n in names[-len(miss):])           # 排在最后（优化器状态能接上）
    st0 = st1 = None
    with torch.no_grad():
        for t in range(3):
            o = _rand_obs(c1, 4, seed=t)
            pc = torch.randint(1, 5, (4,))
            a, st0 = old(o, st0, place_card=pc)
            b, st1 = new(o, st1, place_card=pc)
            for k in ('card', 'value', 'place', 'h'):
                assert torch.equal(a[k], b[k]), k
    o = _rand_obs(c1, 4, seed=9)
    out, _ = new(o, None, place_card=torch.ones(4, dtype=torch.long))
    (out['card'].logsumexp(1).sum() + out['value'].sum() + out['place'].logsumexp(1).sum()).backward()
    for n in ('ext_ab', 'ext_extra', 'ext_hand', 'ext_opp', 'ext_ent'):
        g = getattr(new, n).weight.grad
        assert g is not None and g.abs().sum() > 0, n


def test_extend_adam():
    """旧网络的 AdamW 状态接到打开 obs_ext 的新网络：旧参数的矩原样保留，新参数一阶矩 0、二阶矩为正；能接着走一步。"""
    from cr_ai.model.net import Net, load_weights
    from cr_ai.train.ppo import extend_adam
    c0 = load('cr_ai/configs/bc.yaml', overrides=SMALL)
    c1 = load('cr_ai/configs/bc.yaml', overrides=SMALL + ['model.obs_ext=true'])
    torch.manual_seed(0)
    old = Net(c0).eval()
    opt0 = torch.optim.AdamW(list(old.parameters()), lr=1e-3)
    o = _rand_obs(c0, 4)
    out, _ = old(o)
    (out['card'].logsumexp(1).sum() + out['value'].sum()).backward()
    opt0.step()
    new = Net(c1).eval()
    load_weights(new, old.state_dict())
    newp = [tuple(p.shape) for n, p in new.named_parameters() if n.startswith('ext_')]
    opt1 = torch.optim.AdamW(list(new.parameters()), lr=1e-3)
    osd = extend_adam(opt0.state_dict(), newp)
    opt1.load_state_dict(osd)
    n_old = len(list(old.parameters()))
    for i in range(n_old):
        a, b = opt0.state_dict()['state'].get(i), opt1.state_dict()['state'].get(i)
        if a is not None:
            assert torch.equal(a['exp_avg'], b['exp_avg']) and torch.equal(a['exp_avg_sq'], b['exp_avg_sq'])
    for j in range(len(newp)):
        s = opt1.state_dict()['state'][n_old + j]
        assert not s['exp_avg'].any() and (s['exp_avg_sq'] > 0).all()
    o = _rand_obs(c1, 4)
    out, _ = new(o)
    (out['card'].logsumexp(1).sum() + out['value'].sum()).backward()
    opt1.step()


# ---------------- 和引擎对照（共用一个引擎；换配置用 env.configure，最后换回来） ----------------
def _ext_cfg():
    return load(overrides=['model.entity=true', 'model.obs_ext=true', 'obs.show_effects=true', 'obs.king_shots=true'])


@pytest.fixture
def env_ext(env, cfg):
    env.configure(_ext_cfg())
    try:
        yield env
    finally:
        env.configure(cfg)


def _lane_cell(pm_row, cx=3, far=False):
    cells = [c for c in np.flatnonzero(pm_row) if c % 18 == cx]
    key = lambda c: abs(c // 18 - 15.5)          # noqa: E731
    return int(max(cells, key=key) if far else min(cells, key=key))


def _keep_alive(env):
    for p in (0, 1):
        for w in ('king', 'left', 'right'):
            env.ad.set_tower_hp(p, w, 4000)


def test_rage_visible(env_ext):
    """狂暴：show_effects 开时瓶子/区域是法术（网格法术层、单位列表里都有），关时是特效（都没有）。"""
    env = env_ext
    NM = {v: k for k, v in tables.unit_vocab().items()}
    D0 = ['Rage', 'Knight', 'Archer', 'Giant', 'Musketeer', 'MiniPekka', 'Valkyrie', 'Zap']
    env.reset(seed=7, decks=[D0, D0], forms=[{}, {}])
    cast, seen = None, None
    for t in range(300):
        cm, pm, _a, hand, _n, _s = env.legal(0)
        act = (0, -1)
        if cast is None and 'Rage' in hand and cm[1 + hand.index('Rage')]:
            s = hand.index('Rage')
            act, cast = (1 + s, _lane_cell(pm[s])), t
        elif cast is None and env.ad.elixir(0) > 9 and cm[1:5].any():
            s = int(np.flatnonzero(cm[1:5])[0])
            act = (1 + s, int(np.flatnonzero(pm[s])[0]))
        obs, _r, done, _i = env.step([act, (0, -1)])
        if t % 10 == 0:
            _keep_alive(env)
        u = env.units
        rage = np.isin(u['uid'], [tables.unit_id('Rage'), tables.unit_id('RageBottle')])
        if rage.any():
            seen = t
            assert (u['cls'][rage] == CLS_SPELL).all()
            assert (obs[0]['gi'][4] == tables.unit_id('Rage')).any() or (obs[0]['gi'][4] == tables.unit_id('RageBottle')).any()
            env.ad.show_effects = False
            env._refresh()
            u2 = env.units
            assert (u2['cls'][np.isin(u2['uid'], [tables.unit_id('Rage'), tables.unit_id('RageBottle')])] == CLS_EFFECT).all()
            env.ad.show_effects = True
            break
    assert cast is not None and seen is not None, (cast, seen, [NM.get(int(x)) for x in env.units['uid']])


def test_evo_charge_matches_engine(env_ext):
    """进化骑士连打 3 次：hand_ext 说“下一次是进化”的那次，引擎生成的正好是 Knight_EV1（第 3 次）。"""
    env = env_ext
    D0 = ['Knight', 'Archer', 'Giant', 'Musketeer', 'MiniPekka', 'Valkyrie', 'Zap', 'Bomber']
    obs = env.reset(seed=3, decks=[D0, D0], forms=[{'Knight': 'evo'}, {}])
    ev1 = tables.unit_id('Knight_EV1')
    said, got, pend = [], [], None
    for t in range(1200):
        cm, pm, _a, hand, _n, _s = env.legal(0)
        act = (0, -1)
        if pend is None and env.ad.elixir(0) >= 7 and cm[1:5].any():
            s = hand.index('Knight') if ('Knight' in hand and cm[1 + hand.index('Knight')]) else int(np.flatnonzero(cm[1:5])[0])
            act = (1 + s, int(np.flatnonzero(pm[s])[len(np.flatnonzero(pm[s])) // 2]))
            if hand[s] == 'Knight':
                said.append(bool(obs[0]['hand_ext'][s, 0]))
                pend = (t, int((env.units['uid'] == ev1).sum()))
        obs, _r, done, _i = env.step([act, (0, -1)])
        if t % 10 == 0:
            _keep_alive(env)
        if pend is not None and t - pend[0] >= 8:
            got.append(int((env.units['uid'] == ev1).sum()) > pend[1])
            pend = None
            if len(got) == 3:
                break
    assert said == [False, False, True] and got == [False, False, True], (said, got)


def test_status_and_ability_state(env_ext):
    """英雄冰法打敌方巨人：巨人的单位状态读到减速；技能按下前 ab_ext = [1, 0, 1]，按下后 [0, 1, <1]；观测字段形状和声明一致。"""
    env = env_ext
    D0 = ['IceWizard', 'Knight', 'Archer', 'Musketeer', 'Valkyrie', 'MiniPekka', 'Zap', 'Bomber']
    D1 = ['Giant', 'Knight', 'Valkyrie', 'MiniPekka', 'HogRider', 'Barbarians', 'Prince', 'DarkPrince']
    obs = env.reset(seed=11, decks=[D0, D1], forms=[{'IceWizard': 'hero'}, {}], d_act=0, d_obs=0)
    gid = tables.unit_id('Giant')
    placed_g = placed_iw = False
    before = after = slowed = None
    for t in range(400):
        acts = [(0, -1), (0, -1)]
        cm1, pm1, _a1, hand1, _n1, _s1 = env.legal(1)
        if not placed_g and cm1[1:5].any():
            s = hand1.index('Giant') if ('Giant' in hand1 and cm1[1 + hand1.index('Giant')]) else int(np.flatnonzero(cm1[1:5])[0])
            acts[1] = (1 + s, _lane_cell(pm1[s]))
            placed_g = True
        cm0, pm0, abil0, hand0, _n0, _s0 = env.legal(0)
        u = env.units
        enemy = u[(u['player'] == 1) & (u['cls'] == CLS_TROOP)]
        if not placed_iw and len(enemy) and (enemy['y'] < 15000).any() and 'IceWizard' in hand0 and cm0[1 + hand0.index('IceWizard')]:
            s = hand0.index('IceWizard')
            acts[0] = (1 + s, _lane_cell(pm0[s], far=True))
            placed_iw = True
        if abil0 and cm0[5] and before is None:
            before = obs[0]['ab_ext'][0].copy()
            acts[0] = (5, -1)
        obs, _r, done, _i = env.step(acts)
        if before is not None and after is None and not env.legal(0)[0][5] and env.legal(0)[2]:
            after = obs[0]['ab_ext'][0].copy()
        ent = obs[0]['ent_id'][:, 0]
        g = np.flatnonzero(ent == gid)
        if len(g) and obs[0]['ent_ext'][g[0], 1] < -0.2 and slowed is None:
            slowed = float(obs[0]['ent_ext'][g[0], 1])
        if before is not None and after is not None and slowed is not None:
            break
    assert before is not None and tuple(before) == (1.0, 0.0, 1.0), before
    assert after is not None and after[0] == 0 and after[1] == 1 and after[2] < 1, after
    assert slowed is not None and slowed <= -0.29, slowed
    for p in (0, 1):
        for k, (shape, dt) in env.spec.items():
            assert np.asarray(obs[p][k]).shape == shape and np.asarray(obs[p][k]).dtype == dt, k
    assert env.stats['ab_ready'][0] > 0 and env.stats['abilities'][0] == 1
    ub, rb = env.stats['ab_by'][0], env.stats['ab_ready_by'][0]          # 按技能单位名分开
    assert sum(ub.values()) == 1 and set(ub) <= set(rb) and all('Hero' in nm for nm in ub), (ub, rb)
    assert sum(rb.values()) >= env.stats['ab_ready'][0]


def test_king_shots_flag(env_ext, monkeypatch):
    """国王塔开过火（场上出现它的 KingProjectile）→ 这一方的国王塔记为已激活，之后一直是激活。"""
    from cr_ai.tests.conftest import make_units, towers
    env = env_ext
    D = ['Knight', 'Archer', 'Giant', 'Musketeer', 'MiniPekka', 'Valkyrie', 'Zap', 'Bomber']
    env.reset(seed=5, decks=[D, D], forms=[{}, {}])
    assert env.king_fired == [False, False]
    u = make_units(towers() + [(tables.unit_id('KingProjectile'), 1, 5, 9000, 30000, 0, 0)])
    monkeypatch.setattr(env.ad, 'units', lambda: u)
    env._refresh()
    assert env.king_fired == [False, True]
    o = env._obs(0)
    assert o['scal'][12] == 0 and o['scal'][13] == 1
