"""技能概率下限（ppo.ab_floor，定向探索）：混合分布的公式、只改抽样不改记下的 logp、CUDA Graph 里改 ε 生效且只作用于当前策略、
调度；按技能单位分开的使用率汇总。"""
import math
import types

import numpy as np
import pytest
import torch

from cr_ai.model.infer import _card_sample, ab_floor_mix
from cr_ai.train.ppo import ep_summary
from cr_ai.train.rollout import ab_floor_eps

NEG = -1e9


def _logits():
    torch.manual_seed(0)
    x = torch.randn(4, 7) * 3
    x[0, 5:] = NEG                       # 没有技能可以按
    x[1, 6] = NEG                        # 一个
    x[2, 3] = NEG                        # 两个（另有一张手牌不合法）
    x[3, 5] = -25.0                      # 两个，其中一个策略几乎不按
    return x


@pytest.mark.parametrize('dev', ['cpu', 'cuda'])
def test_mix_formula(dev):
    if dev == 'cuda' and not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    x = _logits().to(dev)
    p = torch.softmax(x, 1)
    for e in (0.0, 0.05, 0.5):
        q = ab_floor_mix(x, torch.tensor(e, device=dev)).exp()
        n = (x[:, 5:] > -1e8).sum(1, keepdim=True).float()
        ee = e * (n > 0).float()
        want = (1 - ee) * p
        want[:, 5:] += torch.where(x[:, 5:] > -1e8, ee / n.clamp(min=1), torch.zeros_like(want[:, 5:]))
        assert torch.allclose(q, want, atol=1e-6), (e, q, want)
        assert torch.allclose(q.sum(1), torch.ones(4, device=dev), atol=1e-6)
    # ε = 0：和原来的 log π 逐位相同；没有技能可以按的行在任何 ε 下都不变
    assert torch.equal(ab_floor_mix(x, torch.tensor(0.0, device=dev)), torch.log_softmax(x, 1))
    assert torch.equal(ab_floor_mix(x, torch.tensor(0.3, device=dev))[0], torch.log_softmax(x, 1)[0])


def test_card_sample_mixes_but_logp_is_policy():
    """抽样按 π'，返回的对数概率是 π 的（学习端的比率按 π 算）；没打开时抽样不受 ab_eps 影响。"""
    N = 20000
    row = torch.tensor([0.0, 1.0, 0.5, NEG, NEG, -12.0, NEG])      # 技能几乎从不按（π ≈ 2e-6）
    out = dict(card=row.repeat(N, 1))
    force = torch.full((N,), -1, dtype=torch.long)
    lp_ref = torch.log_softmax(row, 0)
    net = types.SimpleNamespace(policy_logits=lambda c: c, ab_mix=True, ab_eps=torch.tensor(0.2))
    torch.manual_seed(1)
    sel, lp, _ = _card_sample(net, out, torch.ones(1), force)
    f = (sel == 5).float().mean().item()
    assert abs(f - 0.2) < 0.015, f
    assert torch.allclose(lp, lp_ref[sel], atol=1e-6)
    assert float(lp[sel == 5].max()) < -12                          # 不是 log π'(技能) ≈ log 0.2
    net.ab_mix = False
    sel2, lp2, _ = _card_sample(net, out, torch.ones(1), force)
    assert (sel2 == 5).sum().item() <= 2 and torch.allclose(lp2, lp_ref[sel2], atol=1e-6)


def test_eps_schedule():
    pc = dict(ab_floor=0.03, ab_floor_hold_until=100, ab_floor_until=200)
    assert ab_floor_eps(0, pc) == 0.03 and ab_floor_eps(99, pc) == 0.03
    assert math.isclose(ab_floor_eps(150, pc), 0.015) and ab_floor_eps(200, pc) == 0.0 and ab_floor_eps(10 ** 9, pc) == 0.0
    assert ab_floor_eps(10 ** 9, dict(ab_floor=0.03)) == 0.03          # 没给 until：一直开着
    assert ab_floor_eps(5, {}) == 0.0
    for bad in (dict(ab_floor=0.03, ab_floor_until=200), dict(ab_floor=0.03, ab_floor_hold_until=300, ab_floor_until=200), dict(ab_floor=1.0)):
        with pytest.raises(ValueError):
            ab_floor_eps(0, bad)


def test_league_engine_eps_in_graph():
    """CUDA Graph 录好以后原地改 ab_eps 就生效；只作用于当前策略那一块（BC 对手、历史快照不受影响）；记下的 logp 仍是 π 的。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.config import load
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import LeagueEngine
    from cr_ai.model.net import Net
    from cr_ai.tests.test_net import _batch
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    nets = [Net(cfg).cuda().eval().half() for _ in range(3)]
    nets[0].ab_mix = True
    assert 'ab_eps' not in nets[0].state_dict()                        # 不进存档、不进共享权重
    flat, card_cls = MK.all_masks()
    eng = LeagueEngine(nets[0], nets[1], [nets[2]], flat, card_cls)
    b = _batch()

    def sub(a, n):
        o = {k: np.array((b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n]) for k in list(INPUT_KEYS) + ['pl']}
        o['card_mask'][:, 5] = True                                    # 让第一个技能槽可以按（网络对它给什么分数都行）
        return o
    H = nets[0].H
    z = lambda n: np.zeros((n, H), np.float32)   # noqa: E731
    oa, oo, ox = sub(0, 16), sub(20, 16), sub(40, 16)

    def run():
        return eng.run(oa, z(16), z(16), oo, [ox], [z(16)], [z(16)])
    r0 = run()
    assert (r0['sel'] != 5).any()
    nets[0].ab_eps.fill_(0.999)
    r1 = run()
    hit = r1['sel'] == 5
    assert hit.sum() >= 15, r1['sel']                                 # π' 里技能约 0.999（半精度）
    assert float(r1['logp'][hit].max()) < -0.05, r1['logp']           # 是 log π(技能)，不是 log π' ≈ 0
    assert (r1['opp_sel'] != 5).any() and (r1['hist_sel'][0] != 5).any()
    nets[0].ab_eps.fill_(0.0)
    assert (run()['sel'] != 5).any()


def test_ep_summary_ab_units():
    eps = [dict(opp='self', result=1.0, ret=1.0, len=10, crowns=[1, 0], ab=1, ab_ready=30,
                ab_units={'IceWizardHero': [20, 1], 'ArcherQueen': [10, 0]}),
           dict(opp='self', result=0.0, ret=-1.0, len=10, crowns=[0, 1], ab=3, ab_ready=50,
                ab_units={'IceWizardHero': [5, 0], 'ArcherQueen': [45, 3]}),
           dict(opp='self', result=0.5, ret=0.0, len=10, crowns=[0, 0], ab=0, ab_ready=0, ab_units={})]
    s = ep_summary(eps)
    assert s['ab_units'] == {'IceWizardHero': [2, 1, 1, 25], 'ArcherQueen': [2, 1, 3, 55]}
    assert s['ab_hero_used'] == 0.5 and s['ab_champ_used'] == 0.5
    assert 'ab_units' not in ep_summary([dict(eps[2], ab_units=None)])
