"""网络与 BC 损失：真实分片上的一个 batch 前向/反向，形状、mask、损失有限；放置 mask 跟着镜像翻转。"""
import glob
import os

import numpy as np
import pytest
import torch

from cr_ai.config import ROOT, load
from cr_ai.data.bc_dataset import BCShards, mirror_place_mask


def _batch():
    files = sorted(glob.glob(os.path.join(ROOT, 'data_bc', 'il_replay', 'shard_*.npz')))[:1]
    if not files:
        pytest.skip('没有 BC 分片')
    ds = BCShards([], split='train')
    ds.files = files
    return next(ds.batches(batch_size=64, mirror_prob=0.0, seed=0))


def test_pmask_contains_label():
    b = _batch()
    play = (b['sel'] >= 1) & (b['sel'] <= 4) & ((b['flags'] & 3) == 0)
    assert play.any()
    assert b['pmask'][play, b['cell'][play]].all()
    m = b['pmask'][play][:1]
    assert (mirror_place_mask(mirror_place_mask(m)) == m).all()


def test_forward_backward():
    from cr_ai.model.net import Net
    from cr_ai.train.bc import gauss_kernel, losses, to_dev
    cfg = load('cr_ai/configs/bc.yaml')
    net = Net(cfg)
    b = to_dev({k: torch.as_tensor(v) for k, v in _batch().items()}, 'cpu')
    loss, parts, out, cm, pm, is_play, cell = losses(net, b, gauss_kernel(0.7, 'cpu'), cfg.bc)
    assert torch.isfinite(loss) and out['card'].shape == (64, 7) and out['place'].shape == (64, 576)
    assert (out['card'][~cm] < -1e8).all()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_infer_engine_legal_actions():
    """CUDA Graph 推理：采到的牌在 card_mask 内、手牌的落点在放置 mask 内、技能/不出的落点为 −1；不同批量（分桶）都能跑。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import InferEngine
    from cr_ai.model.net import Net
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    net = Net(cfg).cuda()
    flat, card_cls = MK.all_masks()
    eng = InferEngine(net, flat, card_cls)
    b = _batch()
    for n in (5, 13, 64):
        sub = {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[:n] for k in list(INPUT_KEYS) + ['pl']}
        for _ in range(20):
            r = eng.run(sub)
            sel, cell = r['sel'], r['cell']
            assert sub['card_mask'][np.arange(n), sel].all()
            hand = (sel >= 1) & (sel <= 4)
            assert (cell[~hand] == -1).all()
            for i in np.flatnonzero(hand):
                cid = sub['hand_id'][i, sel[i] - 1]
                assert flat[card_cls[cid], sub['pl'][i], sub['tstate'][i], cell[i]]
            assert np.isfinite(r['logp']).all() and (r['logp'] <= 1e-4).all()


def test_dual_engine_matches_separate():
    """合并图（主角 + BC）：BC 参考分布 = BC 网络在同一观测上的牌头分布；对手动作合法；状态维度正确。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import DualEngine
    from cr_ai.model.net import Net, init_recurrence
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    net = init_recurrence(Net(cfg)).cuda().eval()
    bc = Net(cfg).cuda().eval()
    flat, card_cls = MK.all_masks()
    eng = DualEngine(net, bc, flat, card_cls)
    b = _batch()
    sub = lambda a, n: {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n] for k in list(INPUT_KEYS) + ['pl']}   # noqa: E731
    oa, oo = sub(0, 6), sub(10, 4)
    H = net.H
    r = eng.run(oa, np.zeros((6, H), np.float32), np.zeros((6, H), np.float32), oo)
    assert r['h'].shape == (6, H) and len(r['opp_sel']) == 4
    assert oo['card_mask'][np.arange(4), r['opp_sel']].all()
    o = {k: torch.as_tensor(v).cuda() for k, v in oa.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        out, _ = bc(o)
        ref = torch.log_softmax(bc.policy_logits(out['card'].float()), 1).clamp(min=-30).cpu().numpy()
    assert np.abs(r['bc_card_lp'].astype(np.float32) - ref).max() < 0.05
    hand = (r['sel'] >= 1) & (r['sel'] <= 4)
    for i in np.flatnonzero(hand):                                   # BC 位置分布是以主角所选牌为条件的
        cid = oa['hand_id'][i, r['sel'][i] - 1]
        legal = flat[card_cls[cid], oa['pl'][i], oa['tstate'][i]]
        assert (r['bc_place_lp'][i][~legal] <= -29).all()


def test_league_engine_survives_allocator_churn():
    """录图后显存被别的东西反复分配/释放（例如换历史快照时加载权重），重放结果仍合法（回归：图里引用的常量缓冲区曾被提前释放）。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import LeagueEngine
    from cr_ai.model.net import Net
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    nets = [Net(cfg).cuda().eval().half() for _ in range(3)]
    flat, card_cls = MK.all_masks()
    eng = LeagueEngine(nets[0], nets[1], [nets[2]], flat, card_cls)
    b = _batch()
    sub = lambda a, n: {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n] for k in list(INPUT_KEYS) + ['pl']}   # noqa: E731
    H = nets[0].H
    z = lambda n: np.zeros((n, H), np.float32)   # noqa: E731
    for it in range(30):
        junk = [torch.full((1 << 18,), 7, dtype=torch.long, device='cuda') for _ in range(8)]   # 占用并写脏刚释放的显存
        del junk
        oa, oo, ox = sub(0, 5), sub(10, 3), sub(20, 2)
        r = eng.run(oa, z(5), z(5), oo, [ox], [z(2)], [z(2)])
        assert oa['card_mask'][np.arange(5), r['sel']].all()
        assert oo['card_mask'][np.arange(3), r['opp_sel']].all()
        assert ox['card_mask'][np.arange(2), r['hist_sel'][0]].all()


def test_league_engine_hist_slots_match_their_nets():
    """多个历史快照槽：每个槽的 LSTM 输出 = 那个槽的网络单独前向的结果（各槽权重不同、行数不同、有空槽）；
    没有历史快照对手时（Bx = 0，M3 的情形）也能跑；换掉一个槽的权重后图里立刻生效。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import LeagueEngine
    from cr_ai.model.net import Net, init_recurrence
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    nets = [Net(cfg).cuda().eval() for _ in range(5)]
    for m in nets:
        init_recurrence(m, -1.0)
        with torch.no_grad():
            m.lstm.weight_hh.normal_(0, 0.05)                              # 让状态真的起作用
    nets = [m.half() for m in nets]
    flat, card_cls = MK.all_masks()
    eng = LeagueEngine(nets[0], nets[1], nets[2:], flat, card_cls)
    b = _batch()
    sub = lambda a, n: {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n] for k in list(INPUT_KEYS) + ['pl']}   # noqa: E731
    H = nets[0].H
    rng = np.random.RandomState(0)

    def ref(net, o, h0, c0):
        t = {k: torch.as_tensor(v).cuda() for k, v in o.items()}
        with torch.no_grad():
            out, (h, c) = net(t, (torch.as_tensor(h0).cuda().half(), torch.as_tensor(c0).cuda().half()))
        return h.float().cpu().numpy()

    oa, oo = sub(0, 6), sub(10, 3)
    za = np.zeros((6, H), np.float32)
    for rows in ((2, 0, 3), (0, 0, 0), (1, 9, 0)):
        ox = [sub(20 + 10 * i, n) if n else None for i, n in enumerate(rows)]
        hx = [rng.randn(n, H).astype(np.float32) * 0.1 if n else None for n in rows]
        cx = [rng.randn(n, H).astype(np.float32) * 0.1 if n else None for n in rows]
        r = eng.run(oa, za, za, oo, ox, hx, cx)
        assert oa['card_mask'][np.arange(6), r['sel']].all()
        for i, n in enumerate(rows):
            assert r['hist_h'][i].shape == (n, H) and len(r['hist_sel'][i]) == n
            if n:
                assert ox[i]['card_mask'][np.arange(n), r['hist_sel'][i]].all()
                assert np.abs(r['hist_h'][i] - ref(nets[2 + i], ox[i], hx[i], cx[i])).max() < 2e-2
    # 换掉槽 1 的权重（原地拷贝）：图里立刻用新权重
    nets[3].load_state_dict(nets[0].state_dict())
    ox = [None, sub(30, 4), None]
    hx = [None, np.zeros((4, H), np.float32), None]
    r = eng.run(oa, za, za, oo, ox, hx, hx)
    assert np.abs(r['hist_h'][1] - ref(nets[0], ox[1], hx[1], hx[1])).max() < 2e-2


def test_league_engine_bc_ref_modes():
    """BC 参考分布（KL 项用）：图里算（默认）和图外算（bc_ref_graph=False）都等于 BC 网络单独前向的结果；
    want_bc=False 且没有 BC 对手时 bc_* 全 0、主角的输出不受影响；有 BC 对手时 want_bc=False 也照旧在图里算（对手动作合法）。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import LeagueEngine
    from cr_ai.model.net import Net
    from cr_ai.train.bc import INPUT_KEYS
    cfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    nets = [Net(cfg).cuda().eval().half() for _ in range(3)]
    flat, card_cls = MK.all_masks()
    eg = LeagueEngine(nets[0], nets[1], [nets[2]], flat, card_cls)
    ee = LeagueEngine(nets[0], nets[1], [nets[2]], flat, card_cls, bc_ref_graph=False)
    b = _batch()
    sub = lambda a, n: {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n] for k in list(INPUT_KEYS) + ['pl']}   # noqa: E731
    H, N = nets[0].H, 24
    oa = sub(0, N)
    za = np.zeros((N, H), np.float32)
    t = {k: torch.as_tensor(v).cuda() for k, v in oa.items()}
    fl, cc = torch.as_tensor(flat).cuda(), torch.as_tensor(card_cls).cuda().long()
    cm = oa['card_mask'].astype(bool)

    def ref_bc(sel):
        s = torch.as_tensor(sel).cuda()
        z = torch.zeros(N, H, device='cuda', dtype=torch.half)
        with torch.no_grad():
            out, _ = nets[1](t, (z, z), place_card=s)
        card = torch.log_softmax(nets[1].policy_logits(out['card'].float()), 1).clamp(min=-30).cpu().numpy()
        hid = t['hand_id'].long()[torch.arange(N, device='cuda'), (s - 1).clamp(0, 3)]
        pm = fl[cc[hid], t['pl'].long(), t['tstate'].long()]
        place = torch.log_softmax(out['place'].float().masked_fill(~pm, -1e9), 1).clamp(min=-30).cpu().numpy()
        return card, place, pm.cpu().numpy()

    n_hand = 0
    for _ in range(6):
        rs = []
        for eng in (eg, ee):
            r = eng.run(oa, za, za, None, [None], None, None, want_bc=True)
            card, place, pm = ref_bc(r['sel'])
            assert np.abs(r['bc_card_lp'].astype(np.float32)[cm] - card[cm]).max() < 5e-2
            ia = np.flatnonzero((r['sel'] >= 1) & (r['sel'] <= 4))
            n_hand += len(ia)
            for i in ia:
                assert np.abs(r['bc_place_lp'][i].astype(np.float32)[pm[i]] - place[i][pm[i]]).max() < 5e-2
            assert not r['bc_place_lp'][np.setdiff1d(np.arange(N), ia)].any()
            rs.append(r)
        r0 = ee.run(oa, za, za, None, [None], None, None, want_bc=False)
        assert not r0['bc_card_lp'].any() and not r0['bc_place_lp'].any()
        for r in rs:
            assert np.abs(r0['h'] - r['h']).max() < 1e-2 and np.abs(r0['value'] - r['value']).max() < 1e-2
    assert n_hand > 0
    oo = sub(30, 3)
    r = ee.run(oa, za, za, oo, [None], None, None, want_bc=False)
    assert oo['card_mask'][np.arange(3), r['opp_sel']].all() and len(r['opp_cell']) == 3


def _scal(own=(1.0, 1.0, 1.0), en=(1.0, 1.0, 1.0), time_left=1.0, ot=False):
    """按 env/features.py: scalars 的布局拼一个观测标量（国王激活按同一规则：国王掉血或己方公主塔没了）。"""
    s = np.zeros(15, np.float32)
    s[4], s[5] = float(ot), time_left
    s[6:9], s[9:12] = own, en
    act = lambda t: float(t[0] < 0.999 or t[1] <= 0 or t[2] <= 0)   # noqa: E731
    s[12], s[13] = act(own), act(en)
    return s


def test_win_prior_antisymmetric_and_known_values():
    """胜率先验：开局 = 0；双方互换（我方 ↔ 敌方塔、国王激活）时变号；手算的两个状态对得上。"""
    from cr_ai.model.net import win_prior
    cfg = load('cr_ai/configs/bc.yaml')
    reg, ot = cfg.model.value_prior_reg, cfg.model.value_prior_ot
    s0 = torch.from_numpy(_scal()[None])
    assert win_prior(s0, reg, ot).abs().item() < 1e-7
    rng = np.random.RandomState(0)
    tw = lambda: np.where(rng.rand(3) < 0.3, 0.0, rng.rand(3)).astype(np.float32)   # noqa: E731
    S = torch.from_numpy(np.stack([_scal(tw(), tw(), rng.rand(), rng.rand() < 0.3) for _ in range(256)]))
    M = S.clone()
    M[:, 6:9], M[:, 9:12], M[:, 12], M[:, 13] = S[:, 9:12], S[:, 6:9], S[:, 13], S[:, 12]
    p = win_prior(S, reg, ot)
    assert p.shape == (256,) and (p.abs() < 1).all()
    assert torch.allclose(win_prior(M, reg, ot), -p, atol=1e-6)
    # 常规时间过半、拿掉敌方左公主塔：血差 +1、冠差 +1、最低存活塔差 0、敌方国王激活 → z = 1.35 + (−0.6 + 4.1·0.5) − 0.9 = 1.9
    a = win_prior(torch.from_numpy(_scal(en=(1.0, 0.0, 1.0), time_left=0.5)[None]), reg, ot).item()
    assert abs(a - np.tanh(1.9 / 2)) < 1e-5
    # 加时、我方左塔半血：血差 −0.5、最低塔差 −0.5（我方国王因没掉血也没丢塔，不激活）→ z = 2.8·(−0.5) + 1.2·(−0.5) = −2
    b = win_prior(torch.from_numpy(_scal(own=(1.0, 0.5, 1.0), time_left=0.9, ot=True)[None]), reg, ot).item()
    assert abs(b - np.tanh(-1.0)) < 1e-5
    # 两座公主塔都没了：最低存活塔按 0 算（不是 2）
    c = win_prior(torch.from_numpy(_scal(own=(1.0, 0.0, 0.0), ot=True)[None]), reg, ot).item()
    assert abs(c - np.tanh((2.8 * -2 + 1.2 * -1) / 2)) < 1e-5


def test_value_from_adds_prior():
    """model.value_prior = 0：价值 = tanh(价值头)，和以前一样；= λ 时再加 λ × win_prior；观测里没有 scal 时不加。"""
    from cr_ai.model.net import Net, win_prior
    torch.manual_seed(0)
    net0 = Net(load('cr_ai/configs/bc.yaml'))
    net1 = Net(load('cr_ai/configs/bc.yaml', overrides=['model.value_prior=0.5']))
    net1.load_state_dict(net0.state_dict())
    h = torch.randn(8, net0.H)
    rng = np.random.RandomState(1)
    S = torch.from_numpy(np.stack([_scal(rng.rand(3), rng.rand(3), rng.rand()) for _ in range(8)]))
    with torch.no_grad():
        base = torch.tanh(net0.value_head(h).squeeze(-1))
        assert torch.allclose(net0.value_from(h, {'scal': S}), base)
        v1 = net1.value_from(h, {'scal': S})
        assert v1.dtype == torch.float32
        assert torch.allclose(v1, base + 0.5 * win_prior(S, net1.prior_reg, net1.prior_ot), atol=1e-6)
        assert torch.allclose(net1.value_from(h, {}), base)


def test_infer_engine_value_includes_prior():
    """CUDA Graph 推理（采样端）给的价值 = 网络直接前向的价值（含胜率先验），先验在图里也算对了。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import InferEngine
    from cr_ai.model.net import Net, win_prior
    from cr_ai.train.bc import INPUT_KEYS
    torch.manual_seed(0)
    net = Net(load('cr_ai/configs/bc.yaml', overrides=['model.value_prior=0.5'])).cuda().eval()
    flat, card_cls = MK.all_masks()
    eng = InferEngine(net, flat, card_cls)
    b = _batch()
    sub = {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[:32] for k in list(INPUT_KEYS) + ['pl']}
    r = eng.run(sub)
    o = {k: torch.as_tensor(v).cuda() for k, v in sub.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        ref = net(o)[0]['value'].float().cpu().numpy()
    pri = win_prior(o['scal'], net.prior_reg, net.prior_ot).cpu().numpy()
    assert 0.5 * np.abs(pri).max() > 0.05                             # 这批观测里确实有塔血差（先验的量明显大于容差，不然测不出）
    assert np.abs(r['value'] - ref).max() < 0.02
