"""实体分支（model.entity）：单位列表与网格一致、视角/镜像正确、截断规则；网络前向/反向；CUDA Graph 推理。"""
import glob
import os

import numpy as np
import pytest
import torch

from cr_ai.config import ROOT, load
from cr_ai.data.bc_dataset import BCShards, mirror_batch
from cr_ai.env.engine_adapter import CLS_EFFECT, CLS_SPELL, CLS_TROOP
from cr_ai.env.rasterize import ENT_X, MAX_ENT, entities, rasterize

from .conftest import make_units, towers

ENT_CFG = ['model.entity=true']


def _units():
    return make_units(towers() + [(10, 0, CLS_TROOP, 5300, 10700, 100, 200, 0, 20, 40), (11, 1, CLS_TROOP, 12400, 21300, 50, 50, 1),
                                  (12, 1, CLS_SPELL, 8700, 15200, 0, 0, 0, 0, 0, 0, 2500), (13, 0, CLS_EFFECT, 3000, 3000, 0, 0)])


def test_entity_cells_match_grid_both_views():
    u = _units()
    for p in (0, 1):
        gi, _gf = rasterize(u, p)
        eid, num = entities(u, p)
        n = int((num[:, 0] > 0).sum())
        assert n == len(u) - 1                                         # 特效不进列表
        for i in range(n):
            if num[i, 11 + 4] > 0:                                     # 法术不在单位 ID 槽里
                continue
            cy, cx = divmod(int(eid[i, 1]), 18)
            assert int(eid[i, 0]) in gi[:4, cy, cx], (p, i)            # 列表里的格子 = 网格里这个单位所在的格子
        own = num[:n, 10] == 0
        assert own.sum() == (4 if p == 0 else 5)                       # 视角：自己 3 座塔 + 1 个兵（玩家 1 另有一个法术）
        assert np.all(eid[n:] == 0) and np.all(num[n:] == 0)          # 补齐的行全 0


def test_entity_truncation_keeps_towers():
    rows = towers() + [(10, 0, CLS_TROOP, 500 + (i % 17) * 1000, 8500 + (i // 17) * 400, 10 + i, 1000) for i in range(200)]
    eid, num = entities(make_units(rows), 0)
    assert (num[:, 0] > 0).sum() == MAX_ENT
    assert ((num[:, 13] > 0) | (num[:, 14] > 0)).sum() == 6           # 6 座塔都在
    hp = np.expm1(num[:, 6].astype(np.float64) * 10)
    kept = hp[(num[:, 11] > 0)]
    assert kept.min() >= 87                                            # 兵按血量从高到低留：200 个里留血量最高的 122 个（10 + 78 起）


def test_entity_mirror_matches_mirrored_units():
    """镜像增强（mirror_batch）对单位列表的处理 = 先把单位左右翻转再算单位列表。"""
    u = _units()
    m = u.copy()
    m['x'] = 18000 - m['x']
    m['vx'] = -m['vx']
    eid, num = entities(u, 0)
    b = {'feat_ent_id': eid[None].copy(), 'feat_ent_num': num[None].copy(), 'gi': np.zeros((1, 5, 32, 18), np.int16),
         'gf': np.zeros((1, 18, 32, 18), np.float16), 'cell': np.zeros(1, np.int16), 'feat_scal': np.zeros((1, 15), np.float32),
         'feat_tstate': np.zeros(1, np.int16), 'feat_pend_num': np.zeros((1, 2, 3), np.float32), 'feat_last_num': np.zeros((1, 2), np.float32),
         'mirrored': np.zeros(1, bool)}
    mirror_batch(b, np.array([0]))
    e2, n2 = entities(m, 0)
    # 国王塔正好在 x = 9000（格子边界）：镜像后网格本身也会把它放到第 8 列（与 mirror_grids 一致），不和“翻转单位再算”比格子
    off_edge = (u['x'][u['cls'] < CLS_EFFECT] % 1000) != 0
    assert (b['feat_ent_id'][0][:len(off_edge)][off_edge] == e2[:len(off_edge)][off_edge]).all()
    assert (b['feat_ent_id'][0][:, 0] == e2[:, 0]).all()
    assert np.abs(b['feat_ent_num'][0].astype(np.float32) - n2.astype(np.float32)).max() < 2e-3
    assert abs(float(b['feat_ent_num'][0, 0, ENT_X]) + float(num[0, ENT_X]) - 1.0) < 2e-3


def _ent_batch(n=32, mirror=0.0):
    files = sorted(glob.glob(os.path.join(ROOT, 'data_bc', 'il_replay', 'shard_*.npz')))[:1]
    if not files:
        pytest.skip('没有 BC 分片')
    ds = BCShards([], split='train', entities=True)
    ds.files = files
    return next(ds.batches(batch_size=n, mirror_prob=mirror, seed=0))


def test_entity_net_bc_loss_backward():
    from cr_ai.model.net import Net, n_params
    from cr_ai.train.bc import gauss_kernel, losses, to_dev
    cfg = load('cr_ai/configs/bc.yaml', overrides=ENT_CFG)
    net = Net(cfg)
    assert 'ent_id' in net.input_keys and n_params(net) > n_params(Net(load('cr_ai/configs/bc.yaml')))
    b = to_dev({k: torch.as_tensor(v) for k, v in _ent_batch(mirror=0.5).items()}, 'cpu')
    assert b['ent_id'].shape == (32, MAX_ENT, 2)
    loss, parts, out, cm, pm, is_play, cell = losses(net, b, gauss_kernel(0.7, 'cpu'), cfg.bc)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.ent_in.weight.grad is not None and net.ent_in.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_entity_net_league_engine_graph():
    """实体网络录进 CUDA Graph（手写注意力 + scatter_add）：动作合法，和不录图的前向结果一致。"""
    if not torch.cuda.is_available():
        pytest.skip('没有 GPU')
    from cr_ai.env import masks as MK
    from cr_ai.model.infer import LeagueEngine
    from cr_ai.model.net import Net
    cfg = load('cr_ai/configs/bc.yaml', overrides=ENT_CFG)
    torch.manual_seed(0)
    nets = [Net(cfg).cuda().eval().half() for _ in range(4)]
    flat, card_cls = MK.all_masks()
    eng = LeagueEngine(nets[0], nets[1], nets[2:], flat, card_cls)
    assert 'ent_num' in eng.keys
    b = _ent_batch()
    sub = lambda a, n: {k: (b['feat_' + k] if 'feat_' + k in b else b[k])[a:a + n] for k in eng.keys}   # noqa: E731
    H = nets[0].H
    z = lambda n: np.zeros((n, H), np.float32)   # noqa: E731
    oa, oo, ox = sub(0, 5), sub(8, 3), sub(16, 2)
    for _ in range(3):
        r = eng.run(oa, z(5), z(5), oo, [ox, None], [z(2), None], [z(2), None])
        assert oa['card_mask'][np.arange(5), r['sel']].all()
        assert ox['card_mask'][np.arange(2), r['hist_sel'][0]].all()
    with torch.no_grad():
        out, (h, c) = nets[0]({k: torch.as_tensor(v).cuda() for k, v in oa.items()}, (torch.zeros(5, H).cuda().half(),) * 2)
    assert np.abs(r['h'] - h.float().cpu().numpy()).max() < 2e-2


def test_priv_value_head_and_loading():
    """特权价值头（model.priv_value）：旧存档（没有 priv_*）能加载、只缺这几项；观测里有特权信息才用它，否则用普通价值头；
    critic 热身训练的参数里包括它；结构不同的存档仍然报错。"""
    from cr_ai.model.net import Net, load_weights
    path = os.path.join(ROOT, 'runs', 'bc1', 'bc_final.pt')
    if not os.path.exists(path):
        pytest.skip('没有 bc1 存档')
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    net = Net(load('cr_ai/configs/bc.yaml', overrides=['model.priv_value=true'])).eval()
    miss = load_weights(net, sd)
    assert miss and all(k.startswith('priv_') for k in miss)
    with pytest.raises(RuntimeError):
        load_weights(Net(load('cr_ai/configs/bc.yaml', overrides=['model.deck_pool=mlp'])), sd)
    h = torch.randn(3, net.H)
    o = dict(priv_id=torch.tensor([[5, 6, 7, 8, 9]] * 3), priv_form=torch.zeros(3, 5, dtype=torch.long), priv_num=torch.rand(3, 1))
    with torch.no_grad():
        v0 = net.value_from(h, {})
        v1 = net.value_from(h, o)
        ref = torch.tanh(net.value_head(h).squeeze(-1))
    assert torch.allclose(v0, ref) and not torch.allclose(v1, ref)
    ids = {id(p) for p in net.value_params()}
    assert all(id(p) in ids for p in net.priv_value_head.parameters()) and all(id(p) in ids for p in net.priv_mlp.parameters())
