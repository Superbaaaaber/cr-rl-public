"""扩容手术（model/grow.py）：扩容后和原来是同一个函数（逐步带 LSTM 状态、学习端的整段 LSTM 也一样）；死单元重置、新 BN 校准、
遗忘门偏置之后在样本上仍一致；旧存档（含缺特权头的 BC 存档）能自动补成新结构；AdamW 状态能搬；新参数拿得到梯度。小网络、CPU。"""
import pytest
import torch

from cr_ai.config import load

OLD = ['model.channels=32', 'model.skip_channels=16', 'model.res_blocks=2', 'model.hidden=48', 'model.head_hidden=16',
       'model.entity=true', 'model.entity_dim=32', 'model.entity_layers=1', 'model.entity_grid_channels=8',
       'model.place_res_blocks=1', 'model.place_film_full=true', 'model.deck_pool=mlp', 'model.hand_slot=next', 'model.priv_value=true']
GROW = ['model.hidden=80', 'model.head_hidden=24', 'model.exit_channels=96', 'model.grid_dim=320',
        'model.res_blocks=3', 'model.place_res_blocks=3', 'model.entity_layers=3']


def _cfgs():
    return load('cr_ai/configs/bc.yaml', overrides=OLD), load('cr_ai/configs/bc.yaml', overrides=OLD + GROW)


def _old_net(cfg, seed=0):
    """随机初始化，再把 BN 统计量、FiLM 也弄成非平凡的值（默认 BN 近似恒等、FiLM 是 0，测不出映射错）。"""
    from cr_ai.model.net import Net
    torch.manual_seed(seed)
    net = Net(cfg).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.normal_(0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
                m.weight.normal_(1, 0.1)
                m.bias.normal_(0, 0.1)
        for lin in (net.film, net.film_full):
            lin.weight.normal_(0, 0.02)
            lin.bias.normal_(0, 0.02)
    return net


def _rand_obs(cfg, B, gen):
    from cr_ai.env import tables
    from cr_ai.env.game_env import obs_spec
    from cr_ai.model.net import rl_keys
    nu, nc = tables.n_units(), tables.n_cards()
    keys = set(rl_keys(cfg))
    o = {}
    for k, (shp, dt) in obs_spec(cfg).items():
        if k not in keys:
            continue
        shape = (B,) + tuple(shp)
        if k == 'gi':
            v = torch.randint(0, nu, shape, generator=gen) * (torch.rand(shape, generator=gen) < 0.1)
        elif k == 'ent_id':
            v = torch.stack([torch.randint(0, nu, shape[:-1], generator=gen), torch.randint(0, 576, shape[:-1], generator=gen)], -1)
        elif k == 'ent_num':
            v = torch.rand(shape, generator=gen)
            v[..., 0] = (torch.rand(shape[:-1], generator=gen) < 0.3).float()
        elif k == 'card_mask':
            v = torch.rand(shape, generator=gen) < 0.6
            v[:, 0] = True
        elif k == 'tstate':
            v = torch.randint(0, 16, shape, generator=gen)
        elif k.endswith('_form'):
            v = torch.randint(0, 3, shape, generator=gen)
        elif k.endswith('_id'):
            v = torch.randint(0, nc, shape, generator=gen)
        else:
            v = torch.rand(shape, generator=gen)
        o[k] = v
    return o


def _grown(old, cfg1, seed=1):
    from cr_ai.model.grow import grow_state_dict
    from cr_ai.model.net import Net
    torch.manual_seed(seed)
    new = Net(cfg1).eval()
    new.load_state_dict(grow_state_dict(old.state_dict(), new.state_dict()))
    return new


@torch.no_grad()
def _max_diff(old, new, cfg, steps=6, B=5, seed=0):
    """逐步带状态跑 steps 步：出牌 logits（合法的）、价值（特权 / 普通）、位置 logits（第 1–4 张手牌轮流）、旧单元的 h/c。"""
    gen = torch.Generator().manual_seed(seed)
    H0 = old.H
    s0 = s1 = None
    d = 0.0
    for t in range(steps):
        o = _rand_obs(cfg, B, gen)
        pc = torch.full((B,), 1 + t % 4, dtype=torch.long)
        a, s0 = old(o, s0, place_card=pc)
        b, s1 = new(o, s1, place_card=pc)
        m = o['card_mask']
        plain = {k: v for k, v in o.items() if not k.startswith('priv_')}
        d = max(d, float((a['card'] - b['card'])[m].abs().max()), float((a['value'] - b['value']).abs().max()),
                float((old.value_from(a['h'], plain) - new.value_from(b['h'], plain)).abs().max()),
                float((a['place'] - b['place']).abs().max()), float((s0[0] - s1[0][:, :H0]).abs().max()),
                float((s0[1] - s1[1][:, :H0]).abs().max()))
    return d



def test_grow_is_same_function():
    from cr_ai.model.grow import arch_of, needs_grow
    c0, c1 = _cfgs()
    old = _old_net(c0)
    new = _grown(old, c1)
    a0, a1 = arch_of(old.state_dict()), arch_of(new.state_dict())
    assert a0 == dict(H=48, hh=16, E=64, G=256, res=2, place_res=1, ent_blocks=1)
    assert a1 == dict(H=80, hh=24, E=96, G=320, res=3, place_res=3, ent_blocks=3)
    assert needs_grow(old.state_dict(), new.state_dict()) and not needs_grow(new.state_dict(), new.state_dict())
    assert sum(p.numel() for p in new.parameters()) > 1.3 * sum(p.numel() for p in old.parameters())
    assert _max_diff(old, new, c0) < 1e-5


def test_grow_same_function_learner_sequence():
    """学习端整段 LSTM（ppo.seq_forward，cuDNN 式的 _VF.lstm）也一样。"""
    from cr_ai.train.ppo import seq_forward
    c0, c1 = _cfgs()
    old = _old_net(c0)
    new = _grown(old, c1)
    M, L = 3, 5
    o = _rand_obs(c0, M * L, torch.Generator().manual_seed(3))
    with torch.no_grad():
        h0 = torch.randn(M, old.H) * 0.5
        c0_ = torch.randn(M, old.H) * 0.5
        pad = lambda x: torch.cat([x, torch.randn(M, new.H - old.H)], 1)   # noqa: E731  新单元的初始状态随便给：不影响旧单元
        _, ca, Ha = seq_forward(old, o, h0, c0_, L)
        _, cb, Hb = seq_forward(new, o, pad(h0), pad(c0_), L)
    m = o['card_mask']
    assert float((ca - cb)[m].abs().max()) < 1e-5
    assert float((Ha - Hb[:, :old.H]).abs().max()) < 1e-5
    assert float((old.value_from(Ha, o) - new.value_from(Hb, o)).abs().max()) < 1e-5


def test_redo_calibrate_forget_bias_keep_function_on_sample():
    """融合层里人为弄死几个单元（偏置很负）：找出来、重置、校准新 BN、新单元偏置居中、设遗忘门偏置后，仍是同一个函数；
    重置的单元和新单元都有约一半样本取正；新 BN 通道的统计量变成样本上的值；新 LSTM 单元的遗忘门偏置 = 1。"""
    from cr_ai.model import grow as G
    from cr_ai.model.net import Net
    from cr_ai.tools.grow import FuseProbe
    c0, c1 = _cfgs()
    old = _old_net(c0)
    dead = [3, 7, 20]
    with torch.no_grad():
        old.fuse[0].bias[dead] = -1e3
    batches = [_rand_obs(c0, 16, torch.Generator().manual_seed(10 + i)) for i in range(3)]
    fp = FuseProbe(old)
    with torch.no_grad():
        for o in batches:
            old(o)
    fp.close()
    found = (fp.mx <= 0).nonzero().flatten().numpy()
    assert set(dead) <= set(found.tolist())
    torch.manual_seed(1)
    new = Net(c1).eval()
    tpl = {k: v.clone() for k, v in new.state_dict().items()}
    a0 = G.arch_of(old.state_dict())
    sd = G.grow_state_dict(old.state_dict(), tpl)
    G.redo_dead(sd, tpl, dead, old.H, new.H)       # 只重置真死的：小样本上找到的其它「死」单元在别的输入上会取正，重置它们会改函数
    G.set_forget_bias(sd, old.H, new.H, 1.0)
    new.load_state_dict(sd)
    info = G.calibrate_bn(new, a0, batches)
    cen = G.center_new_units(new, a0, dead, batches)
    assert {'grid_fc', 'fuse', 'card_score', 'value_head', 'priv_value_head'} <= set(cen)
    assert 'down2' in info and 'place_res.2.b2' in info and 'res.2.b1' in info
    assert not torch.allclose(new.down2[1].running_var[64:], torch.ones(32))
    assert torch.equal(new.down2[1].running_var[:64], old.down2[1].running_var)
    assert torch.allclose(new.lstm.bias_ih[new.H + old.H:2 * new.H] + new.lstm.bias_hh[new.H + old.H:2 * new.H], torch.ones(new.H - old.H))
    assert _max_diff(old, new, c0) < 1e-5
    fp = FuseProbe(new)
    with torch.no_grad():
        for o in batches:
            new(o)
    fp.close()
    frac = fp.pos.float() / fp.n
    assert (frac[torch.as_tensor(dead)] > 0.2).all() and (frac[old.H:] > 0.2).float().mean() > 0.9     # 重置的、新加的都不是死的


def test_load_weights_grow_bc_without_priv():
    """BC 存档（没有特权头）是小结构：grow=True 时补成新结构（缺的 priv_* 照样报缺），不开 grow 报错。"""
    from cr_ai.model.net import Net, load_weights
    c0, c1 = _cfgs()
    old = _old_net(c0)
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith('priv_')}
    torch.manual_seed(2)
    new = Net(c1).eval()
    with pytest.raises(RuntimeError):
        load_weights(new, sd)
    miss = load_weights(new, sd, grow=True)
    assert miss and all(k.startswith('priv_') for k in miss)
    o = _rand_obs(c0, 4, torch.Generator().manual_seed(5))
    plain = {k: v for k, v in o.items() if not k.startswith('priv_')}
    with torch.no_grad():
        a, _ = old(plain)
        b, _ = new(plain)
    assert float((a['card'] - b['card'])[o['card_mask']].abs().max()) < 1e-5
    assert float((a['value'] - b['value']).abs().max()) < 1e-5


def test_load_weights_zeroes_missing_ext():
    """开了 model.obs_ext 的网络（采样端的历史槽）先装一个 ext 权重不为 0 的新快照，再装一个没有 ext_* 的旧快照（还是扩容前的小结构）：
    ext 权重要清零，输出和旧快照在原网络上逐位一致（不能沿用上一个快照的 ext 权重）。"""
    from cr_ai.model.net import Net, load_weights
    c0, c1 = _cfgs()
    c1x = load('cr_ai/configs/bc.yaml', overrides=OLD + GROW + ['model.obs_ext=true'])
    old = _old_net(c0)
    slot = Net(c1x).eval()
    with torch.no_grad():
        trained = {k: (torch.randn_like(v) if k.startswith('ext_') else v) for k, v in slot.state_dict().items()}
    load_weights(slot, trained)
    assert slot.ext_hand.weight.abs().sum() > 0
    miss = load_weights(slot, old.state_dict(), grow=True)
    assert {k for k in miss if k.startswith('ext_')} == {k for k in slot.state_dict() if k.startswith('ext_')}
    assert all(float(p.abs().sum()) == 0 for n, p in slot.named_parameters() if n.startswith('ext_'))
    o = _rand_obs(c1x, 4, torch.Generator().manual_seed(4))           # 带 ab_ext / hand_ext / opp_ext / ent_ext（随机值）
    assert 'ent_ext' in o
    with torch.no_grad():
        a, _ = old(o)
        b, _ = slot(o)
    assert float((a['card'] - b['card'])[o['card_mask']].abs().max()) < 1e-5


def test_grow_adam_state():
    """AdamW 状态按同样的映射搬：旧元素原样、新元素一阶矩 0 / 二阶矩 > 0、新块也有状态；新优化器能直接 load。"""
    from cr_ai.model import grow as G
    c0, c1 = _cfgs()
    old = _old_net(c0)
    for p in (old.unit_emb.weight, old.spell_emb.weight):
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in old.parameters() if p.requires_grad], lr=1e-4, eps=1e-5)
    for _ in range(2):
        for p in old.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt.step()
    new = _grown(old, c1)
    a0, a1 = G.arch_of(old.state_dict()), G.arch_of(new.state_dict())
    osd = G.grow_adam(opt.state_dict(), G.opt_names(old), G.opt_names(new), a0, a1, old.state_dict(), dead=[5])
    for p in (new.unit_emb.weight, new.spell_emb.weight):
        p.requires_grad_(False)
    opt2 = torch.optim.AdamW([p for p in new.parameters() if p.requires_grad], lr=1e-4, eps=1e-5)
    opt2.load_state_dict(osd)
    names = [n for n, _ in G.opt_names(new)]
    st = {n: opt2.state[p] for n, p in zip(names, [p for p in new.parameters() if p.requires_grad])}
    so = {n: opt.state[p] for n, p in zip([n for n, _ in G.opt_names(old)], [p for p in old.parameters() if p.requires_grad])}
    H0, H1 = old.H, new.H
    g = G.gate_rows(H0, H1)
    w = st['lstm.weight_hh']
    assert torch.equal(w['exp_avg'][g][:, :H0], so['lstm.weight_hh']['exp_avg'])
    assert (w['exp_avg'][g][:, H0:] == 0).all() and (w['exp_avg_sq'][g][:, H0:] > 0).all()
    assert (st['lstm.weight_ih']['exp_avg'][g][:, 5] == 0).all()                  # 重置的死单元：连到旧单元的那一列
    assert (st['place_res.2.c1.weight']['exp_avg'] == 0).all() and (st['place_res.2.c1.weight']['exp_avg_sq'] > 0).all()
    assert torch.equal(st['card_emb.weight']['exp_avg'], so['card_emb.weight']['exp_avg'])


def test_new_parts_get_gradients():
    """扩容后新参数有梯度：新残差块第二个 BN 的 γ、新实体层的 proj；出牌头里旧隐藏单元连到新 LSTM 单元的那些列、
    新隐藏单元的输出权重（新隐藏单元自己的输入权重要等输出权重不为 0 以后才有梯度）。"""
    c0, c1 = _cfgs()
    old = _old_net(c0)
    new = _grown(old, c1)
    o = _rand_obs(c0, 8, torch.Generator().manual_seed(7))
    out, _ = new(o, place_card=torch.ones(8, dtype=torch.long))
    (out['card'].masked_fill(~o['card_mask'], 0).sum() + out['value'].sum() + out['place'].sum()).backward()
    assert new.place_res[2].b2.weight.grad.abs().sum() > 0
    assert new.ent_blocks[2].proj.weight.grad.abs().sum() > 0
    g0 = new.card_score[0].weight.grad
    assert g0[:16, old.H:new.H].abs().sum() > 0 and g0[16:].abs().sum() == 0
    assert new.card_score[2].weight.grad[:, 16:].abs().sum() > 0


def test_grow_rejects_ln_and_shrink():
    from cr_ai.model.grow import grow_state_dict
    from cr_ai.model.net import Net
    c0, c1 = _cfgs()
    old, big = Net(c0), Net(c1)
    with pytest.raises(ValueError):
        grow_state_dict(big.state_dict(), old.state_dict())
    ln = Net(load('cr_ai/configs/bc.yaml', overrides=OLD + GROW + ['model.ln=true']))
    with pytest.raises(ValueError):
        grow_state_dict(old.state_dict(), ln.state_dict())
