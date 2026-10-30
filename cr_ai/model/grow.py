"""cr_ai/model/grow.py — 训练中途给网络扩容，扩容后的网络和原来是同一个函数（2026-10-28，「手术」：OpenAI Five 训练中多次这样改结构；
Net2Net 式的零输出权重做法）。不用蒸馏，联赛里的旧快照、BC 参考网络也能原样转换（load_weights(..., grow=True)）。

支持的扩容（新旧结构只能差这些，其余必须一样）：
  model.hidden          融合层和 LSTM 加宽：新单元的输入权重随机，输出权重（进旧 LSTM 单元的门、进各个头和 FiLM）为 0
  model.head_hidden     各个头的隐藏层加宽：新隐藏单元的输出权重为 0
  model.exit_channels   主干出口卷积加通道：新通道只连到新的 grid_fc 单元
  model.grid_dim        grid_fc 加宽：新单元只连到新的融合层单元
  model.res_blocks / model.place_res_blocks   主干 / 位置头末尾加残差块：第二个 BN 的 γ、β 置 0 → 恒等
                        （PPO 里 BN 用固定统计量 = 逐通道仿射；块的输入是上一层 ReLU 的输出 ≥ 0，ReLU(x + 0) = x）
  model.entity_layers   实体分支末尾加 Transformer 层：proj 和 ff 最后一层置 0 → 恒等（pre-LN 残差）
不支持：model.ln（LayerNorm 换了宽度就不再等价）、channels / skip_channels / entity_dim。
新部分的初始值取自目标网络现在的参数（template）；函数和这些值无关（所以 hist 槽里残留的值也能当 template）。

    sd = grow_state_dict(sd_old, net_new.state_dict())
只给要接着训练的主模型（tools/grow.py）再做：redo_dead（融合层的死单元重新初始化）、calibrate_bn（新 BN 通道的统计量）、
set_forget_bias（新 LSTM 单元的遗忘门偏置）、grow_adam（AdamW 状态按同样的映射搬过去）。
"""
import numpy as np
import torch

# 末尾可以加块的模块序列，以及新块里置 0 的张量（置 0 后整块 = 恒等）
BLOCK_SEQS = {'res': ('b2.weight', 'b2.bias'), 'place_res': ('b2.weight', 'b2.bias'),
              'ent_blocks': ('proj.weight', 'proj.bias', 'ff.2.weight', 'ff.2.bias')}
HEADS_H = ('noop_head', 'value_head')                    # 输入 = h
HEADS_HX = ('card_score', 'priv_value_head')             # 输入 = [h; 64 维的卡牌/特权编码]


def _n_blocks(sd, seq):
    return len({k.split('.')[1] for k in sd if k.startswith(seq + '.') and k.split('.')[1].isdigit()})


def arch_of(sd):
    """从张量形状读出可扩容的那几项：H（融合层/LSTM）、hh（头）、E（出口通道）、G（grid_fc）、各序列的块数。"""
    a = dict(H=int(sd['lstm.weight_hh'].shape[1]), hh=int(sd['value_head.0.weight'].shape[0]),
             E=int(sd['down2.0.weight'].shape[0]), G=int(sd['grid_fc.0.weight'].shape[0]))
    a.update({s: _n_blocks(sd, s) for s in BLOCK_SEQS})
    return a


def needs_grow(sd_old, sd_new):
    return arch_of(sd_old) != arch_of(sd_new)


def _maps(a0, a1, sd_old):
    """要改形状的张量：每一维「旧下标 → 新下标」（None = 这一维没变）。旧单元排在前面，新单元接在后面；
    LSTM 的 4 个门（输入、遗忘、候选、输出）各自在块内这样排。"""
    H0, H1, h0 = a0['H'], a1['H'], a0['hh']
    pre = np.arange
    gate = np.concatenate([k * H1 + np.arange(H0) for k in range(4)])
    din = int(sd_old['fuse.0.weight'].shape[1])
    fuse_in = np.concatenate([np.arange(a0['G']), a1['G'] + np.arange(din - a0['G'])])     # 融合层输入 = [grid_fc; 其余]

    def hx(k):                                                                               # [h; 其余] 的输入
        n = int(sd_old[k].shape[1])
        return np.concatenate([np.arange(H0), H1 + np.arange(n - H0)])
    M = {'down2.0.weight': (pre(a0['E']), None),
         'grid_fc.0.weight': (pre(a0['G']), pre(a0['E'] * 40)), 'grid_fc.0.bias': (pre(a0['G']),),
         'fuse.0.weight': (pre(H0), fuse_in), 'fuse.0.bias': (pre(H0),),
         'lstm.weight_ih': (gate, pre(H0)), 'lstm.weight_hh': (gate, pre(H0)), 'lstm.bias_ih': (gate,), 'lstm.bias_hh': (gate,)}
    for leaf in ('weight', 'bias', 'running_mean', 'running_var'):
        M['down2.1.' + leaf] = (pre(a0['E']),)
    for k in ('film.weight', 'film_full.weight'):
        if k in sd_old:
            M[k] = (None, hx(k))
    for name in HEADS_H + HEADS_HX:
        if name + '.0.weight' not in sd_old:
            continue
        M[name + '.0.weight'] = (pre(h0), pre(H0) if name in HEADS_H else hx(name + '.0.weight'))
        M[name + '.0.bias'] = (pre(h0),)
        M[name + '.2.weight'] = (None, pre(h0))
    return M


def _put(t_new, t_old, maps, fill=0.0):
    """t_new 的拷贝里放进 t_old：旧的输出单元（第 0 维映射到的那些行）先整行填 fill（= 连到新输入的权重），再按映射放旧值；
    其余行（新单元）保持 t_new。"""
    out = t_new.clone()
    rows = torch.as_tensor(maps[0] if maps[0] is not None else np.arange(t_old.shape[0])).long()
    if t_old.dim() == 1:
        out[rows] = t_old
        return out
    blk = torch.full((len(rows),) + tuple(out.shape[1:]), float(fill), dtype=out.dtype)
    cols = maps[1] if len(maps) > 1 and maps[1] is not None else np.arange(t_old.shape[1])
    assert t_old.shape[2:] == out.shape[2:], (t_old.shape, out.shape)
    blk[:, torch.as_tensor(cols).long()] = t_old
    out[rows] = blk
    return out


def _check_growable(sd_old, template):
    if 'fuse.1.weight' in template or 'fuse.1.weight' in sd_old:
        raise ValueError('带 LayerNorm（model.ln）的结构不能这样扩容：换了宽度就不再是同一个函数')
    a0, a1 = arch_of(sd_old), arch_of(template)
    small = [k for k in a0 if a1[k] < a0[k]]
    if small:
        raise ValueError('只能扩大，不能缩小：%s（%s → %s）' % (small, a0, a1))
    return a0, a1


def grow_state_dict(sd_old, template):
    """sd_old（小结构）→ 目标结构（template = 目标网络的 state_dict）的权重，函数不变。张量都在 CPU 上、浮点为 float32。
    存档里本来就缺的键（如 BC 存档没有 priv_*）照样缺着，由 load_weights 处理。"""
    a0, a1 = _check_growable(sd_old, template)
    M = _maps(a0, a1, sd_old)
    tpl = lambda k: template[k].detach().cpu().float() if template[k].is_floating_point() else template[k].detach().cpu()   # noqa: E731
    out = {}
    for k, v in sd_old.items():
        if k not in template:
            raise KeyError('目标结构里没有 %s（只支持 grow.py 文件头列出的那几种扩容）' % k)
        v = v.detach().cpu()
        if k in M:
            out[k] = _put(tpl(k), v.float(), M[k])
        else:
            if tuple(v.shape) != tuple(template[k].shape):
                raise ValueError('%s 形状 %s → %s，不是支持的扩容' % (k, tuple(v.shape), tuple(template[k].shape)))
            out[k] = v.float() if v.is_floating_point() else v.clone()
    for seq, zero in BLOCK_SEQS.items():
        for i in range(a0[seq], a1[seq]):
            pre = '%s.%d.' % (seq, i)
            for k in template:
                if k.startswith(pre):
                    out[k] = tpl(k).clone()
            for leaf in zero:
                out[pre + leaf].zero_()
    return out


def gate_rows(H0, H1):
    """旧 LSTM 单元在新的 4H1 行里的位置。"""
    return torch.cat([k * H1 + torch.arange(H0) for k in range(4)])


def redo_dead(sd, template, dead, H0, H1):
    """融合层的死单元（旧单元里对观测样本恒 ≤ 0 的，下标 < H0）：输入权重和偏置换成 template 里的随机值；
    它们进旧 LSTM 单元的输入权重置 0（进新 LSTM 单元的照旧随机）。它们原来在样本上输出恒为 0 → 在样本上函数不变（ReDo，Sokar 2023）。"""
    d = torch.as_tensor(np.asarray(dead)).long()
    if not len(d):
        return sd
    sd['fuse.0.weight'][d] = template['fuse.0.weight'][d].detach().cpu().float()
    sd['fuse.0.bias'][d] = template['fuse.0.bias'][d].detach().cpu().float()
    g = gate_rows(H0, H1)
    sd['lstm.weight_ih'][g[:, None], d[None, :]] = 0.0
    return sd


def set_forget_bias(sd, H0, H1, value=1.0):
    """新 LSTM 单元（下标 ≥ H0）的遗忘门偏置 = value（bias_ih + bias_hh；常见做法，起步时记忆别太短）。"""
    r = H1 + torch.arange(H0, H1)
    sd['lstm.bias_ih'][r] = float(value)
    sd['lstm.bias_hh'][r] = 0.0
    return sd


@torch.no_grad()
def calibrate_bn(net, a0, obs_batches, place_card=1):
    """新 BN 通道（出口卷积的新通道、新残差块的两个 BN）的统计量 = 观测样本上的均值/方差（γ = 1、β = 0 不变；新块第二个 BN 的 γ、β 仍是 0）。
    新块 = 恒等，所以各新块的输入和旧网络一样，按「先所有 b1、再所有 b2」两遍就够。obs_batches：可重复迭代的观测 dict 列表。"""
    mods = []
    if net.down2[1].num_features > a0['E']:
        mods.append(('down2', net.down2[1], slice(a0['E'], None)))
    for seq in ('res', 'place_res'):
        blocks = getattr(net, seq)
        for i in range(a0[seq], len(blocks)):
            mods.append(('%s.%d.b1' % (seq, i), blocks[i].b1, slice(None)))
    second = [('%s.%d.b2' % (seq, i), getattr(net, seq)[i].b2, slice(None))
              for seq in ('res', 'place_res') for i in range(a0[seq], len(getattr(net, seq)))]
    info = {}
    for group in (mods, second):
        if not group:
            continue
        acc = {name: [0.0, 0.0, 0] for name, _, _ in group}
        hooks = []
        for name, bn, _ in group:
            def hook(m, inp, out, name=name):
                x = inp[0].detach().double()
                a = acc[name]
                a[0] = a[0] + x.sum((0, 2, 3))
                a[1] = a[1] + (x * x).sum((0, 2, 3))
                a[2] += x.shape[0] * x.shape[2] * x.shape[3]
            hooks.append(bn.register_forward_hook(hook))
        for o in obs_batches:
            out, _ = net(o)
            if any(n.startswith('place_res') for n, _, _ in group):
                net.place_logits(out, torch.full((len(o['tstate']),), place_card, dtype=torch.long, device=out['h'].device))
        for h in hooks:
            h.remove()
        for name, bn, sl in group:
            s, s2, n = acc[name]
            mu = s / n
            var = (s2 / n - mu * mu).clamp(min=1e-5)
            bn.running_mean[sl] = mu[sl].to(bn.running_mean.dtype)
            bn.running_var[sl] = var[sl].to(bn.running_var.dtype)
            info[name] = (float(mu[sl].abs().mean()), float(var[sl].mean()))
    return info


def synth_priv(B, gen, dev):
    """特权输入（对手真实手牌 5 张、真实圣水）：随机生成（观测样本里没有特权信息时，用来走特权价值头）。"""
    from ..env import tables
    return dict(priv_id=torch.randint(1, tables.n_cards(), (B, 5), generator=gen).to(dev),
                priv_form=torch.zeros(B, 5, dtype=torch.long).to(dev), priv_num=torch.rand(B, 1, generator=gen).to(dev))


@torch.no_grad()
def center_new_units(net, a0, dead, obs_batches, seed=0):
    """新加的 ReLU 单元（grid_fc、融合层、各个头隐藏层的新行，以及重置的死单元）：偏置减去样本上预激活的中位数 → 约一半样本取正。
    输入都是 ReLU 之后的非负特征，按默认初始化有不少新单元对所有样本都 ≤ 0，生下来就是死的。它们只连到新单元（或输出权重为 0），
    改偏置不改函数。按 grid_fc → 融合层 → 头的顺序做（后面的输入依赖前面）。→ {层: 居中前取正的比例中位数}。"""
    gen = torch.Generator().manual_seed(seed)
    targets = [('grid_fc', net.grid_fc[0], list(range(a0['G'], net.grid_fc[0].out_features))),
               ('fuse', net.fuse[0], list(range(a0['H'], net.H)) + [int(x) for x in dead])]
    for name in HEADS_H + HEADS_HX:
        if hasattr(net, name):
            lin = getattr(net, name)[0]
            targets.append((name, lin, list(range(a0['hh'], lin.out_features))))
    info = {}
    for name, lin, rows in targets:
        if not rows:
            continue
        r = torch.as_tensor(rows, device=lin.bias.device)
        got = []
        h = lin.register_forward_hook(lambda m, i, o: got.append(o.detach().float().reshape(-1, o.shape[-1])[:, r]))
        for o in obs_batches:
            net({k: v for k, v in o.items() if not k.startswith('priv_')})              # 普通价值头
            if getattr(net, 'priv_value', False):                                        # 特权价值头（样本里没有特权信息就随机生成）
                net(o if 'priv_id' in o else dict(o, **synth_priv(len(o['tstate']), gen, o['tstate'].device)))
        h.remove()
        if not got:
            continue
        x = torch.cat(got)
        info[name] = float((x > 0).float().mean(0).median())
        lin.bias[r] -= x.median(0).values.to(lin.bias.dtype)
    return info


def _analog(name, a0):
    """新块的参数 → 前一个旧块里同名的参数（取它的二阶矩量级）。"""
    p = name.split('.')
    if p[0] in BLOCK_SEQS and p[1].isdigit() and int(p[1]) >= a0[p[0]] and a0[p[0]] > 0:
        return '.'.join([p[0], str(a0[p[0]] - 1)] + p[2:])
    return None


def grow_adam(osd, old_names, new_names, a0, a1, sd_old, dead=()):
    """AdamW 的 state_dict 按和权重一样的映射搬到新结构（old_names / new_names：优化器里参数的顺序 = 名字）。
    一阶矩：新元素 0。二阶矩：新元素 = 同一张量旧元素的中位数（全新的张量取前一个旧块里同名张量的中位数）——取 0 的话
    第一步会迈出约 3 倍学习率、把连到旧单元的那些零权重一下推开。死单元重置的那些元素同样处理（它们以前几乎没梯度，二阶矩≈0）。"""
    old_names = [x[0] if isinstance(x, tuple) else x for x in old_names]
    assert len(osd['param_groups']) == 1 and len(osd['param_groups'][0]['params']) == len(old_names), '优化器参数个数和网络对不上'
    M = _maps(a0, a1, sd_old)
    st_old = {old_names[i]: s for i, s in osd['state'].items()}
    any_s = next(iter(st_old.values()))
    allv = torch.cat([s['exp_avg_sq'].flatten().float().cpu() for s in st_old.values()])
    gmed = float(allv[allv > 0].median()) if (allv > 0).any() else 1e-12
    med = lambda t: float(t[t > 0].median()) if (t > 0).any() else gmed      # noqa: E731
    d = torch.as_tensor(np.asarray(dead)).long()
    g = gate_rows(a0['H'], a1['H'])
    state = {}
    for j, (name, shape) in enumerate(new_names):
        s = st_old.get(name)
        if s is None:
            src = st_old.get(_analog(name, a0) or '')
            v0 = med(src['exp_avg_sq'].float().cpu()) if src is not None else gmed
            state[j] = dict(step=any_s['step'].clone(), exp_avg=torch.zeros(shape), exp_avg_sq=torch.full(shape, v0))
            continue
        m, v = s['exp_avg'].float().cpu(), s['exp_avg_sq'].float().cpu()
        vm = med(v)
        if name in M:
            m = _put(torch.zeros(shape), m, M[name], fill=0.0)
            v = _put(torch.full(shape, vm), v, M[name], fill=vm)
        if len(d) and name in ('fuse.0.weight', 'fuse.0.bias'):
            m[d], v[d] = 0.0, vm
        if len(d) and name == 'lstm.weight_ih':
            m[g[:, None], d[None, :]] = 0.0
            v[g[:, None], d[None, :]] = vm
        assert tuple(m.shape) == tuple(shape), (name, m.shape, shape)
        state[j] = dict(step=s['step'].clone(), exp_avg=m, exp_avg_sq=v)
    pg = dict(osd['param_groups'][0])
    pg['params'] = list(range(len(new_names)))
    return dict(state=state, param_groups=[pg])


def opt_names(net, value_only=False, freeze_unit_emb=True):
    """和 train/ppo.py 的 Learner 一样的优化器参数顺序 → [(名字, 形状)]：opt = 除了冻结的单位/法术嵌入以外的全部参数；opt_v = value_params()。"""
    named = list(net.named_parameters())
    if value_only:
        ids = {id(p): n for n, p in named}
        return [(ids[id(p)], tuple(p.shape)) for p in net.value_params()]
    skip = {'unit_emb.weight', 'spell_emb.weight'} if freeze_unit_emb else set()
    return [(n, tuple(p.shape)) for n, p in named if n not in skip]
