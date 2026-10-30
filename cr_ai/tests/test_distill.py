"""蒸馏（train/distill.py）：学生配置与从老师复制权重、蒸馏损失（学生 = 老师时为 0）、老师出手概率。BC 分片的一个 batch，CPU 上跑。"""
import glob
import os

import pytest
import torch

from cr_ai.config import ROOT, load


def _batch():
    from cr_ai.data.bc_dataset import BCShards
    files = sorted(glob.glob(os.path.join(ROOT, 'data_bc', 'il_replay', 'shard_*.npz')))[:1]
    if not files:
        pytest.skip('没有 BC 分片')
    ds = BCShards([], split='train')
    ds.files = files
    return next(ds.batches(batch_size=64, mirror_prob=0.0, seed=0))


def _obs(b, net):
    return {k: torch.as_tensor(b['feat_' + k] if 'feat_' + k in b else b[k]) for k in list(net.input_keys) + ['pl']}


def test_copy_shared_keeps_trunk_and_widens():
    """学生 = 老师的结构 + 更宽的 LSTM、LayerNorm、更宽的头：主干等原样复制（编码出的特征图和手牌表示完全相同），融合层以后是新的。"""
    from cr_ai.model.net import Net
    from cr_ai.train.distill import copy_shared, student_cfg
    tcfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    teacher = Net(tcfg).eval()
    H = int(tcfg.model.hidden)
    scfg = student_cfg(tcfg, files=(), overrides=['model.hidden=%d' % (2 * H), 'model.ln=true', 'model.head_hidden=192'])
    student = Net(scfg).eval()
    copied, new = copy_shared(student, teacher.state_dict())
    assert 'res.0.c1.weight' in copied and 'stem.1.running_mean' in copied and 'card_emb.weight' in copied
    assert 'lstm.weight_ih' in new and 'fuse.1.weight' in new and 'noop_head.1.weight' in new     # fuse.1 / noop_head.1 = LayerNorm
    assert not any(k.startswith('fuse.1') for k in teacher.state_dict())                           # 默认结构没有 LayerNorm（旧存档照常加载）
    o = _obs(_batch(), teacher)
    with torch.no_grad():
        xt, ft, skt, hdt, _ = teacher.encode(o)
        xs, fs, sks, hds, _ = student.encode(o)
    assert torch.equal(ft, fs) and torch.equal(skt, sks) and torch.equal(hdt, hds)
    assert xs.shape[1] == 2 * xt.shape[1] == 2 * H


def test_distill_loss_zero_when_student_equals_teacher():
    """老师的分布由同一个网络算出：出牌 KL、位置 KL、价值误差都是 0，一致率 1；学生偏离后 KL > 0、梯度有限。"""
    from cr_ai.env import masks as MK
    from cr_ai.model.net import Net
    from cr_ai.train.distill import distill_loss, place_mask
    from cr_ai.train.ppo import seq_forward
    cfg = load('cr_ai/configs/bc.yaml')
    torch.manual_seed(0)
    net = Net(cfg).eval()
    b = _batch()
    o = _obs(b, net)
    M, L = 8, 8
    h0, c0 = torch.zeros(M, net.H), torch.zeros(M, net.H)
    flat, card_cls = MK.all_masks()
    flat, card_cls = torch.as_tensor(flat), torch.as_tensor(card_cls).long()
    sel = torch.as_tensor(b['sel']).long()
    hi = ((sel >= 1) & (sel <= 4)).nonzero().squeeze(1)
    assert len(hi) > 0
    with torch.no_grad():
        out, card, Hs = seq_forward(net, o, h0, c0, L)
        t_card = torch.log_softmax(net.policy_logits(card.float()), 1).clamp(min=-30)
        t_place = torch.zeros(M * L, 576)
        pm = place_mask(o, hi, sel, flat, card_cls)
        sub = {k: out[k][hi] for k in ('h', 'f16', 'skip', 'hand')}
        t_place[hi] = torch.log_softmax(net.place_logits(sub, sel[hi]).float().masked_fill(~pm, -1e9), 1).clamp(min=-30)
        t_value = net.value_from(Hs, o).float()
    D = dict(sel=sel, t_card=t_card, t_place=t_place, t_value=t_value)
    m = torch.ones(M * L)
    _, parts = distill_loss(net, o, h0, c0, L, lambda k: D[k], m, flat, card_cls, 1.0, 0.5)
    assert abs(parts['kl_card']) < 1e-5 and abs(parts['kl_place']) < 1e-4 and parts['vmse'] < 1e-8 and parts['agree'] == 1.0
    with torch.no_grad():
        net.noop_head[-1].bias += 2.0
        net.value_head[-1].bias += 0.5
    loss, parts = distill_loss(net, o, h0, c0, L, lambda k: D[k], m, flat, card_cls, 1.0, 0.5)
    assert parts['kl_card'] > 1e-3 and parts['vmse'] > 1e-4
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_teacher_act_prob_anneals():
    from cr_ai.train.distill import teacher_act_prob
    dc = load('cr_ai/configs/distill.yaml').distill
    s, e = float(dc.teacher_act_start), float(dc.teacher_act_end)
    assert teacher_act_prob(0, dc) == s
    assert abs(teacher_act_prob(int(dc.teacher_act_anneal_steps) // 2, dc) - (s + e) / 2) < 1e-6
    assert teacher_act_prob(10 ** 12, dc) == e
