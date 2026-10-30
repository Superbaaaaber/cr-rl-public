"""cr_ai/train/ppo.py — 循环 PPO（学习进程）。采样在另一个进程（train/rollout.py），两边并行。

    python -m cr_ai.train.ppo --config cr_ai/configs/ppo.yaml --run runs/ppo1 [--resume] [--updates N] [--no-eval]

流程：
  1. 从 BC 模型初始化，转成带状态的 LSTM（model/net.py: init_recurrence，首步与 BC 完全一致）；
  2. 热身 critic：前 critic_warmup_steps 步只训价值头（策略不动）；续训中途改了价值目标（如打开 model.value_prior）时，
     用 ppo.critic_rewarm_until（绝对步数）再热身一段；
  3. 正式训练：每批 W × T 步，按 seq_len 切成序列（LSTM 截断反传），epochs 轮 × minibatches 个小批（显存不够时每个小批再拆 micro_batches 份累积梯度）。
损失：PPO 截断目标（牌头 + 位置头的联合对数概率）+ 价值 MSE − 熵奖励（牌头、位置头分开系数）+ β·KL(π ‖ π_BC)。
日志里的 gns_G2 / gns_trS（ppo.noise_scale，默认开，只读梯度、不影响训练）：梯度噪声尺度的两个分量，见 noise_scale_parts。
ppo.noise_lambdas（默认空 = 关，只读梯度、不影响训练）：影子 λ——同一批数据按别的 GAE λ 算优势，记 adv_var_lam* 和
策略梯度项的单样本梯度方差 gns_trS_lam*（每 noise_lambda_every 次更新测一次），用来比较降 λ 能把临界批量降多少，见 shadow_advantages。
KL 的参考分布 π_BC 由采样进程顺手算好（BC 网络本来就要给对手出动作，主角一方并进同一次前向，几乎不额外花时间），
学习进程不用再为 BC 网络做前向——比“训练时每个小批额外前向一次 BC 网络”省掉约三分之一的训练算力。
输出：runs/<run>/ppo_latest.pt（续训）、ppo_step<N>.pt（每 snapshot_every_steps 一个快照，评测 / M4 历史对手用）、
      train_log.jsonl、arena.jsonl（定期对战评测）、tb/、config.yaml、run_info.json。
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ROOT, dump, limit_torch_threads, load, repo_path
from ..env import masks as MK
from ..model.net import Net, init_recurrence, load_weights, n_params, rl_keys
from .league import ExploitTarget, League
from .rollout import Slot, WeightBox, ab_floor_eps, actor_main, bc_ref_every, slot_bytes, slot_spec

warnings.filterwarnings('ignore', message='RNN module weights are not part of single contiguous chunk')


def gae(rew, val, done, last_v, gamma, lam, trunc=None):
    """trunc：被截断的局的最后一步（引擎出错，后面没有观测）——不当终局，用这一步自己的价值代替下一步的价值自举；
    done 仍为 True（优势不往前一局传、序列在这里断开）。"""
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    nxt_adv = np.zeros(rew.shape[1], np.float32)
    nxt_v = last_v
    for t in range(T - 1, -1, -1):
        nd = 1.0 - done[t].astype(np.float32)
        boot = nxt_v * nd if trunc is None else nxt_v * nd + val[t] * trunc[t]
        delta = rew[t] + gamma * boot - val[t]
        nxt_adv = delta + gamma * lam * nd * nxt_adv
        adv[t] = nxt_adv
        nxt_v = val[t]
    return adv, adv + val


def critic_warm(step, pc):
    """这一批是否只训价值头：训练开头的 critic_warmup_steps 步，或续训中途换了价值目标后、到 critic_rewarm_until（绝对步数）为止。"""
    return step < pc.critic_warmup_steps or step < int(pc.get('critic_rewarm_until', 0) or 0)


def noise_scale_parts(sq_small, b_small, sq_big):
    """梯度噪声尺度（McCandlish 等 2018，arXiv:1812.06162 附录 A）的两个无偏分量：
    sq_small：各小批梯度的 |g_i|²；b_small：各小批大小（加权有效步数）；sq_big：按大小加权平均后的整批梯度 |G_B|²。
    |G|² ≈ (B·|G_B|² − b·|g|²) / (B − b)，tr(Σ) ≈ (|g|² − |G_B|²) / (1/b − 1/B)（b = 小批平均大小，B = 总和）。
    单次估计噪声很大（|G|² 可能为负）：临界批量 B_noise = tr(Σ) / |G|² 要对很多次更新分别求平均后再相除。"""
    B, b, s = float(sum(b_small)), float(np.mean(b_small)), float(np.mean(sq_small))
    return dict(gns_G2=(B * sq_big - b * s) / (B - b), gns_trS=(s - sq_big) / (1.0 / b - 1.0 / B), gns_batch=B)


def policy_targets(rew, val, done, last_v, gamma, lam_value, lam_policy, trunc=None):
    """→ (策略用的优势, 价值目标)。价值目标是按 lam_value 的 λ 回报；lam_policy 不同时（ppo.policy_lambda，2026-10-28）策略的优势另按它算
    （解耦 GAE：降策略的 λ 减小优势里的噪声，价值目标不跟着变短）。"""
    adv, ret = gae(rew, val, done, last_v, gamma, lam_value, trunc)
    if lam_policy is not None and float(lam_policy) != float(lam_value):
        adv = gae(rew, val, done, last_v, gamma, float(lam_policy), trunc)[0]
    return adv, ret


def shadow_advantages(rew, val, done, last_v, gamma, lams, trunc, vmask, adv_train):
    """影子 λ（ppo.noise_lambdas）：同一批数据按别的 GAE λ 算优势。都减各自的均值、除以训练用优势（adv_train）的标准差——
    用同一个尺度，不各自标准化：降 λ 减小的是优势里的噪声，各自标准化会把这部分差别抹掉。→ {λ: 优势}（无效步为 0）"""
    sd = float(adv_train[vmask].std()) + 1e-8
    out = {}
    for lam in lams:
        a = np.where(vmask, gae(rew, val, done, last_v, gamma, lam, trunc)[0], 0.0)
        out[lam] = np.where(vmask, (a - a[vmask].mean()) / sd, 0.0).astype(np.float32)
    return out


def _lstm_seq(net, xs, h0, c0):
    """xs [M, L, D] → [M, L, H]。序列都不跨局（切序列时在换局处断开），整段走 cuDNN LSTM（与 LSTMCell 共用同一组参数）。"""
    lc = net.lstm
    w = [lc.weight_ih, lc.weight_hh, lc.bias_ih, lc.bias_hh]
    hx = (h0[None], c0[None])
    if torch.is_autocast_enabled():
        w = [t.half() for t in w]
        xs, hx = xs.half(), (hx[0].half(), hx[1].half())
    return torch._VF.lstm(xs, hx, w, True, 1, 0.0, torch.is_grad_enabled(), False, True)[0]


def seq_forward(net, o, h0, c0, L, value_only=False):
    """o：拍平的 [M·L, ...]；h0/c0：[M, H]（序列开头之前的 LSTM 状态）。"""
    x, f16, skip, hand, abr = net.encode(o)
    M = h0.shape[0]
    Hs = _lstm_seq(net, x.view(M, L, -1), h0, c0).reshape(M * L, -1)
    if value_only:
        return None, None, Hs
    card = net.card_logits(Hs, hand, abr, o['card_mask'])
    return dict(h=Hs, f16=f16, skip=skip, hand=hand), card, Hs


def make_chunks(done, L, valid=None):
    """按数据流切序列：在换局处断开，只保留这一流由当前策略控制的段（valid；同一局内不变），每段再切成 ≤ L 步。
    → (流, 起点, 长度) 数组。"""
    T, S = done.shape
    out = []
    for w in range(S):
        ends = np.flatnonzero(done[:, w])
        starts = np.concatenate([[0], ends + 1])
        stops = np.concatenate([ends + 1, [T]])
        for a, b in zip(starts, stops):
            if a >= T or (valid is not None and not valid[a, w]):
                continue
            for t0 in range(a, b, L):
                out.append((w, t0, min(L, b - t0)))
    return np.array(out, np.int64).reshape(-1, 3)


class Learner:
    def __init__(self, cfg, run, resume, init=None):
        self.cfg, self.pc, self.run = cfg, cfg.ppo, run
        self.dev = torch.device('cuda')
        pc = self.pc
        self.bc_every = bc_ref_every(cfg)                              # 采样进程每几步算一次 BC 参考分布（只有这些行能算 kl_bc）
        ab_floor_eps(0, pc)                                            # 技能概率下限的配置不对时现在就报错（采样端按步数算 ε，学习端不用）
        self.net = Net(cfg).to(self.dev)
        self.keys = rl_keys(cfg) + ['pl']                              # rollout 槽里要搬上显卡的观测字段（含可选的特权信息）
        # 起点：默认是 bc_ckpt（单步 BC，循环部分没训过 → init_recurrence）；ppo.init_ckpt 可指定带记忆的 BC（train/bc_seq.py，recurrent=True），
        # 那样直接用它的 LSTM。bc_ckpt 仍是 KL 约束的参考策略和 BC 陪练对手（零状态推理）。
        bst = torch.load(repo_path(pc.get('init_ckpt') or pc.bc_ckpt), map_location=self.dev, weights_only=False)
        miss = load_weights(self.net, bst['model'], grow=True)        # 扩过容（model/grow.py）时 BC 存档是小结构；续训的 ppo_latest 必须已是新结构（tools/grow.py，带优化器状态）
        if [k for k in miss if not k.startswith('ext_')]:
            print('起点存档里没有特权价值头（%d 个张量），从头学（critic 热身）' % len([k for k in miss if not k.startswith('ext_')]), flush=True)
        if not bst.get('recurrent'):
            init_recurrence(self.net, pc.forget_bias)
        self.net.eval()                                        # BN 用 BC 的统计量（采样和训练行为一致），参数照常训练
        self.net.to(memory_format=torch.channels_last)
        if pc.get('freeze_unit_emb', True):
            # 单位/法术嵌入沿用 BC：它们的反向（几百万次查表、大量落在 0 号空格上）占编码反向的约 1/3
            self.net.unit_emb.weight.requires_grad_(False)
            self.net.spell_emb.weight.requires_grad_(False)
        self.opt = torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad], lr=pc.lr, weight_decay=0.0, eps=1e-5)
        self.opt_v = torch.optim.AdamW(self.net.value_params(), lr=pc.critic_warmup_lr, eps=1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
        self.step, self.update = 0, 0
        ck = os.path.join(run, 'ppo_latest.pt')
        if not (resume and os.path.exists(ck)) and init:
            ck, resume = repo_path(init), True                        # 新的一次训练从别的存档接着练（例如 M4 从 M3 的终点开始）
        if resume and os.path.exists(ck):
            st = torch.load(ck, map_location=self.dev, weights_only=False)
            miss = load_weights(self.net, st['model'])
            ext_new = [k for k in miss if k.startswith('ext_')]
            if len(ext_new) < len(miss):
                print('存档里没有特权价值头：价值头从头学，优化器从头开始（参数组对不上）', flush=True)
                st.pop('opt', None)
            elif ext_new and 'opt' in st:
                # 刚打开扩展观测（model.obs_ext）：新输入层从 0 开始（函数不变），它们在优化器里排最后 → 旧 AdamW 状态原样接上
                named = [(n, p) for n, p in self.net.named_parameters() if p.requires_grad]
                newp = [(n, tuple(p.shape)) for n, p in named if n.startswith('ext_')]
                if [n for n, _ in named[-len(newp):]] != [n for n, _ in newp]:
                    raise RuntimeError('扩展观测的参数没有排在优化器最后，旧的优化器状态接不上')
                st['opt'] = extend_adam(st['opt'], [s for _, s in newp])
                print('存档里没有扩展观测的输入层（%d 个张量）：从 0 开始，优化器状态接上' % len(ext_new), flush=True)
            if 'opt' in st:
                self.opt.load_state_dict(st['opt'])
                self.opt_v.load_state_dict(st['opt_v'])
                self.scaler.load_state_dict(st['scaler'])
            else:
                print('存档里没有优化器状态（只有模型的快照），优化器从头开始', flush=True)
            self.step, self.update = st['step'], st['update']
            print('续训：step %d update %d' % (self.step, self.update), flush=True)
        flat, card_cls = MK.all_masks()
        self.flat = torch.as_tensor(flat, device=self.dev)
        self.card_cls = torch.as_tensor(card_cls, device=self.dev).long()

    # ---------------- 调度 ----------------
    def lr(self):
        pc = self.pc
        return pc.lr * max(pc.lr_floor, 1.0 - self.step / pc.lr_decay_steps)

    def beta(self):
        pc = self.pc
        f = min(1.0, self.step / max(1, pc.kl_bc_anneal_steps))
        return pc.kl_bc_start + (pc.kl_bc_end - pc.kl_bc_start) * f

    def save(self, path, full=True):
        d = dict(model=self.net.state_dict(), cfg=dict(self.cfg), step=self.step, update=self.update, kind='ppo')
        if full:
            d.update(opt=self.opt.state_dict(), opt_v=self.opt_v.state_dict(), scaler=self.scaler.state_dict())
        torch.save(d, path + '.tmp')
        os.replace(path + '.tmp', path)

    # ---------------- 一次更新 ----------------
    def train_on(self, A):
        pc, dev = self.pc, self.dev
        T, W, P = A['rew'].shape
        S = W * P                                                  # 数据流 = 环境 × 玩家（流号 w·2 + p）
        L = pc.seq_len
        rs = lambda x: x.reshape((T, S) + x.shape[3:])               # noqa: E731
        vmask = rs(A['valid'])
        lw = rs(A['lw']).astype(np.float32)
        done = np.repeat(A['done'], P, axis=1)
        last_v = np.where(A['last_valid'], A['last_value'], 0.0).reshape(S).astype(np.float32)
        val = rs(A['value'])
        trunc = np.repeat(A['trunc'], P, axis=1).astype(np.float32) if 'trunc' in A else None
        adv, ret = policy_targets(rs(A['rew']), val, done, last_v, pc.gamma, pc.gae_lambda, pc.get('policy_lambda'), trunc)
        adv = np.where(vmask, adv, 0.0).astype(np.float32)
        D = {k: torch.from_numpy(rs(A['obs_' + k])).to(dev, non_blocking=True) for k in self.keys}
        for k in ('sel', 'cell', 'logp', 'h_in', 'c_in', 'bc_card_lp', 'bc_place_lp'):
            D[k] = torch.from_numpy(rs(A[k])).to(dev, non_blocking=True)
        D['lw'] = torch.from_numpy(lw).to(dev)
        D['kref'] = torch.from_numpy(np.ascontiguousarray(np.broadcast_to(
            (np.arange(T) % self.bc_every == 0).astype(np.float32)[:, None], (T, S)))).to(dev)   # 有 BC 参考分布的步（bc_ref_every = 1 时全是）
        D['ret'] = torch.from_numpy(ret.astype(np.float32)).to(dev)
        av = adv[vmask]
        D['adv'] = torch.from_numpy((adv - av.mean()) / (av.std() + 1e-8)).to(dev)
        warm = critic_warm(self.step, pc)
        # 影子 λ（2026-10-28，只读梯度、不影响训练）：每 noise_lambda_every 次更新，在第一轮每个小批的第一个微批上，
        # 对按各个 λ 算的优势分别求“只有策略梯度项”的梯度 g → 单样本梯度方差 tr(Σ) ≈ b·|g|²（b ≪ B_noise，|G|² 项可忽略）。
        # 期望梯度（信号）在价值头准确时与 λ 无关，所以各 λ 的 tr(Σ) 之比 ≈ 临界批量之比。另每次更新记各 λ 优势的方差（同一尺度）。
        lams = [float(x) for x in (pc.get('noise_lambdas') or [])] if not warm else []
        shadow = bool(lams) and self.update % max(1, int(pc.get('noise_lambda_every', 4))) == 0
        sh_logs, sh_sq = {}, {lam: [] for lam in lams}
        if lams:
            sa = shadow_advantages(rs(A['rew']), val, done, last_v, pc.gamma, lams, trunc, vmask, adv)
            for lam, a in sa.items():
                sh_logs['adv_var_lam%g' % lam] = float(a[vmask].var())
                if shadow:
                    D['adv_lam%g' % lam] = torch.from_numpy(a).to(dev)
        chunks = make_chunks(done, L, vmask)
        csum = np.cumsum(np.concatenate([np.zeros((1, S), np.float32), lw]), 0)
        cw = csum[chunks[:, 1] + chunks[:, 2], chunks[:, 0]] - csum[chunks[:, 1], chunks[:, 0]]   # 每段的损失权重之和
        ar_L = torch.arange(L, device=dev)
        ch = torch.from_numpy(chunks).to(dev)
        logs = {}
        n_mb = 0
        rng = np.random.RandomState(self.update)
        # 梯度噪声尺度（2026-10-27，只读梯度、不影响训练）：第一轮里每个小批的梯度 g_i（反缩放后、截断前）和按大小加权的总和
        gns = pc.get('noise_scale', True) and not warm
        gn_sum, gn_sq, gn_b = None, [], []
        for _ep in range(pc.epochs):
            perm = rng.permutation(len(chunks))
            for mb in np.array_split(perm, pc.minibatches):
                opt = self.opt_v if warm else self.opt
                for g_ in opt.param_groups:
                    g_['lr'] = pc.critic_warmup_lr if warm else self.lr()
                # 清全网的梯度、只按这个优化器自己的参数截断：critic 热身时特权价值头经共享的卡牌嵌入（card_emb/form_emb）
                # 也会给它们算梯度，它们不在 opt_v 里——不清、不反缩放，算进范数后把价值头的梯度压掉两个数量级
                # （2026-10-27 续训热身 31 批 grad_norm ≈ 80、vf 一点没降）
                self.net.zero_grad(set_to_none=True)
                own = [p for g_ in opt.param_groups for p in g_['params']]
                n_valid_mb = float(cw[mb].sum())
                for j_mic, mic in enumerate(np.array_split(mb, pc.micro_batches)):
                    if len(mic) == 0:
                        continue
                    c3 = ch[torch.from_numpy(mic).to(dev)]
                    ww, t0, ln = c3[:, 0], c3[:, 1], c3[:, 2]
                    valid = ar_L[None] < ln[:, None]                     # [M, L]
                    tt = t0[:, None] + torch.minimum(ar_L[None], ln[:, None] - 1)   # 补齐的步重复最后一个有效步（因果，不影响有效步）
                    wl = ww[:, None].expand_as(tt)
                    o = {k: D[k][tt, wl].reshape((-1,) + D[k].shape[2:]) for k in self.keys}
                    h0 = D['h_in'][t0, ww].float()
                    c0 = D['c_in'][t0, ww].float()
                    g = lambda k: D[k][tt, wl].reshape((-1,) + D[k].shape[2:])   # noqa: E731
                    m = valid.reshape(-1).float() * g('lw')                   # 有效步 × 损失权重（自我对弈时对手一方的权重较小）
                    part = float(m.sum()) / max(n_valid_mb, 1e-6)          # 小批内按权重加权
                    sk = ['adv_lam%g' % lam for lam in lams] if shadow and not warm and _ep == 0 and j_mic == 0 else None
                    with torch.autocast('cuda', dtype=torch.float16):
                        if warm:
                            with torch.no_grad():
                                _o, _c, Hs = seq_forward(self.net, o, h0, c0, L, value_only=True)
                            v = self.net.value_from(Hs.detach(), o).float()
                            vf = 0.5 * (((v - g('ret')) ** 2) * m).sum() / m.sum()
                            loss = vf
                            parts = dict(vf=vf.item())
                        else:
                            loss, parts = self._ppo_loss(o, h0, c0, L, g, m, shadow=sk)
                    if sk:
                        self._shadow_grads(parts.pop('_shadow'), lams, float(m.sum()), sh_sq)
                    self.scaler.scale(loss * part).backward()
                    for k, v in parts.items():
                        logs[k] = logs.get(k, 0.0) + v * part
                self.scaler.unscale_(opt)
                if gns and _ep == 0:
                    with torch.no_grad():
                        gv = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
                                        for p in self.net.parameters() if p.requires_grad])
                        if bool(torch.isfinite(gv).all()):                 # 这一步溢出（scaler 会跳过）就不记
                            gn_sq.append(float(gv.dot(gv)))
                            gn_b.append(n_valid_mb)
                            gn_sum = gv.mul_(n_valid_mb) if gn_sum is None else gn_sum.add_(gv, alpha=n_valid_mb)
                gn = torch.nn.utils.clip_grad_norm_(own, pc.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
                logs['grad_norm'] = logs.get('grad_norm', 0.0) + float(gn)
                n_mb += 1
        logs['n_chunks'] = float(len(chunks))
        logs['pad_frac'] = 1.0 - float(chunks[:, 2].sum()) / (len(chunks) * L)
        logs = {k: v / max(1, n_mb) for k, v in logs.items()}
        logs.update(sh_logs)
        for lam, xs in sh_sq.items():
            if xs:
                logs['gns_trS_lam%g' % lam] = float(np.mean(xs))
        if len(gn_sq) >= 2:
            B = float(sum(gn_b))
            gb = gn_sum.div_(B)
            logs.update(noise_scale_parts(gn_sq, gn_b, float(gb.dot(gb))))
        logs['warmup'] = float(warm)
        ev = 1 - np.var((ret - val)[vmask]) / (np.var(ret[vmask]) + 1e-8)
        logs['explained_var'] = float(ev)
        return logs

    def _shadow_grads(self, losses, lams, b, sh_sq):
        """影子 λ：各个“只有策略梯度项”的损失分别求梯度（保留计算图，训练的反传照常），记 b·|g|²。
        乘上 GradScaler 当前的缩放再求导（半精度激活的梯度不下溢），算完除回去；有溢出的就不记。"""
        params = [p for p in self.net.parameters() if p.requires_grad]
        scale = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
        for lam, pl in zip(lams, losses):
            gs = torch.autograd.grad(pl * scale, params, retain_graph=True, allow_unused=True)
            with torch.no_grad():
                sq = sum(float((x.float() / scale).pow(2).sum()) for x in gs if x is not None)
            if math.isfinite(sq):
                sh_sq[lam].append(b * sq)

    def _ppo_loss(self, o, h0, c0, L, g, m, shadow=None):
        """m：[M·L] 有效步掩码（补齐的步为 0）。各项都是“每个有效步”的平均。
        shadow：影子 λ 的优势键名列表——另外返回同样截断形式、只换了优势的策略梯度损失（parts['_shadow']，张量）。"""
        pc, net = self.pc, self.net
        out, card, Hs = seq_forward(net, o, h0, c0, L)
        sel = g('sel').long()
        cell = g('cell').long()
        is_hand = (sel >= 1) & (sel <= 4) & (m > 0)
        nv = m.sum()
        kr = g('kref') * m                                       # KL 项只在有 BC 参考分布的步上算（bc_ref_every = 1 时 kr = m，和以前一样）
        nk = kr.sum().clamp(min=1e-6)
        value = net.value_from(Hs, o).float()
        N = sel.shape[0]
        ar = torch.arange(N, device=sel.device)
        hi = is_hand.nonzero().squeeze(1)                      # 只在出了手牌的步上算位置头（约 5% 的步）
        K = len(hi)
        if K:
            sub = dict(h=out['h'][hi], f16=out['f16'][hi], skip=out['skip'][hi], hand=out['hand'][hi])
            place = net.place_logits(sub, sel[hi])
        with torch.autocast('cuda', enabled=False):
            pol = net.policy_logits(card.float())
            cm = o['card_mask']
            lp_c = torch.log_softmax(pol, 1)
            cell_lp = torch.zeros(N, device=sel.device)
            ent_p = kl_p = torch.zeros((), device=sel.device)
            if K:
                ak = torch.arange(K, device=sel.device)
                sk, ck = sel[hi], cell[hi].clamp(0)
                hid = o['hand_id'].long()[hi, sk - 1]
                pm = self.flat[self.card_cls[hid], o['pl'].long()[hi], o['tstate'].long()[hi]]
                pm[ak, ck] = True                                 # 保险：实际下过的格子一定在 mask 里
                lp_p = torch.log_softmax(place.float().masked_fill(~pm, -1e9), 1)
                cell_lp = cell_lp.index_put((hi,), lp_p[ak, ck])
                p_p = lp_p.exp()
                ent_p = -torch.where(pm, p_p * lp_p, torch.zeros_like(p_p)).sum() / nv
                bc_p = g('bc_place_lp')[hi].float()
                kl_p = (torch.where(pm, p_p * (lp_p - bc_p), torch.zeros_like(p_p)).sum(1) * g('kref')[hi]).sum() / nk
            new_logp = lp_c[ar, sel] + cell_lp
            old_logp = g('logp')
            # 不算损失的行（m = 0：补齐的步、这一批没写到的旧数据）先置 0：它们的 logp/回报可能是残留的任意值，
            # exp 溢出成 inf 后乘 0 会得到 NaN，把整个小批的损失变成 NaN（有效行的结果不受影响）
            valid = m > 0
            lr_ = torch.where(valid, new_logp - old_logp, torch.zeros_like(new_logp))
            ratio = torch.exp(lr_)
            adv = g('adv')
            pg = -(torch.min(ratio * adv, ratio.clamp(1 - pc.clip, 1 + pc.clip) * adv) * m).sum() / nv
            ret = g('ret')
            vf = 0.5 * (torch.where(valid, (value - ret) ** 2, torch.zeros_like(value)) * m).sum() / nv
            p_c = lp_c.exp()
            ent_c = -(torch.where(cm, p_c * lp_c, torch.zeros_like(p_c)).sum(1) * m).sum() / nv
            bc_c = g('bc_card_lp').float()
            kl_c = (torch.where(cm, p_c * (lp_c - bc_c), torch.zeros_like(p_c)).sum(1) * kr).sum() / nk
            beta = self.beta()
            loss = pg + pc.vf_coef * vf - pc.ent_card * ent_c - pc.ent_place * ent_p + beta * (kl_c + kl_p)
            with torch.no_grad():
                approx_kl = (((ratio - 1) - lr_) * m).sum() / nv
                clipfrac = ((((ratio - 1).abs() > pc.clip).float()) * m).sum() / nv
                rdy = (cm[:, 5:].any(1) & (m > 0)).float()               # 有技能可以按的步：策略本身（不含采样端的技能概率下限）按技能的概率
                ab_p = (p_c[:, 5:].sum(1) * rdy).sum() / rdy.sum().clamp(min=1)
            sh = [-(torch.min(ratio * g(k), ratio.clamp(1 - pc.clip, 1 + pc.clip) * g(k)) * m).sum() / nv for k in (shadow or [])]
        parts = dict(pg=pg.item(), vf=vf.item(), ent_card=ent_c.item(), ent_place=float(ent_p), kl_bc_card=kl_c.item(),
                     kl_bc_place=float(kl_p), approx_kl=approx_kl.item(), clipfrac=clipfrac.item(), ab_p=ab_p.item())
        if shadow:
            parts['_shadow'] = sh
        return loss, parts


def extend_adam(osd, shapes_new):
    """AdamW 的 state_dict 后面接上新参数（形状 shapes_new，按优化器里的顺序排在最后）：一阶矩 0，二阶矩取旧状态的中位数
    （取 0 的话新参数第一步会迈出远大于学习率的步子，同 model/grow.py: grow_adam）。"""
    pg = dict(osd['param_groups'][0])
    n_old = len(pg['params'])
    st = dict(osd['state'])
    if st:
        allv = torch.cat([s['exp_avg_sq'].flatten().float().cpu() for s in st.values()])
        med = float(allv[allv > 0].median()) if (allv > 0).any() else 1e-12
        step = next(iter(st.values()))['step']
        for j, shp in enumerate(shapes_new):
            st[n_old + j] = dict(step=step.clone(), exp_avg=torch.zeros(shp), exp_avg_sq=torch.full(shp, med))
    pg['params'] = list(range(n_old + len(shapes_new)))
    return dict(state=st, param_groups=[pg] + list(osd['param_groups'][1:]))


def ep_summary(eps):
    out = dict(n_episodes=len(eps))
    if not eps:
        return out
    by = {}
    for e in eps:
        by.setdefault(e['opp'], []).append(e)
    for opp, es in by.items():
        r = [e['result'] for e in es if e['result'] is not None]
        out['win_' + opp] = float(np.mean(r)) if r else None
        out['n_' + opp] = len(es)
    for name, sel in (('main', lambda e: e.get('deck') == 'main'), ('pool', lambda e: e.get('deck') not in (None, 'main'))):
        r = [e['result'] for e in eps if e['opp'] == 'bc_pool' and sel(e) and e['result'] is not None]
        if r:
            out['win_bc_pool_by_%s_deck' % name] = float(np.mean(r))
    allr = [e['result'] for e in eps if e['result'] is not None]
    out['win_all'] = float(np.mean(allr)) if allr else None
    out['ep_return'] = float(np.mean([e['ret'] for e in eps]))
    out['ep_len'] = float(np.mean([e['len'] for e in eps]))
    out['crowns_for'] = float(np.mean([e['crowns'][0] for e in eps]))
    out['crowns_against'] = float(np.mean([e['crowns'][1] for e in eps]))
    # 英雄/冠军技能（2026-10-28）：每局发动次数、每局“有技能可以按”的决策步数、按的比例（发动次数 / 可以按的步数）
    ab = [e for e in eps if e.get('ab_ready') is not None]
    if ab:
        out['ab_per_game'] = float(np.mean([e['ab'] for e in ab]))
        out['ab_ready_per_game'] = float(np.mean([e['ab_ready'] for e in ab]))
        n_ready = sum(e['ab_ready'] for e in ab)
        out['ab_use_rate'] = float(sum(e['ab'] for e in ab) / n_ready) if n_ready else None
    # 按技能单位分开（2026-10-30）：ab_units = {名字: [方数（这局有过可以按的时候）, 其中按过的方数, 发动次数, 可以按的步数]}；
    # ab_hero_used / ab_champ_used：英雄（单位名带 Hero）/ 其余（冠军为主，有冷却、能反复按；带技能的进化单位也算这边）按过的方占比，人类回放里英雄 50–80%
    units = {}
    for e in eps:
        for nm, (r, u) in (e.get('ab_units') or {}).items():
            s = units.setdefault(nm, [0, 0, 0, 0])
            s[0] += 1
            s[1] += int(u > 0)
            s[2] += u
            s[3] += r
    if units:
        out['ab_units'] = units
        for key, pick in (('ab_hero_used', lambda nm: 'Hero' in nm), ('ab_champ_used', lambda nm: 'Hero' not in nm)):
            n = sum(s[0] for nm, s in units.items() if pick(nm))
            out[key] = float(sum(s[1] for nm, s in units.items() if pick(nm)) / n) if n else None
    return out


def launch_eval(cfg, run, ckpt, step):
    pc = cfg.ppo
    py = sys.executable
    log = open(os.path.join(run, 'eval_%d.log' % step), 'w', encoding='utf-8')
    cmd = [py, os.path.join(ROOT, 'tools', 'lowprio.py'), py, '-u', '-m', 'cr_ai.eval.arena', '--ckpt', ckpt, '--suite', pc.get('eval_suite', 'rule,bc_main,bc_pool'),
           '--bc-ckpt', repo_path(pc.bc_ckpt), '--games', str(pc.eval_games), '--workers', str(pc.eval_workers), '--stateful',
           '--json', os.path.join(run, 'arena.jsonl'), '--step', str(step)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def launch_gauntlet(cfg, run, ckpt, step):
    """固定考官评测（tools/gauntlet.py）：结果追加到 runs/<run>/gauntlet.jsonl，拟合结果写 gauntlet_elo.json。"""
    pc = cfg.ppo
    py = sys.executable
    log = open(os.path.join(run, 'gauntlet_%d.log' % step), 'w', encoding='utf-8')
    cmd = [py, os.path.join(ROOT, 'tools', 'lowprio.py'), py, '-u', '-m', 'cr_ai.tools.gauntlet', 'play', '--ckpt', ckpt,
           '--examiners', pc.get('gauntlet_examiners', 'cr_ai/configs/examiners.yaml'), '--games', str(pc.get('gauntlet_games', 200)),
           '--workers', str(pc.get('gauntlet_workers', pc.eval_workers)), '--out', os.path.join(run, 'gauntlet.jsonl'), '--step', str(step)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', nargs='+', default=['cr_ai/configs/ppo.yaml'], help='可给多个，依次叠加')
    ap.add_argument('--run', default='runs/ppo1')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--init', default=None, help='没有 ppo_latest.pt 时从这个存档开始（模型 + 优化器 + 步数）')
    ap.add_argument('--updates', type=int, default=None, help='只跑 N 次更新（试跑/测吞吐）')
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    limit_torch_threads()
    cfg = load(*a.config, overrides=a.overrides)
    pc = cfg.ppo
    run = repo_path(a.run)
    os.makedirs(run, exist_ok=True)
    dump(cfg, os.path.join(run, 'config.yaml'))
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    L = Learner(cfg, run, a.resume, a.init)
    W, T = pc.workers, pc.steps_per_env
    assert T % pc.seq_len == 0, 'steps_per_env 必须是 seq_len 的整数倍'
    spec = slot_spec(cfg, T, W, L.net.H)
    NA = max(1, int(pc.get('actors', 1)))                    # 采样进程数：各带 W 个环境、各有 2 个 rollout 槽，交来的批轮流训练
    print('参数量 %.2fM；每批 %d 步（%d 环境 × %d），采样进程 %d 个，rollout 槽 %.0f MB × %d' % (
        n_params(L.net) / 1e6, T * W, W, T, NA, slot_bytes(spec) / 2 ** 20, 2 * NA), flush=True)
    with open(os.path.join(run, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(kind='ppo', total_steps=int(pc.total_steps), steps_per_update=T * W), f)
    ctx = mp.get_context('spawn')
    slots = [Slot(spec) for _ in range(2 * NA)]              # 第 k 个采样进程用 slots[2k]、slots[2k+1]
    n_w = sum(t.numel() for t in WeightBox.tensors(L.net))
    lock = ctx.Lock()
    wbox = WeightBox(n_w, lock=lock)
    wbox.write(L.net)
    Q = {}
    if cfg.league.mode == 'league':
        league = League(cfg, run, init_elo=cfg.league.get('init_elo', 0.0), seed=cfg.seed + L.update)
    elif cfg.league.mode == 'exploit':
        league = ExploitTarget(cfg, run)                     # 剥削者：对手固定为主模型的冻结副本
    else:
        league = None
    n_picks = max(1, int(cfg.league.get('hist_slots', 4)))

    def send_hist(k):
        # 采样进程在取空槽之后读控制消息，所以要先发消息、再放回空槽。
        # 每批按最新的 PFSP 权重抽一组候选（覆盖采样进程里没用完的旧候选）；采样进程只在某个槽手上的对局都打完后才换上候选（league.py: HistSlots）
        if league is not None:
            picks = [h for h in (league.pick() for _ in range(n_picks)) if h is not None]
            if picks:
                Q['ctl'][k].put(('hist', picks))
            dp = league.deck_probs()                              # 卡组 PFSP（league.deck_pfsp > 0）：对手卡组的抽样概率
            if dp is not None:
                Q['ctl'][k].put(('decks', dp))

    def start_actors():
        # 每次（重新）启动都用新的队列：崩掉的进程可能留下半条消息。q_ready 大家共用，q_free / q_ctl 每个采样进程一个
        Q['ready'] = ctx.Queue()
        Q['free'] = [ctx.Queue() for _ in range(NA)]
        Q['ctl'] = [ctx.Queue() for _ in range(NA)]
        ps = []
        for k in range(NA):
            send_hist(k)
            for i in range(2):
                Q['free'][k].put(i)
            p = ctx.Process(target=actor_main, args=(cfg, [slots[2 * k].names, slots[2 * k + 1].names], spec, wbox.shm.name, lock, n_w,
                                                     Q['ready'], Q['free'][k], Q['ctl'][k], L.step, k),
                            daemon=False)                # 采样进程自己还要开环境进程（daemon 进程不能有子进程）
            p.start()
            ps.append(p)
        return ps

    def stop_actors(ps, wait=60):
        for k in range(len(ps)):
            try:
                Q['ctl'][k].put(('stop',))
                Q['free'][k].put('stop')
            except Exception:                                   # noqa
                pass
        for p in ps:
            p.join(timeout=wait)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)

    actors = start_actors()
    n_restart = 0
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run, 'tb'))
    flog = open(os.path.join(run, 'train_log.jsonl'), 'a', encoding='utf-8')
    eval_proc = None
    next_eval = (L.step // pc.eval_every_steps + (1 if L.step > 0 else 0)) * pc.eval_every_steps   # 续训：下一个整点再评（上一次的评测可能还在跑）
    next_snap = (L.step // pc.snapshot_every_steps + 1) * pc.snapshot_every_steps
    g_every = int(pc.get('gauntlet_every_steps', 0) or 0)                 # 0 = 不跑固定考官评测
    gaunt_proc = None
    next_gaunt = (L.step // g_every + 1) * g_every if g_every else None
    if not a.no_eval and L.step == 0:
        p0 = os.path.join(run, 'ppo_step0.pt')
        L.save(p0, full=False)
        eval_proc = launch_eval(cfg, run, p0, 0)
        next_eval = pc.eval_every_steps
    t_last = time.time()
    n_up = 0
    try:
        while L.step < pc.total_steps and (a.updates is None or n_up < a.updates):
            w0 = time.time()
            try:
                m = Q['ready'].get(timeout=pc.get('ready_timeout', 600))
            except queue.Empty:
                # 采样进程卡死（例如环境进程启动失败、一直等不到）：按崩溃处理
                m = ('fatal', '采样进程 %d 秒没有送来数据（卡死）' % pc.get('ready_timeout', 600))
            if m[0] == 'fatal':
                # 采样进程崩了（例如偶发的 CUDA 非法访问：CUDA 上下文坏了只能换进程）：记下来、重启，最多 max_actor_restarts 次
                with open(os.path.join(run, 'actor_crash.log'), 'a', encoding='utf-8') as f:
                    f.write('==== %s step %d\n%s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), L.step, m[1]))
                n_restart += 1
                if n_restart > pc.get('max_actor_restarts', 5):
                    raise RuntimeError('采样进程崩溃次数过多：\n' + m[1])
                print('采样进程崩溃（第 %d 次），全部重启。详情见 actor_crash.log' % n_restart, flush=True)
                stop_actors(actors, wait=30)                     # 其余采样进程也停掉重开：旧队列里可能还有引用这些槽的消息
                time.sleep(pc.get('actor_restart_wait', 20))     # 等显卡从驱动错误里恢复
                actors = start_actors()
                t_last = time.time()
                continue
            si, st = m
            k = int(st.get('aid', 0))
            wait = time.time() - w0
            t1 = time.time()
            logs = L.train_on(slots[2 * k + si].a)
            torch.cuda.synchronize()
            train_secs = time.time() - t1
            if league is not None:
                league.update(st['episodes'])
                if L.update % 20 == 0:
                    n_exp = league.scan_exploiters()              # 别用 k：k 是交来这批的采样进程号，下面还槽要用
                    if n_exp:
                        print('联赛：加入 %d 个剥削者快照' % n_exp, flush=True)
            send_hist(k)
            Q['free'][k].put(si)
            L.step += st['steps']
            L.update += 1
            n_up += 1
            wbox.write(L.net)
            now = time.time()
            rec = dict(step=L.step, update=L.update, sps=st['steps'] / max(1e-6, now - t_last), actor_sps=st['steps'] / max(1e-6, st['secs']),
                       actor_env_wait=st['env_wait'] / max(1e-6, st['secs']), actor_gpu=st['gpu_time'] / max(1e-6, st['secs']),
                       learner_train_s=train_secs, learner_wait_s=wait, actor_id=k, lr=L.lr(), beta=L.beta(), shaping_c=st['coef'], ab_eps=st.get('ab_eps'),
                       weights_lag=n_up - st['ver'],
                       env_errors=st['errors'], env_steps=st.get('env_steps'), env_busy=st.get('env_busy'), env_bot=st.get('env_bot'),
                       cg_cores=st.get('cg_cores'), cg_throttled=st.get('cg_throttled'), hist_id=st.get('hist_id'), hist_ids=st.get('hist_ids'), t=now, **logs, **ep_summary(st['episodes']),
                       **(league.summary() if league is not None else {}))
            t_last = now
            flog.write(json.dumps(rec) + '\n')
            flog.flush()
            for k, v in rec.items():
                if isinstance(v, (int, float)) and v is not None and k not in ('step', 't'):
                    tb.add_scalar(k, v, L.step)
            print('[%d] %.1fM 步  %.0f 步/秒（采样 %.0f，等环境 %.0f%%，GPU %.0f%%；训练 %.1fs，等数据 %.1fs）%s 胜率 %s  pg %.3f vf %.3f klBC %.3f' % (
                L.update, L.step / 1e6, rec['sps'], rec['actor_sps'], 100 * rec['actor_env_wait'], 100 * rec['actor_gpu'], train_secs, wait,
                '热身' if logs.get('warmup') else '', None if rec.get('win_all') is None else round(rec['win_all'], 3),
                logs.get('pg', 0), logs.get('vf', 0), logs.get('kl_bc_card', 0) + logs.get('kl_bc_place', 0))
                + ('  Elo %.0f' % league.elo if league is not None else '')
                + ('  环境忙 %.0f%%（bot %.0f%%）' % (100 * rec['env_busy'], 100 * rec['env_bot']) if rec.get('env_busy') is not None else '')
                + ('  容器 %.1f 核、限流 %.0f%%' % (rec['cg_cores'], 100 * rec['cg_throttled']) if rec.get('cg_cores') is not None else ''), flush=True)
            if L.update % pc.ckpt_every_updates == 0:
                L.save(os.path.join(run, 'ppo_latest.pt'))
            if L.step >= next_snap:
                sp = os.path.join(run, 'ppo_step%d.pt' % L.step)
                L.save(sp, full=False)
                if league is not None:
                    league.add(sp, L.step)
                next_snap += pc.snapshot_every_steps
            if not a.no_eval and L.step >= next_eval and (eval_proc is None or eval_proc.poll() is not None):
                p = os.path.join(run, 'ppo_step%d.pt' % L.step)
                if not os.path.exists(p):
                    L.save(p, full=False)
                eval_proc = launch_eval(cfg, run, p, L.step)
                next_eval = (L.step // pc.eval_every_steps + 1) * pc.eval_every_steps
            # 固定考官评测：和普通评测错开（两者都在跑会和训练抢太多显卡/CPU），普通评测在跑就等下一次更新再看
            if (not a.no_eval and g_every and L.step >= next_gaunt and (gaunt_proc is None or gaunt_proc.poll() is not None)
                    and (eval_proc is None or eval_proc.poll() is not None)):
                p = os.path.join(run, 'ppo_step%d.pt' % L.step)
                if not os.path.exists(p):
                    L.save(p, full=False)
                gaunt_proc = launch_gauntlet(cfg, run, p, L.step)
                next_gaunt = (L.step // g_every + 1) * g_every
    finally:
        # 显卡出错（驱动重置等）时保存本身也可能失败：不保存（沿用上一个存档），但一定要停掉采样进程，
        # 否则它不是 daemon 进程，解释器退出时会一直等它，守护脚本就等不到退出
        try:
            L.save(os.path.join(run, 'ppo_latest.pt'))
            print('已保存 ppo_latest.pt（step %d）' % L.step, flush=True)
        except Exception as e:                                  # noqa
            print('保存失败（沿用上一个存档）：%s' % str(e)[:200], flush=True)
        try:
            stop_actors(actors)
        finally:
            for s in slots:
                s.close(unlink=True)
            wbox.shm.close()
            wbox.shm.unlink()


if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    main()
