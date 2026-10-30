"""cr_ai/train/distill.py — 把训好的模型（老师）蒸馏进更大的学生网络（2026-10-28，方案 A，见 reports/皇室战争训练停滞原因.md）。

    python -m cr_ai.train.distill --teacher runs/ppo_v2_m4/ppo_stepN.pt --run runs/distill_a [--updates N] [覆盖 a.b=值 ...]

学生结构 = 老师存档里的配置 ← configs/distill.yaml（model 段：LSTM 1536、LayerNorm、各个头加宽）← 命令行覆盖。
卷积主干、嵌入、实体分支、输入 MLP、特权信息编码、位置卷积从老师原样复制（COPY_MODULES）；融合层、LSTM、各个头、FiLM 随机初始化后蒸馏。
流程（单进程，采样和训练轮流做）：
  1. 采样：VecEnv 自我对弈（卡组抽样同训练：主角卡组概率 league.agent_main_prob，其余从卡组池抽）。每局开局时每名玩家各抽一次
     由老师还是学生出手（老师出手的概率 teacher_act_start → teacher_act_end，DAgger：后期主要在学生自己会走到的局面上学）。
     每一步老师都打分（带自己的 LSTM 状态）：出牌头的完整分布、实际所选手牌上的位置分布、局面价值；学生的 LSTM 状态也一起推进
     （训练时序列开头的状态）。推理都走 CUDA Graph（model/infer.py: InferEngine）。
  2. 训练：按局切成 ≤ seq_len 的序列，学生前向（和 PPO 同一个 seq_forward），
     损失 = KL(老师‖学生)（出牌头）+ place_coef × KL（位置头，出了手牌的步）+ value_coef × (学生价值 − 老师价值)²。
     前 freeze_copied_updates 次更新只训新的部分；之后复制来的部分用 lr_copied 一起微调。BN 用老师的统计量（eval，和 PPO 一样）。
输出：runs/<run>/distill_latest.pt（续训）、student_step<N>.pt（每 snapshot_every_updates 次更新；arena / 对打直接能用）、train_log.jsonl、config.yaml。
验收：学生对老师直接配对对打（如 runs/tmp/ladder 的做法）≥ 0.47。之后接着做 PPO 时注意（还没做）：采样进程里的 BC 参考网络和历史快照
都按学生的结构建（rollout.py: Net(cfg)），老结构的存档装不进去——bc_ckpt 要换成学生自己（或关掉 KL、bc 对手），联赛的历史快照从学生重新攒。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

from ..config import _merge, _unwrap, _wrap, dump, limit_torch_threads, load, read_yaml, repo_path
from ..env import masks as MK
from ..model.net import Net, load_weights, n_params, rl_keys
from .ppo import make_chunks, seq_forward
from .rollout import WeightBox, slot_spec

# 从老师原样复制的模块（顶层名字）：和 LSTM 宽度无关、学生和老师形状一样
COPY_MODULES = ('unit_emb', 'spell_emb', 'card_emb', 'form_emb', 'ent_uid', 'ent_in', 'ent_blocks', 'ent_norm', 'ent_out',
                'stem', 'down', 'res', 'down2', 'grid_fc', 'extra_mlp', 'hand_mlp', 'ab_mlp', 'opp_mlp', 'deck_mlp', 'deck_fc',
                'priv_mlp', 'skip_proj', 'place_res', 'place_conv', 'noop_bias')


def student_cfg(tcfg, files=('cr_ai/configs/distill.yaml',), overrides=()):
    """默认配置 ← 老师存档里的配置 ← files ← overrides（'a.b=值'，值按 YAML 解析）。"""
    d = _unwrap(load())
    _merge(d, _unwrap(tcfg))
    for f in files:
        _merge(d, read_yaml(f))
    for o in overrides:
        key, val = o.split('=', 1)
        x = d
        ks = key.split('.')
        for k in ks[:-1]:
            x = x.setdefault(k, {})
        x[ks[-1]] = yaml.safe_load(val)
    return _wrap(d)


def copy_shared(student, teacher_sd):
    """把老师 COPY_MODULES 里的张量（参数和 BN 统计量）原样拷进学生，形状对不上就报错。→ (拷了的张量名, 没拷、要蒸馏的张量名)"""
    sd = student.state_dict()
    copied, new = [], []
    for k, v in sd.items():
        if k.split('.')[0] in COPY_MODULES and k in teacher_sd:
            if tuple(teacher_sd[k].shape) != tuple(v.shape):
                raise RuntimeError('复制 %s：学生 %s、老师 %s，形状对不上' % (k, tuple(v.shape), tuple(teacher_sd[k].shape)))
            copied.append(k)
        else:
            new.append(k)
    sd.update({k: teacher_sd[k].to(sd[k].dtype) for k in copied})
    student.load_state_dict(sd)
    return copied, new


def teacher_act_prob(step, dc):
    """老师出手的概率：teacher_act_anneal_steps 步内从 teacher_act_start 线性退到 teacher_act_end。"""
    f = min(1.0, step / max(1, int(dc.teacher_act_anneal_steps)))
    return float(dc.teacher_act_start) + (float(dc.teacher_act_end) - float(dc.teacher_act_start)) * f


def place_mask(o, rows, sel, flat, card_cls):
    """rows 这些行在所选手牌 sel（1–4）上的放置 mask [P, 576]。"""
    s = sel[rows]
    hid = o['hand_id'].long()[rows, s - 1]
    return flat[card_cls[hid], o['pl'].long()[rows], o['tstate'].long()[rows]]


def distill_loss(net, o, h0, c0, L, g, m, flat, card_cls, place_coef, value_coef):
    """g(k)：按序列取出的 [M·L] 字段（sel = 实际出手的选项，t_card / t_place / t_value = 老师的出牌对数分布、所选手牌上的位置对数分布、价值）；
    m：有效步掩码（补齐的步为 0）。出牌项和价值项是有效步上的平均，位置项是出了手牌的有效步上的平均。
    KL 用老师的分布加权（前向 KL，覆盖老师的全部选项）；被 mask 屏蔽的选项不算。"""
    out, card, Hs = seq_forward(net, o, h0, c0, L)
    sel = g('sel').long()
    hi = ((sel >= 1) & (sel <= 4) & (m > 0)).nonzero().squeeze(1)
    place = None
    if len(hi):
        sub = {k: out[k][hi] for k in ('h', 'f16', 'skip', 'hand')}
        place = net.place_logits(sub, sel[hi])
    value = net.value_from(Hs, o)
    with torch.autocast('cuda', enabled=False):
        nv = m.sum().clamp(min=1e-6)
        cm = o['card_mask'].bool()
        lp_s = torch.log_softmax(net.policy_logits(card.float()), 1)
        lp_t = g('t_card').float()
        kl_c = (torch.where(cm, lp_t.exp() * (lp_t - lp_s), torch.zeros_like(lp_s)).sum(1) * m).sum() / nv
        agree = ((lp_s.argmax(1) == lp_t.argmax(1)).float() * m).sum() / nv     # 几乎总是“不出”最可能，这个数天然接近 1
        # 实际出了牌/技能的步（约 5%）上，“出哪张”（不算“不出”）最可能的是不是同一张
        mp = m * (sel >= 1).float()
        agree_play = ((lp_s[:, 1:].argmax(1) == lp_t[:, 1:].argmax(1)).float() * mp).sum() / mp.sum().clamp(min=1e-6)
        kl_p = torch.zeros((), device=lp_s.device)
        if place is not None:
            pm = place_mask(o, hi, sel, flat, card_cls)
            lp_ps = torch.log_softmax(place.float().masked_fill(~pm, -1e9), 1)
            lp_pt = g('t_place')[hi].float()
            kl_p = torch.where(pm, lp_pt.exp() * (lp_pt - lp_ps), torch.zeros_like(lp_ps)).sum(1).mean()
        vmse = (((value.float() - g('t_value').float()) ** 2) * m).sum() / nv
        loss = kl_c + place_coef * kl_p + value_coef * vmse
    return loss, dict(kl_card=kl_c.item(), kl_place=float(kl_p), vmse=vmse.item(), agree=agree.item(), agree_play=agree_play.item(),
                      hand_frac=float(len(hi)) / float(nv))


class Distiller:
    def __init__(self, teacher, run, overrides=()):
        from ..env.vec_env import CTL, VecEnv
        from ..model.infer import InferEngine
        self.dev = torch.device('cuda')
        self.run, self.teacher_path = run, teacher
        os.makedirs(run, exist_ok=True)
        tst = torch.load(repo_path(teacher), map_location='cpu', weights_only=False)
        tcfg = load(overrides=())
        tcfg.update(tst['cfg'])
        self.cfg = cfg = student_cfg(tst['cfg'], overrides=overrides)
        self.dc = dc = cfg.distill
        if dc.steps_per_env % dc.seq_len:
            raise ValueError('distill.steps_per_env 要是 seq_len 的整数倍')
        flat, card_cls = MK.all_masks()
        self.flat = torch.as_tensor(flat, device=self.dev)
        self.card_cls = torch.as_tensor(card_cls, device=self.dev).long()
        # 学生（训练用，float32）
        self.net = Net(cfg).to(self.dev)
        self.step, self.update = 0, 0
        ck = os.path.join(run, 'distill_latest.pt')
        st = torch.load(ck, map_location=self.dev, weights_only=False) if os.path.exists(ck) else None
        if st is None:
            copied, new = copy_shared(self.net, {k: v.to(self.dev) for k, v in tst['model'].items()})
            print('学生 %.2fM 参数（老师 %.2fM）：从老师复制 %d 个张量，蒸馏 %d 个' % (n_params(self.net) / 1e6, sum(v.numel() for v in tst['model'].values()) / 1e6,
                                                                    len(copied), len(new)), flush=True)
        else:
            self.net.load_state_dict(st['model'])
            self.step, self.update = int(st['step']), int(st['update'])
            print('续训：%d 步、%d 次更新' % (self.step, self.update), flush=True)
        self.net.eval()                                                # BN 用老师的统计量（和 PPO 一样）
        self.net.to(memory_format=torch.channels_last)
        for p in (self.net.unit_emb.weight, self.net.spell_emb.weight):
            p.requires_grad_(False)                                    # 单位/法术嵌入沿用老师（PPO 也不训它们）
        named = [(n, p) for n, p in self.net.named_parameters() if p.requires_grad]
        self.copied_params = [p for n, p in named if n.split('.')[0] in COPY_MODULES]
        new_params = [p for n, p in named if n.split('.')[0] not in COPY_MODULES]
        self.opt = torch.optim.AdamW([dict(params=new_params, lr=float(dc.lr)), dict(params=self.copied_params, lr=float(dc.lr_copied))], weight_decay=0.0, eps=1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
        if st is not None:
            self.opt.load_state_dict(st['opt'])
            self.scaler.load_state_dict(st['scaler'])
        dump(cfg, os.path.join(run, 'config.yaml'))
        # 推理：老师 + 学生的半精度副本（每次更新后拷贝新权重，原地拷贝，CUDA Graph 引用的地址不变）
        t_net = Net(tcfg).to(self.dev).eval()
        load_weights(t_net, tst['model'])
        self.s_act = Net(cfg).to(self.dev).eval()
        for m in (t_net, self.s_act):
            m.to(memory_format=torch.channels_last).half()
        self.sync_actor()
        self.t_eng = InferEngine(t_net, flat, card_cls, self.dev)
        self.s_eng = InferEngine(self.s_act, flat, card_cls, self.dev)
        self.keys = rl_keys(cfg) + ['pl']
        if 'ent_id' in self.keys or 'ent_id' in self.t_eng.keys:
            cfg.obs.entities = True
        # 环境：自我对弈，双方都由这里出动作
        W, T = int(dc.workers), int(dc.steps_per_env)
        lc = cfg.league
        spec = dict(controllers=[['agent', 'self', 1]], main_for='agent', opp_main_prob=lc.get('opp_main_prob', 0.0),
                    agent_main_prob=lc.get('agent_main_prob', 1.0), tower_troops=lc.get('tower_troops'))
        cfg.vec_env.workers = W
        self.venv = VecEnv(cfg, n_workers=W, episode_spec=spec, seed=cfg.seed + 7 * self.update)
        self.obs = self.venv.reset()
        self.ctl_ok = (CTL['agent'], CTL['self'])
        self.W, self.T = W, T
        Hs, Ht = self.net.H, t_net.H
        self.hs, self.cs = np.zeros((W, 2, Hs), np.float32), np.zeros((W, 2, Hs), np.float32)
        self.ht, self.ct = np.zeros((W, 2, Ht), np.float32), np.zeros((W, 2, Ht), np.float32)
        self.rng = np.random.RandomState(cfg.seed + self.update)
        self.tact = self.rng.rand(W, 2) < teacher_act_prob(self.step, dc)
        self.buf = {k: np.zeros(shape, dt) for k, (shape, dt) in slot_spec(cfg, T, W, Hs).items()}

    def sync_actor(self):
        with torch.no_grad():
            for ta, tl in zip(WeightBox.tensors(self.s_act), WeightBox.tensors(self.net)):
                ta.copy_(tl)

    # ---------------- 采样 ----------------
    def collect(self):
        venv, B, W, T = self.venv, self.buf, self.W, self.T
        B['valid'][:] = False
        B['done'][:] = False
        p_t = teacher_act_prob(self.step, self.dc)
        acts = np.zeros((W, 2, 2), np.int32)
        st = dict(episodes=0, errors=0, teacher_rows=0, rows=0, gpu=0.0, env=0.0)

        def launch(ws, t):
            ctl = venv.ctl
            cw, cp = [], []
            for w in ws:
                for p in (0, 1):
                    if ctl[w, p] in self.ctl_ok:
                        cw.append(w)
                        cp.append(p)
            cw, cp = np.array(cw, np.int64), np.array(cp, np.int64)
            sub = {k: self.obs[k][cw, cp] for k in self.keys}
            for k in self.keys:
                B['obs_' + k][t, cw, cp] = sub[k]
            B['h_in'][t, cw, cp] = self.hs[cw, cp]
            B['c_in'][t, cw, cp] = self.cs[cw, cp]
            B['valid'][t, cw, cp] = True
            g0 = time.time()
            rs = self.s_eng.run(sub, self.hs[cw, cp], self.cs[cw, cp])
            ta = self.tact[cw, cp]
            # 老师：学生出手的行强制用学生选的牌（位置分布就是在这张牌上的），老师出手的行自己采样；出牌分布不受强制影响
            rt = self.t_eng.run(sub, self.ht[cw, cp], self.ct[cw, cp], force_sel=np.where(ta, -1, rs['sel']), want_logp=True)
            st['gpu'] += time.time() - g0
            sel = np.where(ta, rt['sel'], rs['sel'])
            cell = np.where(ta, rt['cell'], rs['cell'])
            B['sel'][t, cw, cp] = sel
            B['cell'][t, cw, cp] = cell
            B['value'][t, cw, cp] = rt['value']
            B['bc_card_lp'][t, cw, cp] = rt['card_logp']
            B['bc_place_lp'][t, cw, cp] = rt['place_logp']
            self.hs[cw, cp], self.cs[cw, cp] = rs['h'], rs['c']
            self.ht[cw, cp], self.ct[cw, cp] = rt['h'], rt['c']
            acts[cw, cp, 0] = sel
            acts[cw, cp, 1] = cell
            st['rows'] += len(cw)
            st['teacher_rows'] += int(ta.sum())
            venv.step_async(list(ws), acts[ws])

        def collect(ws, t):
            e0 = time.time()
            infos = venv.step_wait(list(ws))
            st['env'] += time.time() - e0
            B['done'][t, ws] = venv._arr['done'][ws]
            for w, inf in zip(ws, infos):
                if inf is None:
                    continue
                self.hs[w] = 0
                self.cs[w] = 0
                self.ht[w] = 0
                self.ct[w] = 0
                self.tact[w] = self.rng.rand(2) < p_t
                st['episodes'] += 1
                st['errors'] += int(bool(inf.get('error')))

        groups = [np.arange(0, W, 2), np.arange(1, W, 2)] if W > 1 else [np.arange(1)]
        for g in groups:
            launch(g, 0)
        for t in range(T):
            for g in groups:
                collect(g, t)
                if t + 1 < T:
                    launch(g, t + 1)
        st['steps'] = int(B['valid'].sum())
        st['teacher_act'] = p_t
        return st

    # ---------------- 训练 ----------------
    def train(self):
        dc, dev, B = self.dc, self.dev, self.buf
        T, W, P = B['valid'].shape
        S, L = W * P, int(dc.seq_len)
        rs = lambda x: x.reshape((T, S) + x.shape[3:])                 # noqa: E731
        vmask = rs(B['valid'])
        done = np.repeat(B['done'], P, axis=1)
        D = {k: torch.from_numpy(rs(B['obs_' + k])).to(dev, non_blocking=True) for k in self.keys}
        for k, src in (('sel', 'sel'), ('h_in', 'h_in'), ('c_in', 'c_in'), ('t_card', 'bc_card_lp'), ('t_place', 'bc_place_lp'), ('t_value', 'value')):
            D[k] = torch.from_numpy(rs(B[src])).to(dev, non_blocking=True)
        D['valid'] = torch.from_numpy(vmask.astype(np.float32)).to(dev)
        chunks = make_chunks(done, L, vmask)
        ch = torch.from_numpy(chunks).to(dev)
        ar_L = torch.arange(L, device=dev)
        freeze = self.update < int(dc.freeze_copied_updates)
        for p in self.copied_params:
            p.requires_grad_(not freeze)
        rng = np.random.RandomState(self.update)
        logs, n_mb = {}, 0
        for _ep in range(int(dc.epochs)):
            perm = rng.permutation(len(chunks))
            for mb in np.array_split(perm, int(dc.minibatches)):
                if len(mb) == 0:
                    continue
                self.net.zero_grad(set_to_none=True)
                n_mb_steps = float(chunks[mb, 2].sum())
                for mic in np.array_split(mb, int(dc.micro_batches)):
                    if len(mic) == 0:
                        continue
                    c3 = ch[torch.from_numpy(mic).to(dev)]
                    ww, t0, ln = c3[:, 0], c3[:, 1], c3[:, 2]
                    valid = ar_L[None] < ln[:, None]
                    tt = t0[:, None] + torch.minimum(ar_L[None], ln[:, None] - 1)   # 补齐的步重复最后一个有效步（因果，不影响有效步）
                    wl = ww[:, None].expand_as(tt)
                    o = {k: D[k][tt, wl].reshape((-1,) + D[k].shape[2:]) for k in self.keys}
                    h0 = D['h_in'][t0, ww].float()
                    c0 = D['c_in'][t0, ww].float()
                    g = lambda k: D[k][tt, wl].reshape((-1,) + D[k].shape[2:])    # noqa: E731
                    m = valid.reshape(-1).float() * g('valid')
                    part = float(m.sum()) / max(n_mb_steps, 1e-6)
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss, parts = distill_loss(self.net, o, h0, c0, L, g, m, self.flat, self.card_cls, float(dc.place_coef), float(dc.value_coef))
                    self.scaler.scale(loss * part).backward()
                    for k, v in parts.items():
                        logs[k] = logs.get(k, 0.0) + v * part
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_([p for p in self.net.parameters() if p.requires_grad], float(dc.grad_clip))
                self.scaler.step(self.opt)
                self.scaler.update()
                logs['grad_norm'] = logs.get('grad_norm', 0.0) + float(gn)
                n_mb += 1
        logs = {k: v / max(1, n_mb) for k, v in logs.items()}
        logs['frozen_copied'] = float(freeze)
        return logs

    # ---------------- 存档 ----------------
    def save(self, snapshot=False):
        d = dict(model=self.net.state_dict(), opt=self.opt.state_dict(), scaler=self.scaler.state_dict(), cfg=dict(self.cfg),
                 step=self.step, update=self.update, kind='distill', teacher=self.teacher_path)
        p = os.path.join(self.run, 'distill_latest.pt')
        torch.save(d, p + '.tmp')
        os.replace(p + '.tmp', p)
        if not snapshot:
            return
        # 学生快照：只有模型和配置（arena / 对打 / 之后的 PPO 起点直接能用）
        s = os.path.join(self.run, 'student_step%d.pt' % self.step)
        torch.save(dict(model=self.net.state_dict(), cfg=dict(self.cfg), step=0, update=0, kind='ppo',
                        distill=dict(teacher=self.teacher_path, steps=self.step, update=self.update)), s + '.tmp')
        os.replace(s + '.tmp', s)

    def loop(self, updates=None):
        dc = self.dc
        logf = open(os.path.join(self.run, 'train_log.jsonl'), 'a', encoding='utf-8')
        n = 0
        try:
            while self.step < int(dc.total_steps) and (updates is None or n < updates):
                t0 = time.time()
                st = self.collect()
                t1 = time.time()
                logs = self.train()
                t2 = time.time()
                self.update += 1
                self.step += st['steps']
                n += 1
                self.sync_actor()
                row = dict(step=self.step, update=self.update, sps=st['steps'] / (t2 - t0), collect_s=t1 - t0, train_s=t2 - t1,
                           infer_s=st['gpu'], env_s=st['env'], teacher_act=st['teacher_act'],
                           teacher_rows=st['teacher_rows'] / max(1, st['rows']), episodes=st['episodes'], env_errors=st['errors'], t=time.time(), **logs)
                logf.write(json.dumps(row) + '\n')
                logf.flush()
                print('[%d] %.2fM 步 %.0f 步/秒（采样 %.0fs，训练 %.0fs）老师出手 %.0f%%  KL 出牌 %.4f 位置 %.4f  价值误差 %.4f  一致率 %.3f（出牌步 %.3f）%s'
                      % (self.update, self.step / 1e6, row['sps'], row['collect_s'], row['train_s'], 100 * row['teacher_rows'],
                         logs.get('kl_card', 0), logs.get('kl_place', 0), logs.get('vmse', 0), logs.get('agree', 0), logs.get('agree_play', 0),
                         '（只训新部分）' if logs.get('frozen_copied') else ''), flush=True)
                if self.update % int(dc.ckpt_every_updates) == 0:
                    self.save(snapshot=self.update % int(dc.snapshot_every_updates) == 0)
        finally:
            self.save(snapshot=True)
            logf.close()
            self.venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True, help='老师存档（PPO 存档）')
    ap.add_argument('--run', default='runs/distill_a')
    ap.add_argument('--updates', type=int, default=None, help='只跑 N 次更新（试跑）')
    ap.add_argument('overrides', nargs='*')
    a = ap.parse_args()
    limit_torch_threads()
    sys.setrecursionlimit(20000)
    Distiller(a.teacher, repo_path(a.run), a.overrides).loop(a.updates)


if __name__ == '__main__':
    main()
