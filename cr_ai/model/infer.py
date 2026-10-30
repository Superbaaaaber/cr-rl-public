"""cr_ai/model/infer.py — 小批量推理加速：把“前向 + 牌头采样 + 位置头采样”录成 CUDA Graph，一次重放代替上百次 kernel 发起。

小批量（评测/RL 采样每次只有几个到几十个样本）时，逐个发起 kernel 的开销远大于计算本身（约 9 ms/次，与批量无关）。
按批量分桶（8/16/32/64/128/256）各录一张图：
  - 输入：所有观测字段打包进一块锁页内存，一次拷进 GPU；图里的输入张量就是这块显存的视图（不再逐字段拷贝）；
  - 输出：sel、cell、logp、value 打包成一个 [B, 4] 张量，一次拷回；LSTM 的 h/c 只在 stateful 时拷回。

    eng = InferEngine(net, place_flat, card_cls)
    out = eng.run(obs_np, h=None, c=None, temperature=1.0, force_sel=None, want_logp=False)
    out：sel [N]、cell [N]（技能/不出 = −1）、logp [N]（所选动作的总对数概率）、value [N]；传了 h/c 时另有 h/c [N, H]
    force_sel [N]：≥0 的行不采样牌头、直接用给定选项（位置头以它为条件采样）——RL 里用来让冻结的 BC 网络
                   在 agent 实际选的牌上给出位置分布（KL 约束的参考分布）。
    want_logp：另返回 card_logp [N, 5+K]、place_logp [N, 576]（温度 1、已按 mask 屏蔽，屏蔽项截到 −30；float16）。

采样用 Gumbel-max（与按 softmax 分布采样等价，且可以录进图里）。
"""
import collections
import contextlib
import math
import os

import numpy as np
import torch

from .net import INPUT_KEYS

BUCKETS = (8, 16, 32, 64, 128, 256)
ALIGN = 16


class InferEngine:
    def __init__(self, net, place_flat, card_cls, device='cuda', use_graph=True):
        self.net = net.eval()
        self.dev = torch.device(device)
        self.flat = torch.as_tensor(place_flat, device=self.dev)            # [类别, 2, 16, 576] bool
        self.card_cls = torch.as_tensor(card_cls, device=self.dev).long()
        self.use_graph = use_graph and self.dev.type == 'cuda'
        self.H = net.H
        # 这张网要的观测字段（实体分支等会多几个）；特权信息（priv_*）是可选的：观测里有才用（RL 采样有，部署/arena 没有）
        self.keys = list(getattr(net, 'input_keys', INPUT_KEYS)) + list(getattr(net, 'priv_keys', ())) + ['pl']
        self.graphs = {}
        self.layout = None

    # ---------------- 图内的计算 ----------------
    def _forward(self, o, h0, c0):
        """网络前向（不含采样）。"""
        half = self.net.card_emb.weight.dtype == torch.float16          # 整网半精度时不用 autocast（省掉每次重放的权重转换）
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False, enabled=not half):
            if half:
                h0, c0 = h0.half(), c0.half()
            out, (h, c) = self.net(o, (h0, c0))
        return out, h, c

    def _sample(self, o, out, h, c, inv_t, force):
        half = self.net.card_emb.weight.dtype == torch.float16
        with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False, enabled=not half):
            pol1 = self.net.policy_logits(out['card'].float())
            pol = pol1 * inv_t
            g1 = -torch.log(-torch.log(torch.rand_like(pol).clamp_(1e-10, 1.0)))
            sel = torch.where(force >= 0, force, (_sample_logits(self.net, pol) + g1).argmax(1))   # 技能概率下限只改抽样（见 ab_floor_mix）
            place = self.net.place_logits(out, sel)
        N = sel.shape[0]
        ar = torch.arange(N, device=sel.device)
        slot = (sel - 1).clamp(0, 3)
        hid = o['hand_id'].long()[ar, slot]
        pm = self.flat[self.card_cls[hid], o['pl'].long(), o['tstate'].long()]
        pl1 = place.float().masked_fill(~pm, -1e9)
        pl = pl1 * inv_t
        g2 = -torch.log(-torch.log(torch.rand_like(pl).clamp_(1e-10, 1.0)))
        cell = (pl + g2).argmax(1)
        is_hand = (sel >= 1) & (sel <= 4)
        lp = torch.log_softmax(pol, 1)[ar, sel] + torch.where(is_hand, torch.log_softmax(pl, 1)[ar, cell], torch.zeros_like(pol[:, 0]))
        cell = torch.where(is_hand, cell, torch.full_like(cell, -1))
        packed = torch.stack([sel.float(), cell.float(), lp, out['value'].float()], 1)
        card_lp = torch.log_softmax(pol1, 1).clamp(min=-30).half()
        place_lp = torch.log_softmax(pl1, 1).clamp(min=-30).half()
        return packed, h.float(), c.float(), card_lp, place_lp

    def _compute(self, o, h0, c0, inv_t, force):
        out, h, c = self._forward(o, h0, c0)
        return self._sample(o, out, h, c, inv_t, force)

    # ---------------- 打包布局 ----------------
    def _make_layout(self, sample):
        self.keys = [k for k in self.keys if k in sample]
        lay, off = [], 0
        for k in self.keys:
            v = np.asarray(sample[k])
            dt = v.dtype
            row = int(np.prod(v.shape[1:], dtype=np.int64)) * dt.itemsize
            lay.append((k, dt, tuple(v.shape[1:]), row))
        self.layout = lay

    def _offsets(self, B):
        offs, off = [], 0
        for k, dt, shp, row in self.layout:
            offs.append(off)
            off += -(-B * row // ALIGN) * ALIGN
        return offs, off

    def _graph(self, B):
        g = self.graphs.get(B)
        if g is not None:
            return g
        offs, total = self._offsets(B)
        gbuf = torch.zeros(total, dtype=torch.uint8, device=self.dev)
        hbuf = torch.zeros(total, dtype=torch.uint8).pin_memory()
        o = {}
        for (k, dt, shp, row), off in zip(self.layout, offs):
            tdt = torch.from_numpy(np.zeros(1, dt)).dtype
            o[k] = gbuf[off:off + B * row].view(tdt).view((B,) + shp)
        h0 = torch.zeros(B, self.H, device=self.dev)
        c0 = torch.zeros(B, self.H, device=self.dev)
        inv_t = torch.ones(1, device=self.dev)
        force = torch.full((B,), -1, dtype=torch.long, device=self.dev)
        o['card_mask'][:, 0] = True                                       # 合法的占位输入（填充行也保持“不出”可选）
        for (k, dt, shp, row), off in zip(self.layout, offs):
            if k == 'card_mask':
                hbuf.numpy()[off:off + B * row].reshape(B, -1)[:, 0] = 1
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._compute(o, h0, c0, inv_t, force)                    # 预热（cudnn 选算法等）
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            packed, h, c, card_lp, place_lp = self._compute(o, h0, c0, inv_t, force)
        g = dict(graph=graph, gbuf=gbuf, hbuf=hbuf, hnp=hbuf.numpy(), offs=offs, h0=h0, c0=c0, inv_t=inv_t, force=force, packed=packed, h=h, c=c,
                 card_lp=card_lp, place_lp=place_lp,
                 out_host=torch.zeros(packed.shape, dtype=torch.float32).pin_memory(), last_t=None)
        self.graphs[B] = g
        return g

    @torch.no_grad()
    def run(self, obs, h=None, c=None, temperature=1.0, force_sel=None, want_logp=False):
        N = len(obs['tstate'])
        if not self.use_graph or N > BUCKETS[-1]:
            o = {k: torch.as_tensor(np.ascontiguousarray(obs[k])).to(self.dev) for k in self.keys if k in obs}
            h0 = torch.as_tensor(h, device=self.dev) if h is not None else torch.zeros(N, self.H, device=self.dev)
            c0 = torch.as_tensor(c, device=self.dev) if c is not None else torch.zeros(N, self.H, device=self.dev)
            fs = torch.as_tensor(force_sel if force_sel is not None else np.full(N, -1), device=self.dev).long()
            packed, hh, cc, clp, plp = self._compute(o, h0, c0, torch.full((1,), 1.0 / temperature, device=self.dev), fs)
            r = self._unpack(packed.cpu().numpy(), hh, cc, h is not None)
            if want_logp:
                r['card_logp'], r['place_logp'] = clp.cpu().numpy(), plp.cpu().numpy()
            return r
        if self.layout is None:
            self._make_layout(obs)
        B = next(b for b in BUCKETS if b >= N)
        g = self._graph(B)
        hnp = g['hnp']
        for (k, dt, shp, row), off in zip(self.layout, g['offs']):
            v = np.ascontiguousarray(obs[k], dtype=dt)
            hnp[off:off + N * row] = v.reshape(-1).view(np.uint8)            # 填充行沿用上一次的内容（合法输入，结果不用）
        g['gbuf'].copy_(g['hbuf'], non_blocking=True)
        stateful = h is not None
        if stateful:
            g['h0'][:N].copy_(torch.as_tensor(h))
            g['c0'][:N].copy_(torch.as_tensor(c))
        elif g['last_t'] != 'zero':
            g['h0'].zero_()
            g['c0'].zero_()
        g['last_t'] = 'state' if stateful else 'zero'
        g['inv_t'].fill_(1.0 / temperature)
        if force_sel is not None:
            g['force'][:N].copy_(torch.as_tensor(np.asarray(force_sel, np.int64)))
            g['force'][N:].fill_(-1)
            g['forced'] = True
        elif g.get('forced'):
            g['force'].fill_(-1)
            g['forced'] = False
        g['graph'].replay()
        g['out_host'].copy_(g['packed'])
        r = self._unpack(g['out_host'].numpy()[:N], g['h'][:N] if stateful else None, g['c'][:N] if stateful else None, stateful)
        if want_logp:
            r['card_logp'] = g['card_lp'][:N].cpu().numpy()
            r['place_logp'] = g['place_lp'][:N].cpu().numpy()
        return r

    @staticmethod
    def _unpack(p, h, c, stateful):
        r = dict(sel=p[:, 0].astype(np.int64), cell=p[:, 1].astype(np.int64), logp=p[:, 2].copy(), value=p[:, 3].copy())
        if stateful:
            r['h'] = h.cpu().numpy()
            r['c'] = c.cpu().numpy()
        return r


class DualEngine:
    """RL 采样用：当前策略（带 LSTM 状态）+ 冻结的 BC 网络，合进同一张 CUDA Graph，一次重放、输入输出各拷一次。

        out = eng.run(obs_a, h, c, obs_o)
    obs_a [N]：主角一方的观测；obs_o [No]（No ≤ N，可为 0）：由 BC 控制的对手一方的观测。
    out：sel、cell、logp、value、h、c（主角）；bc_card_lp [N, 5+K]、bc_place_lp [N, 576]（BC 网络在主角实际所选牌上的分布，KL 参考）；
         opp_sel、opp_cell [No]（BC 对手采样出的动作）。
    """

    def __init__(self, net, bc_net, place_flat, card_cls, device='cuda'):
        self.a = InferEngine(net, place_flat, card_cls, device)
        self.b = InferEngine(bc_net, place_flat, card_cls, device)
        assert self.a.keys == self.b.keys, '合并图里的网络要用同一套观测字段'
        self.keys = self.a.keys
        self.dev = self.a.dev
        self.s2 = torch.cuda.Stream(device=self.dev)
        self.H = net.H
        self.graphs = {}
        self.layout = None

    def _make_layout(self, sample):
        self.keys = [k for k in self.keys if k in sample]
        lay = []
        for blk in ('a', 'o'):
            for k in self.keys:
                v = np.asarray(sample[k])
                lay.append(((blk, k), v.dtype, tuple(v.shape[1:]), int(np.prod(v.shape[1:], dtype=np.int64)) * v.dtype.itemsize))
        for k in ('h0', 'c0'):
            lay.append(((k, None), np.dtype(np.float32), (self.H,), 4 * self.H))
        self.layout = lay

    def _compute(self, oa, oo, h0, c0, inv_t, zf, zs):
        """两条流：BC 网络的前向（不依赖主角的选择）和主角的前向 + 采样并行；汇合后 BC 再以主角所选牌为条件算位置分布。
        小批量时 GPU 大部分 SM 空着，两路并行几乎不互相拖慢。"""
        B = h0.shape[0]
        cur = torch.cuda.current_stream()
        self.s2.wait_stream(cur)
        with torch.cuda.stream(self.s2):
            ob = {k: torch.cat([oa[k], oo[k]]) for k in self.keys}
            out_b, hb, cb = self.b._forward(ob, zs, zs)
        pa, h, c, _clp, _plp = self.a._compute(oa, h0, c0, inv_t, zf[:B])
        cur.wait_stream(self.s2)
        force = torch.cat([pa[:, 0].long(), zf[B:]])
        pb, _h, _c, clp, plp = self.b._sample(ob, out_b, hb, cb, inv_t, force)
        return torch.cat([pa, h, c, pb[B:, :2], clp[:B].float(), plp[:B].float()], 1)

    def _graph(self, B):
        g = self.graphs.get(B)
        if g is not None:
            return g
        offs, off = [], 0
        for _k, _dt, _shp, row in self.layout:
            offs.append(off)
            off += -(-B * row // ALIGN) * ALIGN
        gbuf = torch.zeros(off, dtype=torch.uint8, device=self.dev)
        hbuf = torch.zeros(off, dtype=torch.uint8).pin_memory()
        t = {}
        for ((blk, k), dt, shp, row), o_ in zip(self.layout, offs):
            tdt = torch.from_numpy(np.zeros(1, dt)).dtype
            t[(blk, k)] = gbuf[o_:o_ + B * row].view(tdt).view((B,) + shp)
            if k == 'card_mask':
                t[(blk, k)][:, 0] = True
                hbuf.numpy()[o_:o_ + B * row].reshape(B, -1)[:, 0] = 1
        oa = {k: t[('a', k)] for k in self.keys}
        oo = {k: t[('o', k)] for k in self.keys}
        h0, c0 = t[('h0', None)], t[('c0', None)]
        inv_t = torch.ones(1, device=self.dev)
        zf = torch.full((2 * B,), -1, dtype=torch.long, device=self.dev)
        zs = torch.zeros(2 * B, self.H, device=self.dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._compute(oa, oo, h0, c0, inv_t, zf, zs)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            out = self._compute(oa, oo, h0, c0, inv_t, zf, zs)
        # zf / zs 必须跟图一起活着：图里记的是它们的显存地址，Python 一释放、显存被别的东西复用，重放时就会读到垃圾
        # （强制选项变成乱码 → 索引越界 / 非法访问）
        g = dict(graph=graph, gbuf=gbuf, hbuf=hbuf, hnp=hbuf.numpy(), offs=offs, inv_t=inv_t, out=out, zf=zf, zs=zs,
                 out_host=torch.zeros(out.shape, dtype=torch.float32).pin_memory())
        self.graphs[B] = g
        return g

    @torch.no_grad()
    def run(self, obs_a, h, c, obs_o=None, temperature=1.0):
        N = len(obs_a['tstate'])
        No = 0 if obs_o is None else len(obs_o['tstate'])
        assert No <= N
        if os.environ.get('CR_INFER_EAGER'):                              # 调试：不用 CUDA Graph，配合 CUDA_LAUNCH_BLOCKING=1 定位出错的算子
            dev = self.dev
            oa = {k: torch.as_tensor(np.ascontiguousarray(obs_a[k])).to(dev) for k in self.keys}
            pad = lambda k: np.concatenate([obs_o[k], obs_a[k][:N - No]]) if No else obs_a[k]   # noqa: E731  对手块补齐到 N 行
            oo = {k: torch.as_tensor(np.ascontiguousarray(pad(k))).to(dev) for k in self.keys}
            h0 = torch.as_tensor(np.asarray(h, np.float32), device=dev)
            c0 = torch.as_tensor(np.asarray(c, np.float32), device=dev)
            zf = torch.full((2 * N,), -1, dtype=torch.long, device=dev)
            zs = torch.zeros(2 * N, self.H, device=dev)
            out = self._compute(oa, oo, h0, c0, torch.full((1,), 1.0 / temperature, device=dev), zf, zs)
            torch.cuda.synchronize()
            return self._unpack(out.float().cpu().numpy(), N, No)
        if self.layout is None:
            self._make_layout(obs_a)
        B = next(b for b in BUCKETS if b >= N)
        g = self._graph(B)
        hnp = g['hnp']
        for ((blk, k), dt, shp, row), off in zip(self.layout, g['offs']):
            if blk == 'a':
                v, n = obs_a[k], N
            elif blk == 'o':
                if not No:
                    continue
                v, n = obs_o[k], No
            else:
                v, n = (h if blk == 'h0' else c), N
            hnp[off:off + n * row] = np.ascontiguousarray(v, dtype=dt).reshape(-1).view(np.uint8)
        g['gbuf'].copy_(g['hbuf'], non_blocking=True)
        g['inv_t'].fill_(1.0 / temperature)
        g['graph'].replay()
        g['out_host'].copy_(g['out'])
        return self._unpack(g['out_host'].numpy(), N, No)

    def _unpack(self, p, N, No):
        H = self.H
        r = dict(sel=p[:N, 0].astype(np.int64), cell=p[:N, 1].astype(np.int64), logp=p[:N, 2].copy(), value=p[:N, 3].copy(),
                 h=p[:N, 4:4 + H].copy(), c=p[:N, 4 + H:4 + 2 * H].copy())
        o2 = 4 + 2 * H
        r['opp_sel'] = p[:No, o2].astype(np.int64)
        r['opp_cell'] = p[:No, o2 + 1].astype(np.int64)
        nsel = p.shape[1] - o2 - 2 - 576
        r['bc_card_lp'] = p[:N, o2 + 2:o2 + 2 + nsel].astype(np.float16)
        r['bc_place_lp'] = p[:N, o2 + 2 + nsel:].astype(np.float16)
        return r


def _bucket(n):
    return next(b for b in BUCKETS if b >= n)


class LeagueEngineFused:
    """（旧版，CR_INFER_LEGACY=1 时 rollout 用它；新版见下面的 LeagueEngine）
    M4 采样用：当前策略 + 冻结的 BC 网络 + K 个历史快照槽，合进同一张 CUDA Graph（三条流并行），一次重放。

        out = eng.run(obs_a, h_a, c_a, obs_o=None, obs_x=None, h_x=None, c_x=None)
    obs_a [Na]：当前策略控制的各行（主角一方 + 自我对弈时的对手一方），带 LSTM 状态 h_a/c_a；
    obs_o [No]：BC 控制的对手；
    obs_x：长 K 的列表，第 k 项是第 k 个历史快照槽控制的对手 [Nk]（没有就 None），带状态 h_x[k]/c_x[k]。
    out：sel、cell、logp、value、h、c [Na]；bc_card_lp、bc_place_lp [Na]（BC 在当前策略所选牌上的分布，KL 参考）；
         opp_sel、opp_cell [No]；hist_sel、hist_cell、hist_h、hist_c：长 K 的列表（第 k 项 [Nk]）。
    图按 (B, Bx) 分别录：B 是主角/BC 两块的行数分桶，Bx 是历史快照块里每个槽的行数分桶（各槽都补齐到 Bx）；
    这一步没有历史快照对手时 Bx = 0，整块不算（M3 只有 BC/规则对手）。
    换快照：hist_nets[k].load_state_dict(...)（原地拷贝，图里引用的地址不变）；只在那个槽没有进行中的对局时换（见 train/league.py: HistSlots）。
    """

    def __init__(self, net, bc_net, hist_nets, place_flat, card_cls, device='cuda'):
        if isinstance(hist_nets, torch.nn.Module):
            hist_nets = [hist_nets]
        self.a = InferEngine(net, place_flat, card_cls, device)
        self.b = InferEngine(bc_net, place_flat, card_cls, device)
        self.xs = [InferEngine(m, place_flat, card_cls, device) for m in hist_nets]
        assert all(e.keys == self.a.keys for e in [self.b] + self.xs), '合并图里的网络要用同一套观测字段'
        self.keys = self.a.keys
        self.K = len(self.xs)
        self.dev = self.a.dev
        self.sb = torch.cuda.Stream(device=self.dev)
        self.sx = torch.cuda.Stream(device=self.dev)
        # 默认三块（主角、BC、历史快照）都在当前流上依次算。2026-10-18：arch_v2 的 M4 开了历史快照侧流后，采样进程几分钟内必出
        # CUDA 非法访存（CUDA_LAUNCH_BLOCKING=1 时不出，单独压测 2 万次也不出）；改成单流后连跑不再出，代价是采样慢约 15%。
        # CR_INFER_STREAMS=1 恢复侧流并行（M3 只有 BC 侧流时跑了 7 小时没出过事）。
        self.serial = not os.environ.get('CR_INFER_STREAMS')
        self.H = net.H
        self.graphs = {}
        self.layout = None

    def _side(self, s):
        return contextlib.nullcontext() if self.serial else torch.cuda.stream(s)

    def _make_layout(self, sample):
        """[((块, 字段), dtype, 每行形状, 每行字节, 是否历史快照块)]；状态块的字段为 None。"""
        self.keys = [k for k in self.keys if k in sample]
        lay = []
        for blk in ['a', 'o'] + ['x%d' % i for i in range(self.K)]:
            for k in self.keys:
                v = np.asarray(sample[k])
                lay.append(((blk, k), v.dtype, tuple(v.shape[1:]), int(np.prod(v.shape[1:], dtype=np.int64)) * v.dtype.itemsize, blk != 'a' and blk != 'o'))
        for blk in ['ha', 'ca']:
            lay.append(((blk, None), np.dtype(np.float32), (self.H,), 4 * self.H, False))
        for i in range(self.K):
            for blk in ('hx%d' % i, 'cx%d' % i):
                lay.append(((blk, None), np.dtype(np.float32), (self.H,), 4 * self.H, True))
        self.layout = lay

    def _compute(self, t, B, Bx, inv_t, zf, zs, zfx):
        """t：(块, 字段) → 张量。返回 (主块 [B, ...], 历史快照块 [K·Bx, 2 + 2H] 或 None)。"""
        oa = {k: t[('a', k)] for k in self.keys}
        oo = {k: t[('o', k)] for k in self.keys}
        cur = torch.cuda.current_stream()
        if not self.serial:
            self.sb.wait_stream(cur)
        with self._side(self.sb):
            ob = {k: torch.cat([oa[k], oo[k]]) for k in self.keys}
            out_b, hb, cb = self.b._forward(ob, zs, zs)
        hist = None
        if Bx:
            if not self.serial:
                self.sx.wait_stream(cur)
            with self._side(self.sx):                                     # K 个槽在同一条流上依次算（每个槽只有几行）
                parts = []
                for i, x in enumerate(self.xs):
                    ox = {k: t[('x%d' % i, k)] for k in self.keys}
                    px, hx2, cx2, _c1, _p1 = x._compute(ox, t[('hx%d' % i, None)], t[('cx%d' % i, None)], inv_t, zfx)
                    parts.append(torch.cat([px[:, :2], hx2, cx2], 1))
                hist = torch.cat(parts, 0)
        pa, h, c, _c2, _p2 = self.a._compute(oa, t[('ha', None)], t[('ca', None)], inv_t, zf[:B])
        if not self.serial:
            cur.wait_stream(self.sb)
            if Bx:
                cur.wait_stream(self.sx)
        force = torch.cat([pa[:, 0].long(), zf[B:]])
        pb, _h, _c, clp, plp = self.b._sample(ob, out_b, hb, cb, inv_t, force)
        main = torch.cat([pa, h, c, pb[B:, :2], clp[:B].float(), plp[:B].float()], 1)
        return main, hist

    def _graph(self, B, Bx):
        g = self.graphs.get((B, Bx))
        if g is not None:
            return g
        offs, off = [], 0
        for _k, _dt, _shp, row, is_x in self.layout:
            offs.append(off)
            off += -(-(Bx if is_x else B) * row // ALIGN) * ALIGN
        gbuf = torch.zeros(off, dtype=torch.uint8, device=self.dev)
        hbuf = torch.zeros(off, dtype=torch.uint8).pin_memory()
        t = {}
        for ((blk, k), dt, shp, row, is_x), o_ in zip(self.layout, offs):
            n = Bx if is_x else B
            if not n:
                continue
            tdt = torch.from_numpy(np.zeros(1, dt)).dtype
            t[(blk, k)] = gbuf[o_:o_ + n * row].view(tdt).view((n,) + shp)
            if k == 'card_mask':
                t[(blk, k)][:, 0] = True
                hbuf.numpy()[o_:o_ + n * row].reshape(n, -1)[:, 0] = 1
        inv_t = torch.ones(1, device=self.dev)
        zf = torch.full((2 * B,), -1, dtype=torch.long, device=self.dev)
        zs = torch.zeros(2 * B, self.H, device=self.dev)
        zfx = torch.full((Bx,), -1, dtype=torch.long, device=self.dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._compute(t, B, Bx, inv_t, zf, zs, zfx)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            main, hist = self._compute(t, B, Bx, inv_t, zf, zs, zfx)
        # zf / zs / zfx 必须跟图一起活着：图里记的是它们的显存地址，Python 一释放、显存被别的东西复用，重放时就会读到垃圾
        # （强制选项变成乱码 → 索引越界 / 非法访问）
        g = dict(graph=graph, gbuf=gbuf, hbuf=hbuf, hnp=hbuf.numpy(), offs=offs, inv_t=inv_t, main=main, hist=hist, zf=zf, zs=zs, zfx=zfx,
                 main_host=torch.zeros(main.shape, dtype=torch.float32).pin_memory(),
                 hist_host=torch.zeros(hist.shape, dtype=torch.float32).pin_memory() if hist is not None else None)
        self.graphs[(B, Bx)] = g
        return g

    @torch.no_grad()
    def run(self, obs_a, h_a, c_a, obs_o=None, obs_x=None, h_x=None, c_x=None, temperature=1.0, want_bc=True):
        """want_bc：接口和新版一致；旧版每步都算 BC 参考分布（忽略这个参数）。"""
        K = self.K
        Na = len(obs_a['tstate'])
        No = 0 if obs_o is None else len(obs_o['tstate'])
        obs_x = list(obs_x) if obs_x is not None else [None] * K
        h_x = list(h_x) if h_x is not None else [None] * K
        c_x = list(c_x) if c_x is not None else [None] * K
        assert len(obs_x) == K, '历史快照块要给 %d 个（每个槽一个，没有就 None）' % K
        Nk = [0 if o is None else len(o['tstate']) for o in obs_x]
        if os.environ.get('CR_INFER_EAGER'):                              # 调试：不用 CUDA Graph（配合 CUDA_LAUNCH_BLOCKING=1 定位出错的算子）
            return self._run_eager(obs_a, h_a, c_a, obs_o, obs_x, h_x, c_x, Na, No, Nk, temperature)
        if self.layout is None:
            self._make_layout(obs_a)
        B = _bucket(max(Na, No))
        Bx = _bucket(max(Nk)) if max(Nk, default=0) else 0
        g = self._graph(B, Bx)
        hnp = g['hnp']
        src = {'a': (obs_a, Na), 'o': (obs_o, No), 'ha': (h_a, Na), 'ca': (c_a, Na)}
        for i in range(K):
            src['x%d' % i] = (obs_x[i], Nk[i])
            src['hx%d' % i] = (h_x[i], Nk[i])
            src['cx%d' % i] = (c_x[i], Nk[i])
        for ((blk, k), dt, shp, row, is_x), off in zip(self.layout, g['offs']):
            o, n = src[blk]
            if not n:
                continue                                                  # 空块：沿用上次的内容（合法输入，结果不用）
            v = o if k is None else o[k]
            hnp[off:off + n * row] = np.ascontiguousarray(v, dtype=dt).reshape(-1).view(np.uint8)
        if os.environ.get('CR_INFER_CHECK'):                              # 调试：重放前检查整块输入（含残留行）的编号范围
            from ..env import tables
            lim = dict(gi=tables.n_units(), hand_id=tables.n_cards(), deck_id=tables.n_cards(), opp_id=tables.n_cards(), ab_id=tables.n_cards(),
                       pend_id=tables.n_cards(), last_id=tables.n_cards(), hand_form=3, deck_form=3, opp_form=3, tstate=16, pl=2)
            for ((blk, k), dt, shp, row, is_x), off in zip(self.layout, g['offs']):
                n = Bx if is_x else B
                if k in lim and n:
                    v = hnp[off:off + n * row].view(dt).reshape((n,) + shp)
                    if v.min() < 0 or v.max() >= lim[k]:
                        raise RuntimeError('输入越界：块 %s 字段 %s 范围 [%d, %d] 上限 %d（Na=%d No=%d Nk=%s B=%d Bx=%d）' % (blk, k, v.min(), v.max(), lim[k], Na, No, Nk, B, Bx))
        g['gbuf'].copy_(g['hbuf'], non_blocking=True)
        g['inv_t'].fill_(1.0 / temperature)
        g['graph'].replay()
        g['main_host'].copy_(g['main'])
        if g['hist'] is not None:
            g['hist_host'].copy_(g['hist'])
        return self._unpack(g['main_host'].numpy(), g['hist_host'].numpy() if g['hist'] is not None else None, Na, No, Nk, Bx)

    def _run_eager(self, obs_a, h_a, c_a, obs_o, obs_x, h_x, c_x, Na, No, Nk, temperature):
        dev = self.dev
        B = max(Na, No)
        Bx = max(Nk, default=0)

        def blk(o, n, N):
            src = o if n else obs_a
            idx = np.arange(N) % max(1, n if n else Na)
            return {k: torch.as_tensor(np.ascontiguousarray(src[k][idx])).to(dev) for k in self.keys}

        def st(v, n, N):
            z = np.zeros((N, self.H), np.float32)
            if n:
                z[:n] = v
            return torch.as_tensor(z, device=dev)
        t = {}
        for name, o, n, N in [('a', obs_a, Na, B), ('o', obs_o, No, B)] + [('x%d' % i, obs_x[i], Nk[i], Bx) for i in range(self.K)]:
            if N:
                t.update({(name, k): v for k, v in blk(o, n, N).items()})
        t[('ha', None)], t[('ca', None)] = st(h_a, Na, B), st(c_a, Na, B)
        if Bx:
            for i in range(self.K):
                t[('hx%d' % i, None)], t[('cx%d' % i, None)] = st(h_x[i], Nk[i], Bx), st(c_x[i], Nk[i], Bx)
        main, hist = self._compute(t, B, Bx, torch.full((1,), 1.0 / temperature, device=dev), torch.full((2 * B,), -1, dtype=torch.long, device=dev),
                                   torch.zeros(2 * B, self.H, device=dev), torch.full((Bx,), -1, dtype=torch.long, device=dev))
        torch.cuda.synchronize()
        return self._unpack(main.float().cpu().numpy(), hist.float().cpu().numpy() if hist is not None else None, Na, No, Nk, Bx)

    def _unpack(self, p, q, Na, No, Nk, Bx):
        H = self.H
        r = dict(sel=p[:Na, 0].astype(np.int64), cell=p[:Na, 1].astype(np.int64), logp=p[:Na, 2].copy(), value=p[:Na, 3].copy(),
                 h=p[:Na, 4:4 + H].copy(), c=p[:Na, 4 + H:4 + 2 * H].copy())
        o2 = 4 + 2 * H
        r['opp_sel'] = p[:No, o2].astype(np.int64)
        r['opp_cell'] = p[:No, o2 + 1].astype(np.int64)
        nsel = p.shape[1] - o2 - 2 - 576
        r['bc_card_lp'] = p[:Na, o2 + 2:o2 + 2 + nsel].astype(np.float16)
        r['bc_place_lp'] = p[:Na, o2 + 2 + nsel:].astype(np.float16)
        r['hist_sel'], r['hist_cell'], r['hist_h'], r['hist_c'] = [], [], [], []
        for i, n in enumerate(Nk):
            x = q[i * Bx:i * Bx + n] if n else np.zeros((0, 2 + 2 * H), np.float32)
            r['hist_sel'].append(x[:, 0].astype(np.int64))
            r['hist_cell'].append(x[:, 1].astype(np.int64))
            r['hist_h'].append(x[:, 2:2 + H].copy())
            r['hist_c'].append(x[:, 2 + H:2 + 2 * H].copy())
        return r


# ======================================================================================
# 新版 LeagueEngine（2026-10-18）：位置头只算出了手牌的行（采样进程用；结果与旧版同分布）
# ======================================================================================
# 旧版（LeagueEngineFused）每步：主角 B 行 + BC 2B 行（BC 对手块没人时也补齐到 B）+ K 个历史槽各 Bx 行，每一行都过位置头
# （v2 的位置头约占前向的 45%），而真正出手牌的只有约 5% 的步。新版仍是“每步一张图、一次重放、一次同步”
# （试过分块多图 + 中途同步一次：GPU 被训练占满时每次同步都要排队，反而更慢），区别是：
#   - 图里按牌头采样结果挑出出了手牌的行（只看有效行，补齐行不算），位置头只算这些行，名额固定（主角 4/8、BC 再 +4、每个历史槽 2/4）；
#     超出名额的（极少）重放之后在图外补算；
#   - BC 对手块单独分桶（Bo），没有 BC 对手时整块不算；分桶更细（4/8/12/16/24/32/…）。
LBUCKETS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def _lb(n):
    return next(b for b in LBUCKETS if b >= n)


def _gumbel(x):
    return -torch.log(-torch.log(torch.rand_like(x).clamp_(1e-10, 1.0)))


def _net_forward(net, o, h0, c0):
    half = net.card_emb.weight.dtype == torch.float16              # 整网半精度时不用 autocast
    with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False, enabled=not half):
        if half:
            h0, c0 = h0.half(), c0.half()
        out, (h, c) = net(o, (h0, c0))
    return out, h, c


def ab_floor_mix(pol, eps, k0=5):
    """技能概率下限（定向探索，ppo.ab_floor）：采样分布 π' = (1−ε)·π + ε·（在这一步可以按的技能槽上均分）；没有技能可以按的行不变。
    pol：牌头 logits [B, 5+K]（不合法的项是 −1e9，见 Net.card_logits）；eps：0 维张量。→ log π'（当 logits 用）；ε = 0 时原样返回 log π。
    只改采样：记下的 logp 和学习端的比率/熵/KL 仍按 π。若把下限也算进训练的策略（对 π' on-policy），技能 logit 的期望梯度是原来的
    (1−ε) 倍，仍正比于 π(技能)，推不动；按 π 算比率时，被下限抽中的样本按优势直接推技能 logit（推力 ∝ ε/可按的技能数，和 π 无关）。
    这有偏（等于把技能样本的权重放大），所以要退火到 0。"""
    lp = torch.log_softmax(pol.float(), 1)
    ok = pol[:, k0:] > -1e8
    n = ok.sum(1, keepdim=True)
    e = eps.float() * (n > 0)
    keep = torch.log1p(-e)
    lu = torch.where(ok, -torch.log(n.clamp(min=1).float()), torch.full_like(lp[:, k0:], -math.inf))
    return torch.cat([keep + lp[:, :k0], torch.logaddexp(keep + lp[:, k0:], torch.log(e) + lu)], 1)


def _sample_logits(net, pol):
    """采样用的牌头 logits：打开了技能概率下限的网络（采样进程里的当前策略）混入下限，其余原样。"""
    return ab_floor_mix(pol, net.ab_eps) if getattr(net, 'ab_mix', False) else pol


def _card_sample(net, out, inv_t, force):
    """牌头 Gumbel-max 采样（force ≥ 0 的行直接用给定选项）→ (sel, 所选项的对数概率（带温度）, 温度 1 的 logits)。
    技能概率下限（net.ab_mix）只改抽样的分布，返回的对数概率仍按策略本身（见 ab_floor_mix）。"""
    pol1 = net.policy_logits(out['card'].float())
    pol = pol1 * inv_t
    ps = _sample_logits(net, pol)
    sel = torch.where(force >= 0, force, (ps + _gumbel(ps)).argmax(1))
    lp = torch.log_softmax(pol, 1).gather(1, sel[:, None])[:, 0]
    return sel, lp, pol1


def _place_sample(net, flat, card_cls, act, o, sel, idx, inv_t, want_lp):
    """位置头只算 idx 指定的行（以该行所选的牌为条件）→ ([P, 2]：cell、对数概率（带温度）, 温度 1 的对数分布 [P, 576] 或 None)。"""
    sub = {k: act[k].index_select(0, idx) for k in ('h', 'f16', 'skip', 'hand')}
    s = sel.index_select(0, idx)
    half = net.card_emb.weight.dtype == torch.float16
    with torch.autocast('cuda', dtype=torch.float16, cache_enabled=False, enabled=not half):
        place = net.place_logits(sub, s)
    ar = torch.arange(idx.shape[0], device=idx.device)
    hid = o['hand_id'].index_select(0, idx).long()[ar, (s - 1).clamp(0, 3)]
    pm = flat[card_cls[hid], o['pl'].index_select(0, idx).long(), o['tstate'].index_select(0, idx).long()]
    pl1 = place.float().masked_fill(~pm, -1e9)
    pl = pl1 * inv_t
    cell = (pl + _gumbel(pl)).argmax(1)
    lp = torch.log_softmax(pl, 1).gather(1, cell[:, None])[:, 0]
    return torch.stack([cell.float(), lp], 1), (torch.log_softmax(pl1, 1).clamp(min=-30) if want_lp else None)


def _is_hand(sel):
    return (sel >= 1) & (sel <= 4)


def _pick(sel, valid, P):
    """出了手牌的有效行按行号排在最前，取前 P 个行号（不够 P 个时后面是别的行，结果不用）。"""
    N = sel.shape[0]
    ar = torch.arange(N, device=sel.device)
    return torch.topk((_is_hand(sel) & valid).float() * (2 * N) - ar.float(), P).indices


def _caps(B, Bo, Bx):
    """位置头的名额（每步出牌的行约 5%：主角 20 来行时平均不到 1 行，超出名额的在图外补算）。"""
    ca = min(B, 4 if B <= 32 else 8)
    return ca, min(B + Bo, ca + (4 if Bo else 0)), min(Bx, 2 if Bx <= 8 else 4)


def _capture(fn):
    """预热（侧流上跑 3 次：cudnn 选算法等）后录图 → (图, fn 的输出)。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = fn()
    return g, out


class _Buf:
    """图的输入：一块锁页内存 + 同样大的一块显存，按 spec [(名字, dtype, 每行形状, 行数)] 切成各字段的视图；put 填前 n 行，upload 一次拷上显卡。
    没填到的行沿用上一次的内容（合法输入，结果不用）。"""

    def __init__(self, spec, dev):
        self.off, off = {}, 0
        for name, dt, shp, n in spec:
            dt = np.dtype(dt)
            row = int(np.prod(shp, dtype=np.int64)) * dt.itemsize
            self.off[name] = (off, row, dt, n)
            off += -(-n * row // ALIGN) * ALIGN
        off = max(off, ALIGN)
        self.g = torch.zeros(off, dtype=torch.uint8, device=dev)
        self.h = torch.zeros(off, dtype=torch.uint8).pin_memory()
        self.hnp = self.h.numpy()
        self.t = {}
        for name, _dt, shp, _n in spec:
            o, row, dt, n = self.off[name]
            tdt = torch.from_numpy(np.zeros(1, dt)).dtype
            self.t[name] = self.g[o:o + n * row].view(tdt).view((n,) + tuple(shp))
            if isinstance(name, tuple) and name[1] == 'card_mask':      # 补齐行保持“不出”可选
                self.t[name][:, 0] = True
                self.hnp[o:o + n * row].reshape(n, -1)[:, 0] = 1

    def put(self, name, v, n):
        o, row, dt, _ = self.off[name]
        self.hnp[o:o + n * row] = np.ascontiguousarray(v, dtype=dt).reshape(-1).view(np.uint8)

    def upload(self):
        self.g.copy_(self.h, non_blocking=True)


ACT = ('h', 'f16', 'skip', 'hand')
MAX_GRAPHS = 24                     # M4 实际常用的 (B, Bo, Bx) 组合约 20 种
HOST_KEYS = ('small', 'h', 'c', 'bc_card_lp', 'sel_b', 'ia', 'res_a', 'ib', 'res_b', 'lp_b', 'xsel', 'xh', 'xc', 'xi', 'xres')


class LeagueEngine:
    """采样进程用：当前策略 + 冻结的 BC 网络 + K 个历史快照槽（接口同 LeagueEngineFused）。

        out = eng.run(obs_a, h_a, c_a, obs_o=None, obs_x=None, h_x=None, c_x=None)
    obs_a [Na]：当前策略控制的各行（主角一方 + 自我对弈时的对手一方），带 LSTM 状态 h_a/c_a；
    obs_o [No]：BC 控制的对手；obs_x：长 K 的列表，第 k 项是第 k 个历史快照槽控制的对手 [Nk]（没有就 None），带状态 h_x[k]/c_x[k]。
    out：sel、cell、logp、value、h、c [Na]；bc_card_lp [Na, 5+K]、bc_place_lp [Na, 576]（BC 在当前策略所选牌上的分布，KL 参考；
         bc_place_lp 只在主角出了手牌的行上有值，其余行为 0——学习端只用这些行）；
         opp_sel、opp_cell [No]；hist_sel、hist_cell、hist_h、hist_c：长 K 的列表（第 k 项 [Nk]）。
    图按 (B, Bo, Bx, wb) 录：主角行数分桶、BC 对手行数分桶（0 = 不算）、历史槽每槽行数分桶（0 = 不算）、图里算不算 BC 块。
    换快照：hist_nets[k].load_state_dict(...)（原地拷贝，图里引用的地址不变）。
    BC 参考分布（2026-10-24，ppo.bc_ref_every）：run(..., want_bc=False) 且没有 BC 对手时整块 BC 不算，bc_card_lp / bc_place_lp 全 0；
    bc_ref_graph=False 时，want_bc=True 的那几步也不录 BC 块，重放之后在图外直接前向 BC 网络（KL 系数为 0、只抽样看漂移时用：
    不用为“带 BC / 不带 BC”各录一套图）。有 BC 对手时照旧整块在图里算。
    """

    def __init__(self, net, bc_net, hist_nets, place_flat, card_cls, device='cuda', bc_ref_graph=True):
        if isinstance(hist_nets, torch.nn.Module):
            hist_nets = [hist_nets]
        self.net, self.bc, self.xs = net.eval(), bc_net.eval(), [m.eval() for m in hist_nets]
        self.K = len(self.xs)
        self.dev = torch.device(device)
        self.flat = torch.as_tensor(place_flat, device=self.dev)            # [类别, 2, 16, 576] bool
        self.card_cls = torch.as_tensor(card_cls, device=self.dev).long()
        keys = lambda m: list(getattr(m, 'input_keys', INPUT_KEYS)) + list(getattr(m, 'priv_keys', ())) + ['pl']   # noqa: E731
        self.keys = keys(net)
        assert all(keys(m) == self.keys for m in [bc_net] + list(hist_nets)), '合并推理的网络要用同一套观测字段'
        self.H = net.H
        self.inv_t = torch.ones(1, device=self.dev)
        self.fields = None
        self.graphs = collections.OrderedDict()                             # (B, Bo, Bx, wb) → 图；最多 MAX_GRAPHS 张，满了删最久没用的
        self.n_overflow = 0                                                 # 位置头名额不够、在图外补算的行数（统计用）
        self.bc_ref_graph = bool(bc_ref_graph)

    def _graph(self, B, Bo, Bx, wb=True):
        key = (B, Bo, Bx, bool(wb or Bo))
        wb = key[3]
        g = self.graphs.get(key)
        if g is not None:
            self.graphs.move_to_end(key)
            return g
        if len(self.graphs) >= MAX_GRAPHS:                                  # 每张图有自己的显存池；行数组合太多时删掉最久没用的
            self.graphs.popitem(last=False)
            torch.cuda.empty_cache()
        H, K, names = self.H, self.K, [k for k, _, _ in self.fields]
        ca, cb, cx = _caps(B, Bo, Bx)
        spec = [(('a', k), dt, shp, B) for k, dt, shp in self.fields]
        spec += [(('o', k), dt, shp, Bo) for k, dt, shp in self.fields] if Bo else []
        for i in range(K if Bx else 0):
            spec += [(('x%d' % i, k), dt, shp, Bx) for k, dt, shp in self.fields] + [('hx%d' % i, np.float32, (H,), Bx), ('cx%d' % i, np.float32, (H,), Bx)]
        spec += [('ha', np.float32, (H,), B), ('ca', np.float32, (H,), B), ('n', np.int64, (2 + K,), 1)]
        buf = _Buf(spec, self.dev)
        zf = torch.full((B + Bo,), -1, dtype=torch.long, device=self.dev)
        zs = torch.zeros(B + Bo, H, device=self.dev)
        zfx = torch.full((max(Bx, 1),), -1, dtype=torch.long, device=self.dev)
        keep = {}                                                           # 图外补算位置头要用的张量（图的输出，地址固定）

        def fn():
            nv = buf.t['n'][0]
            oa = {k: buf.t[('a', k)] for k in names}
            out_a, h, c = _net_forward(self.net, oa, buf.t['ha'], buf.t['ca'])
            sel_a, lp_a, _ = _card_sample(self.net, out_a, self.inv_t, zf[:B])
            act_a = {k: out_a[k] for k in ACT}
            ia = _pick(sel_a, torch.arange(B, device=self.dev) < nv[0], ca)
            res_a, _ = _place_sample(self.net, self.flat, self.card_cls, act_a, oa, sel_a, ia, self.inv_t, False)
            out = dict(small=torch.stack([sel_a.float(), lp_a, out_a['value'].float()], 1), h=h.float(), c=c.float(), ia=ia, res_a=res_a)
            keep.update(a=(act_a, oa, sel_a))
            if wb:
                ob = {k: torch.cat([oa[k], buf.t[('o', k)]]) for k in names} if Bo else oa
                out_b, _h, _c = _net_forward(self.bc, ob, zs, zs)
                force = torch.cat([sel_a, zf[B:]]) if Bo else sel_a      # BC 在主角各行上以主角所选的牌为条件（KL 参考），对手块自己采样
                sel_b, _lp, pol1_b = _card_sample(self.bc, out_b, self.inv_t, force)
                act_b = {k: out_b[k] for k in ACT}
                ar_b = torch.arange(B + Bo, device=self.dev)
                ib = _pick(sel_b, (ar_b < nv[0]) | ((ar_b >= B) & (ar_b < B + nv[1])), cb)
                res_b, lp_b = _place_sample(self.bc, self.flat, self.card_cls, act_b, ob, sel_b, ib, self.inv_t, True)
                out.update(bc_card_lp=torch.log_softmax(pol1_b[:B], 1).clamp(min=-30), sel_b=sel_b, ib=ib, res_b=res_b, lp_b=lp_b)
                keep.update(b=(act_b, ob, sel_b))
            if Bx:
                xs = []
                for i, xn in enumerate(self.xs):
                    ox = {k: buf.t[('x%d' % i, k)] for k in names}
                    o_, h_, c_ = _net_forward(xn, ox, buf.t['hx%d' % i], buf.t['cx%d' % i])
                    s_, _lp, _ = _card_sample(xn, o_, self.inv_t, zfx[:Bx])
                    act_x = {k: o_[k] for k in ACT}
                    ii = _pick(s_, torch.arange(Bx, device=self.dev) < nv[2 + i], cx)
                    r_, _ = _place_sample(xn, self.flat, self.card_cls, act_x, ox, s_, ii, self.inv_t, False)
                    xs.append((s_, h_.float(), c_.float(), ii, r_))
                    keep['x%d' % i] = (act_x, ox, s_)
                for j, name in enumerate(('xsel', 'xh', 'xc', 'xi', 'xres')):
                    out[name] = torch.stack([x[j] for x in xs])
            return out
        graph, out = _capture(fn)
        g = dict(graph=graph, buf=buf, out=out, keep=keep, zf=zf, zs=zs, zfx=zfx, B=B, Bo=Bo, Bx=Bx, wb=wb, caps=(ca, cb, cx),
                 host={k: torch.empty(out[k].shape, dtype=out[k].dtype).pin_memory() for k in HOST_KEYS if k in out})
        self.graphs[key] = g
        return g

    def _bc_ref(self, g, Na, sel, ia):
        """图外算 BC 参考分布（bc_ref_graph=False、want_bc=True 的步）：在这张图已经上传的输入上直接前向 BC 网络（零状态），
        → (bc_card_lp [Na, 5+K], bc_place_lp [Na, 576])，口径和图里的 BC 块一样（温度 1、截到 −30；位置只算主角出了手牌的行）。"""
        buf = g['buf']
        oa = {k: buf.t[('a', k)][:Na] for k in self.keys}
        zs = torch.zeros(Na, self.H, device=self.dev)
        out_b, _h, _c = _net_forward(self.bc, oa, zs, zs)
        card = torch.log_softmax(self.bc.policy_logits(out_b['card'].float()), 1).clamp(min=-30)
        place = np.zeros((Na, self.flat.shape[-1]), np.float16)
        if len(ia):
            act = {k: out_b[k] for k in ACT}
            _res, lp = _place_sample(self.bc, self.flat, self.card_cls, act, oa, torch.as_tensor(sel, device=self.dev),
                                     torch.as_tensor(ia, device=self.dev), self.inv_t, True)
            place[ia] = lp.cpu().numpy()
        return card.cpu().numpy().astype(np.float16), place

    def _extra(self, net, kept, rows, want_lp):
        """名额不够时，在图外（直接前向）补算这些行的位置头。"""
        act, o, sel = kept
        self.n_overflow += len(rows)
        res, lp = _place_sample(net, self.flat, self.card_cls, act, o, sel, torch.as_tensor(np.asarray(rows), device=self.dev), self.inv_t, want_lp)
        return res.cpu().numpy(), (lp.cpu().numpy() if lp is not None else None)

    @torch.no_grad()
    def run(self, obs_a, h_a, c_a, obs_o=None, obs_x=None, h_x=None, c_x=None, temperature=1.0, want_bc=True):
        """want_bc=False：这一步不要 BC 参考分布（没有 BC 对手时整块 BC 不算，bc_* 全 0）。"""
        K, H = self.K, self.H
        Na = len(obs_a['tstate'])
        No = 0 if obs_o is None else len(obs_o['tstate'])
        obs_x = list(obs_x) if obs_x is not None else [None] * K
        h_x = list(h_x) if h_x is not None else [None] * K
        c_x = list(c_x) if c_x is not None else [None] * K
        assert len(obs_x) == K, '历史快照块要给 %d 个（每个槽一个，没有就 None）' % K
        wb = bool(No) or (want_bc and self.bc_ref_graph)
        Nk = [0 if o is None else len(o['tstate']) for o in obs_x]
        if self.fields is None:
            self.keys = [k for k in self.keys if k in obs_a]
            self.fields = [(k, np.asarray(obs_a[k]).dtype, tuple(np.asarray(obs_a[k]).shape[1:])) for k in self.keys]
        names = self.keys
        z = lambda n: np.zeros((n, H), np.float32)   # noqa: E731
        g = self._graph(_lb(Na), _lb(No) if No else 0, _lb(max(Nk)) if max(Nk, default=0) else 0, wb)
        B, Bx, buf, keep = g['B'], g['Bx'], g['buf'], g['keep']
        for k in names:
            buf.put(('a', k), obs_a[k], Na)
            if No:
                buf.put(('o', k), obs_o[k], No)
            for i in range(K):
                if Nk[i]:
                    buf.put(('x%d' % i, k), obs_x[i][k], Nk[i])
        buf.put('ha', h_a if h_a is not None else z(Na), Na)
        buf.put('ca', c_a if c_a is not None else z(Na), Na)
        for i in range(K):
            if Nk[i]:
                buf.put('hx%d' % i, h_x[i] if h_x[i] is not None else z(Nk[i]), Nk[i])
                buf.put('cx%d' % i, c_x[i] if c_x[i] is not None else z(Nk[i]), Nk[i])
        buf.put('n', np.array([[Na, No] + Nk], np.int64), 1)
        buf.upload()
        self.inv_t.fill_(1.0 / temperature)
        g['graph'].replay()
        hs = g['host']
        for k, v in hs.items():
            v.copy_(g['out'][k], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        hn = {k: v.numpy() for k, v in hs.items()}
        # ---- 主角 ----
        small = hn['small']
        sel = small[:Na, 0].astype(np.int64)
        logp = small[:Na, 1].copy()
        cell = np.full(Na, -1, np.int64)
        ia = np.flatnonzero(_is_hand_np(sel))
        m = min(len(ia), g['caps'][0])
        rows, res = hn['ia'][:m], hn['res_a'][:m]
        if len(ia) > m:
            r2, _ = self._extra(self.net, keep['a'], ia[m:], False)
            rows, res = np.concatenate([rows, ia[m:]]), np.concatenate([res, r2])
        cell[rows] = res[:, 0].astype(np.int64)
        logp[rows] += res[:, 1]
        # ---- BC：主角出牌行的参考分布 + BC 对手的落点 ----
        if g['wb']:
            opp_sel = hn['sel_b'][B:B + No].astype(np.int64)
            io = np.flatnonzero(_is_hand_np(opp_sel))
            need = np.concatenate([ia, B + io])
            m = min(len(need), g['caps'][1])
            rows, res, lp = hn['ib'][:m], hn['res_b'][:m], hn['lp_b'][:m]
            if len(need) > m:
                r2, l2 = self._extra(self.bc, keep['b'], need[m:], True)
                rows, res, lp = np.concatenate([rows, need[m:]]), np.concatenate([res, r2]), np.concatenate([lp, l2])
            bc_place_lp = np.zeros((Na, self.flat.shape[-1]), np.float16)
            opp_cell = np.full(No, -1, np.int64)
            mine = rows < B
            bc_place_lp[rows[mine]] = lp[mine]
            opp_cell[rows[~mine] - B] = res[~mine, 0].astype(np.int64)
            bc_card_lp = hn['bc_card_lp'][:Na].astype(np.float16)
        else:                                                               # 这张图没有 BC 块（No = 0）
            opp_sel, opp_cell = np.zeros(0, np.int64), np.zeros(0, np.int64)
            if want_bc:
                bc_card_lp, bc_place_lp = self._bc_ref(g, Na, sel, ia)
            else:
                bc_card_lp = np.zeros((Na, 5 + self.net.K), np.float16)
                bc_place_lp = np.zeros((Na, self.flat.shape[-1]), np.float16)
        r = dict(sel=sel, cell=cell, logp=logp, value=small[:Na, 2].copy(), h=hn['h'][:Na].copy(), c=hn['c'][:Na].copy(),
                 bc_card_lp=bc_card_lp, bc_place_lp=bc_place_lp, opp_sel=opp_sel, opp_cell=opp_cell,
                 hist_sel=[], hist_cell=[], hist_h=[], hist_c=[])
        # ---- 历史快照槽 ----
        for i in range(K):
            n = Nk[i]
            if not n:
                for k in ('hist_sel', 'hist_cell'):
                    r[k].append(np.zeros(0, np.int64))
                for k in ('hist_h', 'hist_c'):
                    r[k].append(np.zeros((0, H), np.float32))
                continue
            s_ = hn['xsel'][i, :n].astype(np.int64)
            hc = np.full(n, -1, np.int64)
            ix = np.flatnonzero(_is_hand_np(s_))
            m = min(len(ix), g['caps'][2])
            rows, res = hn['xi'][i, :m], hn['xres'][i, :m]
            if len(ix) > m:
                r2, _ = self._extra(self.xs[i], keep['x%d' % i], ix[m:], False)
                rows, res = np.concatenate([rows, ix[m:]]), np.concatenate([res, r2])
            hc[rows] = res[:, 0].astype(np.int64)
            r['hist_sel'].append(s_)
            r['hist_cell'].append(hc)
            r['hist_h'].append(hn['xh'][i, :n].copy())
            r['hist_c'].append(hn['xc'][i, :n].copy())
        return r


def _is_hand_np(sel):
    return (sel >= 1) & (sel <= 4)
