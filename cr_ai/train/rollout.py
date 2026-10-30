"""cr_ai/train/rollout.py — RL 采样进程（actor）。和学习进程（train/ppo.py）分开跑，两边并行：学习进程训第 k 批时，这里采第 k+1 批。

结构：
  - VecEnv：W 个环境进程（每进程一个引擎），观测/动作走共享内存；
  - 当前策略（带 LSTM 状态）、冻结的 BC 网络、league.hist_slots 个历史快照合进一张 CUDA Graph（model/infer.py: LeagueEngine）：
    · 当前策略控制主角一方，自我对弈时也控制对手一方（两方都是训练数据，对手一方的损失权重 league.self_opp_weight）；
    · BC 网络给出 bc 控制的对手的动作，以及当前策略各行在其实际所选牌上的 BC 分布（KL 约束的参考分布）；
    · 历史快照控制 hist 对手：每局开局时绑定一个槽，整局由同一个快照打（league.py: HistSlots）；
      学习进程（league.py 的 PFSP）每批发一组候选快照，只换手上没有进行中对局的槽；
  - 环境分两组交替：一组在 CPU 上推进引擎时，另一组在 GPU 上推理；
  - 采到的数据写进共享内存的 rollout 槽（2 个槽轮换），写满后通知学习进程；新权重由学习进程写进共享内存，每批开始前加载。

rollout 槽（每个数组形状 [T, W, 2, ...]，第三维是玩家；valid 标出这一步这一方是不是当前策略在控制）：观测各字段、sel、cell、logp、value、
rew、lw（损失权重）、h_in/c_in（这一步之前的 LSTM 状态）、bc_card_lp、bc_place_lp；done [T, W]；
last_value / last_valid [W, 2]（第 T 步之后那个观测的价值，用来自举）。
"""
import os
import queue
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from ..config import ROOT, cpu_count

STAT_KEYS = ('steps', 'secs', 'env_wait', 'gpu_time', 'episodes', 'errors')


def cgroup_cpu_stat():
    """容器的 CPU 用量与限流计数（cgroup v2 的 cpu.stat；v1 或本机读不到时返回 None）。
    usage_usec 两次相减 ÷ 墙钟 = 平均用了几个核；nr_throttled / nr_periods = 被配额卡住的调度周期比例。"""
    try:
        with open('/sys/fs/cgroup/cpu.stat') as f:
            return {k: int(v) for k, v in (ln.split() for ln in f if ln.strip())}
    except (OSError, ValueError):
        return None


def slot_spec(cfg, T, W, H):
    from ..env.game_env import obs_spec
    from ..model.net import rl_keys
    spec = {}
    os_ = obs_spec(cfg)
    for k in rl_keys(cfg) + ['pl']:
        shape, dt = os_[k]
        spec['obs_' + k] = ((T, W, 2) + tuple(shape), dt)
    nsel = 5 + cfg.env.max_abilities
    spec.update(sel=((T, W, 2), np.int16), cell=((T, W, 2), np.int16), logp=((T, W, 2), np.float32), value=((T, W, 2), np.float32),
                rew=((T, W, 2), np.float32), valid=((T, W, 2), np.bool_), lw=((T, W, 2), np.float16), done=((T, W), np.bool_), trunc=((T, W), np.bool_),
                h_in=((T, W, 2, H), np.float16), c_in=((T, W, 2, H), np.float16),
                bc_card_lp=((T, W, 2, nsel), np.float16), bc_place_lp=((T, W, 2, 576), np.float16),
                last_value=((W, 2), np.float32), last_valid=((W, 2), np.bool_))
    return spec


def slot_bytes(spec):
    return sum(int(np.prod(s)) * np.dtype(d).itemsize for s, d in spec.values())


class Slot:
    """共享内存里的一个 rollout 槽（numpy 视图）。create=True 由学习进程创建。"""

    def __init__(self, spec, names=None):
        self.shms, self.a, self.names = {}, {}, {}
        for k, (shape, dt) in spec.items():
            n = max(1, int(np.prod(shape)) * np.dtype(dt).itemsize)
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=n)
            else:
                shm = shared_memory.SharedMemory(name=names[k])
            self.shms[k] = shm
            self.names[k] = shm.name
            self.a[k] = np.ndarray(shape, dtype=dt, buffer=shm.buf)

    def close(self, unlink=False):
        self.a = {}
        for s in self.shms.values():
            s.close()
            if unlink:
                try:
                    s.unlink()
                except FileNotFoundError:
                    pass


class WeightBox:
    """共享内存里的一份网络参数（float32 拍平）+ 版本号；学习进程写、采样进程读（加锁）。"""

    def __init__(self, n, name=None, lock=None):
        self.shm = shared_memory.SharedMemory(create=name is None, size=8 + 4 * n, name=name)
        self.ver = np.ndarray((1,), np.int64, buffer=self.shm.buf[:8])
        self.flat = np.ndarray((n,), np.float32, buffer=self.shm.buf[8:])
        self.lock = lock
        if name is None:
            self.ver[0] = 0

    @staticmethod
    def tensors(net):
        return [v for _k, v in net.state_dict().items() if v.is_floating_point()]

    def write(self, net):
        import torch
        with torch.no_grad():
            flat = torch.cat([t.detach().float().reshape(-1) for t in self.tensors(net)]).cpu().numpy()
        with self.lock:
            self.flat[:] = flat
            self.ver[0] += 1

    def read_into(self, net):
        import torch
        with self.lock:
            v = int(self.ver[0])
            flat = torch.from_numpy(self.flat.copy())
        off = 0
        with torch.no_grad():
            for t in self.tensors(net):
                n = t.numel()
                t.copy_(flat[off:off + n].view_as(t))                 # 原地拷贝：CUDA Graph 引用的地址不变
                off += n
        return v


def _set_idle():
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x40)
    except Exception:                                          # noqa
        pass


def actor_main(cfg, slot_names, spec, wnames, wlock, n_params, q_ready, q_free, q_ctl, start_step, aid=0):
    """采样进程入口。q_free：学习进程放回的空槽号（本进程自己的槽，0/1）；q_ready：(槽号, 统计)，统计里 aid = 第几个采样进程
    （ppo.actors 个采样进程共用一个 q_ready）；q_ctl：('stop',) 或 ('hist', 候选)。崩溃时发 ('fatal', 报错, aid)。"""
    sys.path.insert(0, ROOT)
    sys.setrecursionlimit(20000)
    if cfg.vec_env.get('priority') == 'idle':
        _set_idle()
    import torch
    # 采样进程是 spawn 出来的，不继承主进程的线程设置；torch 默认按 os.cpu_count()（AutoDL 上是宿主机的 208）开线程。
    # 这里只在 GPU 上算，CPU 线程多了只会和环境进程抢配额。
    torch.set_num_threads(max(1, min(int(cfg.ppo.get('actor_threads') or cpu_count()), cpu_count())))
    try:
        _actor(cfg, slot_names, spec, wnames, wlock, n_params, q_ready, q_free, q_ctl, start_step, aid)
    except Exception:                                           # noqa
        q_ready.put(('fatal', traceback.format_exc(), aid))


def bc_ref_every(cfg):
    """ppo.bc_ref_every：每几步算一次 BC 参考分布（KL 项用）。1 = 每步都算（默认，KL 系数 > 0 时必须是 1）；
    N > 1 只在 KL 系数始终为 0 时允许：只在 t % N == 0 的步算，学习端只拿这些行记 kl_bc（看漂移），采样进程省掉其余步的 BC 前向。"""
    pc = cfg.ppo
    n = max(1, int(pc.get('bc_ref_every', 1) or 1))
    if n > 1 and (float(pc.kl_bc_start) > 0 or float(pc.kl_bc_end) > 0):
        raise ValueError('ppo.bc_ref_every = %d 只能在 KL-to-BC 系数为 0 时用（现在 kl_bc_start = %s、kl_bc_end = %s）' % (n, pc.kl_bc_start, pc.kl_bc_end))
    return n


def ab_floor_eps(step, pc):
    """技能概率下限 ε（ppo.ab_floor，见 model/infer.py: ab_floor_mix）。ab_floor_until = 0：一直是 ab_floor；
    否则 ab_floor_hold_until 步之前是 ab_floor，之后线性降到 ab_floor_until 步时的 0（两个都是总步数，续训跨段照算）。"""
    e0 = float(pc.get('ab_floor', 0) or 0)
    s1, s2 = int(pc.get('ab_floor_hold_until', 0) or 0), int(pc.get('ab_floor_until', 0) or 0)
    if not 0 <= e0 < 1:
        raise ValueError('ppo.ab_floor 要在 [0, 1) 里（现在 %s）' % e0)
    if e0 == 0 or s2 <= 0:
        return e0
    if not 0 < s1 < s2:
        raise ValueError('ppo.ab_floor_hold_until（%d）要在 0 和 ab_floor_until（%d）之间' % (s1, s2))
    return e0 if step < s1 else e0 * max(0.0, (s2 - step) / (s2 - s1))


def _actor(cfg, slot_names, spec, wnames, wlock, n_params, q_ready, q_free, q_ctl, start_step, aid=0):
    import torch
    from ..env import masks as MK
    from ..env import tables
    from ..env.reward import shaping_coef
    from ..env.vec_env import CTL, VecEnv
    from ..model.infer import LeagueEngine, LeagueEngineFused
    from ..model.net import Net, load_weights
    from .league import HistSlots, episode_spec, spec_pool
    pc = cfg.ppo
    W, T = pc.workers, pc.steps_per_env
    opp_w = float(cfg.league.get('self_opp_weight', 0.5))
    bc_every = bc_ref_every(cfg)
    slots = [Slot(spec, names=n) for n in slot_names]
    wbox = WeightBox(n_params, name=wnames, lock=wlock)
    dev = 'cuda'
    net = Net(cfg).to(dev).eval()
    net.ab_mix = float(pc.get('ab_floor', 0) or 0) > 0               # 技能概率下限：录图前定下来（之后只原地改 net.ab_eps）
    ver = wbox.read_into(net)
    bst = torch.load(os.path.join(ROOT, pc.bc_ckpt), map_location=dev, weights_only=False)
    bc_net = Net(cfg).to(dev).eval()
    load_weights(bc_net, bst['model'], grow=True)                     # grow：主模型扩过容（model/grow.py）时，小结构的存档按同一个函数补成现在的结构
    K = max(1, int(cfg.league.get('hist_slots', 4)))
    hist_nets = []
    for _ in range(K):
        hn = Net(cfg).to(dev).eval()
        load_weights(hn, bst['model'], grow=True)                     # 没收到快照前先用 BC 占位
        hist_nets.append(hn)
    hs = HistSlots(K, cfg.league.get('hist_games_per_snap', 4))
    spec_dk = [None] * K                                              # 各槽：专项剥削者的卡组池 (卡组列表, 概率)；普通快照 None
    spec_rng = np.random.RandomState(cfg.seed + start_step % 100000 + 31 * aid)
    flat, card_cls = MK.all_masks()
    for m in [net, bc_net] + hist_nets:
        m.to(memory_format=torch.channels_last).half()                # 推理整网半精度（权重读入时自动转换）
    Eng = LeagueEngineFused if os.environ.get('CR_INFER_LEGACY') else LeagueEngine     # 2026-10-18 新版：分块录图、位置头只算出牌的行；CR_INFER_LEGACY=1 用旧版
    eng = Eng(net, bc_net, hist_nets, flat, card_cls, dev, **({} if Eng is LeagueEngineFused else dict(bc_ref_graph=bc_every == 1)))
    torch.manual_seed(cfg.seed + start_step % 100000 + 7919 * aid)    # 多个采样进程：动作采样和对局抽样各用各的种子
    venv = VecEnv(cfg, n_workers=W, episode_spec=episode_spec(cfg), seed=cfg.seed + start_step % 100000 + 100003 * aid)
    obs = venv.reset()
    keys = eng.keys                                                   # 网络要的观测字段（含可选的单位列表）
    H = net.H
    lim = dict(gi=tables.n_units(), hand_id=tables.n_cards(), deck_id=tables.n_cards(), opp_id=tables.n_cards(), ab_id=tables.n_cards(),
               pend_id=tables.n_cards(), last_id=tables.n_cards(), hand_form=3, deck_form=3, opp_form=3, tstate=16, pl=2)
    if 'priv_id' in keys:
        lim.update(priv_id=tables.n_cards(), priv_form=3)
    n_bad = [0]

    if 'ent_id' in keys:
        lim['ent_id'] = max(tables.n_units(), 576)                    # 两列（单位编号、格子）分别再查一遍，见下面

    def _check(o, who, t):
        """送进 GPU 之前查一遍编号范围（越界的嵌入/查表在 GPU 上是非法访问）：越界就记日志并截断。"""
        if 'ent_id' in o:
            np.clip(o['ent_id'][..., 0], 0, tables.n_units() - 1, out=o['ent_id'][..., 0])
            np.clip(o['ent_id'][..., 1], 0, 575, out=o['ent_id'][..., 1])
        for k, hi in lim.items():
            v = o[k]
            if v.size and (v.min() < 0 or v.max() >= hi):
                n_bad[0] += 1
                if n_bad[0] <= 20:
                    print('[rollout] 观测越界 %s.%s：范围 [%d, %d]，上限 %d（t=%d）' % (who, k, v.min(), v.max(), hi, t), file=sys.stderr, flush=True)
                np.clip(v, 0, hi - 1, out=v)

    h = np.zeros((W, 2, H), np.float32)                               # 当前策略各行的 LSTM 状态（按玩家）
    c = np.zeros((W, 2, H), np.float32)
    hx = np.zeros((W, H), np.float32)                                 # 历史快照对手的状态（只在对手一方）
    cx = np.zeros((W, H), np.float32)
    ep_ret = np.zeros((W, 2), np.float64)
    ep_len = np.zeros(W, np.int64)
    ep_hist = np.full(W, -1, np.int64)                                # 这局对手用的快照编号（开局绑定槽时记下，整局不变）
    groups = [np.arange(0, W, 2), np.arange(1, W, 2)] if W > 1 else [np.arange(1)]
    step = start_step
    coef = None
    last_hist = [-1]                                                  # 最近换上的快照编号（日志/看板用）
    me = CTL['agent']

    def install(todo):
        """把 HistSlots.refills() 给出的快照装进对应的槽（那个槽手上没有对局，图里引用的地址不变，原地拷贝）。"""
        for i, sid, path in todo:
            if path is not None and sid != hs.ids[i]:
                sd = torch.load(path, map_location='cpu', weights_only=False)['model']   # 先放内存，不在显卡上临时分配
                load_weights(hist_nets[i], sd, grow=True)                # 扩容之前的旧快照：补成现在的结构（函数不变）
                last_hist[0] = sid
                spec_dk[i] = spec_pool(path)
            hs.installed(i, sid, path)
    while True:
        si = q_free.get()
        if si == 'stop':
            venv.close()
            return
        # 控制消息（学习进程在放回空槽之前发出）：历史快照候选、停止
        try:
            while True:
                m = q_ctl.get_nowait()
                if m[0] == 'stop':
                    venv.close()
                    return
                if m[0] == 'hist':
                    hs.set_picks(m[1])                                # [(快照编号, 路径)]，覆盖没用完的旧候选
                if m[0] == 'decks':
                    venv.set_opp_p(m[1])                              # 对手卡组抽样概率（卡组 PFSP，league.deck_pfsp）
        except queue.Empty:
            pass
        install(hs.refills())                                         # 开头先补：还在用 BC 占位的槽、打完了等着换的槽
        B = slots[si].a
        if int(wbox.ver[0]) != ver:
            ver = wbox.read_into(net)
        c_new = shaping_coef(step, cfg.reward)
        if coef is None or abs(c_new - coef) > 1e-4:
            venv.set_attr('reward_coef', c_new)
            coef = c_new
        ab_eps = ab_floor_eps(step, pc)
        if net.ab_mix:
            net.ab_eps.fill_(ab_eps)
        B['valid'][:] = False
        B['trunc'][:] = False
        st = dict(steps=0, env_wait=0.0, gpu_time=0.0, episodes=[], errors=0, ver=ver, coef=coef, aid=aid, ab_eps=ab_eps)
        t0 = time.time()
        busy0, bot0 = venv.busy()
        cg0 = cgroup_cpu_stat()
        acts = np.zeros((W, 2, 2), np.int32)

        def launch(ws, t):
            ctl = venv.ctl
            cw, cp, ow = [], [], []                                    # 当前策略的行（环境, 玩家）、bc 对手的 (环境, 玩家)
            xw = [[] for _ in range(K)]                                # 各历史快照槽控制的 (环境, 玩家)
            for w in ws:
                for p in (0, 1):
                    k = ctl[w, p]
                    if k == me or k == CTL['self']:
                        cw.append(w)
                        cp.append(p)
                    elif k == CTL['bc']:
                        ow.append((w, p))
                    elif k == CTL['hist']:
                        i = hs.env_slot.get(w)
                        if i is None:                                  # 这局刚开：绑定一个槽，整局不换
                            i = hs.assign(w)
                            ep_hist[w] = hs.ids[i]
                            if spec_dk[i] is not None:                    # 专项剥削者：这局换成它专长的卡组（还没出过动作，重开）
                                dk, dp = spec_dk[i]
                                j = int(spec_rng.choice(len(dk), p=dp))
                                venv.redeck(w, p, dk[j][0], dk[j][1], 'spec%d' % j)
                        xw[i].append((w, p))
            cw, cp = np.array(cw, np.int64), np.array(cp, np.int64)
            sub = {k: obs[k][cw, cp] for k in keys}
            _check(sub, 'cur', t)
            for k in keys:
                B['obs_' + k][t, cw, cp] = sub[k]
            B['h_in'][t, cw, cp] = h[cw, cp]
            B['c_in'][t, cw, cp] = c[cw, cp]
            B['valid'][t, cw, cp] = True
            B['lw'][t, cw, cp] = np.where(ctl[cw, cp] == me, 1.0, opp_w)
            osub = None
            if ow:
                o_w, o_p = np.array([a for a, _ in ow]), np.array([b for _, b in ow])
                osub = {k: obs[k][o_w, o_p] for k in keys}
                _check(osub, 'bc', t)
            xidx = [None] * K                                          # 各槽的 (环境数组, 玩家数组)
            xsub, xh, xc = [None] * K, [None] * K, [None] * K
            for i in range(K):
                if xw[i]:
                    x_w, x_p = np.array([a for a, _ in xw[i]]), np.array([b for _, b in xw[i]])
                    xidx[i] = (x_w, x_p)
                    xsub[i] = {k: obs[k][x_w, x_p] for k in keys}
                    _check(xsub[i], 'hist%d' % i, t)
                    xh[i], xc[i] = hx[x_w], cx[x_w]
            g0 = time.time()
            try:
                r = eng.run(sub, h[cw, cp], c[cw, cp], osub, xsub, xh, xc, want_bc=t % bc_every == 0)
            except Exception:
                # 现场存档：推理出错时把这一步的全部输入存下来，离线复现
                dump = {'a_' + k: v for k, v in sub.items()}
                dump.update({'o_' + k: v for k, v in (osub or {}).items()})
                for i in range(K):
                    if xsub[i] is not None:
                        dump.update({'x%d_%s' % (i, k): v for k, v in xsub[i].items()})
                        dump.update({'h_x%d' % i: xh[i], 'c_x%d' % i: xc[i]})
                dump.update(h_a=h[cw, cp], c_a=c[cw, cp], t=np.array(t), hist_ids=np.array(hs.ids), cw=cw, cp=cp)
                np.savez(os.path.join(ROOT, 'logs', 'infer_fail_%d.npz' % os.getpid()), **dump)
                raise
            st['gpu_time'] += time.time() - g0
            h[cw, cp], c[cw, cp] = r['h'], r['c']
            B['sel'][t, cw, cp] = r['sel']
            B['cell'][t, cw, cp] = r['cell']
            B['logp'][t, cw, cp] = r['logp']
            B['value'][t, cw, cp] = r['value']
            B['bc_card_lp'][t, cw, cp] = r['bc_card_lp']
            B['bc_place_lp'][t, cw, cp] = r['bc_place_lp']
            acts[cw, cp, 0] = r['sel']
            acts[cw, cp, 1] = r['cell']
            if ow:
                acts[o_w, o_p, 0] = r['opp_sel']
                acts[o_w, o_p, 1] = r['opp_cell']
            for i in range(K):
                if xidx[i] is not None:
                    x_w, x_p = xidx[i]
                    acts[x_w, x_p, 0] = r['hist_sel'][i]
                    acts[x_w, x_p, 1] = r['hist_cell'][i]
                    hx[x_w], cx[x_w] = r['hist_h'][i], r['hist_c'][i]
            venv.step_async(list(ws), acts[ws])

        def collect(ws, t):
            e0 = time.time()
            infos = venv.step_wait(list(ws))
            st['env_wait'] += time.time() - e0
            rew = venv._arr['rew']
            done = venv._arr['done']
            B['rew'][t, ws] = rew[ws]
            B['done'][t, ws] = done[ws]
            ep_ret[ws] += rew[ws]
            ep_len[ws] += 1
            for w, inf in zip(ws, infos):
                if inf is None:
                    continue
                h[w] = 0
                c[w] = 0
                hx[w] = 0
                cx[w] = 0
                ctls = inf.get('controllers') or ['?', '?']
                decks = inf.get('deck_names') or ['', '']
                ag = 0 if ctls[0] == 'agent' else 1
                opp = ctls[1 - ag]
                if opp == 'bc':
                    opp = 'bc_main' if decks[1 - ag] == 'main' else 'bc_pool'
                win = inf.get('winner')
                res = None if inf.get('error') else (0.5 if win in (None, -1) else float(win == ag))
                if inf.get('error'):
                    st['errors'] += 1
                    B['trunc'][t, w] = True                            # 引擎出错：这局被截断（不是真的终局），GAE 用价值自举
                cr = inf.get('crowns') or [0, 0]
                es = inf.get('stats') or {}
                abu = None                                             # 按技能单位名：{名字: [可以按的步数, 发动次数]}
                if es.get('ab_by') is not None:
                    rb, ub = es['ab_ready_by'][ag], es['ab_by'][ag]
                    abu = {nm: [int(rb.get(nm, 0)), int(ub.get(nm, 0))] for nm in set(rb) | set(ub)}
                st['episodes'].append(dict(opp=opp, result=res, ret=float(ep_ret[w, ag]), len=int(ep_len[w]), crowns=[int(cr[ag]), int(cr[1 - ag])],
                                           deck=decks[ag], opp_deck=decks[1 - ag], hist=int(ep_hist[w]) if opp == 'hist' else None,
                                           ab=int((es.get('abilities') or [0, 0])[ag]),
                                           ab_ready=int(es['ab_ready'][ag]) if es.get('ab_ready') else None, ab_units=abu))
                ep_ret[w] = 0.0
                ep_len[w] = 0
                ep_hist[w] = -1
                if hs.release(w) is not None:
                    install(hs.refills())                              # 这个槽可能刚打完最后一局，换上新快照

        for g in groups:
            launch(g, 0)
        for t in range(T):
            for g in groups:
                collect(g, t)
                if t + 1 < T:
                    launch(g, t + 1)
        # 自举价值：第 T 步之后的观测（不推进 LSTM 状态、不出动作）
        ctl = venv.ctl
        cw, cp = np.nonzero((ctl == me) | (ctl == CTL['self']))
        sub = {k: obs[k][cw, cp] for k in keys}
        B['last_value'][:] = 0.0
        B['last_value'][cw, cp] = eng.run(sub, h[cw, cp], c[cw, cp], want_bc=bc_every == 1)['value']   # 结果不用；默认（每步都算 BC）时和以前一样走带 BC 块的图，不多录图
        B['last_valid'][:] = False
        B['last_valid'][cw, cp] = True
        st['steps'] = int(B['valid'].sum())
        st['env_steps'] = T * W
        st['hist_id'] = last_hist[0]
        st['hist_ids'] = list(hs.ids)
        st['secs'] = time.time() - t0
        busy1, bot1 = venv.busy()
        st['env_busy'] = float((busy1 - busy0).sum()) / (W * st['secs'])       # 环境进程平均有多少时间在干活（其余在等采样进程）
        st['env_bot'] = float((bot1 - bot0).sum()) / (W * st['secs'])          # 其中规则/随机 bot 出动作占的比例
        cg1 = cgroup_cpu_stat()
        if cg0 and cg1:
            st['cg_cores'] = (cg1['usage_usec'] - cg0['usage_usec']) / 1e6 / st['secs']
            st['cg_throttled'] = (cg1.get('nr_throttled', 0) - cg0.get('nr_throttled', 0)) / max(1, cg1.get('nr_periods', 0) - cg0.get('nr_periods', 0))
        step += st['steps']
        q_ready.put((si, st))
