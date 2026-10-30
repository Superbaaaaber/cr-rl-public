"""cr_ai/env/vec_env.py — 多进程向量化环境：每个 worker 进程一个引擎/一局（引擎内存是进程级全局的），观测/动作/奖励走共享内存。

    venv = VecEnv(cfg, n_workers=8, episode_spec={'controllers': [['agent', 'rule', 1]]})
    obs = venv.reset()                          # dict：key → ndarray [W, 2, ...]（共享内存的视图，下一次 step 会被覆盖，需要就拷贝）
    obs, rew, done, infos = venv.step(actions)   # actions：int32 [W, 2, 2]（sel, cell）；只有 agent 控制的一方被采用
    venv.agent_mask                              # bool [W, 2]：当前这局哪一方由 agent 控制（机器人控制的一方由 worker 自己出动作）
    venv.ctl                                     # int8 [W, 2]：控制方编码 CTL（agent / bc / rule / random）；agent、bc 两种都由主进程推理后写进 act
    venv.set_attr('reward_coef', c)              # 改所有 worker 里 GameEnv 的属性（训练器按步数退火塑形系数）
异步：step_async(ids, actions) / step_wait(ids) —— 把环境分成两组交替推理，CPU 和 GPU 不互相等待。
一局结束 worker 自动开下一局：返回的 obs 是新一局的第一个观测，done=True、rew 是上一局最后一步的奖励，infos[w] 里有上一局的摘要。
引擎陷阱（EngineError）：这一局作废（done=True、infos[w]['error']），worker 重新装载引擎后继续。
"""
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np

from ..config import ROOT

CTL = {'agent': 0, 'bc': 1, 'rule': 2, 'random': 3, 'self': 4, 'hist': 5}   # self：当前策略当对手（自我对弈）；hist：历史快照


def _set_idle():
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")                                   # 私有实例：不改全局 windll 的函数签名
        k32.GetCurrentProcess.restype = ctypes.c_void_p                  # 伪句柄 −1 必须按 64 位传，否则调用静默失败
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x40)
    except Exception:                                          # noqa
        pass


def _attach(specs, names):
    shms, arrs = {}, {}
    for k, (shape, dtype) in specs.items():
        shm = shared_memory.SharedMemory(name=names[k])
        shms[k] = shm
        arrs[k] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shms, arrs


def _worker(wid, conn, cfg, specs, names, episode_spec, seed):
    sys.path.insert(0, ROOT)
    sys.setrecursionlimit(20000)
    if cfg.vec_env.get('priority') == 'idle':
        _set_idle()
    from .game_env import GameEnv
    from .episodes import EpisodeSampler
    from .engine_adapter import EngineError
    from ..bots.random_bot import RandomBot
    from ..bots.rule_bot import RuleBot
    shms, A = _attach(specs, names)
    env = GameEnv(cfg)
    sampler = EpisodeSampler(cfg, episode_spec, seed=seed * 1000 + wid)
    rng = np.random.RandomState(seed * 7919 + wid)
    st = {}

    def write_obs(obs):
        for p in (0, 1):
            for k, v in obs[p].items():
                A['obs_' + k][wid, p] = v

    def new_episode(given=None):
        while True:
            ep = given if given is not None else sampler.sample()
            try:
                obs = env.reset(ep['seed'], ep['decks'], ep['forms'], tower_troop=ep['tower_troop'])
            except EngineError as e:
                st['errors'] = st.get('errors', 0) + 1
                given = None                                    # 指定的这局开不了：改用正常抽样
                continue
            bots = {}
            for p, c in enumerate(ep['controllers']):
                if c == 'rule':
                    bots[p] = RuleBot(np.random.RandomState(rng.randint(1 << 30)))
                elif c == 'random':
                    bots[p] = RandomBot(np.random.RandomState(rng.randint(1 << 30)), p_play=episode_spec.get('random_p_play', 0.1))
            st.update(ep=ep, bots=bots, obs=obs, ret=[0.0, 0.0], steps=0, t0=time.time())
            A['agent_mask'][wid] = [c == 'agent' for c in ep['controllers']]
            A['ctl'][wid] = [CTL[c] for c in ep['controllers']]
            write_obs(obs)
            return

    try:
        while True:
            while not conn.poll(5.0):                           # 父进程（主进程/采样进程）死了就退出，免得留下占内存的孤儿进程
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    raise EOFError
            cmd = conn.recv()
            if cmd == 'close':
                break
            if isinstance(cmd, tuple) and cmd[0] == 'set':
                setattr(env, cmd[1], cmd[2])
                conn.send(None)
                continue
            if isinstance(cmd, tuple) and cmd[0] == 'opp_p':            # 对手卡组抽样概率（不回复：可以夹在进行中的 step 之间发）
                sampler.set_opp_p(cmd[1])
                continue
            if isinstance(cmd, tuple) and cmd[0] == 'redeck':
                # 这局还没出过动作：把第 p 方换成指定卡组重开（种子、塔兵、另一方卡组不变）。专项剥削者当对手时用（rollout）
                _, p, cards, forms, name = cmd
                ep = st['ep']
                decks, fms, names = list(ep['decks']), list(ep['forms']), list(ep['deck_names'])
                decks[p], fms[p], names[p] = list(cards), dict(forms), name
                try:
                    obs = env.reset(ep['seed'], decks, fms, tower_troop=ep['tower_troop'])
                except EngineError:
                    obs = env.reset(ep['seed'], ep['decks'], ep['forms'], tower_troop=ep['tower_troop'])   # 换不了就照原样重开
                    decks, fms, names = ep['decks'], ep['forms'], ep['deck_names']
                ep.update(decks=decks, forms=fms, deck_names=names)
                st.update(obs=obs, ret=[0.0, 0.0], steps=0, t0=time.time())
                write_obs(obs)
                conn.send(None)
                continue
            if isinstance(cmd, tuple) and cmd[0] == 'episode':
                # 按给定设定（种子、双方卡组/形态/塔兵、控制方）重开这局。配对评测用（arena paired：同一对卡组换边各打一局）
                new_episode(dict(cmd[1]))
                A['rew'][wid] = 0
                A['done'][wid] = False
                conn.send(None)
                continue
            if cmd == 'reset':
                new_episode()
                A['rew'][wid] = 0
                A['done'][wid] = False
                conn.send(None)
                continue
            # step
            s0 = time.perf_counter()
            acts = []
            for p in (0, 1):
                b = st['bots'].get(p)
                acts.append(b.act(env, p, st['obs'][p]) if b is not None else (int(A['act'][wid, p, 0]), int(A['act'][wid, p, 1])))
            A['bot_s'][wid] += time.perf_counter() - s0
            info = None
            try:
                obs, rew, done, inf = env.step(acts)
            except EngineError as e:
                obs, rew, done, inf = None, [0.0, 0.0], True, dict(error=str(e)[:200])
            st['steps'] += 1
            for p in (0, 1):
                st['ret'][p] += rew[p]
            A['rew'][wid] = rew
            A['done'][wid] = done
            if done:
                ep = st['ep']
                info = dict(winner=inf.get('winner'), crowns=inf.get('crowns'), tick=inf.get('tick'), error=inf.get('error'), steps=st['steps'],
                            ret=list(st['ret']), controllers=ep['controllers'], deck_names=ep['deck_names'], tower_troop=ep['tower_troop'],
                            stats={k: list(v) for k, v in env.stats.items()} if obs is not None else None,
                            void=[dict(env.aq[p].void_reasons) for p in (0, 1)], secs=time.time() - st['t0'], seed=ep['seed'])
                for p in (0, 1):
                    env.aq[p].void_reasons.clear()
                new_episode()
            else:
                st['obs'] = obs
                write_obs(obs)
            A['busy_s'][wid] += time.perf_counter() - s0
            conn.send(info)
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:                                           # noqa
        conn.send(dict(fatal=traceback.format_exc()))
    finally:
        for s in shms.values():
            s.close()


class VecEnv:
    def __init__(self, cfg, n_workers=None, episode_spec=None, seed=None):
        from .game_env import obs_spec
        self.cfg = cfg
        self.W = int(n_workers or cfg.vec_env.workers)
        W = self.W
        threads = str(cfg.vec_env.get('omp_threads', 1))
        for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[k] = threads                              # 子进程继承：OpenBLAS 不再按 12 线程预留缓冲区
        specs = {'obs_' + k: ((W, 2) + tuple(shape), dt) for k, (shape, dt) in obs_spec(cfg).items()}
        specs.update(act=((W, 2, 2), np.int32), rew=((W, 2), np.float32), done=((W,), np.bool_), agent_mask=((W, 2), np.bool_), ctl=((W, 2), np.int8),
                     busy_s=((W,), np.float64), bot_s=((W,), np.float64))   # 各 worker 累计：处理 step 的总耗时、其中规则/随机 bot 出动作的耗时
        self.specs = specs
        self._shms, self._arr, names = {}, {}, {}
        for k, (shape, dt) in specs.items():
            n = int(np.prod(shape)) * np.dtype(dt).itemsize
            shm = shared_memory.SharedMemory(create=True, size=max(1, n))
            self._shms[k] = shm
            names[k] = shm.name
            self._arr[k] = np.ndarray(shape, dtype=dt, buffer=shm.buf)
            self._arr[k][...] = 0
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        seed = cfg.seed if seed is None else seed
        for w in range(W):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(w, b, cfg, specs, names, dict(episode_spec or {}), seed), daemon=True)
            pr.start()
            self.conns.append(a)
            self.procs.append(pr)
        self.obs = {k[4:]: v for k, v in self._arr.items() if k.startswith('obs_')}

    @property
    def agent_mask(self):
        return self._arr['agent_mask']

    @property
    def ctl(self):
        return self._arr['ctl']

    def busy(self):
        """各 worker 累计耗时 (处理 step 的总秒数 [W], 其中 bot 出动作的秒数 [W])；两次读数相减 ÷ 墙钟时间 = 环境进程的忙碌比例。"""
        return self._arr['busy_s'].copy(), self._arr['bot_s'].copy()

    def set_attr(self, name, value):
        for c in self.conns:
            c.send(('set', name, value))
        for w in range(self.W):
            self._recv(w)

    def set_opp_p(self, p):
        """下发对手卡组抽样概率（卡组 PFSP）。worker 不回复，随时可发：管道有序，worker 在处理下一个命令前先换上。"""
        p = np.asarray(p, np.float64)
        for c in self.conns:
            c.send(('opp_p', p))

    def redeck(self, w, p, cards, forms, name):
        """环境 w 这局刚开、还没出动作时，把第 p 方换成指定卡组重开（同步等它重开完；之后 self.obs 里是新的观测）。"""
        self.conns[w].send(('redeck', int(p), list(cards), dict(forms or {}), str(name)))
        self._recv(w)

    def set_episode(self, w, ep):
        """环境 w 改开指定的一局（ep 的格式同 EpisodeSampler.sample()；同步等它开好，之后 self.obs 里是新的观测）。"""
        self.conns[w].send(('episode', dict(ep)))
        self._recv(w)

    def _recv(self, w):
        r = self.conns[w].recv()
        if isinstance(r, dict) and 'fatal' in r:
            raise RuntimeError('worker %d 崩溃：\n%s' % (w, r['fatal']))
        return r

    def reset(self):
        for c in self.conns:
            c.send('reset')
        for w in range(self.W):
            self._recv(w)
        return self.obs

    def step_async(self, ids, actions=None):
        if actions is not None:
            self._arr['act'][ids] = actions
        for w in ids:
            self.conns[w].send('step')

    def step_wait(self, ids):
        return [self._recv(w) for w in ids]

    def step(self, actions):
        ids = list(range(self.W))
        self.step_async(ids, actions)
        infos = self.step_wait(ids)
        return self.obs, self._arr['rew'], self._arr['done'], infos

    def close(self):
        for c in self.conns:
            try:
                c.send('close')
            except Exception:                                   # noqa
                pass
        for p in self.procs:
            p.join(timeout=10)
        for s in self._shms.values():
            s.close()
            s.unlink()
