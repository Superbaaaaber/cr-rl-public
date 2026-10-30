"""cr_ai/train/league.py — 对手采样（episode_spec）和联赛管理（历史快照池、PFSP、Elo）。

league.mode：
  fixed（M3）：对手只用固定池——冻结的 BC 网络（各卡组）+ 规则 AI；
  league（M4）：当前策略（自我对弈，self）+ 历史快照（hist，PFSP）+ 固定锚点（bc、rule），比例见 league.opponents。
agent 一方用主角卡组的概率 league.agent_main_prob（M4 多卡组训练：其余从对手池按使用量抽，299 套）。

快照池（League）：
  - 每 snapshot_every_steps 存一个快照加入池（含 M3 的快照，league.seed_snapshots 指定的目录），最多 max_snapshots 个，满了丢最旧的；
  - PFSP：每批按权重 ∝ (1 − 当前策略对该快照的胜率)² 抽 league.hist_slots 个候选快照发给采样进程；胜率用带衰减的计数 + 先验（没打过的按 50%）；
  - 采样进程（HistSlots）同时加载 hist_slots 个快照，每局开局时绑定一个槽、整局由同一个快照打；
    换快照只换手上没有进行中对局的槽（以前每批直接换掉唯一的历史网络，一局横跨约 3 批，被记账的快照平均只打了约 1/4 局）；
  - Elo：BC（各卡组平均）锚定为 0；当前策略的分数按它对 BC / 历史快照的每局结果更新；快照入池时冻结为当时当前策略的分数。
  - 剥削者（league.exploiter_dir，可选）：tools/league_loop.py 训出的剥削者快照放在这个目录（*.pt + 同名 .json），
    这里定期扫描、作为 kind='exp' 的快照加入同一个 PFSP 池（主模型打不过它们，PFSP 自然会多选）；最多 max_exploiters 个，满了丢最旧的。
    专项剥削者（.json 里有 deck_pool，2026-10-20）：轮到它当对手时，这局它一方换成它专长的卡组（spec_pool、rollout 里的 redeck）。

exploit 模式（ExploitTarget）：剥削者训练用，对手固定为 league.target（主模型的冻结副本），只记对它的胜率。

卡组 PFSP（DeckPFSP，league.deck_pfsp = λ > 0 时打开，2026-10-20）：按对手卡组统计主模型的胜率（带衰减，向全局平均收缩），
对手一方抽卡组的概率 = (1 − λ) × 使用量 + λ × 使用量 × ((1 − 胜率)² + 下限) 归一化。一半照旧按使用量（老卡组不会被冷落），
一半往输得多的卡组倾斜。只影响对手一方；agent 自己用的卡组照旧按使用量抽。概率每批随历史快照候选一起发给采样进程。
统计来源 league.deck_pfsp_source（2026-10-24）：all = 全部对局（默认，以前的行为）；hist = 只用对历史快照的对局
（80% 自我对弈时，自我对弈的胜负主要反映两边卡组的强弱，不是“主模型打不好这套卡组”；代价是局数少、统计稀）；
main = 只用 agent 一方是主卡组的对局（只为主卡组挑难打的对手卡组）。
"""
import glob
import json
import os
import re

import numpy as np


def episode_spec(cfg):
    lc = cfg.league
    ctl = []
    for opp, w in lc.opponents.items():
        if w <= 0:
            continue
        ctl.append(['agent', opp, w / 2])
        ctl.append([opp, 'agent', w / 2])
    return dict(controllers=ctl, main_for='agent', opp_main_prob=lc.get('opp_main_prob', 0.0),
                agent_main_prob=lc.get('agent_main_prob', 1.0), tower_troops=lc.get('tower_troops'))


def _step_of(path):
    m = re.search(r'step(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def elo_of(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 400.0 * np.log10(p / (1 - p))


class DeckPFSP:
    """按对手卡组的 PFSP（见文件头）。卡组按牌名（排序后）认，换了卡组池也能接上已有的统计。"""

    def __init__(self, cfg, saved=None):
        from ..config import read_yaml, repo_path
        lc = cfg.league
        self.lam = float(lc.get('deck_pfsp', 0.0) or 0.0)
        self.decay = float(lc.get('deck_stat_decay', 0.998))           # 每批衰减（约 500 批的记忆）
        self.prior = float(lc.get('deck_prior_games', 4.0))           # 向全局平均胜率收缩的伪局数（局数少的卡组别被几局结果带偏）
        self.floor = float(lc.get('deck_pfsp_floor', 0.01))
        self.source = str(lc.get('deck_pfsp_source', 'all') or 'all')
        if self.source not in ('all', 'hist', 'main'):
            raise ValueError('league.deck_pfsp_source 只能是 all / hist / main：%s' % self.source)
        pool = read_yaml(repo_path(cfg.paths.deck_pool))['decks']
        self.keys = ['|'.join(sorted(d['cards'])) for d in pool]
        u = np.array([d['weight'] for d in pool], np.float64)
        self.u = u / u.sum()
        saved = saved or {}
        self.n = np.array([float(saved.get(k, (0.0, 0.0))[0]) for k in self.keys])
        self.w = np.array([float(saved.get(k, (0.0, 0.0))[1]) for k in self.keys])
        self.extra = {k: v for k, v in saved.items() if k not in set(self.keys)}   # 不在当前池里的卡组：原样留着（换回旧池时还在）

    def update(self, episodes):
        self.n *= self.decay
        self.w *= self.decay
        for e in episodes:
            od = e.get('opp_deck') or ''
            if e.get('result') is None or not od.startswith('pool'):
                continue
            if (self.source == 'hist' and e.get('opp') != 'hist') or (self.source == 'main' and e.get('deck') != 'main'):
                continue
            i = int(od[4:])
            if i < len(self.n):
                self.n[i] += 1
                self.w[i] += e['result']

    def winrates(self):
        m = (self.w.sum() + 1.0) / (self.n.sum() + 2.0)
        return (self.w + self.prior * m) / (self.n + self.prior)

    def probs(self):
        g = (1.0 - self.winrates()) ** 2 + self.floor
        t = self.u * g
        return (1.0 - self.lam) * self.u + self.lam * t / t.sum()

    def state(self):
        out = dict(self.extra)
        out.update({k: [round(float(n), 3), round(float(w), 3)] for k, n, w in zip(self.keys, self.n, self.w)})
        return out

    def summary(self):
        wr, p = self.winrates(), self.probs()
        seen = self.n >= 3
        out = dict(deck_tv=float(0.5 * np.abs(p - self.u).sum()), deck_seen=int(seen.sum()))     # deck_tv：抽样分布偏离使用量的程度（0–1）
        if seen.any():
            out.update(deck_min_winrate=float(wr[seen].min()),
                       deck_mean_winrate=float((self.u[seen] * wr[seen]).sum() / self.u[seen].sum()))
        return out

    def hardest(self, k=10):
        """胜率最低的 k 套（至少打过 3 局）：[(牌名…, 胜率, 局数, 使用量, 现在的抽样概率)]。"""
        wr, p = self.winrates(), self.probs()
        idx = [i for i in np.argsort(wr) if self.n[i] >= 3][:k]
        return [dict(cards=self.keys[i], winrate=round(float(wr[i]), 3), games=round(float(self.n[i]), 1),
                     usage=round(float(self.u[i]), 4), prob=round(float(p[i]), 4)) for i in idx]


class League:
    def __init__(self, cfg, run, init_elo=0.0, seed=0):
        self.lc = cfg.league
        self.run = run
        self.path = os.path.join(run, 'league.json')
        self.rng = np.random.RandomState(seed)
        self.snaps = []                      # dict(id, path, step, elo, n, w)
        self.elo = float(init_elo)
        self.next_id = 0
        self.exp_seen = []                   # 已经处理过的剥削者快照（被挤出池的也算，免得重启后又加回来）
        deck_stats = None
        if os.path.exists(self.path):
            d = json.load(open(self.path, encoding='utf-8'))
            self.snaps, self.elo, self.next_id = d['snaps'], d['elo'], d['next_id']
            self.exp_seen = d.get('exp_seen', [])
            deck_stats = d.get('deck_stats')
        self.decks = DeckPFSP(cfg, deck_stats) if float(self.lc.get('deck_pfsp', 0.0) or 0.0) > 0 else None
        if not os.path.exists(self.path):
            self._seed()

    def _seed(self):
        """初始快照：league.seed_snapshots（M3 的 ppo_step*.pt），Elo 按 league.seed_elo（步数 → 分数的已知点，线性插值）。"""
        pat = self.lc.get('seed_snapshots')
        if not pat:
            return
        files = sorted(glob.glob(os.path.join(os.path.dirname(self.run), pat) if not os.path.isabs(pat) else pat), key=_step_of)
        files = [f for f in files if _step_of(f) >= 0]
        pts = sorted((int(k), float(v)) for k, v in (self.lc.get('seed_elo') or {}).items())
        for f in files:                                                    # 全部加进来，由 _prune_main 按保留规则挑
            st = _step_of(f)
            e = float(np.interp(st, [p[0] for p in pts], [p[1] for p in pts])) if pts else 0.0
            self.add(f, st, elo=e, save=False)
        self.save()

    def save(self):
        tmp = self.path + '.tmp'
        d = dict(snaps=self.snaps, elo=self.elo, next_id=self.next_id, exp_seen=self.exp_seen)
        if self.decks is not None:
            d.update(deck_stats=self.decks.state(), deck_hardest=self.decks.hardest())
        json.dump(d, open(tmp, 'w', encoding='utf-8'), indent=0, ensure_ascii=False)
        os.replace(tmp, self.path)

    def deck_probs(self):
        """对手卡组抽样概率（卡组 PFSP 打开时），否则 None。"""
        return None if self.decks is None else self.decks.probs()

    def add(self, path, step, elo=None, save=True, kind='main', info=None):
        s = dict(id=self.next_id, path=path, step=int(step), elo=float(self.elo if elo is None else elo), n=0.0, w=0.0)
        if kind != 'main':
            s.update(kind=kind, info=info or {})
        self.snaps.append(s)
        self.next_id += 1
        self._prune_main()
        mine = [x for x in self.snaps if x.get('kind') == 'exp']
        for x in mine[:max(0, len(mine) - self.lc.get('max_exploiters', 10))]:     # 剥削者满了丢最旧的
            self.snaps.remove(x)
        if save:
            self.save()

    def _prune_main(self):
        """主模型快照的保留规则：
        - 配了 league.recent_snapshots：最近 recent 个全留；更早的最多留 league.old_snapshots 个，按几何间隔挑（越老越稀）：
          每次去掉“去掉后留下的空档 ÷ 它的年龄”最小的那个；最老的一个总是留着（保住最长的时间跨度，防止转圈克制看不出来）；
        - 没配：最多 max_snapshots 个，满了丢最旧的（旧规则）。"""
        lc = self.lc
        mine = sorted((x for x in self.snaps if x.get('kind', 'main') == 'main'), key=lambda x: x['step'])
        rec = lc.get('recent_snapshots')
        if rec is None:
            for x in mine[:max(0, len(mine) - lc.max_snapshots)]:
                self.snaps.remove(x)
            return
        n_old = max(0, len(mine) - int(rec))
        cap = max(0, int(lc.get('old_snapshots', 0)))
        if n_old <= cap:
            return
        old, now = mine[:n_old], mine[-1]['step']
        nxt = mine[n_old]['step']                                          # 最老的“最近”快照：最后一个旧快照右边的邻居
        while len(old) > cap:
            if cap == 0:
                victim = old[0]
            else:
                steps = [x['step'] for x in old] + [nxt]
                score = [(steps[j + 1] - steps[j - 1]) / max(1, now - steps[j]) for j in range(1, len(old))]
                victim = old[1 + int(np.argmin(score))]
            old.remove(victim)
            self.snaps.remove(victim)

    def scan_exploiters(self):
        """把 league.exploiter_dir 里新出现的剥削者快照（*.json 描述 + *.pt）加入池。返回新加入的个数。"""
        d = self.lc.get('exploiter_dir')
        if not d:
            return 0
        d = d if os.path.isabs(d) else os.path.join(os.path.dirname(self.run), d)
        known = set(self.exp_seen)
        new = 0
        for jf in sorted(glob.glob(os.path.join(d, '*.json'))):
            pt = jf[:-5] + '.pt'
            if pt in known or not os.path.exists(pt):
                continue
            info = json.load(open(jf, encoding='utf-8'))
            # 它对“当时的主模型”的胜率 → 比当前分数高多少（主模型这段时间也在变，只是个起点，之后按对局更新的胜率决定 PFSP 权重）
            self.add(pt, info.get('step', 0), elo=self.elo + elo_of(info.get('winrate', 0.5)), save=False, kind='exp', info=info)
            self.exp_seen.append(pt)
            new += 1
        if new:
            self.save()
        return new

    def winrate(self, s):
        return (s['w'] + 1.0) / (s['n'] + 2.0)                         # 先验：没打过的按 50%

    def pick(self):
        """PFSP：权重 ∝ (1 − 胜率)²，加一点下限，保证每个快照都有机会。→ (id, path) 或 None。"""
        if not self.snaps:
            return None
        w = np.array([(1.0 - self.winrate(s)) ** 2 + self.lc.get('pfsp_floor', 0.01) for s in self.snaps])
        i = int(self.rng.choice(len(self.snaps), p=w / w.sum()))
        return self.snaps[i]['id'], self.snaps[i]['path']

    def update(self, episodes):
        """一批对局的结果 → 快照胜率计数（带衰减）+ Elo。"""
        decay = self.lc.get('stat_decay', 0.995)
        if self.decks is not None:
            self.decks.update(episodes)
        for s in self.snaps:
            s['n'] *= decay
            s['w'] *= decay
        by_id = {s['id']: s for s in self.snaps}
        K = self.lc.get('elo_k', 4.0)
        for e in episodes:
            if e.get('result') is None:
                continue
            r = e['result']
            if e['opp'] == 'hist' and e.get('hist') in by_id:
                s = by_id[e['hist']]
                s['n'] += 1
                s['w'] += r
                ro = s['elo']
            elif e['opp'] in ('bc_main', 'bc_pool'):
                ro = 0.0
            else:
                continue
            exp = 1.0 / (1.0 + 10 ** ((ro - self.elo) / 400.0))
            self.elo += K * (r - exp)
        self.save()

    def summary(self):
        if not self.snaps:
            return dict(n_snapshots=0, elo=self.elo, **(self.decks.summary() if self.decks is not None else {}))
        main = [s for s in self.snaps if s.get('kind', 'main') == 'main']
        exp = [s for s in self.snaps if s.get('kind') == 'exp']
        wr = [self.winrate(s) for s in main] or [0.5]
        out = dict(n_snapshots=len(main), elo=self.elo, pfsp_min_winrate=float(min(wr)), pfsp_mean_winrate=float(np.mean(wr)))
        if exp:
            out.update(n_exploiters=len(exp), exp_mean_winrate=float(np.mean([self.winrate(s) for s in exp])),
                       exp_min_winrate=float(min(self.winrate(s) for s in exp)))
        if self.decks is not None:
            out.update(self.decks.summary())
        return out


_SPEC_CACHE = {}


def spec_pool(path):
    """快照旁的 .json 里有 deck_pool（专项剥削者，league_loop --exp-variants 训出）→ (卡组列表 [(牌, 形态)], 概率)；否则 None。
    主模型联赛里轮到它当对手时，rollout 把这局对手一方换成这个池里的卡组——它只会用这些卡组，随机卡组上它没有专长。"""
    if not path:
        return None
    jf = os.path.splitext(path)[0] + '.json'
    if jf in _SPEC_CACHE:
        return _SPEC_CACHE[jf]
    out = None
    try:
        dp = json.load(open(jf, encoding='utf-8')).get('deck_pool') if os.path.exists(jf) else None
        if dp:
            from ..config import read_yaml, repo_path
            decks = read_yaml(repo_path(dp))['decks']
            w = np.array([d.get('weight', 1) for d in decks], np.float64)
            out = ([(list(d['cards']), dict(d.get('forms') or {})) for d in decks], w / w.sum())
    except Exception:                                    # noqa  json/池文件坏了就当普通快照
        out = None
    _SPEC_CACHE[jf] = out
    return out


class HistSlots:
    """采样进程里的历史快照槽（纯簿记，不碰显卡）：每局开局时绑定一个槽，整局由同一个快照打。

    - 开局（assign）：从“接新局”的槽里挑进行中局数最少的；
    - 一个槽累计接满 games_per_snap 局后不再接新局（前提是还有别的槽在接，保证随时至少有一个槽可用），
      等它手上的局都打完（release），refills() 给出要换上的快照：学习进程最近一批发来的 PFSP 候选（set_picks），没有候选就沿用原快照；
    - 调用方真正加载完权重后调用 installed()，槽重新开始接新局。
    槽里的快照在有进行中对局时永远不变，所以每局的结果都记在真正打这局的快照上。
    """

    def __init__(self, k, games_per_snap):
        self.k = int(k)
        self.per = max(1, int(games_per_snap))
        self.ids = [-1] * self.k                 # 槽里的快照编号（-1 = 还没收到快照，用 BC 占位）
        self.paths = [None] * self.k
        self.active = [set() for _ in range(self.k)]   # 正在这个槽的快照手上的环境
        self.started = [0] * self.k              # 换上当前快照以来接的局数
        self.accepting = [True] * self.k
        self.pending = []                        # 学习进程发来、还没用上的候选 [(id, path)]
        self.env_slot = {}                       # 环境 → 槽（一局里只有一方是历史快照）

    def set_picks(self, picks):
        """学习进程每批发一组新抽的候选，覆盖没用完的旧候选（PFSP 权重是最新的）。"""
        self.pending = [(int(i), p) for i, p in picks]

    def assign(self, w):
        """环境 w 开了一局历史快照对局 → 槽号。"""
        cand = [i for i in range(self.k) if self.accepting[i]] or list(range(self.k))
        i = min(cand, key=lambda j: (len(self.active[j]), self.started[j]))
        self.active[i].add(w)
        self.started[i] += 1
        self.env_slot[w] = i
        if self.started[i] >= self.per and any(self.accepting[j] for j in range(self.k) if j != i):
            self.accepting[i] = False
        return i

    def release(self, w):
        """环境 w 的一局结束 → 它所在的槽号（不是历史快照对局则 None）。"""
        i = self.env_slot.pop(w, None)
        if i is not None:
            self.active[i].discard(w)
        return i

    def refills(self):
        """→ [(槽号, 快照编号, 路径)]：手上没有对局、该换快照的槽（不再接新局的；或还在用 BC 占位、已经有候选的）。"""
        out = []
        for i in range(self.k):
            if self.active[i]:
                continue
            if not self.accepting[i] or (self.ids[i] < 0 and self.pending):
                sid, path = self.pending.pop(0) if self.pending else (self.ids[i], self.paths[i])
                out.append((i, sid, path))
        return out

    def installed(self, i, sid, path):
        self.ids[i], self.paths[i] = int(sid), path
        self.started[i] = 0
        self.accepting[i] = True


class ExploitTarget:
    """剥削者训练（league.mode = exploit）：对手固定为 league.target（主模型的冻结副本，只放模型的 .pt），只统计对它的胜率。
    接口和 League 一样（pick / update / add / summary / elo），ppo.py 不用区分。"""

    def __init__(self, cfg, run):
        from ..config import repo_path
        t = cfg.league.get('target')
        if not t:
            raise ValueError('exploit 模式要给 league.target（主模型快照）')
        self.path = repo_path(t)
        self.id = max(0, _step_of(self.path))
        self.decay = cfg.league.get('stat_decay', 0.995)
        self.n = self.w = 0.0
        self.elo = 0.0

    def pick(self):
        return self.id, self.path

    def deck_probs(self):
        return None

    def add(self, *a, **k):
        pass

    def scan_exploiters(self):
        return 0

    def update(self, episodes):
        self.n *= self.decay
        self.w *= self.decay
        for e in episodes:
            if e.get('result') is not None and e['opp'] == 'hist':
                self.n += 1
                self.w += e['result']
        self.elo = elo_of((self.w + 1.0) / (self.n + 2.0))              # 相对目标（主模型）的分差

    def summary(self):
        return dict(target_step=self.id, target_winrate=(self.w + 1.0) / (self.n + 2.0), elo=self.elo)
