"""cr_ai/model/net.py — 策略/价值网络（BC 与 PPO 共用）。

    net = Net(cfg)
    out, state = net(obs, state=None, place_card=None)      # obs：dict[str, Tensor]（与 GameEnv 观测同名，batch 维在前，可选时间维见 forward_seq）
    out['card']   [B, 5+K]  牌头 logits（已按 card_mask 屏蔽；0 = 不出，1-4 手牌，5.. 技能）
    out['place']  [B, 576]  位置头 logits（未屏蔽；以 place_card 选中的牌为条件，None 时不算）
    out['value']  [B]       价值头（tanh）

结构（规格书第 5 节）：
  网格 5 个 ID 通道（单位嵌入 16 维 ×4 槽 + 法术嵌入 8 维）+ 18 个数值通道 = 90ch@32×18
    → Conv3×3 64（skip）→ Conv s2 96@16×9 → 4×ResBlock(96) → Conv s2 64@8×5 → FC 256
  标量/塔状态/技能/待执行/上一个动作 → MLP 256；手牌 5 张 → 每张 MLP 64；对手已见卡 → 每张 MLP 后池化 64；
  自己卡组 → 嵌入平均 64（model.deck_pool = mlp：每张先 MLP、再 mean + max 池化 128）
  → FC 384 → LSTM 384 = h
  牌头：手牌/技能 score = MLP([h; 卡表示])，不出 = MLP(h)
  位置头：FiLM(选中卡表示 + h) 作用于 16×9 特征 → 上采样 + skip →（可选：32×18 上再 FiLM 一次、model.place_res_blocks 个残差块）→ Conv3×3 64 → Conv1×1 → 576
  价值头：MLP(h) → tanh（model.priv_value：RL 时另用 MLP([h; 对手真实手牌/圣水的编码])，见 value_from）

“不出”的类别权重：BC 里“不出”类的损失权重是 noop_w（规格 0.1–0.2），这会让模型高估出牌概率（先验被乘了 1/noop_w）。
推理时用 policy_logits() 给“不出”加 log(1/noop_w) 还原到真实频率（logit adjustment），评测与 RL 都用这个。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import tables
from ..env.features import AB_EXT, FORM_EXT, N_HAND, N_SCAL, N_STATUS

GH, GW = 32, 18
NCELL = GH * GW
N_FORMS = 3
N_TSTATE = 16

INPUT_KEYS = ('gi', 'gf', 'scal', 'hand_id', 'hand_form', 'hand_num', 'deck_id', 'deck_form', 'opp_id', 'opp_form', 'opp_num',
              'ab_id', 'ab_num', 'pend_id', 'pend_num', 'last_id', 'last_num', 'card_mask', 'tstate')
ENTITY_KEYS = ('ent_id', 'ent_num')                    # 单位列表（model.entity，见 env/rasterize.py: entities）
PRIV_KEYS = ('priv_id', 'priv_form', 'priv_num')       # 特权信息（model.priv_value）：对手真实手牌 4 张 + 下一张、真实圣水；只给价值头
N_PRIV = 5
EXT_KEYS = ('ab_ext', 'hand_ext', 'opp_ext')           # 扩展观测（model.obs_ext，见 env/features.py）；有单位列表时另有 ent_ext


def input_keys(cfg):
    """网络的输入观测字段（BC、RL、推理共用）：基础字段 + 可选的单位列表 + 可选的扩展观测。"""
    ent = cfg.model.get('entity', False)
    ext = list(EXT_KEYS) + (['ent_ext'] if ent else []) if cfg.model.get('obs_ext', False) else []
    return list(INPUT_KEYS) + (list(ENTITY_KEYS) if ent else []) + ext


def rl_keys(cfg):
    """RL 采样/学习要存的观测字段：网络输入 + 可选的特权信息（只有模拟器里有；BC 数据和部署都没有）。"""
    return input_keys(cfg) + (list(PRIV_KEYS) if cfg.model.get('priv_value', False) else [])


def win_prior(scal, reg, ot):
    """按塔的状态估计胜率：→ tanh(z/2) = 2·p̂ − 1，[B]（每个玩家用自己的观测算，双方互换时变号；开局 = 0）。
    scal 的布局见 env/features.py: scalars——[4] 是否加时、[5] 剩余时间比例、[6:9] 我方 [国王, 左, 右] 血量比、[9:12] 敌方、
    [12] 我方国王已激活、[13] 敌方国王已激活。reg / ot 是常规时间 / 加时的系数（configs/default.yaml: model.value_prior_*）。
    只用张量运算（采样端在 CUDA Graph 里跑）。"""
    s = scal.float()
    own, en = s[:, 6:9], s[:, 9:12]
    d_hp = (own[:, 1] + own[:, 2]) - (en[:, 1] + en[:, 2])
    crowns = lambda t: (t[:, 1] <= 0).float() + (t[:, 2] <= 0).float() + 3.0 * (t[:, 0] <= 0).float()   # noqa: E731
    d_cr = (crowns(en) - crowns(own)).clamp(-1.0, 1.0)

    def live_min(t):
        m = torch.where(t[:, 1:3] > 0, t[:, 1:3], torch.full_like(t[:, 1:3], 2.0)).min(1).values
        return torch.where(m > 1.5, torch.zeros_like(m), m)               # 两座公主塔都没了 → 0
    d_min = live_min(own) - live_min(en)
    d_king = own[:, 0] - en[:, 0]
    d_act = s[:, 13] - s[:, 12]
    tau = (1.0 - s[:, 5]).clamp(0.0, 1.0)
    z_reg = reg[0] * d_hp + (reg[1] + reg[2] * tau) * d_cr + reg[3] * d_min + reg[4] * d_king + reg[5] * d_act
    z_ot = ot[0] * d_hp + ot[1] * d_min + ot[2] * d_king
    return torch.tanh(torch.where(s[:, 4] > 0.5, z_ot, z_reg) * 0.5)


def _mlp_add(mlp, x, add=None):
    """_mlp 前向，可在第一层线性的输出上加一项（扩展观测的输入从这里进；add 的权重从 0 开始 = 和不加完全一样）。"""
    y = mlp[0](x)
    if add is not None:
        y = y + add
    for layer in mlp[1:]:
        y = layer(y)
    return y


def _mlp(i, h, o, act_out=True, ln=False):
    """ln：隐藏层在 ReLU 前加 LayerNorm（model.ln，2026-10-28；不加时结构和以前完全一样，旧存档照常加载）。"""
    layers = [nn.Linear(i, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(inplace=True), nn.Linear(h, o)]
    if act_out:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class EntBlock(nn.Module):
    """实体分支的一层 Transformer（pre-LN）。手写注意力：nn.TransformerEncoder 推理时走嵌套张量快速路径，录不进 CUDA Graph。"""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, d))

    def forward(self, x, am):
        """x [B, N, d]；am [B, 1, 1, N] bool（True = 可以看这个单位）。"""
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        x = x + self.proj(y.transpose(1, 2).reshape(B, N, D))
        return x + self.ff(self.n2(x))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(x + self.b2(self.c2(y)), inplace=True)


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        self.K = cfg.env.max_abilities
        C, CS, H = mc.channels, mc.skip_channels, mc.hidden
        nu, nc = tables.n_units(), tables.n_cards()
        self.unit_emb = nn.Embedding(nu, 16)
        self.spell_emb = nn.Embedding(nu, 8)
        self.card_emb = nn.Embedding(nc, 32)
        self.form_emb = nn.Embedding(N_FORMS, 8)
        cin = 4 * 16 + 8 + 18
        self.input_keys = input_keys(cfg)
        # 2026-10-16 可选：实体分支（model.entity）。网格每格每方只记血量最高的 2 个单位、数量 /4 截断、位置量化到整格、法术后写覆盖先写；
        # 单位列表（精确坐标、血量、类别……，见 env/rasterize.py: entities）过 entity_layers 层小 Transformer，
        # 每个单位的输出向量按所在格子加回 32×18 网格（entity_grid_channels 个通道，和原有通道拼在一起进卷积主干），另外池化一个向量进融合层。
        self.entity = bool(mc.get('entity', False))
        d_ent = 0
        if self.entity:
            from ..env.rasterize import N_ENT_F
            de, self.ent_gc = int(mc.get('entity_dim', 64)), int(mc.get('entity_grid_channels', 32))
            self.ent_uid = nn.Embedding(nu, 32)
            self.ent_in = nn.Linear(32 + N_ENT_F, de)
            self.ent_blocks = nn.ModuleList([EntBlock(de, 4) for _ in range(int(mc.get('entity_layers', 2)))])
            self.ent_norm = nn.LayerNorm(de)
            self.ent_out = nn.Linear(de, self.ent_gc)
            cin += self.ent_gc
            d_ent = de
        # 网格主干。出口（16×9 → 8×5 → 向量，出牌决策只看得到这一路）的宽度：model.exit_channels / model.grid_dim（2026-10-28，默认 64 / 256 = 旧结构）
        E, G = int(mc.get('exit_channels', 64)), int(mc.get('grid_dim', 256))
        self.stem = nn.Sequential(nn.Conv2d(cin, CS, 3, padding=1, bias=False), nn.BatchNorm2d(CS), nn.ReLU(inplace=True))
        self.down = nn.Sequential(nn.Conv2d(CS, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[ResBlock(C) for _ in range(mc.res_blocks)])
        self.down2 = nn.Sequential(nn.Conv2d(C, E, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(E), nn.ReLU(inplace=True))
        self.grid_fc = nn.Sequential(nn.Linear(E * 8 * 5, G), nn.ReLU(inplace=True))
        # 向量特征
        ce = 32 + 8                                             # 卡牌嵌入 + 形态嵌入
        n_extra = N_SCAL + N_TSTATE + self.K * (ce + 2) + 2 * (ce + 3) + (ce + 2)
        self.extra_mlp = _mlp(n_extra, 256, 256)
        # 手牌槽位：onehot = 5 维槽位 one-hot（旧结构；打出的牌原位补新牌，槽位没有策略含义，BC 会学到人类“爱点某个槽”的习惯）；
        # next = 只标“是不是下一张”
        self.hand_slot = mc.get('hand_slot', 'onehot')
        self.hand_mlp = _mlp(ce + 3 + (N_HAND if self.hand_slot == 'onehot' else 1), 64, 64)
        self.ab_mlp = _mlp(ce + 2, 64, 64)
        self.opp_mlp = _mlp(ce + 3, 64, 64)
        # 卡组编码：mean = 8 张嵌入取平均 → 一层线性（旧结构，表达不了牌之间的组合）；mlp = 每张先过 MLP，再 mean + max 池化（同对手卡那一路）
        self.deck_pool = mc.get('deck_pool', 'mean')
        if self.deck_pool == 'mlp':
            self.deck_mlp = _mlp(ce, 64, 64)
            d_deck = 128
        else:
            self.deck_fc = nn.Sequential(nn.Linear(ce, 64), nn.ReLU(inplace=True))
            d_deck = 64
        # 2026-10-28 可选（蒸馏进更大的网络时用，见 train/distill.py）：model.ln 在融合层和各个头的隐藏层 ReLU 前加 LayerNorm
        # （BC 训出来的融合层有 44% 的单元对所有观测都是负的、永远拿不到梯度）；model.head_hidden 是各个头的隐藏宽度（默认 128 = 旧结构）
        ln = bool(mc.get('ln', False))
        hh = int(mc.get('head_hidden', 128))
        self.fuse = nn.Sequential(nn.Linear(G + 256 + N_HAND * 64 + 128 + d_deck + d_ent, H), *([nn.LayerNorm(H)] if ln else []), nn.ReLU(inplace=True))
        self.lstm = nn.LSTMCell(H, H)
        # 头
        self.noop_head = _mlp(H, hh, 1, act_out=False, ln=ln)
        self.card_score = _mlp(H + 64, hh, 1, act_out=False, ln=ln)
        self.film = nn.Linear(H + 64, 2 * C)
        self.skip_proj = nn.Conv2d(CS, C, 1)
        # 2026-10-16 可选：位置头在 32×18 全分辨率上也用选中牌做条件（FiLM），再过几个残差块（默认都关 = 旧结构，旧存档照常加载）
        self.film_full = nn.Linear(H + 64, 2 * C) if mc.get('place_film_full', False) else None
        if self.film_full is not None:
            nn.init.zeros_(self.film_full.weight)
            nn.init.zeros_(self.film_full.bias)
        self.place_res = nn.Sequential(*[ResBlock(C) for _ in range(int(mc.get('place_res_blocks', 0)))])
        self.place_conv = nn.Sequential(nn.Conv2d(C, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.value_head = _mlp(H, hh, 1, act_out=False, ln=ln)
        # 2026-10-16 可选：特权价值头（model.priv_value）。在模拟器里训练时价值头可以看对手的真实手牌和圣水，降低优势估计的方差；
        # 策略头不用，部署不受影响。BC 数据里没有这些信息，BC 照旧只训 value_head；priv_* 在 PPO 的 critic 热身里从头学。
        self.priv_value = bool(mc.get('priv_value', False))
        self.priv_keys = tuple(PRIV_KEYS) if self.priv_value else ()
        if self.priv_value:
            self.priv_mlp = _mlp(N_PRIV * ce + 1, 128, 64)
            self.priv_value_head = _mlp(H + 64, hh, 1, act_out=False, ln=ln)
        # 2026-10-27 可选：价值输出加按塔的状态估计的胜率先验（model.value_prior，见 win_prior）；0 = 关
        self.value_prior = float(mc.get('value_prior', 0.0) or 0.0)
        self.prior_reg = [float(x) for x in mc.get('value_prior_reg', [0.0] * 6)]
        self.prior_ot = [float(x) for x in mc.get('value_prior_ot', [0.0] * 3)]
        self.register_buffer('noop_bias', torch.tensor(math.log(1.0 / cfg.bc.noop_w) if cfg.get('bc') else 0.0))
        # 技能概率下限（ppo.ab_floor，定向探索）：只有采样进程里的当前策略打开（model/infer.py: ab_floor_mix），只改采样；
        # ab_eps 不进存档、不进共享权重（采样进程按步数自己算、原地改，CUDA Graph 里读的是同一块显存）
        self.ab_mix = False
        self.register_buffer('ab_eps', torch.zeros(()), persistent=False)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.H = H
        # 2026-10-28 扩展观测（model.obs_ext）：技能状态、进化充能、单位状态（见 env/features.py）。每一路都是一个线性层，输出加到
        # 对应输入 MLP 第一层的输出上；权重全 0 → 打开后和不打开是同一个函数，旧存档照常加载（load_weights 允许缺 ext_*）、续训时学会用。
        # 放在最后注册：优化器参数顺序里排在最后，续训时旧的 AdamW 状态直接接上（train/ppo.py: extend_adam）。
        self.obs_ext = bool(mc.get('obs_ext', False))
        if self.obs_ext:
            self.ext_ab = nn.Linear(AB_EXT, 64, bias=False)                      # → 技能表示（牌头）
            self.ext_extra = nn.Linear(self.K * AB_EXT, 256, bias=False)       # → 标量那一路（融合层 / 价值）
            self.ext_hand = nn.Linear(FORM_EXT, 64, bias=False)                # → 每张手牌的表示
            self.ext_opp = nn.Linear(FORM_EXT, 64, bias=False)                 # → 对手每张见过的牌
            self.ext_ent = nn.Linear(N_STATUS, d_ent, bias=False) if self.entity else None   # → 每个单位（实体分支）
            for m in (self.ext_ab, self.ext_extra, self.ext_hand, self.ext_opp, self.ext_ent):
                if m is not None:
                    nn.init.zeros_(m.weight)

    # ---------------- 编码 ----------------
    def _card(self, ids, forms):
        return torch.cat([self.card_emb(ids.long()), self.form_emb(forms.long())], -1)

    def encode(self, o):
        """→ (融合特征 [B, H], 16×9 特征图, 32×18 skip, 手牌表示 [B, 5, 64], 技能表示 [B, K, 64])。"""
        gi = o['gi'].long()
        B = gi.shape[0]
        dt = self.card_emb.weight.dtype                                   # 推理时整网可转半精度（采样进程），输入跟着转
        g = [self.unit_emb(gi[:, :4]).permute(0, 1, 4, 2, 3).reshape(B, 64, GH, GW), self.spell_emb(gi[:, 4]).permute(0, 3, 1, 2), o['gf'].to(dt)]
        ent_vec = None
        X = self.obs_ext
        if self.entity:
            ent_grid, ent_vec = self.encode_entities(o['ent_id'], o['ent_num'].to(dt), B, o['ent_ext'].to(dt) if X else None)
            g.append(ent_grid)
        g = torch.cat(g, 1)
        skip = self.stem(g.contiguous(memory_format=torch.channels_last))     # channels_last：卷积走张量核更快（实测 −10%）
        f16 = self.res(self.down(skip))
        gv = self.grid_fc(self.down2(f16).flatten(1))
        ab = torch.cat([self._card(o['ab_id'], torch.zeros_like(o['ab_id'])), o['ab_num'].to(dt)], -1)          # [B, K, 42]
        pend = torch.cat([self._card(o['pend_id'], torch.zeros_like(o['pend_id'])), o['pend_num'].to(dt)], -1)
        last = torch.cat([self._card(o['last_id'][:, 0], torch.zeros_like(o['last_id'][:, 0])), o['last_num'].to(dt)], -1)
        ts = F.one_hot(o['tstate'].long(), N_TSTATE).to(dt)
        ex = _mlp_add(self.extra_mlp, torch.cat([o['scal'].to(dt), ts, ab.flatten(1), pend.flatten(1), last], -1),
                      self.ext_extra(o['ab_ext'].flatten(1).to(dt)) if X else None)
        if self.hand_slot == 'onehot':
            slot = torch.eye(N_HAND, device=gi.device, dtype=dt).expand(B, N_HAND, N_HAND)
        else:
            slot = (torch.arange(N_HAND, device=gi.device) == N_HAND - 1).to(dt).view(1, N_HAND, 1).expand(B, N_HAND, 1)
        hand = _mlp_add(self.hand_mlp, torch.cat([self._card(o['hand_id'], o['hand_form']), o['hand_num'].to(dt), slot], -1),
                        self.ext_hand(o['hand_ext'].to(dt)) if X else None)                                       # [B, 5, 64]
        hand = hand * (o['hand_id'] > 0).unsqueeze(-1).to(dt)
        abr = _mlp_add(self.ab_mlp, ab, self.ext_ab(o['ab_ext'].to(dt)) if X else None) * (o['ab_id'] > 0).unsqueeze(-1).to(dt)
        om = (o['opp_id'] > 0).unsqueeze(-1).to(dt)
        opp = _mlp_add(self.opp_mlp, torch.cat([self._card(o['opp_id'], o['opp_form']), o['opp_num'].to(dt)], -1),
                       self.ext_opp(o['opp_ext'].to(dt)) if X else None) * om
        opp = torch.cat([opp.sum(1) / om.sum(1).clamp(min=1), opp.max(1)[0]], -1)
        if self.deck_pool == 'mlp':
            dk = self.deck_mlp(self._card(o['deck_id'], o['deck_form']))                      # [B, 8, 64]（ReLU 输出 ≥ 0，空位乘 0 不影响 max）
            dm = (o['deck_id'] > 0).unsqueeze(-1).to(dt)
            dk = dk * dm
            deck = torch.cat([dk.sum(1) / dm.sum(1).clamp(min=1), dk.max(1)[0]], -1)
        else:
            deck = self.deck_fc(self._card(o['deck_id'], o['deck_form']).mean(1))
        x = self.fuse(torch.cat([gv, ex, hand.flatten(1), opp, deck] + ([ent_vec] if ent_vec is not None else []), -1))
        return x, f16, skip, hand, abr

    def encode_entities(self, eid, num, B, status=None):
        """单位列表 → (散射到网格的特征 [B, Cg, 32, 18], 池化向量 [B, d])。eid [B, N, 2]：单位编号、所在格子；num [B, N, F]，第 0 列 = 有效；
        status [B, N, N_STATUS]：单位状态（扩展观测，可选）。"""
        valid = num[..., 0] > 0
        x = self.ent_in(torch.cat([self.ent_uid(eid[..., 0].long()), num], -1))
        if status is not None:
            x = x + self.ext_ent(status)
        am = valid.clone()
        am[:, 0] = True                                                  # 全空的行（推理补齐的行）也至少看一个，免得注意力出 NaN
        am = am[:, None, None, :]
        for blk in self.ent_blocks:
            x = blk(x, am)
        x = self.ent_norm(x)
        m = valid.unsqueeze(-1).to(x.dtype)
        e = self.ent_out(x) * m                                          # [B, N, Cg]，无效单位置 0
        cell = eid[..., 1].long().clamp(0, NCELL - 1)
        grid = torch.zeros(B, self.ent_gc, NCELL, dtype=e.dtype, device=e.device)
        grid = grid.scatter_add(2, cell.unsqueeze(1).expand(-1, self.ent_gc, -1), e.transpose(1, 2))
        vec = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return grid.view(B, self.ent_gc, GH, GW), vec

    def init_state(self, B, device):
        z = torch.zeros(B, self.H, device=device)
        return (z, z.clone())

    def forward(self, o, state=None, place_card=None):
        """place_card：[B] 选中的牌头选项（1-4 手牌才算位置头）；None 不算位置头。"""
        x, f16, skip, hand, abr = self.encode(o)
        B = x.shape[0]
        if state is None:
            state = self.init_state(B, x.device)
        h, c = self.lstm(x, state)
        out = dict(h=h, f16=f16, skip=skip, hand=hand)
        out['card'] = self.card_logits(h, hand, abr, o['card_mask'])
        out['value'] = self.value_from(h, o)
        if place_card is not None:
            out['place'] = self.place_logits(out, place_card)
        return out, (h, c)

    def value_from(self, h, o):
        """价值（tanh）：有特权价值头且观测里带特权信息时用它，否则用普通价值头（BC、部署、旧存档）。
        开了胜率先验（model.value_prior > 0）时再加上 value_prior × win_prior（float32）。"""
        if self.priv_value and 'priv_id' in o:
            dt = h.dtype
            pe = self.priv_mlp(torch.cat([self._card(o['priv_id'], o['priv_form']).flatten(1).to(dt), o['priv_num'].to(dt)], -1))
            v = torch.tanh(self.priv_value_head(torch.cat([h, pe], -1)).squeeze(-1))
        else:
            v = torch.tanh(self.value_head(h).squeeze(-1))
        if self.value_prior > 0 and 'scal' in o:
            v = v.float() + self.value_prior * win_prior(o['scal'], self.prior_reg, self.prior_ot)
        return v

    def value_params(self):
        """critic 热身时训练的参数。"""
        ps = list(self.value_head.parameters())
        if self.priv_value:
            ps += list(self.priv_mlp.parameters()) + list(self.priv_value_head.parameters())
        return ps

    def card_logits(self, h, hand, abr, mask):
        B = h.shape[0]
        reps = torch.cat([hand[:, :4], abr], 1)                                           # [B, 4+K, 64]
        n = reps.shape[1]
        s = self.card_score(torch.cat([h.unsqueeze(1).expand(B, n, h.shape[1]), reps], -1)).squeeze(-1)
        logits = torch.cat([self.noop_head(h), s], 1)
        return logits.float().masked_fill(~mask.bool(), -1e9)

    def place_logits(self, out, place_card):
        """以选中的手牌为条件的 576 格 logits。place_card 不是手牌（0 或技能）的行，用第 1 张手牌占位（结果不用）。"""
        h, hand, f16, skip = out['h'], out['hand'], out['f16'], out['skip']
        B = h.shape[0]
        slot = (place_card.long() - 1).clamp(0, 3)
        rep = hand[torch.arange(B, device=h.device), slot]
        cond = torch.cat([h, rep], -1)
        gamma, beta = self.film(cond).chunk(2, -1)
        y = f16 * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        y = F.interpolate(y, size=(GH, GW), mode='nearest') + self.skip_proj(skip)
        if self.film_full is not None:
            g2, b2 = self.film_full(cond).chunk(2, -1)
            y = y * (1 + g2[:, :, None, None]) + b2[:, :, None, None]
        return self.place_conv(self.place_res(F.relu(y))).flatten(1).float()

    def policy_logits(self, card_logits):
        """推理/RL 用的牌头 logits：“不出”加回类别权重造成的先验偏移。"""
        adj = torch.zeros_like(card_logits)
        adj[:, 0] = self.noop_bias
        return card_logits + adj


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_weights(net, sd, grow=False):
    """加载存档权重：只允许缺特权价值头（priv_*：BC 存档、打开 priv_value 之前的存档都没有）和扩展观测的输入层（ext_*：权重本来就从 0 开始），其余必须严格对上。
    grow=True：存档是扩容（model/grow.py）之前的小结构时，先按同一个函数补成网络现在的结构（BC 参考网络、联赛里的旧快照用）。
    → 缺了哪些键（空列表 = 完全对上）。"""
    if grow:
        from .grow import grow_state_dict, needs_grow
        if needs_grow(sd, net.state_dict()):
            sd = grow_state_dict(sd, net.state_dict())
    r = net.load_state_dict(sd, strict=False)
    bad = [k for k in r.missing_keys if not k.startswith(('priv_', 'ext_'))]
    if bad or r.unexpected_keys:
        raise RuntimeError('存档与网络结构对不上：缺 %s，多 %s' % (bad[:8], list(r.unexpected_keys)[:8]))
    # 存档里没有 ext_* = 它不用扩展观测：清零，而不是留着网络里原来的值——采样端的历史槽反复装不同快照，
    # 上一个（开了 obs_ext 训过的）快照的 ext 权重会留下来，悄悄改掉后装进来的旧快照的行为
    ext = {k for k in r.missing_keys if k.startswith('ext_')}
    if ext:
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in ext:
                    p.zero_()
    return list(r.missing_keys)


def init_recurrence(net, forget_bias=-4.0):
    """BC 是单步、零状态训练的：循环部分（W_hh、遗忘门）从没被训练过。转成带状态推理前：
    W_hh 置 0、遗忘门的输入权重置 0、偏置设成很负（遗忘门 ≈ σ(−4) ≈ 0.018）→ 首步与 BC 完全一致、之后也几乎一致；
    PPO 再从这里学怎么用记忆（W_hh 的梯度不为 0）。PyTorch LSTMCell 的门顺序：输入、遗忘、候选、输出。"""
    H = net.H
    lstm = net.lstm
    with torch.no_grad():
        lstm.weight_hh.zero_()
        lstm.weight_ih[H:2 * H].zero_()
        lstm.bias_ih[H:2 * H].fill_(forget_bias)
        lstm.bias_hh[H:2 * H].zero_()
    return net
