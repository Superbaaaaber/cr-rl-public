"""PPO 的纯数值部分（不需要引擎和显卡）。"""
import numpy as np

from cr_ai.train.ppo import gae
from cr_ai.train.rollout import bc_ref_every


def test_gae_truncation_bootstraps_with_own_value():
    """一局在第 2 步被截断（引擎出错）：那一步不当终局（用自己的价值自举，奖励 0 时 δ ≈ (γ−1)·V），
    优势也不往前一局传；真正的终局（done、不截断）则自举为 0。"""
    rew = np.array([[0.0], [0.0], [0.0], [1.0]], np.float32)
    val = np.array([[0.5], [0.6], [0.8], [0.4]], np.float32)
    done = np.array([[False], [False], [True], [True]])
    trunc = np.array([[0.0], [0.0], [1.0], [0.0]], np.float32)
    g, lam = 0.99, 0.9
    adv, ret = gae(rew, val, done, np.zeros(1, np.float32), g, lam, trunc)
    assert abs(adv[3, 0] - (1.0 - 0.4)) < 1e-6                       # 真终局：δ = r − V
    assert abs(adv[2, 0] - (g * 0.8 - 0.8)) < 1e-6                    # 截断：δ = γ·V(s) − V(s)，不接下一局
    d1 = g * 0.8 - 0.6
    assert abs(adv[1, 0] - (d1 + g * lam * adv[2, 0])) < 1e-6
    adv0, _ = gae(rew, val, done, np.zeros(1, np.float32), g, lam)  # 不给 trunc：旧行为，截断步当终局
    assert abs(adv0[2, 0] - (0.0 - 0.8)) < 1e-6


def test_noise_scale_parts_recovers_signal_and_noise():
    """梯度 = 真梯度 G + 每个样本独立噪声（协方差 σ²I）：多次估计取平均，|G|² 和 tr(Σ) 都要对上；小批大小不等也行。"""
    from cr_ai.train.ppo import noise_scale_parts
    rng = np.random.RandomState(0)
    d, sigma = 40, 2.0
    G = rng.randn(d)
    sizes = [90, 100, 110, 100]
    acc = []
    for _ in range(3000):
        gs = [G + rng.randn(d) * sigma / np.sqrt(b) for b in sizes]       # 小批梯度 = b 个样本的平均
        gb = sum(b * g for b, g in zip(sizes, gs)) / sum(sizes)
        acc.append(noise_scale_parts([float(g @ g) for g in gs], sizes, float(gb @ gb)))
    G2 = np.mean([a['gns_G2'] for a in acc])
    trS = np.mean([a['gns_trS'] for a in acc])
    assert abs(G2 - G @ G) < 0.05 * (G @ G), (G2, G @ G)
    assert abs(trS - d * sigma ** 2) < 0.05 * d * sigma ** 2, (trS, d * sigma ** 2)
    assert acc[0]['gns_batch'] == 400


def test_policy_targets_decoupled_lambda():
    """策略 λ 不设或等于价值 λ：和 gae 完全一样；设成别的：优势按策略 λ 算，价值目标（λ 回报）不变。"""
    from cr_ai.train.ppo import policy_targets
    rng = np.random.RandomState(1)
    T, S = 60, 4
    rew = (rng.rand(T, S) < 0.05).astype(np.float32) * rng.choice([-1.0, 1.0], size=(T, S)).astype(np.float32)
    val = rng.uniform(-0.5, 0.5, size=(T, S)).astype(np.float32)
    done = rng.rand(T, S) < 0.03
    last_v = rng.uniform(-0.5, 0.5, size=S).astype(np.float32)
    g = 0.9995
    a0, r0 = gae(rew, val, done, last_v, g, 0.97)
    for lp in (None, 0.97):
        a, r = policy_targets(rew, val, done, last_v, g, 0.97, lp)
        assert np.array_equal(a, a0) and np.array_equal(r, r0)
    a, r = policy_targets(rew, val, done, last_v, g, 0.97, 0.95)
    assert np.array_equal(r, r0)
    assert np.allclose(a, gae(rew, val, done, last_v, g, 0.95)[0]) and not np.allclose(a, a0)


def test_shadow_advantages_same_scale_as_training():
    """影子 λ：训练用的 λ 那份和训练时的标准化优势完全一样；价值准确（这里恒为 0 = 抛硬币的真实价值）、终局 ±1 时，
    λ 越小，噪声传得越短，同一尺度下的方差越小。"""
    from cr_ai.train.ppo import shadow_advantages
    rng = np.random.RandomState(0)
    T, S, n = 400, 8, 50
    rew = np.zeros((T, S), np.float32)
    done = np.zeros((T, S), bool)
    done[n - 1::n] = True
    rew[done] = rng.choice([-1.0, 1.0], size=int(done.sum()))
    val = np.zeros((T, S), np.float32)
    vmask = np.ones((T, S), bool)
    vmask[:, 0] = False                                                # 一条流整条无效：不参与均值/方差，输出为 0
    last_v = np.zeros(S, np.float32)
    g, lt = 0.9995, 0.97
    adv = np.where(vmask, gae(rew, val, done, last_v, g, lt)[0], 0.0)
    sa = shadow_advantages(rew, val, done, last_v, g, [0.9, 0.95, lt], None, vmask, adv)
    av = adv[vmask]
    want = (adv - av.mean()) / (av.std() + 1e-8)
    assert np.allclose(sa[lt][vmask], want[vmask], atol=1e-5)
    assert (sa[0.9][~vmask] == 0).all()
    v = {lam: sa[lam][vmask].var() for lam in sa}
    assert abs(v[lt] - 1.0) < 1e-4 and v[0.9] < v[0.95] < v[lt], v


def test_bc_ref_every_only_without_kl():
    """ppo.bc_ref_every > 1（抽样算 BC 参考分布）只能在 KL-to-BC 系数始终为 0 时用；默认 1。"""
    import pytest
    from cr_ai.config import load
    cfg = load('cr_ai/configs/ppo.yaml')
    assert bc_ref_every(cfg) == 1
    with pytest.raises(ValueError):
        bc_ref_every(load('cr_ai/configs/ppo.yaml', overrides=['ppo.bc_ref_every=16']))
    assert bc_ref_every(load('cr_ai/configs/ppo.yaml', overrides=['ppo.bc_ref_every=16', 'ppo.kl_bc_start=0.0', 'ppo.kl_bc_end=0.0'])) == 16


def test_critic_rewarm_window():
    """只训价值头：训练开头 critic_warmup_steps 步；另设 critic_rewarm_until（绝对步数）时，续训到那一步之前也是。"""
    from cr_ai.config import load
    from cr_ai.train.ppo import critic_warm
    pc = load('cr_ai/configs/ppo.yaml').ppo
    assert critic_warm(0, pc) and not critic_warm(pc.critic_warmup_steps, pc)
    assert not critic_warm(545_829_208, pc)
    pc = load('cr_ai/configs/ppo.yaml', overrides=['ppo.critic_rewarm_until=547829208']).ppo
    assert critic_warm(0, pc) and critic_warm(545_829_208, pc) and not critic_warm(547_829_208, pc)
