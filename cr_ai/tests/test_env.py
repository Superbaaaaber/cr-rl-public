"""奖励、单局环境集成。"""
import numpy as np

from cr_ai.env.reward import potential, shaping_coef, step_reward


def test_potential_and_terminal(cfg):
    rc = cfg.reward
    assert potential([1, 1, 1], [1, 1, 1], rc) == 0
    assert abs(potential([1, 1, 1], [1, 0, 1], rc) - 0.3 * 0.25) < 1e-9
    r, parts = step_reward(0.1, 0.2, False, 0, False, 1.0, rc)
    assert abs(r - (rc.gamma * 0.2 - 0.1)) < 1e-9
    r, parts = step_reward(0.1, 0.5, True, 1, True, 1.0, rc)
    assert abs(parts['shaping'] + 0.1) < 1e-9 and parts['win'] == 1.0 and parts['overflow'] == -rc.overflow_penalty
    assert shaping_coef(0, rc) == 1.0 and abs(shaping_coef(rc.shaping_anneal_steps * 2, rc) - 0.2) < 1e-9


def test_full_game_random(env):
    from cr_ai.bots.random_bot import RandomBot
    from cr_ai.config import load_deck
    D, F = load_deck('cr_ai/configs/deck.yaml')                       # 固定用写这个测试时的卡组（种子 31 下这局恰好 0 次作废；默认主卡组 2026-10-17 改成了墓园）
    H = ['HogRider', 'Musketeer', 'IceGolemite', 'IceSpirits', 'Skeletons', 'Cannon', 'Fireball', 'Log']
    obs = env.reset(seed=31, decks=[D, H], forms=[F, {'Cannon': 'evo'}], tower_troop=['tower-princess', 'cannoneer'])
    bots = [RandomBot(np.random.RandomState(3), 0.15), RandomBot(np.random.RandomState(4), 0.15)]
    n, total = 0, [0.0, 0.0]
    while True:
        obs, rew, done, info = env.step([bots[p].act_obs(obs[p]) for p in (0, 1)])
        n += 1
        total = [total[p] + rew[p] for p in (0, 1)]
        if done:
            break
    assert info['winner'] in (-1, 0, 1)
    assert sum(env.stats['plays']) > 10
    assert env.stats['void'] == [0, 0]                                 # 按 mask 出的动作执行时都合法
    if info['winner'] != -1:
        w = info['winner']
        assert total[w] > total[1 - w]
    for p in (0, 1):                                                    # 观测形状与声明一致
        for k, (shape, dt) in env.spec.items():
            assert np.asarray(obs[p][k]).shape == shape and np.asarray(obs[p][k]).dtype == dt, k


def test_tiebreak_lowest_tower_loses(env):
    """冠数相同：比双方存活塔里最低的绝对血量（引擎加时结束后所有塔同速掉血、先倒的输）；相等为平局。"""
    from cr_ai.tests.conftest import make_units, towers
    rows = towers()
    rows[4] = rows[4][:5] + (2500,) + rows[4][6:]          # 玩家 1 左公主塔掉到 2500（满血 3052）→ 玩家 1 输
    env.units = make_units(rows)
    assert env._winner([0, 0]) == 0
    rows = towers()
    rows[3] = rows[3][:5] + (2000,) + rows[3][6:]          # 玩家 1 国王塔掉到 2000（低于公主塔 3052）→ 玩家 1 输
    env.units = make_units(rows)
    assert env._winner([0, 0]) == 0
    rows = towers()
    rows[3] = rows[3][:5] + (4000,) + rows[3][6:]          # 国王塔掉血但仍高于两边最低的公主塔（3052 = 3052）→ 平局
    env.units = make_units(rows)
    assert env._winner([0, 0]) == -1
    assert env._winner([1, 0]) == 0
