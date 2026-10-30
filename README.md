# cr-rl：皇室战争 AI 的训练流程和权重

在一个 1:1 还原皇室战争（16.402.17）战斗逻辑的引擎上，先用真人回放做行为克隆（BC），再用联赛式自我对弈 PPO 训练出的对战模型。
这个仓库包含**训练代码、配置、实际在用的训练参数和权重**。

**不包含**：战斗引擎（`crx`，来自另一个仓库 cr-engine）、实机部署（cr-live）、游戏数据表和引擎快照、回放数据。
所以这里的代码不能直接跑起来：环境层通过 `cr_ai/env/engine_adapter.py` 调引擎（`from crx.engine import Engine`），
要跑需要自己提供一个接口相同的引擎。权重可以单独加载、看结构、做推理实验。

本项目和 Supercell 没有关系；“皇室战争 / Clash Royale”是 Supercell 的商标。

## 权重（见 Releases）

| 文件 | 内容 |
|---|---|
| `ppo_step724051358.pt` | 主模型，PPO 训到 7.24 亿步（2026-10-31）。3,538 万参数，fp32 |
| `bc_v2_ft3.pt` | 行为克隆模型（PPO 的起点和 KL 参考），扩容前的结构，1,467 万参数 |

存档是一个普通 dict（`torch.load(..., weights_only=True)` 就能读）：`model`（state_dict）、`cfg`（训练时的完整配置，建网用它就够了）、`step`、`kind`。

```python
import torch
from cr_ai.config import _wrap
from cr_ai.model.net import Net, load_weights

st = torch.load('ppo_step724051358.pt', map_location='cpu', weights_only=True)
net = Net(_wrap(st['cfg'])).eval()        # _wrap：dict → 支持属性访问的配置
load_weights(net, st['model'])
# out, state = net(obs, state)   # obs 的字段见 cr_ai/env/game_env.py: obs_spec
```

BC 存档是扩容前的结构，装进新结构要用 `load_weights(net, sd, grow=True)`（自动补成新结构，函数不变）。

## 模型

- 输入：32×18 网格（单位/法术嵌入 + 数值通道）、单位列表（最多 128 个，带状态特征）、标量（圣水、时间、塔血、国王塔是否激活……）、
  手牌 5 张（含进化充能）、自己卡组、对手见过的牌、英雄/冠军技能状态。只用部署时真实看得到的信息（对手手牌和圣水只用估计）。
  另有一个只给价值头的特权分支（训练时用对手的真实信息，推理时不用）。
- 主干：卷积残差网络 + 实体 Transformer 分支 → 融合层 → LSTM（1536）。
- 输出：牌头（不出 / 手牌 4 张 / 技能 2 个）、位置头（576 格，以选中的牌为条件）、价值头。
- 细节见 `cr_ai/model/net.py` 开头的说明。

## 训练流程

1. **回放 → BC 数据**：`cr_ai/data/replay_parser.py` 解析回放，`cr_ai/data/replay_resim.py` 在引擎里逐 tick 重放并采样（双方视角都采），
   `cr_ai/tools/reshard_bc.py` 把分片打散。
2. **行为克隆**：`python -m cr_ai.train.bc --config cr_ai/configs/bc.yaml cr_ai/configs/arch_v2.yaml --run runs/bc_v2`，
   再用新版本回放微调（`cr_ai/configs/bc_ft_scrape.yaml`）。
3. **PPO**：`cr_ai/train/ppo.py`（学习进程）+ `cr_ai/train/rollout.py`（采样进程，推理走 CUDA Graph，见 `cr_ai/model/infer.py`）。
   循环 PPO、GAE、对 BC 的 KL 约束、价值头带胜率先验。
4. **联赛**：`cr_ai/tools/league_loop.py` 分段训练主模型，每段结束后在后台对固定考官评测。对手 80% 是当前策略自己、20% 是历史快照
   （PFSP：越打不过的快照越常被选）。卡组从卡组池（`cr_ai/configs/deck_pool_mix.yaml`，505 套）按 PFSP 抽，30% 的对局用主卡组。
   `league_loop` 也支持剥削者（专门找主模型漏洞的模型），2026-10-21 起关掉了（`--no-exploit`）。
5. **扩容**：`cr_ai/tools/grow.py` 在训练中途把网络加宽（LSTM 768 → 1536 等），扩容后和原网络是同一个函数，接着训。
6. **评测**：`cr_ai/eval/arena.py`（对打）、`cr_ai/tools/gauntlet.py`（对固定考官打，Bradley–Terry 拟合 Elo）。
7. **看板**：`python -m cr_ai.tools.dashboard`（读 `runs/<run>/` 下的日志）。

`scripts/` 里是训练机上实际用的启动命令和覆盖参数：
- `scripts/m4_common.txt`：主模型和剥削者共用的覆盖（模型尺寸、卡组池、BC 存档……）；
- `scripts/m4_main.txt`：只给主模型的覆盖，每一行改动的原因和止损条件都写在注释里，是这个模型训练过程的记录；
- `scripts/v2_m4h.sh`：`league_loop` 的启动命令。

配置按顺序叠加：`cr_ai/configs/default.yaml` ← `ppo.yaml` ← `m4.yaml` ← `arch_v2.yaml` ← 覆盖文件（每行一个 `a.b=值`）。

## 目录

| 路径 | 内容 |
|---|---|
| `cr_ai/env/` | 观测（`features.py`、`rasterize.py`）、动作掩码（`masks.py`）、奖励、对手牌追踪、多进程环境（`vec_env.py`）、引擎适配层 |
| `cr_ai/model/` | 网络（`net.py`）、CUDA Graph 推理（`infer.py`）、扩容（`grow.py`） |
| `cr_ai/train/` | BC、序列 BC、PPO、采样、联赛、蒸馏 |
| `cr_ai/data/` | 回放解析、重模拟、BC 数据加载 |
| `cr_ai/eval/`、`cr_ai/tools/` | 评测、考官、联赛循环、扩容工具、看板、卡组池生成 |
| `cr_ai/configs/` | 全部配置和卡组池 |
| `cr_ai/tests/` | 单元测试（大部分要引擎才能跑） |
| `scripts/` | 训练机上的启动脚本和覆盖参数 |

## 依赖

Python 3.10+、PyTorch 2.x（CUDA）、`requirements.txt` / `requirements-linux.txt`。

## 许可证

代码和 Releases 里的权重都按 MIT 许可证发布（见 `LICENSE`）。
