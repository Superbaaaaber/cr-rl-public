#!/usr/bin/env bash
# M4 续跑（2026-10-20，用户：固定考官 + 专项评测、按卡组挑难的练、剥削者）：tools/league_loop.py 分时训练——
# 主模型每段 1,200 万步 ↔ 剥削者 400 万步。配置覆盖在 runs/m4_common.txt（共用）、runs/m4_main.txt（只给主模型）。
# 10-20 修正：固定考官由 league_loop 在每段主模型结束后后台跑（ppo 内的关掉）；剥削者一半用主卡组（exploit_v2.yaml），
# 比这一代起点胜率高 10 个百分点就入选、高 20 个就开新一代
cd /root/cr-rl
PY=/root/miniconda3/bin/python
log() { echo "$(date '+%m-%d %T') $*" >> runs/chain.log; }
log "M4 v2 续跑：league_loop（主模型 1,200 万步 ↔ 剥削者 400 万步，剥削者按进步入选）+ 卡组 PFSP + 每段后跑固定考官 v2"
for p in $(pgrep -f "cr_ai.keepalive_h"); do kill $p; done
$PY -u -m cr_ai.tools.league_loop --main-run runs/ppo_v2_m4 --main-init runs/ppo_v2/ppo_latest.pt \
    --main-config cr_ai/configs/ppo.yaml cr_ai/configs/m4.yaml cr_ai/configs/arch_v2.yaml \
    --exp-config cr_ai/configs/exploit.yaml cr_ai/configs/exploit_v2.yaml \
    --overrides-file runs/m4_common.txt --main-overrides-file runs/m4_main.txt --main-log runs/ppo_v2_m4.log --exp-log runs/exp_v2.log \
    --exp-prefix runs/exp_v2 --main-steps 12e6 --exp-steps 4e6 --gen-max 12e6 --add-gain 0.10 --success-gain 0.20 \
    --gauntlet-examiners cr_ai/configs/examiners_v2.yaml --gauntlet-games 100 --gauntlet-workers 6 --no-exploit >> runs/league_loop_v2.log 2>&1
log "league_loop 结束 EXIT=$?"
