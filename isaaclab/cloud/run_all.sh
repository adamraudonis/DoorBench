#!/usr/bin/env bash
# The whole DoorBench Isaac Lab pipeline in one unattended command (task board I4).  Run on the GPU box:
#
#   source isaaclab/cloud/env.sh && tmux new -d -s run 'bash isaaclab/cloud/run_all.sh > logs/run_all.log 2>&1'
#
# Stages (each writes a STAGE_<name>_EXIT=<code> marker to the log and the next stage runs regardless):
#   validate  headless Isaac Sim import + settle + actuate of door.usda and door_rl.usda  (VALIDATE_LIMIT doors)
#   train     RSL-RL PPO on DoorBench-Open-Hand-v0                                    (TRAIN_ENVS x TRAIN_ITERS)
#   hero      wide render of hundreds of different doors being opened at once -> docs/media/isaaclab_hero.{png,mp4}
#   eval      the trained policy AND a random policy over all 1000 doors (EVAL_SEEDS seeds) -> results/*.json
# Env overrides: VALIDATE_LIMIT (40), TRAIN_ENVS (1024), TRAIN_ITERS (300), EVAL_SEEDS (3), TASK (DoorBench-Open-Hand-v0).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
mkdir -p logs results docs/media
VALIDATE_LIMIT="${VALIDATE_LIMIT:-40}"; TRAIN_ENVS="${TRAIN_ENVS:-1024}"; TRAIN_ITERS="${TRAIN_ITERS:-300}"
EVAL_SEEDS="${EVAL_SEEDS:-3}"; TASK="${TASK:-DoorBench-Open-Hand-v0}"
t0=$(date +%s); stage() { echo "== [$(date -u +%H:%M:%S)] $*"; }

stage "validate ($VALIDATE_LIMIT doors, both USD kinds)"
$ILAB scripts/isaaclab/validate_usd_isaacsim.py --all --headless --limit "$VALIDATE_LIMIT" --batch 20 --out assets/usd_validation_isaacsim.json
echo "STAGE_validate_EXIT=$?"

stage "train $TASK ($TRAIN_ENVS envs x $TRAIN_ITERS iterations)"
$ILAB scripts/isaaclab/train.py --task "$TASK" --headless --num_envs "$TRAIN_ENVS" --max_iterations "$TRAIN_ITERS"
echo "STAGE_train_EXIT=$?"
CKPT=$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1); echo "checkpoint: ${CKPT:-none}"

stage "hero shot"
$ILAB scripts/isaaclab/record_hero.py --task "$TASK" --headless --enable_cameras ${CKPT:+--checkpoint "$CKPT"}
echo "STAGE_hero_EXIT=$?"

stage "eval: random policy over all doors ($EVAL_SEEDS seeds)"
$ILAB scripts/isaaclab/eval_all_doors.py --headless --random --seeds "$EVAL_SEEDS" --out results/isaaclab_random.json
echo "STAGE_eval_random_EXIT=$?"
if [ -n "${CKPT:-}" ]; then
  stage "eval: trained policy $CKPT over all doors ($EVAL_SEEDS seeds)"
  $ILAB scripts/isaaclab/eval_all_doors.py --headless --checkpoint "$CKPT" --seeds "$EVAL_SEEDS" --out results/isaaclab_ppo_hand.json
  echo "STAGE_eval_policy_EXIT=$?"
fi
echo "== RUN_ALL DONE in $(( ($(date +%s) - t0) / 60 )) min"
