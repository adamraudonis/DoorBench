#!/usr/bin/env bash
# The whole DoorBench Isaac Lab pipeline in one unattended command (task board I4 / I5).  Run on the GPU box:
#
#   source isaaclab/cloud/env.sh && tmux new -d -s run 'bash isaaclab/cloud/run_all.sh > logs/run_all.log 2>&1'
#
# The GPU's job is to find door defects (I5 Isaac parity gate); training is only a data-validation tool and is OFF
# by default.  Stages (each writes a STAGE_<name>_EXIT=<code> marker to the log and the next stage runs regardless;
# a stage that is switched off writes STAGE_<name>_EXIT=skipped):
#   validate  the Isaac parity runner over ALL doors when the repo has one (see PARITY_RUNNER below), otherwise the
#             headless import + settle + actuate check scripts/isaaclab/validate_usd_isaacsim.py over ALL doors
#             (both USD kinds)                                                           -> results/isaac_parity/ | assets/usd_validation_isaacsim.json
#   train     [TRAIN=1] RSL-RL PPO on $TASK                                               (TRAIN_ENVS x TRAIN_ITERS)
#   hero      [HERO=1, default = TRAIN] wide render of hundreds of doors being opened at once -> docs/media/isaaclab_hero.{png,mp4}
#   eval      [EVAL=1, default = TRAIN] random policy AND (if a checkpoint exists) the trained policy over all doors -> results/*.json
# Env overrides:
#   TRAIN (0)            1 = run the train stage (and, unless overridden, hero + eval)
#   HERO, EVAL           default to $TRAIN
#   VALIDATE_LIMIT (0)   0 = all doors; N = first N doors (quick probe, e.g. 40)
#   VALIDATE_BATCH (20)  doors per Isaac stage in the fallback validator
#   PARITY_RUNNER        path of the Isaac-side parity runner script; auto-detected from scripts/isaaclab/*parity*.py,
#                        scripts/parity/*isaac*.py and scripts/*parity*.py when unset.  PARITY_ARGS overrides its
#                        arguments (default: "--all --headless", plus --limit $VALIDATE_LIMIT when non-zero).
#   TRAIN_ENVS (1024), TRAIN_ITERS (300), EVAL_SEEDS (3), TASK (DoorBench-Open-Hand-v0)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
set +e   # _env.sh switches -e on; here a failing stage must be reported and the next stage must still run
mkdir -p logs results docs/media
TRAIN="${TRAIN:-0}"; HERO="${HERO:-$TRAIN}"; EVAL="${EVAL:-$TRAIN}"
VALIDATE_LIMIT="${VALIDATE_LIMIT:-0}"; VALIDATE_BATCH="${VALIDATE_BATCH:-20}"
TRAIN_ENVS="${TRAIN_ENVS:-1024}"; TRAIN_ITERS="${TRAIN_ITERS:-300}"
EVAL_SEEDS="${EVAL_SEEDS:-3}"; TASK="${TASK:-DoorBench-Open-Hand-v0}"
t0=$(date +%s); stage() { echo "== [$(date -u +%H:%M:%S)] $*"; }
skip() { stage "$1 (off: set $2=1 to enable)"; echo "STAGE_$1_EXIT=skipped"; }
# run_stage <name> <cmd...>: run one stage, tee its output to logs/stage_<name>.log and print STAGE_<name>_EXIT=<code>.
# Isaac Sim's app shutdown can mask a Python traceback's exit status (the 2026-09-05 pod run printed STAGE_train_EXIT=0
# right after the dump_pickle ImportError), so a traceback in the output counts as a failure too.
run_stage() {
  local name="$1"; shift
  local log="logs/stage_${name}.log"
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = "0" ] && grep -q "Traceback (most recent call last)" "$log"; then
    echo "STAGE_${name}_NOTE=traceback in the output although the process exited 0 (simulator shutdown masked it)"; rc=1
  fi
  echo "STAGE_${name}_EXIT=$rc"
  return "$rc"
}

# ---------------------------------------------------------------------------------------------------- validate
find_parity_runner() {
  if [ -n "${PARITY_RUNNER:-}" ]; then echo "$PARITY_RUNNER"; return; fi
  local f
  for f in scripts/isaaclab/*parity*.py scripts/parity/*isaac*.py scripts/*parity*isaac*.py scripts/*parity*.py; do
    [ -f "$f" ] && { echo "$f"; return; }
  done
}
PARITY_RUNNER="$(find_parity_runner)"
LIMIT_ARG=""; [ "$VALIDATE_LIMIT" != "0" ] && LIMIT_ARG="--limit $VALIDATE_LIMIT"
if [ -n "$PARITY_RUNNER" ] && [ -f "$PARITY_RUNNER" ]; then
  stage "validate: Isaac parity runner $PARITY_RUNNER over ${VALIDATE_LIMIT/#0/all} doors"
  # shellcheck disable=SC2086
  run_stage validate $ILAB "$PARITY_RUNNER" ${PARITY_ARGS:---all --headless $LIMIT_ARG}
else
  stage "validate: no parity runner found, falling back to validate_usd_isaacsim.py (${VALIDATE_LIMIT/#0/all} doors, both USD kinds)"
  # shellcheck disable=SC2086
  run_stage validate $ILAB scripts/isaaclab/validate_usd_isaacsim.py --all --headless $LIMIT_ARG --batch "$VALIDATE_BATCH" --out assets/usd_validation_isaacsim.json
fi

# ------------------------------------------------------------------------------------------------------- train
CKPT=""
if [ "$TRAIN" = "1" ]; then
  stage "train $TASK ($TRAIN_ENVS envs x $TRAIN_ITERS iterations)"
  run_stage train $ILAB scripts/isaaclab/train.py --task "$TASK" --headless --num_envs "$TRAIN_ENVS" --max_iterations "$TRAIN_ITERS"
  CKPT=$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1); echo "checkpoint: ${CKPT:-none}"
else
  skip train TRAIN
fi

# -------------------------------------------------------------------------------------------------------- hero
if [ "$HERO" = "1" ]; then
  stage "hero shot"
  run_stage hero $ILAB scripts/isaaclab/record_hero.py --task "$TASK" --headless --enable_cameras ${CKPT:+--checkpoint "$CKPT"}
else
  skip hero HERO
fi

# -------------------------------------------------------------------------------------------------------- eval
if [ "$EVAL" = "1" ]; then
  stage "eval: random policy over all doors ($EVAL_SEEDS seeds)"
  run_stage eval_random $ILAB scripts/isaaclab/eval_all_doors.py --headless --random --seeds "$EVAL_SEEDS" --out results/isaaclab_random.json
  if [ -n "${CKPT:-}" ]; then
    stage "eval: trained policy $CKPT over all doors ($EVAL_SEEDS seeds)"
    run_stage eval_policy $ILAB scripts/isaaclab/eval_all_doors.py --headless --checkpoint "$CKPT" --seeds "$EVAL_SEEDS" --out results/isaaclab_ppo_hand.json
  fi
else
  skip eval EVAL
fi
echo "== RUN_ALL DONE in $(( ($(date +%s) - t0) / 60 )) min"
