#!/usr/bin/env bash
# Evaluate a checkpoint over all 1000 doors x N seeds -> results/isaaclab_<task>_<date>.json
#   bash isaaclab/cloud/eval.sh logs/rsl_rl/doorbench_hand/<run>/model_300.pt [--seeds 3] [--doors all] [--task DoorBench-Open-Hand-Play-v0]
#   bash isaaclab/cloud/eval.sh --random                            # random-policy baseline
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
if [ $# -ge 1 ] && [ -f "$1" ]; then
  CKPT="$1"; shift
  $ILAB scripts/isaaclab/eval_all_doors.py --headless --checkpoint "$CKPT" "$@"
else
  $ILAB scripts/isaaclab/eval_all_doors.py --headless "$@"
fi
