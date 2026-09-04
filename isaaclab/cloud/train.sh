#!/usr/bin/env bash
# Train a DoorBench policy (headless).  Examples:
#   bash isaaclab/cloud/train.sh --task DoorBench-Open-Hand-v0 --num_envs 1024 --max_iterations 300
#   bash isaaclab/cloud/train.sh --task DoorBench-Open-G1-v0 --num_envs 2048 --max_iterations 3000 --doors easy-100 --video
# Logs + checkpoints: logs/rsl_rl/<experiment>/<date>/model_<iter>.pt ; tensorboard --logdir logs/rsl_rl
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
$ILAB scripts/isaaclab/train.py --headless "$@"
