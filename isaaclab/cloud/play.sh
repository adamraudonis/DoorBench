#!/usr/bin/env bash
# Play a checkpoint and record a video (headless).  Usage:
#   bash isaaclab/cloud/play.sh --task DoorBench-Open-Hand-Play-v0 --checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
$ILAB scripts/isaaclab/play.py --headless --video "$@"
