#!/usr/bin/env bash
# Hero shot: 512 environments with 512 different doors in one scene -> docs/media/isaaclab_hero.{png,mp4}
#   bash isaaclab/cloud/hero.sh                                   # random policy, all doors
#   bash isaaclab/cloud/hero.sh --checkpoint logs/rsl_rl/doorbench_hand/<run>/model_300.pt --num_envs 512
#   bash isaaclab/cloud/hero.sh --task DoorBench-Open-G1-v0 --checkpoint logs/rsl_rl/doorbench_g1/<run>/model_3000.pt
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
$ILAB scripts/isaaclab/record_hero.py --headless --enable_cameras "$@"
