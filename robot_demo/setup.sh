#!/usr/bin/env bash
# Fetch the third-party robot model + pretrained locomotion policy used by run_g1_door.py.
# Everything lands in robot_demo/third_party/ (git-ignored); all of it is BSD-3-Clause, see LICENSES.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

MENAGERIE_SHA=e4049d0a3bfd58d2a3081614e6777d4007e3f86a   # google-deepmind/mujoco_menagerie, 2026-09-01
RLGYM_SHA=276801e46c5d433564f24658bac64f254b7d2d4b       # unitreerobotics/unitree_rl_gym, 2025-07-25

if [ ! -d "$TP/mujoco_menagerie/.git" ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TP/mujoco_menagerie"
  git -C "$TP/mujoco_menagerie" sparse-checkout set unitree_g1 unitree_h1
fi
git -C "$TP/mujoco_menagerie" checkout -q "$MENAGERIE_SHA"

if [ ! -d "$TP/unitree_rl_gym/.git" ]; then
  git clone --filter=blob:none https://github.com/unitreerobotics/unitree_rl_gym.git "$TP/unitree_rl_gym"
fi
git -C "$TP/unitree_rl_gym" checkout -q "$RLGYM_SHA"

# Python deps (CPU torch is enough: the policy is a small LSTM + MLP)
PY="${PYTHON:-python}"
if "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m pip install torch pyyaml imageio imageio-ffmpeg
else
  uv pip install --python "$PY" torch pyyaml imageio imageio-ffmpeg
fi
echo "ok: $(git -C "$TP/mujoco_menagerie" rev-parse --short HEAD) menagerie, $(git -C "$TP/unitree_rl_gym" rev-parse --short HEAD) unitree_rl_gym"
