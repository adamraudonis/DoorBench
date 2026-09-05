#!/usr/bin/env bash
# Isaac parity gate, PhysX side (task board I5).  Run on the GPU box after the MuJoCo reference exists:
#
#   PYTHONPATH=$PWD python scripts/parity_reference_mujoco.py --doors all --workers 8     # CPU, anywhere; commit results/parity/mujoco.json
#   bash isaaclab/cloud/parity.sh                        # all doors, both USD kinds -> results/parity/isaac_full.json, isaac_rl.json
#   bash isaaclab/cloud/parity.sh --limit 40 --which full
#   bash isaaclab/cloud/parity.sh --hz 240 --iters 32,8 --tag _dt240   # solver-sensitivity rerun
#   python scripts/parity_compare.py                     # verdicts -> results/parity/compare.json + summary
#
# Env overrides: PARITY_DOORS (all), PARITY_BATCH (20).  Extra arguments are passed to scripts/isaaclab/isaac_parity.py.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
mkdir -p results/parity logs
PARITY_DOORS="${PARITY_DOORS:-all}"; PARITY_BATCH="${PARITY_BATCH:-20}"
if [ ! -f results/parity/mujoco.json ]; then
  echo "[parity] results/parity/mujoco.json missing: per-door inputs will be derived from qa.json (run scripts/parity_reference_mujoco.py for exact parity inputs)" >&2
fi
$ILAB scripts/isaaclab/isaac_parity.py --doors "$PARITY_DOORS" --batch "$PARITY_BATCH" --headless "$@"
echo "STAGE_parity_EXIT=$?"
