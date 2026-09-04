#!/usr/bin/env bash
# One-command setup of Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + DoorBench on a fresh Ubuntu 22.04 GPU box.
#
#   git clone https://github.com/adamraudonis/DoorBench.git && cd DoorBench && bash isaaclab/cloud/setup.sh
#   source isaaclab/cloud/env.sh
#
# Thin wrapper around scripts/pod_bootstrap.sh, the exact script executed on a RunPod Secure Cloud L40S on
# 2026-09-04 (docs/RUNPOD.md has the replicable pod flow, costs and the troubleshooting log).
# Env overrides: DOORBENCH_WORK (default /workspace: venv + IsaacLab live there), ISAACSIM_VERSION, ISAACLAB_TAG.
set -euo pipefail
DOORBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$DOORBENCH_DIR/scripts/pod_bootstrap.sh" "$@"
