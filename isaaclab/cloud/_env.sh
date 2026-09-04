#!/usr/bin/env bash
# shared preamble for the cloud scripts: locate Isaac Lab and DoorBench, define $ILAB (the Isaac Lab python)
set -euo pipefail
DOORBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$DOORBENCH_DIR/isaaclab/cloud/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$DOORBENCH_DIR/isaaclab/cloud/env.sh"
fi
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
if [ -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
  ILAB="$ISAACLAB_DIR/isaaclab.sh -p"
elif [ -x /isaac-sim/python.sh ]; then       # inside the nvcr.io/nvidia/isaac-sim container
  ILAB="/isaac-sim/python.sh"
elif command -v python >/dev/null 2>&1 && python -c "import isaaclab" 2>/dev/null; then
  ILAB="python"
else
  echo "Isaac Lab not found: run isaaclab/cloud/setup.sh first (or set ISAACLAB_DIR)" >&2; exit 1
fi
export DOORBENCH_DIR DOORBENCH_ASSETS="${DOORBENCH_ASSETS:-$DOORBENCH_DIR/assets}" OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
cd "$DOORBENCH_DIR"
