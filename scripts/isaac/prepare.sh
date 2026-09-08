#!/usr/bin/env bash
# Same entry point on RunPod and an SSH-accessible Linux GPU cluster.
set -euo pipefail
DB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${DOORBENCH_WORK:-/workspace}"
cd "$DB"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color
export DOORBENCH_GENERATE_IDS="${DOORBENCH_GENERATE_IDS:-db0055_swing_single}"
export PATH="$HOME/.local/bin:$PATH"
R="$DB/out/isaac-ready"
mkdir -p "$R"
rm -f "$R/ready.json"
trap 'status=$?; if [ "$status" -ne 0 ]; then echo "READINESS_FAILED exit=$status"; fi' EXIT
phase() {
  echo "$1"
  python3 "$DB/scripts/isaac/status.py" "$DB/out/launch/pipeline.json" "$1"
}
STAMP="$W/doorbench-runtime-bootstrap.sha256"
HASH=$(sha256sum scripts/pod_bootstrap.sh | cut -d ' ' -f 1)
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$HASH" ] || ! "$W/venv/bin/python" scripts/isaaclab/check_g1_runtime.py; then
  phase '== [1/5] Install or repair pinned runtime'
  bash scripts/pod_bootstrap.sh
  echo "$HASH" > "$STAMP"
else
  phase '== [1/5] Verify cached runtime'
  "$W/venv/bin/python" scripts/isaaclab/check_g1_runtime.py
  "$W/asset-venv/bin/python" -m pip --version >/dev/null 2>&1 || true
  uv pip install --python "$W/venv/bin/python" --no-deps -e "$DB"
  uv pip install --python "$W/asset-venv/bin/python" --no-deps -e "$DB"
fi
AP="$W/asset-venv/bin/python"
IP="$W/venv/bin/python"
# A new source checkout also needs an activation file when runtime installation
# is reused. Explicit PYTHONPATH prevents another checkout's editable install
# from silently selecting a different DoorBench implementation.
mkdir -p "$DB/isaaclab/cloud"
{
  printf 'source %q\n' "$W/venv/bin/activate"
  printf 'export ISAACLAB_DIR=%q DOORBENCH_DIR=%q DOORBENCH_ASSETS=%q PYTHONPATH=%q\n' "$W/IsaacLab" "$DB" "$DB/assets" "$DB"
  echo 'export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y TERM=xterm-256color'
} > "$DB/isaaclab/cloud/env.sh"
export PYTHONPATH="$DB${PYTHONPATH:+:$PYTHONPATH}"
phase '== [2/5] Generate and validate the development door'
if ! "$AP" scripts/isaac/check_assets.py assets --ids "$DOORBENCH_GENERATE_IDS"; then
  "$AP" scripts/generate_dataset.py --out assets --ids "$DOORBENCH_GENERATE_IDS" --workers 1 --no-thumbs
fi
"$AP" scripts/isaac/check_assets.py assets --ids "$DOORBENCH_GENERATE_IDS"
phase '== [3/5] Prepare pinned robot and explicit motor contract'
"$AP" scripts/dexterous/setup_robot.py --upstream "$W/humanoid-bench" --output "$R/h1-shadow.xml"
"$AP" scripts/dexterous/prepare_isaac_import.py --robot "$R/h1-shadow.xml" --output "$R/h1-import.xml"
"$AP" scripts/isaac/make_smoke_reference.py --robot "$R/h1-shadow.xml" --output "$R/reference.json"
phase '== [4/5] Import into Isaac Sim and validate live PhysX'
IMPORT="$R/import-$(date -u +%Y%m%dT%H%M%S)"
"$IP" -u scripts/dexterous/isaac_import_audit.py --mjcf "$R/h1-import.xml" --output "$IMPORT"
# Never use a previous trial as evidence for this invocation.
TRIAL="$R/trial-$(date -u +%Y%m%dT%H%M%S)"
"$IP" -u scripts/dexterous/isaac_opening.py --robot-usd "$IMPORT/robot.usda" \
  --door-usd "$DB/assets/doors/db0055_swing_single/door.usda" --motors "$R/h1-import.motors.json" \
  --reference "$R/reference.json" --output "$TRIAL" --seconds 1 --headless --device cuda:0 --record
"$AP" scripts/dexterous/check_isaac_fk.py --robot "$R/h1-shadow.xml" --run "$TRIAL"
phase '== [5/5] Save readiness receipt and exact runtime'
uv pip freeze --python "$IP" > "$R/requirements-isaac.lock.txt"
uv pip freeze --python "$AP" > "$R/requirements-assets.lock.txt"
"$AP" scripts/isaac/write_ready.py --trial "$TRIAL" --output "$R/ready.json"
echo 'ISAAC_ENVIRONMENT_READY'
