#!/usr/bin/env bash
# Headless Isaac Sim validation of every door USD (I1).  Usage: bash isaaclab/cloud/validate.sh [--limit 50] [--which rl]
# Writes assets/usd_validation_isaacsim.json (static pxr validation is in assets/usd_validation.json).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python scripts/isaaclab/validate_usd_static.py --workers 8 --quiet --out assets/usd_validation.json
$ILAB scripts/isaaclab/validate_usd_isaacsim.py --all --headless "$@"
