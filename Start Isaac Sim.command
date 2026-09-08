#!/bin/bash
# Double-click on macOS. Uses the saved RunPod credential and opens Run Center.
set -e
cd "$(dirname "$0")"
exec python3 scripts/isaac/launch.py "$@"
