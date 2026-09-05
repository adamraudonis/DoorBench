# Door stops must be mounted: the floating cylinder on db0024 (and every stop/bumper) — part of task G7

Read `handoffs/README.md` first. Start from `master` on a new branch (the larger `attachment-gate.md` brief may
also fix this; coordinate or check whether `master` already has it).

## Why (owner's words)

"https://adamraudonis.github.io/DoorBench/#/door/db0024_swing_single there is some a floating cylinder which is
meant to be a door stop. However I don't like that it is floating. Please ensure nothing floats."

## Current state

`db0024_swing_single`: kinematics `hinge_vertical`, max open 90 deg, `stop: wall_bumper`, closer
`magnetic_hold`. The wall bumper is built in `doorbench/geometry/hinged.py` (search `wall_bumper` / `bumper`:
"wall bumper computed from joint pos at z 0.35"): a cylinder placed where the leaf hits the wall, but not
attached to anything (no base plate / stem touching the wall, or offset from the wall surface).

## Goal

1. Every stop type (wall bumper, floor stop / dome, hinge-pin stop, baseboard stop, overhead stop, magnetic
   hold-open, kick-down holder, bumper rails) is physically mounted: a base plate / stem in contact with the wall,
   floor or frame it belongs to (<= 1 mm gap), the rubber tip where the leaf actually strikes (check with the leaf
   at its max-open angle: the strike point must be on the leaf, not in the air, and not through the leaf).
2. A deterministic check that no static geom hangs in the air: each static/world geom is within 3 mm of the frame,
   wall, floor, ceiling or another static geom (if `doorbench/attachment.py` exists on your branch, add the rule
   there; otherwise add `stops_mounted` to `doorbench/qa.py`).
3. Render before/after for db0024 (`docs/media/stop_db0024_{before,after}.jpg`).

## Done when

All stop-bearing doors pass the new check; regenerate -> 1000 signed off; clearance 1000/1000; tests green.
