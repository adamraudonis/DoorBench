# Continuation on the revised Door55 asset

The earlier continuation 009/010 used door XML `d3b367b62eed…`. The continuous
walking/opening 002 run used `5b12a0a19c8b…`, with different handle, strike and
latch geometry/mechanics. Its final root also has a different heading. The old
continuation result does not qualify this revised scene.

`PostOpeningTeacher(..., stow_profile='sequential-v2')` adds an explicit alternate
stow route. The default remains `original-v1`, preserving the earlier force
behavior. A two-second replay of009 still matches within 2.42e−13 Nm. Each profile
builds and screens its route from the actual newly attained state; it cannot
accept a frozen successful pose in place of current measurements.

The revised route releases along the measured panel normal, then closes the
left fingers, reorients the left shoulder yaw, and lowers that arm. The right
fingers close next; the right shoulder lifts before its yaw turns and the arm
lowers. The torso returns last. All 909 static samples retain the original
3 mm nonfoot penetration and 20 mrad joint/loopback bounds. Waypoints are motor
targets; no active root or foot pose is written.

The initial original-route preflight was rejected before physics: 27/404 samples
had thumb/slab or elbow/hip collisions. The bounded search's unsuccessful
alternatives remain in `out/post-opening/route-screen-001/report.json`.

## Reproduce an initialized continuation

```bash
PYTHONPATH=. "$ASSET_PYTHON" scripts/dexterous/probe_post_opening.py \
  --robot "$ROBOT_V2_XML" --door "$ACTUAL_DOOR_DIRECTORY" \
  --motors "$MOTOR_CONTRACT" \
  --initial-trajectory "$WALKING_OPENING_002/trajectory.npz" \
  --body-reset configs/dexterous/door55-readiness-v2/body-reset.json \
  --checkpoint "$OFFICIAL_H1_CHECKPOINT" \
  --portable --stow-profile sequential-v2 --phase-seconds 5 \
  --passage --inward-roll .07 --seconds 69 \
  --output out/post-opening/new-revised-door-run
```

Use `--plan-only` for the actual-state initialization and fresh static screen.
The source robot and door hashes must match the archived run's manifest. The
exact terminal velocities are preserved. The new episode starts at local time 0
and is explicitly an **initialized continuation**, not an uninterrupted opening
claim. Walking/opening002 itself failed its right-pad contact gate (28/29).

## Retained development result 011

The first sequential-profile trial used four seconds per arm stage. It completed
65 simulated seconds and traversed, but failed its zero-QP-failures gate: one
stance solve reached its 100,000-iteration limit at 10.620 s during right-arm yaw.
The prior capped leg force was retained for that interval, as declared by the
existing controller. The result remains **24/25, failed**.

All measured physical gates passed: maximum torso tilt 2.540°, joint-stop
excess 0.775 mrad, nonfoot penetration 0.495 mm and motor-delivery error 0. The final
one-second maximum horizontal speed was 0.692 mm/s. The trailing full-body
collision extent was 0.933 m beyond the door plane at traversal completion.
The independent archive audit passed 5/5, checking all 32,500 state transitions,
motor caps/delivery and 325 sampled FK reconstructions.

## Warning coverage

`native_warning_audit.py` records all active MuJoCo warning counters before and
after each actual physical step, including contact/constraint capacity warnings.
Any increase fails the interval and prevents the next portable-controller call
from receiving physical qualification. The initialized counters are retained
separately; they are not confused with later physical intervals.

This closes a coverage gap in the older numerical-warning field, which only
included bad position/velocity/acceleration/control warnings. Two arena warnings
were observed during rejected static route-search poses at time 0. Such warnings
are not physical contact evidence. Trial 011 predates the new per-step warning
receipt and does not receive that additional check retroactively.

The bounded follow-up 012 keeps the same plant, route, motor caps and solver
settings, uses five-second arm stages, and records all warning counters. It
passed **26/26** over 69 seconds and 34,500 actual transitions: zero QP failures,
zero MuJoCo warnings, maximum tilt 2.531°, joint-stop excess 0.775 mrad, nonfoot
penetration 0.495 mm and motor-delivery error zero. Final horizontal speed was
0.287 mm/s and final one-second excursion 0.178 mm. The leaf ended at 1.61546 rad;
the robot root ended at Y=1.05782 m beyond the door plane.
The independent archive audit passed 6/6, including all 34,500 continuous
warning-counter intervals and 345 FK reconstructions with zero frame error.

Compact component, rejection and independent-audit receipts are stored under
`results/dexterous/2026-09-08/post-opening/`. Complete local evidence and the
recorded video are in `out/post-opening/native-012/`. No Isaac, sensor-only actor,
or uninterrupted opening qualification is implied by these component experiments.
