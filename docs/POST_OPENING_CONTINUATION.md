# Release, stow, rise and restart: one initialized H1 component

Native trial **007 passes 19/19 checks** over 34 seconds (17,001 timestamped
states, including reset). It starts from the *attained* terminal qpos **and qvel**
of `full-push-portable-001`, physically releases the left palm, retracts each
arm, closes the hands, turns the waist back, rises to 1.000 m and hands control
to the official H1 walking policy. A short gait-settle/phase-brake sequence ends
in a quiet standing posture at 1.047 m.

This is an **initialized continuation**, not an uninterrupted approach/open/
traverse demonstration, an Isaac result, or a vision/touch policy. World pose,
contact loads and unstepped inverse dynamics are privileged controller inputs.
No root, foot or door pose is written after reset. All efforts go through the
original 61 bounded motors. Robot masses, passive dynamics, geometry, friction,
joint limits and eight corrected Shadow loopbacks stay unchanged.

| Measurement | Trial 007 |
|---|---:|
| Left palm released and cleared | 4.002 s |
| Left / right arm stowed | 8.004 / 12.004 s |
| Waist returned | 16.004 s |
| Walking policy received control | 23.006 s |
| Maximum torso tilt | 2.060° |
| Maximum nonfoot penetration | 0.090 mm |
| Maximum joint-stop excess, including door joints | 0.714 mrad |
| Failed stance QP updates | 0 |
| Final-second maximum horizontal speed | 0.0154 m/s |
| Final-second horizontal excursion | 7.91 mm |
| Final leaf angle | 1.623 rad |

The source leaf starts at 1.201 rad with 0.511 rad/s velocity. Its existing
momentum carries it farther open after hand release; no opening torque is added.
The body's final XY is `[-0.12117, -0.71411]` m, compared with the initial
`[-0.15600, -0.58732]` m. This approximately 13 cm settling displacement matters
for subsequent navigation. Passage remains untested: the body is outside the
previously screened straight corridor, and the door must remain sufficiently
open during any later crossing.

## What changed

A direct joint interpolation was rejected by the static screen. The accepted
404-state route first withdraws the loaded hand 14 cm along the measured contact
normal, moves the left arm, moves the right arm, then turns the waist. Reversing
that order drives the hand into the door. The screen uses a separate unstepped
`MjData`; each live 2 ms step still independently checks actual contact, limits,
uprightness, finite state and motor caps with refreshed post-step geometry.

Reference progress waits for measured left-hand clearance and sufficiently
small arm motor error. The stance QP only computes motor forces. Its support
polygons are planning assumptions, not simulator constraints. Failed numerical
updates retain the last bounded effort and are counted as failures. A six-case
solver comparison selected `rho=.001`, adaptive update interval 25 and 100,000
maximum iterations; residual tolerances remain the existing 1e-4 values.

Standing directly to 1.035 m with the fixed-foot QP failed. Rising to 1.000 m and
handing over to the H1 gait controller works for this attained posture. Immediate
zero-phase control drifts; 3.4 s of gait phase followed by a one-second phase
fade produces the qualified stop. The checkpoint is the pinned **H1**, not G1,
policy documented in [H1 locomotion](DEXTEROUS_LOCOMOTION.md).

## Reproduce and inspect

Use the established native Python environment, original H1 checkpoint, corrected
robot audit, matching motor contract, and original terminal trajectory. The
runner verifies that the source manifest's robot and door XML hashes match the
active continuation plant. It refuses an unscreened route or reused output
folder. Its output freezes input and source hashes, the geometric route, all
physics records, and the exact terminal state.

```bash
PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/probe_post_opening.py \
  --robot "$H1_V2_XML" \
  --door "$DOOR55_DIRECTORY" \
  --motors "$H1_V2_MOTOR_CONTRACT" \
  --initial-trajectory "$FULL_PUSH_RUN/trajectory.npz" \
  --body-reset configs/dexterous/door55-readiness-v2/body-reset.json \
  --checkpoint "$H1_MOTION_PT" \
  --seconds 34 --output out/post-opening/my-run

PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/render_post_opening.py \
  --trial out/post-opening/my-run
```

The video renders recorded physical states; it never generates the motion.
The committed [trial 007 receipt](../results/dexterous/2026-09-08/post-opening/trial-007.json)
contains exact development paths and evidence hashes. Generated robot assets,
trajectory arrays and videos are excluded from git.

## Retained failures

| Trial | Outcome |
|---|---|
| 001 | 8 s release prefix physically safe; incomplete stow and 66 QP failures. |
| 002 |Full-height QP rise falls; 214 QP failures. |
| 003 |Measured-hand-load variant still falls during full-height rise; 196 QP failures. |
| 004 |Preserving prior forces fixes support continuity; 20 QP failures and substantial backward drift remain. |
| 005 |Solver failures eliminated; immediate zero-phase restart drifts and violates upright/joint gates. |
| 006 |Complete motion records retained; final report serialization failed. |
| 007 |Clean identical replay of 006 after serialization fix; 19/19 checks pass. |

These are development trials on one attained state, not a robustness matrix.
The full opening trajectory that supplied the reset had stale derived-pose
logging; this component refreshes poses and contacts at reset and after every
physical step and does not inherit the earlier component's qualification.
A changed aperture, stance or grasp endpoint must be geometrically screened and
physically tested again. Continuous passage and the Isaac port remain follow-up
work.
