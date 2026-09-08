# Initialized H1 release, stow, rise and passage

Native **trial 009 passes all 24 checks** over 65 seconds and 32,500 actual
2 ms physics transitions. From an attained open-door state, it releases the
left hand, stows both arms, returns the waist, rises, restarts the official H1
walking policy, aligns with the opening, walks through and stops quietly.
A separate archive audit verifies continuous state and motor delivery across
every transition, and exactly matches 325 sampled contact-epoch body poses.

This is an **initialized continuation from one attained qpos/qvel**, not an
uninterrupted approach/open/traverse demonstration, an Isaac result, or a
vision/touch policy. The controller uses privileged body pose, contact loads
and inverse dynamics. The earlier opening component does not transfer its
qualification to this result. The entire sequence still needs a continuous
physical run from a closed, untouched door.

| Measurement | Trial 009 |
|---|---:|
| Left hand release stage completed | 4.002 s |
| Left / right arm stowed | 8.004 / 12.004 s |
| Waist returned | 16.004 s |
| H1 walking controller received control | 23.006 s |
| Alignment finished / whole-body passage declared | 39.000 / 49.400 s |
| Trailing collision extent at passage declaration | 0.943 m beyond the door plane |
| Maximum torso tilt | 2.983° |
| Maximum nonfoot penetration | 0.090 mm |
| Maximum joint-stop excess, including door joints | 0.714 mrad |
| Failed stance QP updates | 0 |
| Actual motor-command delivery error | 0 Nm |
| Final-second maximum horizontal speed | 0.00326 m/s |
| Final-second horizontal excursion | 1.68 mm |
| Final root XYZ | [0.01574, 1.06324, 1.04745] m |
| Final leaf angle | 1.623 rad |

No root, foot or door pose is written after the single reset. The original 61
motor force caps, masses, joint ranges, passive dynamics, friction, collision
shapes and eight corrected Shadow loopbacks remain unchanged. The stance QP's
support constraints exist only in its planner; the simulated feet remain free.
The independent actual-contact archive contains inherited left-hand/panel
contacts during the first 4 ms and no later nonfoot environment contacts.

## Controller and evidence

The source terminal preserves the exact robot and door qpos, qvel and motor
commands. Its leaf starts at 1.201 rad with 0.511 rad/s velocity. That existing
momentum carries it farther open after release; this continuation commands no
door effort. It does not establish passage for a door that closes before the
body clears it.

A 404-state geometric route on separate unstepped data withdraws the loaded
palm 14 cm along its measured outward normal, stows the left arm, stows the right
arm, and then returns the waist. Reversing that order causes collision. Both
shoulders roll inward by 0.07 rad during stow, giving the hands more clearance.
Reference progression waits for actual hand clearance and bounded arm error.

The original-capped stance controller rises to 1.000 m, then hands its legs to
the pinned **H1**, not G1, policy described in [H1 locomotion](DEXTEROUS_LOCOMOTION.md).
A 3.4-second gait settle and one-second phase fade produce a quiet standing
state. Bounded privileged waypoint commands then align and traverse. The guard
checks the actual leaf aperture (at least 1.57 rad) and native robot/environment
shapes before passage, including receiving-only colliders and recorded gait
root orientations. A sampled static corridor cannot certify future dynamic
clearance; every physical interval remains audited. Completion requires the
whole collision shape beyond the frame and a quiet final second.

Starting with 009, [NativeTransitionRecorder](NATIVE_TRANSITION_AUDIT.md) copies
the actual `mj_step` contact solution, force frames, matching pre-state body
transforms, controls and actual motor forces before any refresh. Endpoint joint
limits and body pose are checked independently. Only `mj_kinematics` refreshes
the active plant's endpoint poses. Inertia, Jacobians and servo calculations use
separate unstepped planning data; its hand loads come from the preceding actual
interval, with explicit interval timestamps. The compressed raw archive retains
every transition for independent reevaluation.

## Reproduce and inspect

The runner validates the exact source robot and door hashes, the corrected
hand mechanics profile, motor contract and original H1 checkpoint. It requires
a fresh output directory and refuses a failed stow screen. Exact local paths,
input hashes and evidence hashes are in the committed
[009 receipt](../results/dexterous/2026-09-08/post-opening/trial-009.json).

```bash
PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/probe_post_opening.py \
  --robot "$H1_V2_XML" --door "$DOOR55_DIRECTORY" \
  --motors "$H1_V2_MOTOR_CONTRACT" \
  --initial-trajectory "$FULL_PUSH_RUN/trajectory.npz" \
  --body-reset configs/dexterous/door55-readiness-v2/body-reset.json \
  --checkpoint "$H1_MOTION_PT" \
  --passage --inward-roll .07 --seconds 65 \
  --output out/post-opening/my-run

PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/audit_post_opening_record.py \
  --trial out/post-opening/my-run

PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/render_post_opening.py \
  --trial out/post-opening/my-run --azimuth 120 --distance 4.5 \
  --reverse-after 42 --reverse-azimuth 270

PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/inspect_post_opening_hands.py \
  --trial out/post-opening/my-run --times 39.952 41.002 42.002 49.402
```

The video renders recorded physical states. Its explicit camera cut exposes
both sides of the wall; it does not change the motion. Hand close-ups at the
strike crossing were inspected separately. Generated robot assets, raw
trajectories and videos remain outside git.

## Retained failures and corrections

| Trial | Outcome |
|---|---|
| 001 | 8 s release prefix; incomplete stow and 66 QP failures. |
| 002 / 003 | Full-height QP rise falls; 214 / 196 QP failures. |
| 004 | Prior-force retention improves support; 20 QP failures and drift remain. |
| 005 | QP failures eliminated; immediate zero-phase restart drifts and violates upright/joint gates. |
| 006 | Motion retained; final report serialization failed. |
| 007 | Identical replay of 006 originally reported 19/19; contact qualification subsequently withdrawn. |
| 008 | Traverses and stops, but right index J4 exceeds its stop by 26.17 mrad, beyond the unchanged 20 mrad gate; 29 failing samples. |
| 009 | Correct actual-step audit and inward stow; 24/24 checks plus 5/5 independent archive checks pass. |

Trials 001–008 recomputed contact forces using `mj_forward` at each endpoint.
Those forces are counterfactual solves, not the forces used by the preceding
physical transition. Their original reports remain immutable; the
[timing review](../results/dexterous/2026-09-08/post-opening/contact-timing-review.json)
withdraws contact qualification while retaining real integrated states and
coherent endpoint geometry.

In 008, the right index struck the included `stop_strike` collider because
forward gait left the stopped/settling posture envelope. Receiving-only collider
hardening fixes a general omission in candidate selection, but that omission
did not cause this door's failure: this scene has no receiving-only geometry.
The new stow was screened against the recorded failing gait before the physical
repeat. These results cover one initialized state, not a robustness matrix.
Changed opening endpoints and the future uninterrupted native/Isaac sequence
require their own actual physical qualification.
