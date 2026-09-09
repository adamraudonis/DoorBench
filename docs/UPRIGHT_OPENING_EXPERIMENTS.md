# Upright opening: experiment and portability guide

These are development experiments on `db0055_swing_single`, using H1 with two
Shadow hands and the corrected `shadow-loopback-v2` mechanics. They use privileged
simulator state. They are not a learned vision/tactile policy or a catalogue score.
See the [current execution ledger](DEXTEROUS_NEXT_STEPS.md) for live status.

## What is established

| Experiment | Evidence | Limit |
|---|---|---|
| Native panel006 — superseded qualification | Whole-handle re-audit failed despite33 runtime and12 earlier checks; continuous 120 s from closed door, upright grasp, lever operation, intentional release, supported 42.5° opening | Fixed feet during opening; no traversal |
| Native panel007 | Supported segment handoff and contact accounting reproduced | Failed to reach the wider target; stalled at 43.8° |
| Native panel008 | Contact accounting reproduced; recorded hinge-torque analysis explains the stall | Failed to reach the wider target; stopped at 63.0° |
| Isaac pressure014 | Independent audit and off-pod archive verified | Failed grasp; recorded thumb contact on an excluded surface; stopped at 27.4 s |
| Isaac lead015 | Its entire 20 s physical prefix exactly matches pressure014 | The lead limit never intervened; the identical failure is not a correction |

[Original lever-only opening report](evidence/native-standing-panel-006.json) ·
[Wider-opening force diagnosis](evidence/native-standing-panel-008.json) ·
[Isaac contact failure](evidence/isaac-standing-pressure-014.json) ·
[Controlled comparison](evidence/isaac-standing-lead-015.json).

## Run a fresh partial-opening comparison

First use the [one-click environment preparation](ISAAC_ONE_CLICK.md). Require
its actual `ready: true` receipt and corrected v2 mechanics. Use a frozen source
bundle and its source manifest, not a checkout another process can edit. The
standing coordinator verifies every source hash before launching physics.

Arm and verify the owned pod's local and remote teardown guards using the
[RunPod procedure](RUNPOD.md). The coordinator deadline must precede the guard
and leave an evidence-export reserve. A deadline argument alone does not arm a
teardown guard. Leave another worker's node untouched.

Run the coordinator from the prepared source directory. Replace the capitalized
paths and deadline with the values from that preparation and guard receipt:

```sh
python3 scripts/isaac/run_standing_operation.py \
  --source FROZEN_SOURCE \
  --ready READY_JSON \
  --reference SOURCE_REFERENCE_JSON \
  --output NEW_RUN_DIRECTORY \
  --work WORKSPACE \
  --deadline-unix VERIFIED_COORDINATOR_DEADLINE \
  --actual-material-pads \
  --material-pad-profile measured-pressure-v1 \
  --operation-leaf-target-rad .082
```

The coordinator rescreens geometry on the destination, runs a 36 s native
prerequisite, independently audits contacts, then admits the Isaac trial. A
native failure blocks Isaac. Final task success requires the actual Isaac result
and its independent audit. Register the run and attach the evidence collector
before launch, as described in the [Run Center guide](GPU_RUN_CENTER.md).
Recorded operation runs now save both `live-isaac.mp4` and
`live-isaac-hand.mp4`; neither video substitutes for the physical audit.

Declare one new experimental change before running it:

- `--operation-leaf-lead-limit-rad .012` limits the motor reference's lead over
  measured leaf travel. This particular setting did not change the failed
  pressure014 trajectory. Do not describe a configured limit as an intervention
  without evidence that it bound the reference.
- `--operation-operator-follow-after-leaf-rad .02` blends the hand's operator
  reference toward the measured handle rotation over one second, after actual
  leaf clearance. This allows the handle spring to return instead of continuing
  to demand full depression. Follow016 passed its native prerequisite; its Isaac
  outcome was pending when this guide was written.

The declared partial-opening acceptance remains 0.075–0.10 rad. These switches
are restricted to standalone operation experiments; they do not silently alter
an existing full-sequence controller.

## Qualify a wider panel continuation

1. Start from an archived actual pose, with the robot/door bytes verified.
2. Generate a fresh unstepped whole-body path with
   `scripts/dexterous/screen_whole_body_panel.py`. Numerical warm starts must match
   the exact source state and aperture grid. They confer no qualification.
3. Run `scripts/dexterous/audit_whole_body_panel_screen.py` on at least 2,001
   samples. Check hand and foot frames, joint limits, upright posture, clearance,
   and derivative bounds. Sampled endpoint fits alone are insufficient.
4. Bind the resulting target-plan SHA-256 into the withdrawal configuration.
   Each next segment requires a measured half-second supported target hold and
   validates the actual attained state before starting.
5. Execute a fresh physical trial and run the independent grasp/release and
   external-support audits. Keep failed reports and original targets unchanged.

The panel-force controller only requests left-palm load through original motors.
Its target cap is 6 N. The optional `stiction_assist: true` on a subsequent panel
segment increases the integral only when requested progress is positive and
measured motion is stalled. It retains terminal braking and support. A force
increase cannot make an infeasible geometric path acceptable.

## Move to another cluster or robot

Preserve the source manifest, prepared robot/door and motor contracts, controller
configuration, planner inputs, commands, dependency versions, seed/reset state,
and full interval recordings. Absolute XML paths can affect source identity;
rescreen against destination assets instead of editing stored hashes to pass an
audit. The renderer rejects mismatched native XML. For original Isaac evidence,
`render_isaac_contact_snapshot.py` can render measured hand-body transforms and
contact points on the original node (`MUJOCO_GL=egl` for headless rendering).
It labels the result as a source-mesh diagnostic, not an Isaac camera image.

The current H1 controller is **not a generic robot adapter**. A different robot
requires a new actuator/transmission mapping, joint limits and passive-coupling
checks, hand-surface calibration, frame conventions, balance/locomotion skills,
and fresh geometric and physical qualification. Native teacher poses use WXYZ
quaternions; archived PhysX contact-body transforms use XYZW. Preserve that
explicit distinction when building the next adapter.

A larger GPU does not establish larger-batch correctness. First repeat one
qualified episode on the new node, then profile state-only, RGB, and RGB+tactile
batches with matching physics and sensor clocks. Report environment throughput,
peak memory, and success checks together. Complete traversal, repeatability,
sensor-only learning and unseen-door coverage remain separate milestones in the
[approved plan](DEXTEROUS_NEXT_STEPS.md).


## September9 continuation: measured torque and released hand

Panel008's final measured contact moment was0.4557117Nm, below the declared
0.457638Nm hinge friction. The prospective `bounded-7N-v1` continuation load
profile raises the palm target cap from6 to7N and the integral cap from3 to4.5N.
It must explicitly enable terminal stiction assistance. The initial qualified
segment retains the original6N profile. The same motor caps, contact checks,
geometry, terminal braking and69° acceptance remain in effect. This changes
the controller command range, not the success criteria or a physics parameter.

Released-hand planner options can explicitly follow the root, retreat up to12cm,
and permit up to0.35rad orientation deviation after release. These define a
new free-hand motion; they do not alter grasp qualification. Screen021 using
these options failed foot-pose checks and is not eligible for physical replay.
Default world-fixed hand behavior remains unchanged.

Follow016's actual hand camera at24s was inspected at close range. Independent
raw-contact reconstruction under the existing volar profile still rejects
lateral little-finger middle-link contact, and at17.368s thumb loading is absent.
The historical trial remains failed; no prospective profile change is justified
by these observations. A graceful failed-prefix export was requested.


The prospective clearance017 comparison uses handle-frame palm offset
`(-.002,-.003,.0025)m`, versus follow016's `(.004,-.003,.0025)m`.
The lever extends along negative local X from its hub, so this moves the palm
reference6mm away from the hub. It retains measured-pressure-v1, the .082rad
leaf target and .02rad handle-follow threshold. The coordinator now records
and passes this bounded offset identically to both backends. This is a
hypothesis tested by a fresh native prerequisite; it is not an acquired result.


Clearance017 failed the native prerequisite: maximum handle angle0.783645rad,
no opening transition, lost grip and a physical joint-limit failure. Isaac was
not launched. Clearance018 is a prospective smaller2mm shift using offset
`(.002,-.003,.0025)m` and identical frozen source. Neither is qualified.

Use `scripts/dexterous/audit_leaf_contact_moment.py --run RUN --time SECONDS
--output FRESH.json` to diagnose an archived panel interval. It verifies source
XML and chunk hashes, reconstructs only kinematics at the contact epoch and
checks them against recorded body transforms. It decomposes actual leaf-subtree
contact torque into normal, tangent and couple contributions. It does not
recompute dynamics, assert complete torque balance or qualify the task.


## Whole-handle correction

The additional native gate rejects loaded RH contacts against any handle
collider other than the declared grasped lever. Panel006 fails it with45,352
hub patches, maximum7.7866N, from24.234 through69.244s. The earlier45 checks
remain historical evidence of their limited scope, not a qualified whole-hand
episode. The coordinator now requires `native-whole-handle-audit.json` before
any new Isaac dispatch. Full sequence qualification is pending a hub-safe grasp.


Panel009 completed210s at1.200033rad (68.8°), with34 original runtime and12
original independent checks passing. Its new whole-handle check fails with the
same45,352 earlier hub patches as panel006; it is not a qualified episode.
The complete lossless archive was uploaded, downloaded and hash-verified before
local raw files were evicted. Receipts and contact diagnostics are retained.

The next native-only experiment uses `--operation-handle-hub-avoidance`:
`little-finger-3N-v1` starts avoiding the hub within4mm, with800N/m stiffness,
3Ns/m damping and a3N repulsive target cap. The force is mapped only through
existing finger motors and clipped to their original limits; it does not apply
an external wrench. It uses privileged geometric queries in the private teacher
model. Both the original grip audit and full handle-assembly audit are required.


Native hub002 passes18 original runtime checks,6 independent pad checks and the
new assembly check over36s: final aperture0.0758125rad, zero invalid lever
patches, zero hub contacts. Hub001 never stepped: its initial implementation
incorrectly assumed the private acquisition model already contained the hub.
Hub002 explicitly copies the original hub cylinder geometry into that private,
non-colliding calculator and selects proximal/middle little-finger geoms even
when pressure feedback uses distal geoms only. The active plant is unchanged.

A fresh Isaac comparison uses the identical hub descriptor emitted by its new
native prerequisite, with its hash preserved. The coordinator verifies actual
activation of `little-finger-3N-v1` in both backends. Full native assembly gating
is mandatory; old passing reports cannot authorize dispatch. A bounded75-minute
owned-pod window is being armed for native verification, Isaac and collection;
the lifecycle ceiling is explicitly extended to12h to allow this corrective
run after the newly discovered verification gap. This is not permission to
leave the GPU running indefinitely.


Isaac hub020's destination-native prerequisite passed the original checks and
full assembly gate with zero extra handle contacts. Isaac itself stopped during
startup because the new descriptor Path was not JSON-serializable. No physical
Isaac episode is claimed. Hub021 corrects only argument serialization and uses
a fresh source tree and run directory; the original error log remains retained.

`rebase_standing_transfer.py` now regenerates the later receiving-hand route
from a fully audited attained grasp, using the older route only as a posture
preference. Paths001/002 failed interpolation at their first interval. The
planner now eases from the measured initial soft-limit posture into its25mrad
planning margin over20nodes, instead of forcing an immediate wrist/posture jump.
Path003 passes1,001 samples with0.257mm maximum fixed-hand/foot error,0.000665rad
rotation error,2.55° root tilt and zero extra handle penetration. Physics remains
unqualified. The dense auditor now checks hub geometry as well as its original
constraints. Robot joint bounds and physics checks remain unchanged.


The first physical rebased transfer preserved opposed pads and zero hub contact,
but the stance QP reached50,000iterations at36.14s (primal residual0.00012288).
Its status persisted for five physics intervals and the episode remains failed.
A prospective100,000-iteration run changes only the declared solver work budget;
absolute/relative tolerances,25-iteration rho updates, motor limits and all
physical/contact gates remain unchanged. The default is still50,000.


September 9, 19:45 UTC: transfer002 was interrupted at14.942s by local disk
exhaustion, before its transfer phase. It has an explicit execution-failure
receipt and incomplete raw evidence; no physical outcome is inferred. Failed
transfer001 and clearance018/019 archives were uploaded, independently downloaded
and SHA-verified before raw local eviction. Transfer003 repeats the100,000
iteration experiment from the closed-door reset in a fresh directory.

`plan_standing_return.py` now prepares the next lever-return route from a source
that passes runtime, pad and whole-handle checks. It binds the actual recorded
state and model hashes, screens41 candidate nodes, then independently audits401
interpolated poses including hub clearance. It emits a reusable configuration
only after both geometric screens pass. A configuration still requires a fresh
physical run and the same full contact audits; planning is not execution.


Transfer003 completed50s and passed21 runtime checks,6 independent pad checks
and the independent full-handle audit: final aperture0.08240294rad, original
motor/contact limits, zero extra handle contacts. The100k solver budget resolved
the prior convergence failure without changing tolerances. Its attained state
produced a41-node lever-return route passing401 dense samples: maximum fixed
pose error0.293mm, rotation0.000786rad, torso2.311°, reference speed0.399rad/s.
Runtime source/target admission also passed. A fresh64s physical return episode
starts from the original closed-door reset; this candidate is not yet a return
success. The held-finger controller replaces the inner operation finger command,
so inherited hub-profile metadata alone does not establish delivered avoidance
through this phase. Final raw contact audits remain mandatory.


Hub-return001 passed24 runtime checks,6 independent pad checks and the whole
handle audit over64s. The lever and bolt returned to rest with left support
and final aperture0.08855423rad; no extra handle contact or invalid lever patch
was recorded. A63s exact-model hand close-up was personally inspected. This
qualifies the native return component only. The next unstepped withdrawal
candidate is being regenerated from this attained state.
