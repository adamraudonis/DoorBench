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


The first regenerated withdrawal failed25/2,001 dense samples: the little
fingertip crossed the original hub by approximately0.25mm during retreat.
Thumb, fixed-pose and final clearance checks otherwise passed. A declared0.04rad
LFJ4 abduction, smoothly blended over candidate times4–5.5s and removed over8–9.5s,
clears this collision. Candidate003 passes all2,001 samples with0.569mm maximum
fixed-pose error,0.003982rad rotation error,3.975° torso tilt,1.144rad/s peak joint
reference speed and47.8mm final hand clearance. `refine_thumb_withdrawal.py`
records the correction and frozen source hashes. The separate80.6s physical
withdrawal run retains original force/geometry gates and starts at the original
closed-door reset. It is not yet qualified.


Withdrawal001 was interrupted during evidence recording near77s by renewed
local storage exhaustion. Its partial observations include transient loss of
left support; no release is qualified. The native runner retained every rich
physics/controller dictionary in memory. `BoundedEvidence` now serializes these
observations into lossless chunks, permitting repeated audit passes without
retaining entire episodes. It atomically exports the existing gzip JSON-array
format. All72 chunks through36s in withdrawal002 match the qualified reference
exactly, including physics states, contacts and forces.37 storage/acquisition/
withdrawal tests passed; runtime qualification remains separate.

Isaac hub021 retains its original distal-pad protocol. A23.002s raw snapshot
shows loaded middle phalanges on the lever. The pre-existing volar-phalange
anatomy formula accepts that one snapshot, including all loaded patches; this
is diagnosis only, never a retrospective upgrade. The coordinator now supports
an explicitly predeclared Isaac grasp profile for a future run, retaining the
stricter distal/whole-handle native prerequisite. Independent auditing rejects
a profile that differs from the recorded launch, and future coordinator success
also requires zero invalid loaded patches over the entire episode, beyond the
existing final sustained-grasp window. No new protocol has yet run.


The complete bounded-recording withdrawal002 failed joint range, loaded lever
surface and full-handle checks despite reaching0.350164rad. Hub contact begins
at69.072s (316 patches,87 intervals, maximum42.93N); invalid distal surfaces run
69.442–70.912s, and a16-interval joint excursion peaks at0.020101rad. All evidence
is retained in a verified draft archive and a public failure receipt. The
attained-grasp-increment alternative failed its geometric screen and was not run.

A composition issue was identified: attained finger tracking overwrites the
operation controller's hub feedback. Withdrawal003 prospectively enables
`final_hub_avoidance`, applying the existing bounded3N feedback after finger
tracking, before the original capped motor handoff. This changes only the
withdrawal phase; it is a fresh physical experiment, not a repaired report.

Isaac hub-follow022 is queued after hub021. It prospectively follows the measured
lever after0.025rad leaf clearance, with hub avoidance active, while retaining
`distal-pad-v1` and the newly mandatory whole-episode loaded-patch gate. No volar
protocol is dispatched. Both shutdown guards were verified for21:40:15UTC,
within the12h allocation lifetime ceiling; the coordinator reserves export time.


### September 9, 20:36 UTC: failed comparisons retained

Isaac hub021 completed all 36 simulated seconds but passed only 17/19 runtime
checks: distal grasp and held partial opening failed. Final aperture was
0.070147 rad. A separate read-only audit reconstructed the raw contacts and
confirmed accounting completeness, with 25,631 invalid loaded patches; this
does not qualify the task. The coordinator exhausted its older export budget
before its own independent audit, so the supplemental audit is identified
separately in [the evidence](evidence/isaac-standing-hub-021.json).

Hub-follow022 failed its native prerequisite with a balance/solver failure.
Hub-follow023 added the predeclared measured-pressure controller: it completed
36 seconds and retained the final opposed grasp, but failed joint range, held
aperture and whole-handle checks. Neither dispatched Isaac. See the
[022](evidence/isaac-standing-hub-follow-022.json) and
[023](evidence/isaac-standing-hub-follow-023.json) receipts. Following a freely
returning lever has not established a stable opening strategy.

Native withdrawal003 also failed both independent contact audits, despite
reapplying hub avoidance after finger tracking. The joint-range gate now
passes. Withdrawal004 tests 4 N receiving-palm support, up from 2.25 N, within
the original controller range. It starts from the same closed-door reset and
retains all existing gates; it is running, not qualified.


Withdrawal004 again shows a transient receiving-palm unload around69s despite
a4N target. Its 80.6 s rollout completed but failed joint range, final left-palm support and both contact audits. The whole-handle audit records 306 extra patches across 84 intervals (69.070–69.316 s). A hand close-up at 69.202 s was personally inspected; see [the failure receipt](evidence/native-standing-hub-withdrawal-004.json). The
attained finger controller damps absolute joint motion while the arm controller
already tracks reference velocity. The optional `finger_velocity_feedforward`
comparison adds the derivative of screened finger positions to that damping
term, mapped through the original coupled motor transmission. It rejects
incomplete/nonfinite inputs and reference speeds above2rad/s, retains the same
stiffness and original motor caps, and defaults off. Five focused hand/withdrawal
tests pass. This is a prospective controller change, not a physical success.

GPU pressure024 removes lever-following from023 while retaining measured pad
pressure and hub avoidance. Its source is the same frozen archive; native and
whole-handle gates still precede Isaac. Both owned shutdown guards were verified
for21:57:21UTC, within the12h allocation ceiling, with collection time reserved.


Pressure024 completed the native rollout with only the held-aperture runtime
check failing: final aperture 0.07421235 rad against the unchanged 0.075 rad
threshold. Its original distal-pad and whole-handle runtime checks passed;
independent reconstruction remains mandatory. Pressure025 commands 0.085 rad
instead of 0.08 rad, with the same success threshold, original motor limits,
controller and frozen source. It waits for024 to close before its own complete
native prerequisite and any Isaac dispatch. Its Isaac budget is3300 wall seconds
with the existing guard/export reserve; no outcome is claimed.


The finger-velocity hypothesis was disproved for this embodiment: all18 right
finger actuators have zero velocity bias, and the acquisition controller adds
zero finger damping. The optional feedforward therefore adds exactly zero
torque. Withdrawal005 reaches the identical terminal physical state as004;
its independent audit is still running. The optional API is tested, but it is
not an effective correction for this motor contract.

A separate direct-release planner now starts from the verified resting grasp,
without adapting an old canonical regrasp. Candidate001 separates material pads
radially, holds body/feet/support pose, then lifts the arm. Its dense audit fails
67/2001 samples because a curled middle fingertip catches the lever during the
lift. Pose errors, upright posture, speed and final clearance otherwise meet
the unchanged checks. Candidate002 instead extends the four fingers first;
it remains an unstepped candidate. Planner source bytes are frozen beside each
report and bound into the independent audit, alongside all prior source checks.


Withdrawal005's162 raw transition chunks are byte-identical to004's verified
archive; the velocity-gain explanation is therefore confirmed rather than
inferred from a matching endpoint. [Failure and equivalence receipt](evidence/native-standing-hub-withdrawal-005.json).

The direct extension candidate002 failed geometry; neutral finger extension
is not clear of the door in this attained wrist orientation. Candidate003
slides the radially separated hand toward the lever's free end before lifting.
It has no invalid contact samples, but fails5 interpolated pose samples, the
2rad/s speed limit and final4cm clearance. Candidate004 adds bounded upright
whole-body IK with fixed feet and left palm; it is awaiting the dense audit.

Pressure025 held the required aperture but lost valid index/middle fingertip
surfaces in its native prerequisite. No Isaac rollout was dispatched.
Pressure026 commands0.0815rad, between the two tested commands, under unchanged
acceptance thresholds. To allow the native gate, Isaac run and export to finish
on the already prepared node, the owned allocation's administrative lifetime
ceiling was explicitly raised from12h to13h; both replacement shutdown guards
were verified. This avoids another bootstrap and adds only a bounded interval.
The current deadline is recorded in the guard-renewal receipt, not an open-ended
allocation. No other agent's node was changed.


Direct-release004 passes all2,001 independent dense samples: maximum fixed-pose
error8.15µm, rotation error0.000024rad, torso tilt2.868°, joint speed0.602rad/s
and final hand clearance56.3mm. It keeps the original body/feet/support-palm
constraints. Native hub-withdrawal006 will test this route with the4N support
profile, actual measured-release phase and full unchanged contact gates from
the original closed-door reset. Geometry is not physical qualification.


Direct physical withdrawal006 passes the whole-handle assembly audit and
finishes near65.7° with upright torso and17.7cm minimum final hand clearance,
but still fails the original distal-surface, joint-range and final support
checks. There are4,010 middle-phalange patches during69.338–72.558s. A diagnostic
using the existing alternative volar formula accepts3,983 of them, but27 still
miss the cylindrical-side margin; the original task remains failed and its
protocol is unchanged. At77.89s the left wrist yaw and right thumb J3 exceed
the0.02rad gate. Actual hand and body close-ups were personally inspected.

Geometric direct-release005 begins the free-end slide during pad separation and
ramps a35mrad thumb-J3 planning margin from the attained pose. It passes all
2,001 dense samples under unchanged tolerances. Physical withdrawal007 will
combine this route with2.25N left support, reducing the uncontrolled opening
that takes the left wrist to its reach limit. The original force caps, anatomy
checks and joint stops remain unchanged.

Controller JSON snapshots now preserve direct references and their planning
reports/audits before initialization. Twelve focused snapshot/hand/standing
tests pass. Withdrawal006 received an explicitly labeled supplemental postlaunch
copy; this is not a retroactive initialization attestation. Model assets and
complete source episode archives remain separate requirements.


### September 9, 21:29 UTC — withdrawal007 remains failed; live Isaac monitoring corrected

Withdrawal007 passes 28/30 runtime checks: the thumb margin and 2.25 N receiving-palm target fix the earlier joint-limit and final support failures. Independent records still reject 703 ring-finger middle-segment patches at 69.524–70.798 s and 30 little-finger hub patches over 23 intervals at 68.976–69.024 s. Final aperture is 0.108101 rad; the hand ends 64.36 mm clear. Actual close-up at 70.002 s was personally inspected. [Scoped evidence](evidence/native-standing-hub-withdrawal-007.json). No gate was relaxed.

Prospective withdrawal008 uses 30 mm rather than 18 mm radial finger separation, retaining the coordinated free-end retreat, thumb margin, and 2.25 N support. Its unstepped 2,001-sample audit passes: 3.64 micrometers maximum palm error, 2.867 degrees torso tilt, 1.091 rad/s maximum reference velocity, 42.75 mm final clearance. Full motor physics and independent audits are running; this is not yet a qualified release.

The local Run Center now recognizes actual Isaac operation records, displays elapsed/target simulated seconds, and waits for independent coordinator acceptance before calling a run complete. Newly registered and live runs take polling priority over unavailable historical nodes; dead archives back off to five minutes. Eighteen focused tests pass. The live Isaac pressure026 page was verified visually, including its hand close-up. Its coordinator deadline is labelled separately from pod teardown. The browser link is http://127.0.0.1:5193/?run=standing-hub-pressure-isaac-026-trial . Pressure026 has lost grip during lever pressing; final records are still pending.


### Moving-leaf release verification

Withdrawal008 also fails: 28/30 runtime checks pass, but independent reconstruction finds 1,337 invalid distal-contract patches and 69 little-finger hub patches. Increasing radial separation alone is insufficient. Final aperture is 0.109503 rad, with 54.43 mm hand clearance; these terminal values do not repair the earlier contact failure. [Evidence](evidence/native-standing-hub-withdrawal-008.json).

A separate `audit_release_leaf_motion.py` now checks non-distal lever and all hub clearances across 3,003 unstepped configurations, including a declared ±0.012 rad leaf envelope that grows over the first three route seconds. It supplements the original static anatomy screen and physical audit. On direct-release006 it detects 228 intersecting configurations, with a worst 5.24 mm penetration; no force or physical success is inferred. A 12 mm early palm retreat away from the door (direct-release007) fails both static and moving-door checks, so it is not dispatched to physics. A 20 mm early upward palm motion is being screened next. Original reports remain unchanged.


Direct-release008, with 20 mm early upward palm travel, passes both geometric screens: 2,001 original samples (4.10 micrometers maximum palm error; 2.870 degrees torso tilt; 1.091 rad/s joint reference speed; 42.75 mm final clearance), and all 3,003 moving-leaf configurations (minimum tested non-distal/assembly clearance 0.124 mm). The latter is a geometric envelope, not proof of dynamic robustness. A source-bound optional admission check now rejects changed audit files, changed source/route inputs and failed moving-leaf screens; ten targeted admission, snapshot, withdrawal and hand tests pass. Native withdrawal009 will test this exact route through the original capped motors and complete contact audits.


Launch correction: withdrawal009 executed no physics. The ad hoc monitor registration created its reserved output directory before the probe checked for a fresh directory. Its error and launch receipt are retained. Withdrawal010 uses the same audited route/configuration and waits for the probe's own PID file before attaching monitoring; it is running with both contact-audit waiters. No output-directory check was weakened.


### Prospective Isaac opening-transition comparison

Pressure026's actual Isaac close-ups at steps 8,500 and 10,500 were personally inspected. At 17.502 s the lever is 0.769656 rad and bolt retraction is 0.011088 m; opening is still withheld by the controller's 0.80 rad trigger. By 21.002 s the palm compliance correction has grown to 0.13544 rad while the lever remains 0.79478 rad; grip is then lost. This supports testing transition timing, but does not prove it is the only cause.

A new explicit `--operation-opening-trigger-rad` option is forwarded identically to native and Isaac controllers. The prospective comparison uses 0.75 rad while retaining the 0.011 m measured bolt threshold, five-second pressing reference, 0.87 rad handle target, original motor limits, and final 0.80 rad operator/0.011 m bolt acceptance gates. The controller starts its continuous opening ramp; it never commands the door directly. Twenty-six operation/withdrawal tests pass, including the requirement that the lower trigger cannot bypass a blocked bolt. No historical failed result is rescored.


Pressure026 completed: 17/19 Isaac runtime checks pass, but sustained fingertip grip and held aperture fail. Maximum handle angle is 0.800545 rad; maximum bolt retraction is 11.536 mm; final aperture is only 0.007311 rad. Independent raw contact accounting passes and finds 6,223 invalid patches. All 427 final output files were verified off-pod. [Actual run evidence and timestamps](evidence/isaac-standing-hub-pressure-026.json). Trigger027 is running its native prerequisite on frozen source `dc8dfa3335efa950a7b178750a9e0db5a2843940d1264115bae92c757ef60c37`. Both guards were renewed for only the owned node; the current teardown deadline is September 9, 22:57:51 UTC, with the coordinator stopping two minutes earlier.

Withdrawal010 remains failed (28/30 runtime; 411 ring-finger middle-segment patches, and 65 little-fingertip hub patches during 68.952–68.992 s). The actual hand frame at 68.962 s was inspected. [Evidence](evidence/native-standing-hub-withdrawal-010.json). Hub feedback had covered only little-finger middle/proximal geometry; a prospective withdrawal-only option now includes the fingertip, retaining the same 3 N feedback cap and original motor limits. The moving-leaf screen also supports a smoothly introduced 2 mm clearance reserve for tracking error. A 30 mm upward retreat passes zero-penetration geometry but misses the new reserve by 0.102 mm at ten configurations, so it is not dispatched. The next 40 mm upward candidate retains the original joint, pose, root and torso bounds; the early-travel search bound is widened to 50 mm, not a physical acceptance limit.


Withdrawal011 passes 29/30 runtime checks and the independent whole-handle audit (zero extra assembly contacts). Its remaining failure is 228 ring-finger middle-segment patches at 68.604–68.802 s. At the first contact, the coupled ring-finger J1/J2 goals lead the actual joints by about 0.0298/0.0296 rad; palm error is about 1.30 mm. No motor clipping occurred. The 40 mm upward path had passed the moving-leaf reserve screen; physical tracking still needs more clearance. [Evidence](evidence/native-standing-hub-withdrawal-011.json). A 50 mm upward candidate is being screened under the same original motor, joint, pose and torso bounds.

Trigger027 passes all native prerequisites but dispatches no Isaac run: queue time left less than the coordinator's required complete physics/export reservation. The failed coordinator record is preserved and its native archive is being verified for retention. Trigger028 repeats the same frozen source and controller parameters with sufficient reservation. The owned guard pair was rearmed for September 9, 23:15:06 UTC (coordinator two minutes earlier); only this allocation's guards were replaced. The administrative lifetime ceiling was increased from 13 to 14 hours to accommodate the measured 49.8-minute Isaac operation and its prerequisites on the already prepared node. This remains a bounded comparison, not permission for an open-ended allocation.


Withdrawal012 (50 mm early upward travel) remains failed: 29/30 runtime checks pass, 321 ring-finger middle-segment patches remain, and independent whole-handle audit passes with zero extra contacts. [Evidence](evidence/native-standing-hub-withdrawal-012.json). More vertical clearance alone did not fix the tracking defect. Prospective withdrawal013 returns to the matched 40 mm route from011 and adds screened distal material-point feedback for all four fingers, using the same bounded 6 N point feedback already used for the thumb and only the original clipped motor outputs. Thumb feedback retains its original output fields and numeric force calculation through a shared implementation. Ten feedback/admission/withdrawal tests pass, including preservation of unrelated motor commands and no physical pose or external-wrench writes. Full runtime/independent checks remain mandatory.

A separate `py-spy` 0.4.1 installation was attempted for a 20 s nonblocking profile of the owned Isaac process. The container denied process inspection before any samples were collected. No performance attribution or batch-capacity claim is derived from this attempt; future profiling must be launched with the process or use in-process timing. No simulator package or running controller was changed.


### Release 013 and bounded segment feedback (2026-09-09)

The four-finger material-point controller did not qualify the release: 29/30 runtime checks passed, but the independent pad audit found 279 invalid loaded ring-middle patches from 68.610 to 68.838 s (peak 0.8634 N). Independent whole-handle accounting passed with zero extra loaded patches. Final leaf angle was 0.09961046 rad; this is not an opening/traversal success.

Trial 014 retains trial 011's screened 40 mm upward route and disables the unsuccessful four-finger point feedback. Its sole added controller is release-only ring-middle/proximal clearance from the measured analytic lever. It uses the existing 4 mm activation distance, 3 N bounded repulsion law, and original capped finger motors; it writes no physical poses or external forces. Both runtime and independent contact gates remain unchanged. This is privileged geometry feedback, not a vision/tactile policy. The source-bound configuration is `out/standing-direct-release-010/withdrawal-segment-avoidance.json`.


### Release 014: clean lever pads, residual hub contact

The ring-segment controller removed all invalid loaded lever patches. However, the whole-handle audit found 24 little-finger middle-segment hub patches over seven intervals between 68.324 and 68.370 s, peaking at 1.5406 N. The episode therefore failed, and the source-qualified panel planner refused dispatch. Replay at 68.322 s found the middle segment only 0.0276 mm from the hub, versus 1.1588 mm for the distal segment. The controller correctly selected the middle segment and was already at its 3 N cap: this was insufficient advance clearance, not selection of the wrong segment.

Trial 015 keeps the same screened route, ring feedback, original motors and 3 N hub-force cap. It moves the withdrawal-only hub activation distance from 4 to 6 mm so repulsion begins earlier. The physical audit tolerances are unchanged. The experimental configuration is `out/standing-direct-release-010/withdrawal-segment-hub6mm.json`. No wider-opening qualification is claimed.


### Release 015 failed; middle-segment trajectory comparison

The 6 mm hub activation preserved zero invalid lever patches but failed 76 little-finger middle hub patches over 19 intervals (68.212–68.270 s; peak 1.3602 N). The earlier activation shifted rather than removed the collision. Trial 016 restores trial 014's 4 mm activation and adds tracking of source-bound material points on the ring and little middle segments. Their goals come from the same dense-audited 40 mm upward route. Each uses the existing 1200 N/m, 3 Ns/m, 6 N capped point tracker, mapped through the original motor transmission and caps, only after intentional release. The distal thumb controller and 3 N collision avoidance remain. This tests tracking of the problematic surfaces rather than additional repulsion. Physical and independent contact gates are unchanged; full opening and traversal remain unqualified.


### CPU-only acquired-posture hold comparison 029

Isaac trigger028 reached a small opening with opposed contact, then lost index contact around 25 s and later unloaded most digits; its full audit remains pending. A close-up from its recorded 25 s state was inspected. CPU comparison 029 tests the existing `QualifiedHandHold` at the aperture stage: an actual qualifying grasp must persist for 0.5 s before the original coupled finger targets are captured, followed by a one-second motor handoff. This replaces measured-pressure feedback for this comparison; both are not combined. The target remains 0.0815 rad and opening trigger 0.75 rad, with unchanged final physical thresholds and whole-handle audit. Source identity is recorded in `out/standing-hub-hold-029/source-manifest.json`; the exact CPU coordinator and commands are retained beside its launch receipt. It runs on the owned pod's CPU at reduced process priority, does not launch Isaac, and uses the existing guarded allocation. A new GPU comparison is conditional on full native qualification.


### Qualified upright release 016

All 30 runtime checks and both independent audits pass over 80.6 s: zero invalid loaded lever patches, zero extra whole-handle patches, original motor/loopback/joint limits, receiving-palm support and final RH clearance. Final aperture is 0.10415695 rad (5.97°). The recorded hand close-up at 68.722 s was personally inspected. This qualifies release, not full opening or traversal.

Panel screen 001 starts at the exact attained state at 80.58 s and targets 0.75 rad. Its first dense audit stopped on floating-point overshoot in a quintic retreat phase; the bounded phase fix is separately tested, with no geometry tolerance change. The completed audit found no pose/clearance violations but rejected the conservative 0.149 rad/s phase envelope (joint acceleration 5.242 rad/s²). A fresh audit at 0.1 rad/s and 0.05 rad/s² passes all 2,001 poses and conservative derivative bounds (joint acceleration 2.394 rad/s²). The forthcoming physical continuation uses this slower source-bound plan and the existing bounded-pi-stop-v2 panel motor controller.


Panel physical attempt 001 stopped during controller construction, with zero physics steps: the exact 80.58 s leaf velocity was -3.41e-8 rad/s, outside the phase controller's nonnegative initialization contract. The admission limit was retained. Screen 002 instead uses the qualified episode's exact 76.0 s state, where the hand is already clear and the measured leaf velocity is +1.22e-7 rad/s. Its 2,001-sample geometry and conservative rate audit passes at 0.1 rad/s and 0.05 rad/s². Physical panel002 runs the complete closed-door prefix again before this continuation; it is not initialized at an already-open door.

The acquired-posture CPU comparison029 passed its physical, pad and whole-handle reports. Isaac hold030 reuses that immutable source (`a87dada4c39f5530` prefix), repeats native admission, and queues behind trigger028. Both verified shutdown guards are armed for September 10, 00:16:10 UTC, with the experiment coordinator ending two minutes earlier. This bounded extension uses only the owned L40S allocation and reserves native, Isaac and export time. Trigger028 completed with failed sustained grasp/held-opening checks; it is not upgraded by reaching an aperture transiently.

Verified draft-release archives retain every failed release and qualified release016; the latter keeps source chunk152 locally for panel planning. Collector processes were briefly paused for local disk capacity while the remote experiment continued, then resumed after archive download/hash verification and raw-copy offloading. Archive receipts and restore commands are under `DoorBench-runs/remote-archives`.


Trigger028's final independent audit confirms complete 18,000-interval raw contact accounting but rejects 25,410 loaded patches. Runtime checks pass17/19; maximum lever angle0.802339 rad, maximum bolt travel11.562 mm, maximum leaf angle0.086863 rad, final leaf angle0.070927 rad. All431 final files were verified against the stopped remote run before offloading its raw trial archive. The README links the full scoped result and completion time.

Evidence collector correction: a final same-size progress rewrite can retain the same filesystem mtime as an earlier snapshot. rsync's normal quick check then skips it, while the SHA audit correctly rejects the copy. Final transfers now use checksum comparison; a real local rsync regression reproduces and repairs this case. The CPU-only029 wrapper also lacked its process receipt; its actual recorded, completed coordinator PID was added so final archival can be verified. The replacement collectors retain their original remote runs and deadlines.


Panel002 remains supported but stalled near its initial 0.104157 rad opening. At 80 s its 5 mrad reference lead produces only a 2.336 N palm target: the ordinary PI integral grows too slowly against stiction. The existing bounded stiction assistance, previously available only on later panel segments, is now optionally available on the initial segment through `panel_initial_stiction_assist`. The prospective comparison keeps the standard6N load profile and original motor limits; it does not alter the geometry or physical success gates. Initial assistance requires an explicit screened terminal plan. Panel002 will retain its own complete result.


Panel002 executed all 60,000 physical intervals but disk exhaustion interrupted final export. Its closed raw archive and all step chunks survived. A read-only recovery rebuilt 60,001 JSON records (reset plus steps), verified the 2 ms clock through 120 s, and recorded every chunk hash without advancing physics. Its final task report remains incomplete, so it is classified as an infrastructure failure with the observed stalled aperture, not qualified opening. Original partial output, recovery receipt and full records were independently downloaded and hash-verified in an 825,896,960-byte research archive before local raw copies were offloaded. Final-state/policy success was not inferred from the recovery.

The local disk reserve was restored by offloading only hash-verified archived traces/videos and record copies. The dead collector was restarted against the same remote run and deadline. Run Center now exposes the four measured Isaac wall-time phases; its syntax and collector tests pass, and the live API returned actual timing data. Direct visual review was unavailable while the Mac was locked. Native export also received a byte-equivalent per-record write optimization and explicit reduction/export/audit stage labels.


### Panel003: valid contact, unfinished aperture; longer matched comparison

Panel003 completes 120 seconds from the closed-door reset. It passes 32/34 runtime checks, with zero invalid loaded lever patches and zero extra whole-handle patches. The independent pad audit remains failed because it correctly requires the full physical task report to pass. The two failed checks are panel reference completion and held target aperture: the leaf reaches 0.518265 rad (29.7°), below the 0.75 rad target, while still moving at roughly 0.0092 rad/s. Torso tilt remains below the gate. At the end the target load is 4.30 N and the integral is 2.10 N, below their unchanged 6 N and 3 N caps; this does not establish force saturation. [Detailed result](evidence/native-standing-hub-panel-003.json).

Panel004 changes only the overall horizon from 120 to 160 seconds, retaining the exact screened plan, stiction configuration and all physical gates. It restarts from the closed door and re-executes the complete qualified prefix. The previous failed run was uploaded and independently downloaded/hash-verified before local raw copies were offloaded; more than 2 GiB was reserved before dispatch. Full opening and traversal remain unfinished.


### Owner-requested restart checkpoint

Panel004 reaches0.7954055rad over160s, passing33/34 runtime checks and the contact-specific portions of all independent audits. Its final receiving-palm load fluctuates below2N, so it remains failed. Prospective panel005 adds a smooth2.4N terminal target floor with unchanged6N maximum and original motor limits; eight force tests pass. The owner requested a safe restart, so005 was interrupted before its panel phase and its raw archive closed incomplete. Isaac030 finished with failed grasp/held opening, and all432 final files were verified locally. CPU031's remote result must be recovered from the stopped pod. [Restart handoff and exact resumption instructions](../handoffs/RESTART_2026-09-10.md).

## September10: longer wider-opening003 and flatter-palm candidate

The300s replay passes33/35 runtime checks but fails reference completion and held target aperture, ending at1.06568rad. Independent pad and panel-support contact checks pass their contact-specific criteria; their overall status remains failed because the physical task failed. Whole-handle audit passes. The actual final contact moment is0.412425Nm: normal contribution0.624008Nm minus tangential opposition0.211584Nm, below the authored0.457638Nm hinge-friction limit. This is recorded contact evidence, not an inferred solver multiplier. [Evidence](evidence/native-standing-panel-wide-003.json).

Candidate screen004 flattens the palm, retaining panel006 at156.722s as its qualified source, the same4° upright bound,8cm height drop and4cm inward shift. All2001 dense geometry samples pass. The original0.09rad/s,0.045rad/s² envelope fails the unchanged3rad/s² joint-acceleration limit. A slower0.045rad/s,0.01125rad/s² envelope passes (maximum2.223rad/s²). It is not yet a physical result. Source-bound plan: `out/standing-panel-wide-004-audit-slow/target-plan.json`. Isaac032 separately failed its native sustained-grasp and whole-handle prerequisite; Isaac dispatch was blocked.

## September10: failed captured-grasp prerequisites032/033

Both native prerequisites fail sustained pad grasp and whole-handle contact, so neither dispatches Isaac physics. Each has109 finally hash-verified collected files. Moving hub avoidance after the attained finger hold fixes an overwrite bug but does not qualify033:26,514 extra hub patches remain, compared with30,968 in032. [Original checks and audit identities](evidence/grasp-prerequisites-032-033.json). The033 trace shows about3.9mm clearance at capture16.1s, falling to0.6mm at17.9s as the posture error grows; no finger motors are clipped at those samples. Trial034 therefore tests earlier8mm activation with the same3N cap. It is currently running, not a success. Local wider-opening004 remains independently in progress.

## September 10, 07:44 UTC: bounded load follow-up

Wider004 completed 300 seconds with 33/35 runtime checks, failing completed panel reference and held target aperture. It ended at 1.02922 rad; whole-handle and contact-specific reductions pass, but the aggregate task remains failed. Personally inspected the recorded final left-hand close-up. The flatter palm path gave only 0.389920 N m net opening moment, below the unchanged 0.457638 N m hinge friction parameter and worse than original-path003 (0.412425 N m).

Remote prerequisite036 also failed. Measured operator following with captured fingers reduced invalid lever-pad patches to 14 but produced 9,901 extra handle-assembly patches, peaking at 37.30 N. Isaac dispatch was blocked; these are native results on the GPU host, not Isaac results. [Compact evidence](evidence/opening-wide004-grasp036.json).

Next candidate005 returns to the original dense-screened wide003 path and explicitly requests `bounded-9N-v1`: at most 9 N target and 6.5 N integral, with unchanged motor caps, physics and acceptance. The initial qualified panel segment stays on its original 6 N profile. The final 2.4 N support floor and terminal braking remain. Duration is prospectively 260 seconds to fit storage admission after verified offload of closed004. The same 1.2 rad target and full contact checks are required; timeout remains failure. Twenty-nine focused controller/path tests pass. This is a privileged controller experiment, not learned sensor-only control.
