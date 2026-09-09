# A sensor-based locomotion foundation

The complete vision/touch door policy remains unfinished. Learning 61 raw motor
forces from one full trajectory and short corrections still falls quickly.
This component reuses the pinned official Unitree H1 walking network and derives
its orientation inputs from the robot's own IMU and encoders.

The runtime receives no root pose, base velocity, door state, contact identities
or environmental geometry. The private robot model removes articulated torso
motion from the mounted IMU. Native instantaneous samples and Isaac finite
delta-angle samples have separate, explicit timing/composition paths. The only
orientation prior is a declared upright reset, with arbitrary local yaw and XY.
The network's gait oscillator is an internal controller clock, not a task phase.

| MuJoCo component | Duration | Forward displacement | Maximum root tilt | Independent checks |
|---|---|---|---|---|
| Zero body command | 5.000 s | −0.217 m | 2.015° | 12/12 |
| Constant 0.1 m/s forward command | 5.000 s | +0.220 m | 1.975° | 12/12 |

Both use actual physics, original motor limits and passive hand mechanics. Every
submitted force is delivered exactly. The audits replay all 2,500 decisions,
compare the estimated gravity with evaluator-only actual orientation, reconstruct
the gyro, and check warnings and loaded scene contacts. Maximum gravity-vector
error is about 0.0011 and gyro reconstruction error below 3e-8 rad/s.

Zero command **does not hold position**: the robot drifts backward. Forward speed
also differs from the request. These are stable locomotion components, not precise
navigation, learned vision control, acquisition or opening. Upper-body targets
are a disclosed fixed motor posture. RGB and tactile packets are recorded and
validated but do not drive this baseline.

## Reproduce

`TEACHER` identifies the admitted native capture and its exact physical reset.
`H1_POLICY` is the original pinned Unitree checkpoint documented in
[the locomotion module](../doorbench/dexterous/locomotion.py).

```bash
python scripts/dexterous/evaluate_native_sensor_policy.py \
  --teacher-run "$TEACHER" --locomotion-checkpoint "$H1_POLICY" \
  --body-command .1 0 0 --seconds 5 --output out/sensor-forward
python scripts/dexterous/audit_sensor_locomotion.py --run out/sensor-forward
```

The separate five-second Isaac adapter requires
`--sensor-locomotion-calibration`, `--sensor-locomotion-robot` and
`--sensor-locomotion-checkpoint`, together with the frozen reset proof and actual
sensor layout. It rejects simultaneous teacher controls or another actor mode.
The calibration binds only motor posture, constant command, robot/motor identity,
checkpoint identity and the upright assumption. Actual reset encoders are sampled
before the first step; no IMU/touch history is fabricated.

## Actual Isaac result — September 9, 2026, 07:10 UTC

`sensor-locomotion-isaac-002` ran 2,500 physical steps on the L40S using
`backend-dry-v2`. It passed 15/15 runtime physical checks and
[19/19 independent replay checks](evidence/isaac-sensor-locomotion-002.json).
The five-second trial moved 0.187 m in the floor plane, with maximum torso tilt
1.934°. Every replayed motor command and gyro component matched exactly;
maximum gravity-vector error was 2.44e-7 and actual-body FK rotation error was
6.38e-7 rad. No runtime robot pose writes or teacher actions occurred.

The first attempt failed reset-container preflight before physics. The second
used two identical standing-reset rows, as required by that existing container;
no configuration trajectory was replayed. Its coordinator's historical gyro
audit failed because that auditor expects grasp-specific files. A separate
locomotion auditor now consumes the actual locomotion records and stores its
receipt separately; the coordinator failure remains preserved.

```bash
python scripts/dexterous/audit_isaac_sensor_locomotion.py \
  --run "$TRIAL" --robot "$ROBOT_XML" --checkpoint "$H1_POLICY" \
  --calibration "$CALIBRATION" --original-layout "$SENSOR_LAYOUT" \
  --output out/independent-locomotion-audit.json
```

The audit never steps physics and supplies only recorded actor packets to the
controller. It independently reconstructs mounted-IMU rotations, encoders,
gravity estimates and every motor command. Contact and force-delivery checks
remain bound runtime evidence; accelerometer and tactile values are not
independently reconstructed by this audit. This is a stable five-second
component, not a complete door task or a learned vision/touch policy.

The previous grasp's `legacy-tanh-v1` result is a different passive-joint profile.
The next architecture should learn perception, steering and dexterous interaction
above validated locomotion/balance components, with explicit tested transitions.

## Walking to a quiet stance

`--stop-after-s 3` selects a separate ten-second native component. It requests
braking at the next appropriate gait phase, sets body velocity to zero and
reduces the H1 oscillator amplitude over one second. Only after three seconds
of braking, sustained local foot touch and estimated speed below 0.02 m/s does
it transfer to the sensor balance controller. The new stance initializes only
its private unstepped estimator and motor targets from encoders and the prior
IMU estimate. No active robot pose is written. It is a timed stop experiment,
not visual approach control.

Trial001 failed after one stance interval: fleeting two-foot support did not
establish a valid stop. Trial002 never transitioned because its walking gait
continued. Both are preserved. Trial003 included gait braking and passed
[18 independent checks](evidence/native-sensor-walk-stop-003.json) over 5,000
physical steps: 0.133 m forward travel, stance handoff at 6.582 s, maximum root
tilt 1.976°, final-second speed below 0.000051 m/s and minimum actual foot
support 213.65 N. Every motor command replayed exactly with no QP failures.
Original joint-stop and loopback development tolerances remained unchanged.

The Isaac adapter uses `--sensor-locomotion-stop-after-seconds 3 --seconds 10`
with the same bound locomotion calibration and checkpoint. It has separate
handoff/quiet-state checks. The actual `sensor-walk-stop-isaac-002` trial completed on September 9, 2026 at
07:36:32 UTC. It passed 18/18 runtime checks and
[22/22 independent checks](evidence/isaac-sensor-walk-stop-002.json).
It moved 0.0883 m in the floor plane, handed off at 6.494 s, and stayed below
1.950° torso tilt. All 5,000 motor commands and gyro components replayed exactly.
The original `backend-dry-v2` passive profile and all physical checks remained
active. The first launch failed before physics on an older system Python;
using the pinned environment interpreter resolved setup without changing the
controller. Its failed receipt is retained.


Further lowering experiments remain unqualified. A first native descent reached
0.872 m but rotated about 24°; its older height/balance audit did not check
heading. A stricter heading objective held orientation but failed to descend.
A wider-stance attempt lost support and accumulated QP failures. Robot-only IK
posture guidance is under development with original joint bounds and explicit
foot-frame residual checks. None of these results qualify handle interaction.

The optional native lowering flags (`--lower-to-m`, `--stance-yaw-weight`,
`--hip-spread`, `--stance-leg-path`) are explicit development experiments.
Seven attempts have not established a qualified lowering transition. The
robot-only leg planner tests original joint bounds and foot-pose residuals;
optimizer convergence is disclosed separately from geometric feasibility.
A geometrically valid wider-stance path still lost support during actual
execution. These options do not change the default qualified stopping protocol.

A standing grasp alternative is now promising: the original 401-pose hand route
can be fitted at the measured standing height by using the existing torso-yaw
joint. It passes a separate 1,001-pose full-scene collision screen. A native
physical trial using preserved actual foot frames passed 12 grasp checks, and
a fresh trial with full transition recording passed 14, including every-interval
stance solves and warning checks. Peak torso tilt was 0.4673° and final palm
tracking error below 0.577 mm. Body and close-up hand views were inspected.
This is privileged acquisition from a pregrasp reset, not continuous walking,
opening, independent replay qualification or a sensor-only hand policy.

Reproduce candidate generation from a qualified standing capture:

```bash
python scripts/dexterous/prepare_standing_acquisition.py \
  --stance-run "$STANDING_RUN" \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --output out/standing-candidate
python scripts/dexterous/rescreen_acquisition_reference.py \
  --robot "$ROBOT_XML" --door "$DOOR_DIR" \
  --reference out/standing-candidate/reference.json --output out/standing-screen
python scripts/dexterous/probe_acquisition_teacher.py \
  --robot "$ROBOT_XML" --door "$DOOR_DIR" --motors "$MOTORS" \
  --reference out/standing-screen/reference.json --stance-profile landed-foot-v1 \
  --record-transitions --seconds 10.6 --output out/standing-physical
```

Use the pinned environment interpreter on the GPU; its system Python does not
provide the same hashing/runtime APIs. Rescreen and physically test against the
exact destination robot bytes before claiming cross-engine results.

The complete recorded native standing grasp also passes the
[independent contact audit](evidence/native-standing-grasp-004.json): all 5,300
intervals agree with recomputed pad classifications and forces, no loaded patch
uses the wrong surface, and the opposed hold lasts 3.522 s through the endpoint.
Final-half-second minimum forces are 1.900/1.545/1.226/1.682 N on the four finger
pads and 7.460 N on the opposed thumb. This verifies the standing grasp, not an
opening. The first standing operation trial retracts the latch but fails its
held partial-opening criterion; bounded compliance compensation is being tested.


On September 9 at 08:07 UTC, the first destination Isaac standing-grasp trial
completed 5,300 physics intervals under `backend-dry-v2`: **14/15 runtime checks,
failed sustained grasp**. It remained upright and its endpoint had all five
opposed pads loaded, but the middle finger unloaded at six samples in the final
half-second. The [contact accounting receipt](evidence/isaac-standing-acquisition-001.json)
recomputes anatomical formulae and load sums, finds no invalid loaded surfaces,
and records a longest uninterrupted hold of 0.532 s. This older archive lacks
raw world-to-hand contact transforms, so independent raw-contact reconstruction
is explicitly incomplete. A separately declared 3 N middle-finger preload trial
is running; the original motor limits and 0.5 s endpoint hold gate remain fixed.
Future acquisition recordings retain synchronized raw contact frames.

Standing operation trials remain unqualified. Trial002 physically released the
latch and opened the leaf 0.077 rad, but failed grasp quality. Trial004 explicitly
targeted distal finger segments for pressure and opened 0.083 rad, with fingers
rolling onto middle segments. Trial005 also recentred the palm by +4 mm in the
handle's X frame (the asset's lever axis is **negative** X), eliminating endcap
contacts but still failing the original distal-only grasp gate. Trial003 used
the opposite recenter direction and failed joint stops/release; it is retained.
All trials use measured state through bounded motors without direct door forces
or runtime pose writes. Neither partial opening nor a valid final photograph is
reported as a complete door task. Close hand views and all actual transitions are
retained for inspection.


Trial006 subsequently passes **18/18 native runtime checks and 6/6 independent
actual-contact checks**, using the original distal-pad contract. Its declared
handle-frame offset `[0.004, -0.003, 0.0025]` m ramps over one second, recentring
axially and moving the palm slightly radially outward. Distal pressure targeting
and compliance gain 0.2 are explicit. The handle reaches release and the leaf
holds 0.0774 rad; all 11,000 raw interval classifications and pad loads agree
exactly, with zero invalid loaded patches. Five-pad loading is briefly
intermittent during operation, so this is a qualified final held partial opening,
not uninterrupted five-digit contact or full traversal. The final-half-second
minimum pad loads are 1.069/0.949/5.299/4.422/11.664 N. See the
[independent receipt](evidence/native-standing-operation-006.json).
The same frozen variant is being prepared for destination-native verification
and an actual Isaac operation trial.


The destination Isaac standing-grasp trial002 finished at **2026-09-09 08:20:19
UTC** and passes **15/15 runtime checks** with the explicit 3 N middle-finger
preload. Its complete 124-file archive is byte-verified locally. Contact
accounting confirms zero invalid loaded patches and a 0.584 s uninterrupted hold
through the endpoint; [receipt](evidence/isaac-standing-acquisition-002.json).
Raw contact transforms are absent from this frozen version, so this is a runtime
pass with independent accounting, not a complete independent raw-contact audit.
The separate standing-operation trial001 passed its exact destination-native
18/18 prerequisite and is now executing actual Isaac with raw contact recording.


Native standing-operation trial007 also passes 18 runtime and six independent
contact checks with the 3 N middle-finger preload; see
[the raw-contact receipt](evidence/native-standing-operation-007.json). Its exact
destination-native repeat passes 18/18. Isaac operation trial001 timed out at its
old wall deadline after delayed grasp qualification (operation only began at
17.076 s); its final partial archive is byte-verified, with no completed task
report. Trial002 starts operation at 10.6 s using the previously qualified preload
and a renewed, separately guarded wall deadline. It finished on **2026-09-09
08:59:34 UTC**: 18/19 runtime checks passed, but sustained distal-pad grasp
failed. The handle reached 0.81361 rad, latch retraction 11.727 mm, and final leaf
opening 0.077532 rad (4.44 degrees). Three fingers loaded middle segments at the
endpoint. The [independent raw-contact audit](evidence/isaac-standing-operation-002.json)
reproduces all 11,000 intervals, including the failure. This is physical latch
release and partial opening evidence, not a qualified opening or traversal.

The standing bimanual transfer experiment uses a distinct 101-node geometric
route, independently screened at 1,001 interpolated poses: fixed right-hand/foot
errors stay below 0.627 mm and 0.002 rad, root tilt below 2.6 degrees, and no
collision check fails. The first physical trial keeps balance and the partial
opening but loses right-hand contact and leaves the left palm 1.28 mm short of
the panel. Trial002 adds fixed material-point pad tracking and an 8 mm bounded
normal approach: the left palm sustains roughly 4 N, but right-hand grasp still
fails. These are failed transfers, not full-opening evidence. Trial003's stronger pad tracking worsened grasp and violated a joint stop.
Trial004 followed the measured spring-returning operator, but the door closed and
the robot destabilized early. Both failed archives are retained. Trial005 restores
the more stable controller and blends fixed-pad targets toward the same commanded
handle transform as the palm over one second. Trial005 completed 36 seconds with 19/20 runtime checks, including left-palm support and balance, but failed right-hand grasp. Close-up recorded-state renders show the lever moving from distal pads toward middle segments. Consistent targets alone did not fix the grasp. No joint,
contact, opposition, penetration or motor limit is relaxed.

Trial006 raises commanded distal preload smoothly to 4 N per finger and 8 N
for the thumb during transfer. It preserves balance, physical limits and left
support (19/20), with four digits on qualified pads at the endpoint; the index
still loads its middle segment. Trial007 isolates a 6 N index preload while
retaining the other trial006 settings. Neither is yet a qualified transfer.

Trial007 loads all five distal pads at its endpoint but also loads the index
middle segment, so it remains failed. Trial008's 1.8 mm recenter also fails that
index-contact gate. Both retain balance, motor limits and left support. Next,
a diagnostic holds the body/left route fixed to isolate the controller handoff
from body motion; it deliberately cannot qualify as a completed transfer.
Experimental preloads now require an explicit CLI profile; the default preserves
existing preloads. All trial sources and failures remain archived.

The held-route diagnostic009 also loses index-pad qualification while the body/
left route remains at its initial target. Body route motion is therefore not
necessary for this failure; this does not isolate which handoff component is
responsible. Next ablation: held route with unchanged preloads, then disable the
added pad tracker separately. Compare actual motor changes and raw contact onset
at 22 seconds before changing geometry again. [Runtime trial inventory](evidence/standing-transfer-progress.json).

Both the local and GPU independent audits of Isaac operation002 produce identical
receipts. Its 172 final files, and the ready environment's 808-file recovery
archive, were byte-verified before terminating owned pod d30lnidzxkvv01 at
2026-09-09 09:11:17 UTC. No other pod was touched. Trials001–009 and planner
sources are retained under DoorBench-runs/2026-09-09. Local free space fell to
approximately 280 MB despite clearing 1.2 GB of old regenerable browser HTTP
cache; more persistent storage is required before another fully recorded run.
No new GPU allocation or Hugging Face upload is running.

The raw motor audit found a 24.9578 N·m command jump at the transfer boundary.
Diagnostic010's explicit one-second bumpless switch reduces the boundary jump
to 1.8e-15 N·m, but the contact gate still fails. [Measured comparison](evidence/standing-transfer-motor-handoff.json).
A longer **operation-only** baseline008 also loses index qualification at
22.620 s, without any transfer controller, and fails its 36-second hold. This
supersedes the hypothesis that transfer initiation alone causes the grip loss.
The previous 22-second result remains a short-duration qualification, not proof
of a stable longer hold. Stabilize the long hold before attempting transfer again.
Trial009 tests a bounded -0.025 rad index proximal reference adjustment during
operation; original motor and collision limits remain enforced.

Storage recovery is now available: [persistent run archives](PERSISTENT_RUN_ARCHIVES.md).
Two large historical archives were hash-verified on the new network volume before
local eviction, recovering roughly 1.2 GB. A guarded CPU pod handles transfers;
the GPU remains off. No extra local drive is required for the immediate next runs.

### September 9, 2026: sustained native grasp qualified

Native standing operation012 passes all 18 runtime checks and all six independent
raw-contact checks over 36 seconds (18,000 physics intervals). A 3 N index preload
established during acquisition produces 17.968 seconds of continuous qualified
opposed distal-pad contact through the endpoint, with zero invalid loaded patches.
The original joint, collision, motor and upright limits remain unchanged. The
handle reaches 0.8643 rad, the bolt retracts 12.52 mm and the leaf opens 4.34 degrees.
[Independent receipt](evidence/native-standing-operation-012.json).

This is a privileged native partial-opening result, not an Isaac repeat, full
traversal or learned vision/tactile policy. Extended baseline008 and posture
corrections009–010 fail sustained contact; experimental pad controller011 also
fails opening. Its subsequently corrected material-target code remains unqualified.
A fresh coordinated route is being screened from operation012's actual 36-second
state before another force-driven left-hand transfer. Transfer timing is explicit;
start-pose and all physical tolerances remain unchanged.

The previous storage block is resolved for immediate trials. Closed failed-trial
recordings are hash-verified on the persistent RunPod volume; the temporary CPU
transfer pod and idle GPU are terminated. Full operation009–012 artifacts also
have verified permanent local archives. The overall approved plan remains open.

### September 9, 2026, 10:15 UTC: isolate moving-body grasp loss

The fresh operation012 route passes 1,001 independent geometric samples, with
maximum fixed hand/foot error 0.558 mm and root tilt 2.534 degrees. Native transfer011
and012 both reach about 4 N left-palm support, but fail right-pad qualification
(19/20 runtime checks); independent raw-contact audits retain those failures.
Trial012 removes extra transfer pad feedback, so that feedback alone does not
explain the failure. Held-route diagnostic013 retains all five qualified pads
through 44 seconds; its left-support check correctly fails because no reach was
requested. This isolates moving-body compensation from an unavoidable static slip.

Recorded wrist flexion is near its original stop. The geometric route preserves
the attained palm pose, whereas the operation IK pursues a different, loaded
Cartesian reference. Trial014 tests the screened joint route with retained initial
motor preload and changing gravity feedforward. It keeps original motor caps,
physical gates, and the explicit one-second handoff. This is experimental until
its actual force-driven recording passes.

Owned L40S `4jqu6fih3f0cc0` ($1.09/hour) is preparing with local and remote deadline
guards. `scripts/isaac/run_standing_operation.py` waits for corrected-hand readiness,
checks frozen source hashes, rescreens on the destination, runs and audits the
36-second native prerequisite, then runs actual Isaac and audits its raw contacts.
A detached collector retains the run under `DoorBench-runs/2026-09-09/standing-sustain-isaac-003`.
The deadline is fixed in the owned-pod journal and Run Center. No Isaac result is
claimed until this pipeline finishes. Full opening/traversal and the learned
vision/tactile actor remain unfinished.

### September 9, 2026: native standing transfer qualified

Trial016 passes **20/20 runtime checks and 6/6 independent raw-contact checks**
over 50 seconds (25,000 actual physics intervals). Its final 31.968 seconds retain
continuous opposed distal-pad contact, with zero invalid loaded patches. Left-palm
support is 3.69 N at the endpoint and the leaf remains at 0.08274 rad (4.74 degrees).
Original motor caps, joints, collisions, loopback mechanics and upright gates pass.
The final hand close-up and whole-body frame were personally inspected.
[Independent receipt](evidence/native-standing-transfer-016.json).

The successful experimental mode tracks the screened torso and right-arm joint
route together, retaining initial motor preload plus changing gravity feedforward.
The left-arm solver compensates at the measured torso instead of replacing that
route. Trial015 tracked only the right arm and failed; trial014 rejected a reference
velocity incorrectly differentiated at physics substeps. The corrected rate uses
actual reference-update times. No physical state is written after the initial reset.

This qualifies acquisition, lever/latch operation, partial opening and left-palm
transfer in native MuJoCo. Release, substantial opening and walking through in this
standing sequence remain open; this is a privileged teacher, not a learned policy.
The separate actual Isaac 36-second operation repeat is preparing on the guarded
L40S. All failed trials remain labeled and archived.

Controller exceptions now retain an incomplete raw archive, the actual completed
physics intervals, terminal state, and a failing report. A deliberately injected
failure at 0.008 seconds retained all four completed intervals and correctly failed.

### September 9 continuation: standing lever return qualified

Native return004 passes **23/23 runtime checks and 6/6 independent raw-contact
checks** over 64 seconds / 32,000 physical intervals. The final **45.968 seconds**
retain continuous opposed distal-pad contact; zero loaded patches violate the
original anatomy gate. The operator and bolt return to rest while the left palm
supports the partially open leaf. Final hand close-up, intermediate hand view and
whole-body frame were personally inspected. [Receipt](evidence/native-standing-return-004.json).

Return001–003 each failed the same grip check (22/23): the ring finger loaded its
middle segment. Finger-posture stiffness and lower left support pressure did not
fix it. Measured palm drift was approximately 2 mm / 1.5 degrees. Return004 adds
bounded measured-palm error integration to the original arm motor references;
its independent FK model never steps or writes the physical plant. Finger contact,
motor caps, upright posture and joint limits remain unchanged acceptance gates.

The next stage is physical withdrawal. A first unstepped retargeted release cleared
its sampled contact checks but tilted the torso 6.69 degrees; it is rejected for
this standing route. A second candidate explicitly bounds rotation within the
standing planner’s 4-degree design target before independent dense screening. Neither candidate
is a physical release result. Full opening, traversal and the learned sensor-only
policy remain unfinished.

The owned L40S is still preparing the frozen Isaac repeat with local and remote
teardown guards. Its current bottleneck is downloading/extracting PyTorch's CUDA
library; process I/O is advancing. The separate evidence-volume attachment attempt
never started a container and was terminated; its persistent volume is retained.
No new off-pod transfer is claimed. Closed return001–003 are SHA-verified in the
permanent local run directory. An inactive, unopened downloaded application-update
cache was removed to recover disk space; no unique experiment evidence was deleted.
