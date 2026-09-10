# From initialized opening to complete sensor-driven traversal

Execution plan approved by the owner on September 8, 2026 (UTC). This is the next work package under [the full research plan](DEXTEROUS_PLAN.md), not a replacement for its catalogue coverage targets.

## Restart checkpoint

The owner requested an application restart on September10. See [the current restart handoff](../handoffs/RESTART_2026-09-10.md) before acting: local workers are stopped, the GPU is stopped, panel004 reaches45.6° but fails final palm support, and panel005 was deliberately interrupted. Earlier live-status paragraphs below are historical.

## Starting point

September 9, 2026 correction: the upright42.5° native episode passed its
original lever-only checks, but **is not fully qualified**. A new whole-handle
audit found45,352 loaded little-finger hub patches across14,573 intervals
([recorded evidence](evidence/native-standing-panel-006-whole-handle.json)).
The native prerequisite had omitted these colliders while Isaac's raw contact
view included them. The full assembly gate is now mandatory before future Isaac
dispatch. Historical reports remain unchanged; they cannot establish whole-hand
qualification. The same source sequence underlies the current wider-opening
experiment, so greater aperture cannot repair this earlier contact defect.

Native panel009 finished at68.8° upright under its declared7N continuation
profile. Its original34+12 checks pass, but its whole-handle check fails.
The corrected native hub002 test now passes18 original runtime checks,6 pad
audits and the new assembly audit over36s, with zero invalid lever patches and
zero extra handle contacts ([evidence](evidence/native-standing-hub-002.json)).
It opens4.34°; it does not traverse. Native hub-transfer003 now also passes21 runtime checks and both independent contact audits over50s, reaching4.72° with left-palm support ([evidence](evidence/native-standing-hub-transfer-003.json)). Native hub-return001 then passed24 runtime checks and both independent contact audits over64s: lever and bolt at rest, left palm supporting a5.07° opening ([evidence](evidence/native-standing-hub-return-001.json)). Initial right-hand withdrawal attempts002 and003 failed independent contact checks. Isaac hub021 completed36s but failed sustained grip and held aperture (17/19 runtime checks; [evidence](evidence/isaac-standing-hub-021.json)). Follow022 and023 then failed native prerequisites, correctly blocking Isaac dispatch. Withdrawal007 fixed joint-limit and final receiving-palm support failures but still failed 703 ring-finger middle-segment patches and 30 little-finger hub patches. Withdrawal008 also failed contact checks. Withdrawal009 failed launcher setup before physics. Withdrawal011 now passes whole-handle checks but retains 228 ring-finger middle-segment patches. Withdrawal012 also failed ring-finger release contact despite extra upward clearance. Withdrawal013 failed 279 ring-middle patches despite explicit distal material-point tracking; whole-handle clearance passed. Withdrawal014 removed invalid lever contact but failed 24 little-finger hub patches. Withdrawal015 also failed hub contact. Withdrawal016 now passes all 30 runtime checks and both independent contact audits with zero invalid loaded patches. A source-bound 0.75 rad panel path passes dense geometry and conservative rates at 0.1 rad/s; its physical continuation is next. The release screen also needs to account for motion of the supported door: replaying the nominal route against the measured leaf angle reveals up to 2.9 mm middle-segment intersection that is absent with the door frozen. Isaac pressure026 finished with 17/19 runtime checks and failed independent contact qualification. Trigger028 is running with an earlier controller opening transition; its final acceptance still requires the original measured lever/bolt thresholds. No new Isaac success is claimed.
Isaac follow016 remains failed. Clearance017 and018 failed native prerequisites;
clearance019's coordinator was stopped to prevent dispatch under the incomplete
audit. Its native process finished with failed runtime and whole-handle audits. The immediate
priority is hub-safe grasp acquisition and operation, then re-running the whole
sequence under the complete contact contract.

Live trials are listed in [local Run Center](http://127.0.0.1:5193/).

The earlier complete native traversal used a deep stance. Keep that result
separate from this upright sequence. The sensor actor still fails unassisted
balance, and all repeatability and catalogue completion gates remain open.

The v1 H1/dual-Shadow simulation produced opening of `db0055_swing_single` in four fixed-start Isaac repetitions, but those results are **not mechanically qualified**: the imported hand omitted the manufacturer's passive finger loopback constraints. Two fresh prepared environments reached 95 degrees. These are initialized, privileged demonstrations: the hand starts near the handle, the teacher uses simulator geometry/state, and the robot does not approach or traverse. They are not a robustness score or a vision/tactile policy. See [the evidence and reproduction commands](ISAAC_HANDLE_DEMO.md).

The current H1 actor passes the native Door55 approach in 9/9 varied-start development trials, and straight walking/stopping passed a live Isaac check. The corrected hand now acquires an opposed grasp in native simulation. Live Isaac acquisition passes its physical checks but has brief digit unloading; complete task integration remains unresolved. A later initialized Isaac opening repeat failed because the teacher released too early. The G1 walking checkpoint belongs to a different embodiment. Component passes do not establish the complete task.

## Ordered milestones

| Stage | Implementation | Acceptance evidence |
|---|---|---|
| 1. Complete one physical task | Start with hands away from the handle; approach, acquire an opposed grasp, operate the lever, open, traverse, release safely and settle. Coordinate reaching, balance and walking on the audited H1 model. Release timing depends on the door rather than a universal fixed order. | An uncut Isaac rollout, wide and hand close-up views, and independent checks of mechanism operation, motor limits, collisions, balance and whole-body passage. No runtime root pose writes, supports, welds or direct door torques. |
| 2. Establish repeatability | Randomize declared starting positions, handle locations, friction and door resistance. Add recovery from missed grasps, slips and closing doors. | Proposed development gate: at least 95/100 complete trials. Freeze feasible ranges, seeds, timeouts and numerical success checks before scoring. Keep every failure. This is a one-door development gate, not catalogue generalization. |
| 3. Remove privilege early | Build finite local tactile observations, camera streams, proprioception and recurrent student training. Begin on stage-1 trajectories rather than waiting for all-door teacher coverage. | Audit the entire runtime stack: no exact handle/door transforms, door angles, object IDs, contact identity labels or hidden oracle routing. Teacher/critic/reward may use state during training only. Score sensor-only performance separately. |
| 4. Learn from student errors | Imitate the teacher, collect teacher corrections on states visited by the student, then fine-tune with RL. Train recovery, observation delays and contact/visual uncertainty. | Matched-seed teacher/student reports, sensor ablations and failure videos. No replacement of unsuccessful rollouts by attractive selected examples. |
| 5. Expand by mechanism | Push/pull, opposite hinge sides, knobs, closers, sliding, double doors and unlocking procedures, followed by broader coverage. | Per-family opening and traversal scores; CAD/mechanism-lineage holdouts. Human-feasible robot failures remain in the denominator. Pet doors remain a downloadable supplement. |

The final proposed targets remain at least 95% reliable coverage of the independently frozen human-openable set, 9/10 seeded successes per required scenario, at least 90% coverage in each populated family, and separately reported traversal and unseen-door results. These are ambitious research targets, not promised outcomes.

## Why this approach

Separating physical motor capability from partial-observation learning makes failures diagnosable and reduces the initial exploration problem. Useful low-level walking/reaching skills are supported by [HumanoidBench](https://arxiv.org/abs/2403.10506). [DoorMan](https://arxiv.org/abs/2512.01061) provides a relevant privileged-teacher to visual-policy strategy, while [DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates correcting student-visited states. Their results do not establish success for our robot, hands or catalogue.

## Portability, resources and progress

Use the [one-click Isaac environment](ISAAC_ONE_CLICK.md); preserve source revision, dependency versions, robot adapter, task configuration, inputs, checkpoint, seeds and evaluation commands. Profile state-only, RGB and RGB+tactile throughput before large allocations. Keep bounded owned-pod allocations, teardown guards and off-pod evidence. Do not interfere with another agent's node.

Each implementation milestone must update the [execution ledger](DEXTEROUS_EXPERIMENTS.md), with measured results and limitations. GPU runs must be registered in Run Center with their stage, heartbeat, cost and evidence paths. Hand inspection must use close-ups as well as wide footage. A geometric fit is a candidate, not a physical success; an initialized opening is not traversal. The [sensor imitation guide](DEXTEROUS_SENSOR_IMITATION.md) documents the actor/data boundary and the remaining closed-loop evaluation requirement.

## Work log

- September 8, 22:24 UTC: execution continues with three parallel workers and the integration owner. An isolated L40S test reproduces the [PhysX pose/velocity discrepancy](PHYSX_POSE_VELOCITY_FIXTURE.md); a standalone GPU arithmetic fixture matches its small-angle rotation-rate loss. The new, explicitly declared [own-IMU delta-angle sensor](POSE_DERIVED_GYROSCOPE.md) reduces saved-state palm estimation error from 3.827 to 0.389 mm with the estimator unchanged. Its Isaac recorder integration passed 90 focused tests, and a fresh physical grasp comparison is being prepared; offline improvement is not physical success. [Thumb coordination](SCRIPTED_THUMB_COORDINATION.md) failed its actual 36-second trial despite a valid static candidate: the thumb remained overloaded and crossed its joint stop, so target relocation is not accepted as an improvement. A timed panel recontact trial stopped at 70.018 seconds on the unchanged waist-correction bound as the door continued moving; the next planner accounts for measured door motion across the whole body. The guarded GPU allocation expires at 23:16:56 UTC. Complete traversal, repeatability, learned vision/tactile control and catalogue coverage remain unfinished.

- September 8, 2026 UTC: plan recorded before execution. First engineering task is moving from the verified near-handle pose to contact-free reaching and acquisition, without changing the known-good opening baseline. Full-sequence evaluation and the sensor audit follow. No new full-sequence or sensor-only result exists yet.
- September 8, 2026 UTC: [acquisition development started](DEXTEROUS_ACQUISITION.md). One geometric approach passed 1,001 sampled configurations; seven native physical development attempts failed acquisition. The current explicit motor adapter reaches within 8.83 mm while upright, but thumb/index contact and a little-finger joint-limit violation remain unresolved. Reports, source snapshots and close-up previews are retained. No GPU was allocated and no new benchmark success is claimed.
- September 8, 07:50 UTC: execution continues in three parallel tracks. The [official H1 walking actor](DEXTEROUS_LOCOMOTION.md) moves the unchanged H1/dual-Shadow plant 4.02 m in a 12-second native trial; the integration worktree independently reproduced it. Stopping and continuous doorway interaction remain separate gates. A fresh guarded L40S environment passed readiness at 07:32 UTC. The [live sensor fixture](DEXTEROUS_SENSORS.md) passed at 07:43 UTC; actual H1-mounted sensor validation is running. The grasp audit exposed both coupled-tendon seating and a shallow wrong-side contact route in the old geometric candidate. Eleven root physical acquisition attempts have failed; contact-clearance planning is being corrected before further integration. The full sequence and sensor-only policy remain incomplete.
- September 8, 08:30 UTC: [live Isaac walking and quiet stopping](ISAAC_H1_WALKING.md) passed 14 checks; the [native continuous lowering/rising transition](DEXTEROUS_LOCOMOTION_TRANSITION.md) passed 9/9 seeded trials with safety checked every 2 ms. Actual H1 sensor capture passed its stream and initialized-opening audits, but native eye views miss the handle. A disclosed fixed wider/downward camera profile is being checked in Isaac. Acquisition remains the critical path: seventeen root development trials failed, including reversed measured release, Cartesian integral, operator following and force-replay variants. Work continues on contact control, a standing-height grasp and a real approach in the complete door scene. These component results do not complete stage 1 or establish a sensor-only policy.

- September 8, 08:52 UTC: [actual Door55 approach](DEXTEROUS_DOOR_APPROACH.md) passed 9/9 native cases; its live Isaac port is running. Corrected fixed eye cameras show the working hand/lever and passed all 16 sensor-stream checks, but the same initialized teacher failed the opening gate at 25.7°. The early-release/coasting dependency is now explicit in the demo and project README. Native maintained-contact palm pushing is being independently audited before Isaac testing. Grasp development continues with strict physical/anatomical checks and a bounded shooting search; initialized search rollouts are not counted as acquisition. All failures remain preserved.

- September 8, 09:08 UTC: manufacturer-source review identified the missing unilateral `J1 <= J2` Shadow loopback mechanism. Native acquisition-013 violates it by about 1.3 rad; both saved Isaac sensor/opening repeats also violate it (up to 1.383 rad). The v1 opening claim is explicitly withdrawn from mechanical qualification, while all original assets/results remain reproducible. A new versioned passive-tendon model and matched native/PhysX fixtures are being implemented before further controller training. The strict pad audit also rejects dorsal thumb/index and tip-only substitutions; 30 focused contact/route/verification tests pass. These findings change the critical path from controller tuning to correcting the robot model and requalifying its capabilities.

- September 8, 09:35 UTC: [the opt-in corrected model](SHADOW_LOOPBACK_MECHANICS.md) preserves original robot geometry, mass, motors and force limits. Native v2 approach passes 9/9 varied starts, and its six-second initialized hold passes the strict pad and every-step physical checks. A live PhysX two-link fixture passes 7/7 unilateral/slack checks; this is a behavior test, not a matched compliance calibration. Its earlier endpoint-comparison failure remains preserved. The full v2 Isaac import is running separately from v1. Native acquisition-v2-001 failed: only the little finger loaded, it pushed the lever down early, and LFJ1 exceeded its individual lower stop by 27.4 mrad despite the new loopback gate passing. All traces and closeups remain available. Controller work now uses the corrected model, with a measured release and a contact-sequenced approach under development. Stage 1 remains incomplete.


- September 8, 09:53 UTC: [native contact-free Door55 acquisition](DOOR55_PRE_CURL_ACQUISITION.md) passes every 2 ms physical and volar-pad check on the corrected hand. The route pre-curls the distal fingers before lowering them, starts 30 mm farther back from the panel, and shifts the final grasp 3 mm away from the handle end. Earlier shallow initial contact and inadequate endcap margin failures remain retained. Root independently reproduced the grasp with a reusable FK/motor controller that never steps its analytical model; its [native result](../results/dexterous/2026-09-08/acquisition-shared-v2.json) also passes. The Isaac acquisition adapter and same-anatomy live pad audit are ready for a separate GPU trial. The full robot sequence and sensor-only actor remain incomplete.

- September 8, 10:24 UTC: corrected native acquisition passes the original start and three frozen perturbed starts. The [continuous native test](DOOR55_CONTINUOUS_OPERATION.md) acquires the grasp, presses the lever, retracts the latch and holds a 0.086 rad partial opening. Live Isaac acquisition passes 13/14 checks in [the original pressure trial](../results/dexterous/2026-09-08/isaac-acquisition-v2-001.json) and [a middle-finger pressure trial](../results/dexterous/2026-09-08/isaac-acquisition-v2-002.json); both fail the frozen final 0.5 s hold due to isolated 2 ms middle-finger unloads. The original has a separate 0.636 s valid five-pad window. The next trial triggers lever operation after a measured qualifying grasp window; it does not relabel either fixed-hold failure. Close-up side footage and every-step audits are archived. Bimanual dynamic transfer, traversal, repeated task success and sensor-only learning remain active work.

- September 8, 10:55 UTC: [the continuous native walking chain](DOOR55_CONTINUOUS_WALK_GRASP.md) passes its physical audit over 55 s: real approach, lowering, contact-free arm preparation, opposed acquisition, lever operation and partial opening, with no runtime pose resets. The older 2° intermediate heading target remains failed and disclosed; the hand adapts to the actual landed pose. A separate [bimanual contact test](BIMANUAL_OPENING_SCREEN.md) holds left-hand contact while preserving the right pads; palm-only load verification is being added. Live [Isaac operation-001](../results/dexterous/2026-09-08/isaac-operation-v2-001.json) fails its final grasp/partial-opening hold: actual release occurred before the press timer ended, so opening never started. A measured-release transition [passes native verification](../results/dexterous/2026-09-08/native-operation-clear-v2.json) and is running in Isaac with robot sensor recording. The [opt-in v2 one-click environment](ISAAC_V2_READY.md) passed actual GPU readiness. Sensor-only imitation ingestion and a recurrent actor are implemented and checked on synthetic fixtures; no learned robot result is claimed. Release, whole-body passage and closed-loop sensor learning remain active.

- September 8, 11:35 UTC: continuous walking-to-partial-opening is integrated into the live Isaac runner and independently reviewed before its first GPU execution. It consumes actual foot and hand loads, screens the preparation at the landed pose, preserves native motor caps and audits delivered torque and contact patches at 500 Hz. Two synchronized diagnostic cameras record the body and hand; the robot's sensor cameras remain fixed. The first two sensor-recorded opening attempts remain failed: one retains the grasp but does not fully release the latch; the other opens but shifts index contact away from the required fingertip. [Independent archive review](ISAAC_OPERATION_SENSOR_REVIEW.md). Bounded wrist compensation now freezes on measured release and passes the native regression; its Isaac test is running. Sensor-only inference boundary and optimizer tests pass, but no learned closed-loop task result is claimed. Work continues through full opening, passage, repeatability and broader coverage.

- September 8, 12:18 UTC: a fresh [native loaded bimanual opening](BIMANUAL_OPENING_SCREEN.md) passes 20/20 physical checks to 1.201 rad, with final left-palm load 3.373 N; it starts in the lowered stance and does not traverse. An independent extraction replay reproduces its numeric state/control arrays. Isaac operation-005 uses the predeclared [volar phalange profile](SHADOW_GRASP_PROFILES.md): joint, loopback, collision, torque-delivery and partial-opening checks pass, but the final sustained grasp fails. The original distal-only result remains failed. Independent review separates insufficient opposition from actual digit unloading; the old counter's name conflates them. The first continuous Isaac walking trial stalled before arm preparation and lacks a complete numeric archive because native interruption bypassed export. It remains a failed diagnostic. Trial 002 is running with measured readiness details, periodic numeric checkpoints, and a graceful stop file. Sensor-only inference now has a live runner branch with no teacher fallback, but has not been evaluated on a learned checkpoint. Work continues on full task integration, grasp stability and early sensor learning.

- September 8, 13:04 UTC: [Isaac operation006](ISAAC_OPERATION006_REVIEW.md) passes 19/19 declared checks and 28/28 independent archive checks over 24 s. It starts with the hand clear, acquires an opposed volar grasp, depresses the lever and holds a 0.09778 rad partial opening. The final hand close-up was personally inspected; 109 transient unloading steps and 777 opposition failures remain disclosed. The [continuous Isaac walking trial002](../results/dexterous/2026-09-08/isaac-full-v2-002.json) completes 55 s, including real approach, both feet swinging, lowering, actual-pose preparation, acquisition and partial opening. It passes 26/27 checks but fails the frozen final sustained grasp. No runtime root pose writes or direct door commands occurred. The native bimanual result reported earlier mixed integrated joints with one-step-old derived poses at its handoffs; that timing limitation is now explicit, and the synchronized replacement is being physically qualified. [Sensor imitation](SENSOR_IMITATION_FIRST_RUN.md) now has two actual closed-loop Isaac failures: actor001 falls at 0.522 s; startup-supervised actor002 falls at 0.762 s. Exact student-visited states are recorded separately from actor inputs for offline teacher corrections. Full opening, safe passage, repeatability and broad sensor-only coverage remain active work.


- September 8, 13:56 UTC: [continuous Isaac trial003](../results/dexterous/2026-09-08/isaac-full-v2-003.json) finishes 55 seconds with 26/27 checks; its increased index preload does not fix the final grip (seven isolated 2 ms index unloads in the final 251 samples). All independent archive checks pass, and no loaded pad patches are misplaced. The unchanged trial002 failure is retained. [Native initialized continuation009](POST_OPENING_CONTINUATION.md) now passes 24/24 physical checks and 5/5 independent archive checks over 65 seconds: arm stow, rise, actual walking, whole-body passage and quiet stop. It starts from an earlier attained open-door state with leaf momentum; it does not establish uninterrupted opening or a difficult loaded release. Earlier native contact-force qualifications that used an extra forward solve are explicitly withdrawn in the timing review. The corrected actual-step recorder retains force/pose epochs separately. Four sensor-only Isaac candidates have failed; the longest lasts 1.042 seconds, with incidental hand collisions but no acquired grasp or operated handle. Offline corrections and a longer recurrent history are being trained. Full bimanual Isaac opening is running; full-sequence composition, robustness and broad coverage remain unfinished.


- September 8, 14:54 UTC: [continuous native walking/opening002](DOOR55_WALKING_FULL_OPENING.md) reaches 1.200824 rad after real walking, preparation, acquisition, lever operation, LH support and RH release. It remains 28/29 failed because withdrawal loads invalid RH finger surfaces. Actual-state LH replanning fixes the old path's landed-pose mismatch; two leaf-following alternatives fail and are retained. [Isaac full-opening002](../results/dexterous/2026-09-08/isaac-full-opening-v2-002.json) fails transfer after 37.2 s;003 fails directory setup before physics.004 is running with destination-audited actual-state LH targets and 3 mm intermediate clearance. Portable passage passed on an older Door55 scene, but new-door continuation requires fresh screening and physics; its first fresh route had collisions and the next physical run hit one QP iteration limit. Five sensor actors failed; longer GPU fits 006/007 do not clear all frozen fitting criteria. The sampler's missing 64–126 ms labels are fixed in a separate protocol; actor-owned command-history training is next. A two-hour owned-pod extension is verified with both teardown guards. Stage 1, repeatability, sensor-only task success and catalogue expansion remain unfinished.

- September 8, 15:15 UTC: revised-door continuation012 passes 26/26 physical and 6/6 independent checks, with zero warning-counter increments at every actual 2 ms transition. This closes the changed-door continuation gap, while continuous walking/opening002 remains 28/29 because of right-hand release contact. A continuous controller composition is being implemented without resets; coordinated arm/waist release and sensor training008 are active. [Evidence](POST_OPENING_REVISED_DOOR.md). Stage 1 remains incomplete.

- September 8, 15:55 UTC: [native walking/opening005](DOOR55_WALKING_FULL_OPENING.md) passes 29/30 checks with the explicit 6 N transfer target; original RH withdrawal still fails. Isaac full-opening004 remains failed with 17/24 checks passing, with invalid RH pad contact preceding sustained LH support; its complete source/output archive is verified off-pod and hand close-ups were personally inspected.005 tests the same bounded load change in Isaac. The continuous native runner passes a 500-step bit-exact walking-interface comparison; the complete task remains failed. [Isaac traversal integration](ISAAC_CONTINUOUS_TRAVERSAL.md) passes CPU contracts and awaits a live smoke. Sensor fits008/009 produce no qualifying checkpoint. A reproducible frozen-weight diagnostic identifies omitted recurrent history and late optimization regression, motivating a separately declared continuous-history fit. Full episode, repeatability, sensor-only success and catalogue coverage remain unfinished.


- September 8, 16:38 UTC: the [executed Isaac traversal-interface smoke](ISAAC_TRAVERSAL_SMOKE_RESULT.md) passes 26/26 independent checks over 500 actual physics steps; its one-second whole-task timeout remains failed. [Native walking/opening006](../results/dexterous/2026-09-08/native-walking-opening-006.json) reproduces all 30,000 whole-body lever-return transitions bit for bit, with every mechanical and anatomical gate intact; it stops before ungrip and full aperture. [Isaac full-opening005](../results/dexterous/2026-09-08/isaac-full-opening-v2-005.json) remains 17/24 failed; all 152 outputs and 514 frozen source files are verified off-pod. Independent [transfer-overlap analysis](ISAAC_TRANSFER_OVERLAP.md) identifies right-grip geometry deterioration and interrupted simultaneous support. The new staged withdrawal completes its route but still fails brief distal-pad contact and left support; both failed variants are retained. The first sensor checkpoint from training010 passes all six frozen fitting gates and 16 runtime-boundary checks. Its actual Isaac acquisition006 starts at 16:37:08 UTC with 304 runtime/input files identical to actor005, only the selected checkpoint changed, no teacher fallback and a 13-second simulation maximum. No learned task success is claimed. Stage 1, repeatability and catalogue expansion remain unfinished.

- September 8, 16:45 UTC: [sensor actor006](../results/dexterous/2026-09-08/sensor-actor-acquisition-v2-006.json) fails after 0.702 seconds, with 14/17 checks passing. The six offline fitting gates did not establish closed-loop stability. Original motors, caps, couplings, collisions and command submission pass; upright posture, full duration and acquisition fail. No grasp or mechanism operation occurred. All 65 outputs and 309 frozen source files are verified off-pod. Same-state teacher corrections become infeasible at 0.290 seconds, before the 12-degree tilt limit is crossed at 0.352 seconds. A separately declared sensor-only stance controller is being investigated alongside the reference hand-release work.

- September 8, 17:00 UTC: the [sensor-only stationary balance controller](SENSOR_BALANCE.md) passes two final-source five-second native trials (nominal and a declared 0.05 m/s reset velocity), 19/19 checks each. It uses only encoders, IMU, local touch and a constant joint-posture calibration; RGB is unused. This addresses the immediate balance subproblem and is awaiting actual Isaac qualification. The full-palm orientation comparison eliminates gross support loss during withdrawal but retains 38 samples below the 2 N palm threshold, six invalid fingertip contacts and later left-joint violations. Shorter, clearance-screened withdrawal and the complete opening/traversal remain active.

- September 8, 17:26 UTC: [actual Isaac stationary balance001](SENSOR_BALANCE.md) completes all 2,500 steps, passing 12 balance checks and 14 original physics checks, with peak tilt 0.3566 degrees. The runtime receives only robot sensors and static robot/posture calibration; RGB is unused. All 81 outputs and 509 frozen source files are verified off-pod. Native reproduction matches every state and sensor packet. A separate modest scripted-arm native test passes 22 checks and is being prepared for Isaac. Shortened reference withdrawal still has nine invalid middle-finger contacts; the later panel push develops growing contact forces and loses stance. These remain failed development results. Next work separates arm control from balance, corrects release contact and stabilizes the panel transition before an uninterrupted full-task attempt. Stages 1–5 remain open.

- September 8, 18:02 UTC: [actual Isaac scripted-arm balance001](SENSOR_ARM_BALANCE.md) passes16/16 six-second task checks and14/14 mechanical checks, with3,000 steps, peak tilt0.3577degrees and maximum joint tracking error0.0027724rad. All95 outputs/528 source-input files are verified off-pod, and actual hand/wide frames were personally inspected. This uses analytical sensor feedback plus explicit joint goals; there is no learned or vision-based opening claim. Native torso/finger approach is being qualified next. The reference release retains three brief invalid fingertip contacts. A target-ownership cleanup reproduced its failed panel trajectory bit for bit, ruling out that overwrite as the cause. A separately declared eight-joint waist/arm force projection is being compared against the original seven-joint controller; full opening/traversal remains incomplete.

- September 8, 19:00 UTC: [actual Isaac coordinated reach001](SENSOR_REACH_RUNTIME.md) passes 20/20 reach and 14/14 physical checks over 5,500 steps, with 5.05 mm palm endpoint error and peak torso tilt 0.384 degrees. All 111 outputs and 551 source/input files are hash-verified off-pod; wide and hand views were personally inspected. The passive finger split differs from native, so grasp transfer is tested separately. [Native full acquisition001](SENSOR_SCRIPTED_ACQUISITION.md) passes 25/25 with a 2.628-second opposed five-pad hold and all 9,500 contact intervals independently reproduced. Its first uninterrupted lever press fails when fingers unload and the thumb middle segment takes load. Tactile preload/progression and actual Isaac acquisition are next. The new bounded whole-body panel path completes 95 seconds without the prior collapse, but still fails aperture, palm-specific support and inherited RH release contact. Full sequence, repeatability, learned sensor-only task success and catalogue coverage remain open.

- September 8, 19:45 UTC: [actual Isaac acquisition001](SENSOR_ACQUISITION_ISAAC_001.md) completes all 9,500 steps but fails the original distal-contact anatomy and opposed-hold gates. Independent raw reduction finds 1,549 invalid little-finger patches, including middle-segment loading; FF/MF/RF end unloaded. An evaluator arithmetic correction reproduces the producer's float32 sum while retaining the independent force threshold; it corrects the reported duration to 19 seconds and leaves task failure intact. All 138 outputs and 566 source/input files are verified off-pod. Hand and wide views were personally inspected. Native tactile feedback advances the lever to 0.3265 rad with valid opposed pads, but remains below the release threshold. The flat-palm reference path now sustains support without collapse, while insufficient hinge moment and three earlier RH withdrawal contacts still prevent qualification. Run Center now includes wide and hand previews. Full-task completion, repeatability, learned sensor-only success and catalogue expansion remain open.

- September 8, 20:15 UTC: the [corrected passive-joint adapter](ISAAC_PASSIVE_JOINT_PROFILE.md) passes its first [actual Isaac reach qualification](../results/dexterous/2026-09-08/sensor-reach-dry-isaac-001.json), launched at 20:04:51 UTC: 20/20 reach and 14/14 physical checks over 5,500 steps. All 69 original passive coefficients and armatures remain exact throughout; the initial free-space finger split drift is greatly reduced. Peak tilt is 0.3785 degrees and palm endpoint error 5.164 mm. The old explicit friction approximation is retained under its historical profile; a native/PhysX isolated fixture independently exposes its persistent velocity oscillation. All 114 outputs and 588 source/input files are verified off-pod, with hand and wide views personally inspected. A separate corrected-profile 19-second grasp test is starting; the original grasp failure remains failed. Native index coordination reaches 0.5206 rad of lever rotation but still fails release, and a 0.25 mm palm shift does not improve it. Bounded local force feedback and the full-sequence reference remain in development. Stages 1–5 remain unfinished.

- September 8, 20:42 UTC: the corrected-friction [actual Isaac grasp](SENSOR_ACQUISITION_DRY_ISAAC_001.md), launched at 20:14:28 UTC, completes 9,500 steps with 21/23 task and 14/14 physical checks, but fails distal-contact anatomy and sustained opposition. All 141 outputs and 594 source/input files are hash-verified off-pod; independent raw-contact and passive-property audits reproduce the failure exactly. Body placement accounts for most of the remaining hand-position difference, motivating bounded Cartesian compensation before further GPU grasp trials. Native direct tactile force feedback retains a qualified grasp but stalls at 0.4426 rad; exact motor decomposition shows competing posture effort. Contact-normal force/posture separation and actual-base panel tracking are in development. A portable ready-host sensor preparation command passes its real-host dry runs; full preparation execution is next. Full sequence, repeatability, sensor-only learning and catalogue coverage remain unfinished.

- September 8, 21:30 UTC: the portable sensor preparation command completed all five native phases on the ready v2 host, including 9,500 exact causal packet decisions and a fresh relocated-asset reset preflight. Its exact emitted Isaac command ran at 20:59:07 UTC and completed 19 seconds; the actual grasp remains failed with 21/23 task checks, 1,987 invalid patches and a 6 ms opposed hold. All 667 run/source/input files are hash-verified off-pod, and independent raw evidence reproduces the failure. [Preparation and execution](ISAAC_SENSOR_DEMO_PREPARATION.md). The [IMU audit](RECORDED_IMU_DIAGNOSTIC.md) rules out packet frame/clock errors and isolates a recorded PhysX pose-versus-velocity discrepancy; a bounded backend fixture is next. Native palm feedback restores sustained support but still misses usable aperture, while tactile contact-mode repair and actual-state release planner extraction continue. Full sequence, repeatability, learned vision/tactile success and catalogue expansion remain unfinished.

- September 9, 01:38 UTC: execution continues locally; no worker agents are active. The previous pod was terminated by its guard during the usage interruption. Its final own-IMU trial summary reports failure, but the raw run was not copied before deletion; only [retained partial evidence](evidence/own-imu-grasp001-interrupted.json) is available. A fresh owned L40S is installing Isaac Lab, with teardown at 04:52 UTC and independent local evidence collectors active. The two-reset GPU synchronization fixture is queued behind actual environment readiness. [Evidence collection](GPU_EVIDENCE_COLLECTION.md) is now automatic for new environment launches. Two additional tactile thumb trials failed their unchanged stop-margin guard; both full raw archives are preserved. A revised whole-body planner passes a six-second geometric extrapolation, and its uninterrupted physical continuation is running with a required byte-identical 69.5-second prefix. [Results and limitations](THUMB_AND_RECONTACT_CONTINUATION.md). Full opening/traversal, repeatability, learned vision/tactile control and catalogue coverage remain unfinished.

- September 9, 02:00 UTC: `walked-moving-body-recontact-003` passes 30/30 native opening checks after an uninterrupted approach, grasp, handle operation, transfer and right-hand release. It reaches 1.2299 rad at 75.168 s with the required half-second palm-load hold. Independent audits verify all 37,584 raw intervals and all 2,682 new panel-target intervals; the 69.5-second baseline prefix is byte-identical. [Evidence and retained failures](THUMB_AND_RECONTACT_CONTINUATION.md). A fresh static stow plan passes 909 samples and an initialized traversal diagnostic is running. No uninterrupted traversal, repeatability, Isaac opening, or learned sensor-policy claim follows. GPU environment installation and its queued reset/native-preparation checks continue under the existing timer and collectors.

- September 9, 02:28 UTC: the [uninterrupted native sequence](CONTINUOUS_NATIVE_TRAVERSAL.md) passes 47/47 recorded task checks in 127.990 s: approach, opposed grasp, handle operation, opening, release, stow, traversal and quiet finish. Independent reconstruction confirms every position/velocity boundary, motor delivery, sampled body frames and final-second passage geometry, but its aggregate fails because the inherited recorder lacks full warning-counter history. The fresh repeat captures those counters and requires identical dynamics through 127.5 s. All first-run evidence and source are preserved off the temporary worktree. The initialized traversal diagnostic also passes 26/26 task and 6/6 independent checks. GPU installation is still progressing on slow network storage; readiness and the corrected reset/gyro trials remain pending. No Isaac, varied-start repeatability or sensor-policy success is claimed.

- September 9, 02:37 UTC: `continuous-opening-traversal-002` passes **47/47 task and 16/16 independent archive checks**, including complete warning-counter history. All 63,995 physical transitions are byte-identical to the first run through its terminal state; the contact audit finds zero invalid right distal patches, and all 2,682 selected panel-target intervals pass. Both complete archives, source packages and 749 model-asset/input mappings are retained outside `/tmp`. Wide cutaway and hand close-up videos use actual recorded states and were inspected. The README now distinguishes this privileged native success from the still-incomplete Isaac/sensor-policy task. GPU runtime package checks pass; first Isaac startup and downstream readiness/reset/grasp checks continue. Next: finish live GPU correction checks, port the attained-state opening continuation, then varied-start and sensor-only evaluations; stage 1 still requires actual complete Isaac evidence.

- September 9, 03:18 UTC: fresh Isaac environment readiness passes; reset synchronization fixture002 passes 7/7 actual GPU checks. Native grasp preparation003 passes all five phases, including replay of all 9,500 causal decisions with maximum motor-force error below 1e-9 Nm. The actual corrected Isaac grasp trial is running, with a separate live collector and Run Center entry. Native full-sequence capture003 reproduces every transition and passes 47/47 task, 16/16 physics and 15/15 sensor-join checks. Original camera timing/view defects are retained and excluded; a separate fixed-eye variant passes 9/9 pixel/reset checks. Both complete archives are preserved outside the worktree. The new native demonstration adapter admits initial-to-terminal causal examples; this does not establish learned policy performance.

- September 9, 03:45 UTC: corrected own-IMU Isaac grasp003 finishes **22/23** task checks. Independent physical reconstruction reproduces the failure and all 16 gyro audit checks pass; there are zero invalid loaded contact patches, but sustained opposed loading still fails. The full archive and 1,317 source/asset inputs are verified locally. A bounded four-finger tactile preload candidate passes 60 focused tests and is undergoing fresh native qualification before any new GPU trial. Native full-sequence training completes one full-source optimizer update in 512.43 CPU seconds; its unassisted sensor actor falls at 0.296 s, with exact motor delivery. Checkpoint, data admission, failed rollout and reset fixture are archived. See [grasp comparison](ISAAC_TACTILE_GRASP.md) and [sensor training](NATIVE_CONTINUOUS_SENSOR_CAPTURE.md). These failures remain visible in the README.

- September 9, 04:29 UTC: the first Isaac tactile correction remains a failed sustained grasp (longest hold 0.032 s versus 0.002 s baseline), with both independent audits complete and all 8,500 pre-reflex intervals exactly identical. The 1.5 N candidate passes native grasp but fails motor tracking (0.06706 rad versus the unchanged 0.04 rad gate), so Isaac is withheld. An intermediate 0.8 N candidate is undergoing fresh native qualification. Its first preparation was rejected before physics for a mismatched feedback version; the corrected retry has a new source/trace and all three checked-in schedules now have admission tests (40 focused tests pass). Exact failed-run inputs and evidence are retained off-pod. No complete Isaac traversal or sensor-policy success is claimed.

- September 9, 05:03 UTC: actual Isaac pressure004 completes 19 s and independently reproduces a 0.822 s opposed hold, but remains 22/23 because the final index pad minimum is 0.171 N (required 0.2 N). All four preload offsets saturate at 0.08 rad. Exact reset states nevertheless diverge before preload because the balance QP's automatic rho interval uses wall time; the next paired experiment fixes that interval at 25 iterations and compares 0.08/0.12 rad preload limits at the same 0.8 N force target. Both native preparations are underway, with serial actual Isaac launches guarded by completed preparation and archive checks; 96 focused tests pass. The native sensor-fed acquisition/press diagnostic reaches 0.53793 rad then stops at 25.81 s on unintended contact. The five-pass sensor-student comparison is also running on CPU and will execute a real rollout afterward. Owned L40S allocation renewed with verified local/remote guards through 06:32 UTC; no other pod touched.

- September 9, 05:21 UTC: fixed-QP grasp preparations005/006 pass all five checks;
  every native trajectory array is exactly equal between the two pressure limits.
  Both preparation/input archives are verified off-pod, and actual Isaac005 is
  running before the serial006 trial. The five-update student process exited
  unexpectedly after three saved updates; the cause is not established. Recovery
  now restores weights, Adam moments and Torch RNG in a new output directory,
  rejects data/protocol changes, and records missing historical losses explicitly.
  Nine focused training tests pass and updates 4–5 are running before the planned
  physical rollout. The destination return adapter feeds the original 41-node
  solver with measured Isaac poses; exact geometric equivalence and 16 focused
  tests pass. This is not yet an actual release/opening/traversal qualification.

- September 9,05:53 UTC: actual fixed-QP pressure006 passes 23/23 grasp checks and
  both independent audits; all evidence is verified locally. Paired005/006
  pre-reflex states and forces are exactly identical across 8,500 intervals.
  The 0.12 rad preload cap resolves the terminal index-load failure without
  changing original anatomy, force or motion gates. Five complete CPU student
  updates still give 0/1 tasks (fall at 0.304 s). A separately frozen 1,000-update
  GPU short-window experiment is running, with an automatic unassisted rollout
  after verified collection. Press plans now bind the exact attained acquisition
  configuration and controller source; regenerated candidates pass geometry.
  Their first execution was rejected before physics by a Git-only provenance
  assumption; the bundled-source fix is committed and fresh retries are running.
  Full Isaac operation/traversal, robust student success and catalogue coverage
  remain unfinished.

- September 9, 07:20 UTC: the new [sensor locomotion foundation](SENSOR_LOCOMOTION_BASELINE.md)
  passes five-second native and actual Isaac trials. Isaac passes 15/15 runtime
  and 19/19 independent checks with exactly replayed motor commands and mounted
  gyro readings, under the original `backend-dry-v2` passive profile. The H1
  walking network receives its orientation inputs from own IMU and encoders;
  constant command and fixed upper posture remain explicit. No vision steering,
  contact acquisition or full task success is claimed. Earlier raw-force students
  still fall at 0.392–0.580 s; original motor-target feedback extends the latest
  attempt to 1.306 s but is not usable. Privileged physical corrections are now
  verified and separated from actual applied action history. The architecture
  proceeds through validated locomotion, stopping and hand-control transitions
  before more whole-body imitation. Complete Isaac traversal and final policy
  remain unfinished.

- September 9, 07:36 UTC: actual Isaac sensor walking/braking/stance trial002
  passes 18 runtime and 22 independent checks over ten seconds, with exact replay
  of every motor command and gyro reading. Its source, failed system-Python setup
  attempt and verified final evidence archive are retained. Native walking/stopping
  also passes 18 checks. Lowering exposed a heading defect missed by its initial
  audit; height and heading are now separate acceptance checks. The complete
  vision/tactile door policy and full Isaac opening/traversal remain unfinished.


- September 9, 08:16 UTC: standing native acquisition passes 14 runtime checks
  and six independent actual-contact checks. Destination Isaac trial001 fails
  one of 15 checks: middle-finger contact is intermittent, despite a valid final
  grasp. Trial002 tests an explicit 3 N middle-finger preload after a passing
  destination-native run. Native partial opening physically works but rolls onto
  finger middle segments; the original strict grasp criteria still fail. Bounded
  palm recentering and distal pressure targeting are experimental, with failures
  retained. No whole-task or learned hand-policy success is claimed.


- September 9, 08:26 UTC: actual Isaac standing acquisition trial002 passes
  15/15 with an explicit 3 N middle-finger preload; the byte-verified archive
  and limited independent contact-accounting receipt are retained. Native
  standing partial-opening trial006 passes 18/18 runtime and 6/6 independent
  raw-contact checks (11,000 steps, original pad and physics limits), with a
  declared 5.6 mm palm offset and distal pressure targeting. Destination-native
  repeats 18/18; actual Isaac partial-opening trial001 is running from b9311e1e0.
  Existing left-hand transfer targets fail at the standing pose; coordinated
  bimanual geometry is being screened before any physical transfer attempt.

### September 9, 2026, 09:07 UTC

Isaac standing operation002 completed: physical handle depression, latch release,
and 4.44-degree partial opening, but failed sustained fingertip contact (18/19).
The independent audit reconstructs every raw contact interval. Native standing
transfer003 and004 failed; stronger tracking and spring-following are not fixes.
Retain their archives. Next test: use consistent palm/fingertip commanded handle
transforms while retaining press torque, then qualify transfer before release.
The full Isaac opening/traversal and sensor-only learned hand policy remain open.

### September 9, 2026, continuation after Isaac operation002

Standing transfer001–008 remain unqualified; additional preload improves distal
loads but leaves index-middle contact. Diagnostic009 holds the body/left route
and still loses index qualification, locating the next investigation at the
controller handoff rather than assuming body motion is the cause. Next run:
held route, unchanged preload; then separately remove added fixed-pad tracking.
Keep all original contact, joint, force and collision gates. Explicit preload
profiles and bounded recenter options are recorded in the run manifest; default
settings preserve the existing preload. Thirty-three focused tests pass.

The completed Isaac evidence is byte-verified and independently audited locally
and on GPU. The idle owned GPU is terminated. More local or external persistent
storage is needed before further fully recorded physics runs (about 280 MB free).
The full approved plan remains incomplete; no learned vision/tactile opening or
full Isaac traversal has been demonstrated by these experiments.

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

### September 9 continuation: withdrawal admission and physical failures

The fifth upright withdrawal geometry candidate passes an independent 2,001-sample
unstepped audit: maximum torso tilt 3.977 degrees, palm position error 0.182 mm,
and final hand/environment clearance 46.46 mm. It uses the previously measured
`clearance-lift-v4` release with a 10 mm early lift, retimed over 16 seconds, from
the actual return004 terminal state. This is geometric admission, not physical success.

Physical withdrawal001 lost balance when the route replaced the loaded stance
controller's equilibrium offsets. Withdrawal002 preserves those offsets and stays
upright, but loses opposed grip during repositioning. Its release prerequisite
correctly aborts the run. Neither is a passing reference. Withdrawal003 tests a
world-fixed right-hand goal, matching the screened route, and reduces left support
target from 4 N to 2.25 N over two seconds; the measured support gate stays 2 N.
The partial-opening envelope is required until intentional release. After release,
the left palm may open the leaf farther; joint, collision, upright and motor gates
still apply, and the final hand must remain at least 4 cm clear for half a second.

The raw archives for failed transfer011/012/013/015 and withdrawal001 were streamed
to a **draft** GitHub research release, independently downloaded and SHA-256
verified before their duplicate local raw files were evicted. Reports, trajectories,
source snapshots and per-file restore receipts remain local. This is not a dataset
release; Hugging Face was not updated. Restore commands are in each run's
`.remote-artifacts-github.json` and the permanent `remote-archives` directory.

The owned L40S has finished the CUDA/Isaac Lab install stage and is installing the
remaining dependencies. The frozen source and independent teardown guards remain
intact. A future bootstrap change explicitly installs the CUDA Torch build before
Isaac Sim to avoid downloading both the PyPI and CUDA builds; this optimization
has not yet been timed on a fresh node. Full standing traversal, Isaac operation
qualification, and learned vision/tactile control remain open.

Withdrawal003 also fails: all five distal pads remain correctly loaded through
71 seconds, but grip adjustment builds ring/little finger loads to about 19 N;
an invalid loaded patch first occurs at 71.36 seconds, followed by complete loss
of grip. Original arm motor saturation appears during this adjustment. Trial004
explicitly begins intentional unloading at the start of the screened adjustment,
rather than retaining the old squeeze preload until its end. This is a new control
experiment, not a reinterpretation of trial003. It still requires the preceding
half-second opposed grasp and left support, original loaded-surface checks, and
final clearance. The same screened route, motor limits and support target apply.

Withdrawal004 completes all 40,300 physical intervals and clears the right hand:
independent final half-second clearance is 46.80 mm, with exact raw-contact label
reconstruction. It still **fails**: 3,668 invalid loaded patches, a maximum sampled
right-thumb THJ4 limit overshoot of 0.0202 rad, and no final left-palm support.
The leaf reaches 1.337 rad, but that is not a qualified opening result. Hand views
at 69/71/71.36 seconds in trial003 and 73 seconds in trial004 were inspected; the
contact failure is visible during the slide off the handle.

Trial005 applies the existing hybrid normal-force projection to left support,
using measured leaf velocity and the original palm collision surface. It blends
in over two seconds, retains the 2.25 N target and all original motor caps, and
keeps the same screened withdrawal. This tests the large left-palm reacquisition
load (26 N near 78 seconds in trial004). Fourteen focused tests pass; physical
qualification is pending. No successful full traversal or learned policy is claimed.

Trial005 completes with the same three failed checks; independent invalid loaded
patches decrease to 2,383 and final hand clearance is 43.28 mm. Measured force
feedback improves intermediate support but does not qualify the release. Trial006
changes the thumb route: its three base joints move toward the screened open
configuration earlier, while tip extension waits until the later release segment.
The chosen candidate passes a fresh 2,001-sample full geometry/anatomy audit.
Coarse-search candidates and the exact postprocessor are preserved in the permanent
`standing-thumb-release-development` archive. No passing physical result is inferred
from this screen. GPU setup has reached DoorBench asset-environment preparation.

Trial006 retains the same three failed checks (2,428 independently invalid loaded
patches; final hand clearance 43.57 mm). Its thumb-only modification is insufficient.
Inspection of the route found right wrist yaw and WRJ2 targets just 1 mrad from
their authored stops, restricting measured-palm corrections. Planning now exposes
a bounded wrist reserve and a separate right-palm orientation objective; defaults
retain the old behavior. Eighteen focused tests pass.

The 40 mrad reserve candidate initially failed the fixed-foot audit. A subsequent
solve prioritizes foot/left-hand placement over approximately three degrees of
right-palm orientation. The feasible solved right-palm poses become explicit new
waypoints; original requested poses and fit errors are retained. Candidate011,
including the earlier thumb-base release, passes all 2,001 dense samples: maximum
position error 0.500 mm, rotation error 0.003965 rad, torso tilt 3.977 degrees and
final hand clearance 46.45 mm. Trial007 tests this new route physically. Neither
pose projection nor geometric admission is a physical success claim.

Trial007 passes the physical joint-limit check and reduces invalid loaded contact
to 10 physical samples / 39 independently reconstructed patches. All remaining
invalid patches are on the thumb near or past the lever's axial end during
69.578–69.642 seconds; the original 1 mm axial-clearance rule remains enforced.
Final clearance is 46.07 mm. Left-palm support recovers to about 3 N at the endpoint
but is not continuous across the required half-second window, so the trial fails.
Candidate012 adds earlier thumb-base separation in THJ5/THJ3 while preserving
THJ4's limit margin. It passes the unchanged 2,001-sample audit; trial008 is running.

The owned GPU has passed runtime dependency checks, Door55 generation, and a live
Isaac Sim startup (about 20 seconds). Pinned robot checkout/import and the separate
actual physics readiness proof precede the queued standing-operation experiment.

Continuation, 2026-09-09 13:25 UTC: the fresh Isaac environment passed at
13:07:57 UTC; its copied evidence hashes independently match. The destination-native
36-second prerequisite passed 18/18 runtime checks and 6/6 raw-contact checks,
with zero invalid loaded patches. The actual Isaac run is now executing on the
owned L40S under its existing 14:04:30 UTC teardown deadline. This is not yet an
Isaac opening result. See the timestamped receipt in `docs/evidence/`.

Native withdrawal008 still failed: four invalid intervals / eleven independently
classified thumb end-cap patches and intermittent left-palm support. Increasing
thumb separation and support force in009 made things worse (31 intervals / 88
patches, plus a stance-solver failure). Both remain rejected; neither contact nor
support thresholds changed. Close-up inspection at69.502 and73.002 seconds shows
the thumb scraping past the lever end before separating.

Trial010 tests an explicit left-arm-only IK phase with the candidate012 route and
2.25 N support target. The earlier shared solve still constrained the old right
palm and waist while separate controllers moved them; the new option drops those
constraints only after the actual qualified return. It preserves measured waist
and right-arm coordinates in the private solve and still uses original capped
motors. Twenty focused tests pass. A separate unstepped thumb candidate targets
radial separation before arm withdrawal; it requires fresh dense admission and
physical testing. Full standing traversal and learned control remain unqualified.

At 13:32 UTC, Isaac operation003 has reached 0.07678 rad of leaf opening
with a 0.79679 rad lever angle and 11.48 mm latch retraction; the 36-second
sustained test is still running. Native withdrawal010 completed but failed palm
support and thumb surface checks (19 invalid intervals / 44 independently audited
patches). Its final right-hand clearance was 45.86 mm.

Radial thumb candidate014 passes the unchanged 2,001-sample geometry audit. Its
25 mm material-pad radial goal is not fully reachable (15.29 mm maximum residual
is explicitly recorded), so only its actual solved poses are admitted. Trial011
was interrupted by ENOSPC at17.922 seconds, before withdrawal, and has a separate
infrastructure-failure receipt rather than a fabricated completed report. After
verified remote archiving freed space, trial012 restarted with the same config
and a 1.4 GiB preflight reserve. Original successful evidence remains local.
A supporting-left-hand close-up replay option is now available. Bounded optional
left-arm target velocity uses the 100 Hz IK clock, with a 40 ms filter and 2 rad/s
cap; 21 focused tests pass. It is not enabled in trial012.

At 13:43 UTC, native withdrawal012 still fails: five invalid intervals / twelve
independently classified thumb patches, all near the lever end at69.446–69.468 s.
Final hand clearance is46.16 mm. In the last half-second,66/250 recorded samples
had zero palm load, and none were finger-only support; substituting palm load for
total hand load would therefore not fix this failure. Recorded FK shows about
11 degrees of palm roll relative to the panel while the old IK only constrains
the palm normal. A new opt-in full-orientation target retains its actual attained
orientation relative to the moving door.

Trial013 was interrupted by ENOSPC at1.142 s while an independent audit overlapped
new recording; it supplies no withdrawal result. Heavy audits and physics trials
are now serialized, and trial014 waits for at least2000 MiB free. Verified draft
release archives now include failed older walking trials, preserving their raw
evidence before local eviction. Trial014 combines full palm orientation, bounded
IK-clock velocity feedforward and candidate015's axial thumb reserve. Candidate015
passes all2001 geometric samples; physical success remains unqualified.

At 13:58 UTC, withdrawal014 has only one invalid physical interval / three
independently classified thumb patches, at69.524 s with about0.10 mm axial
clearance (the original minimum remains1 mm). Full palm orientation plus velocity
tracking eliminates zero-load samples from the final half-second: minimum/mean/
maximum palm load1.834/2.454/3.141 N;15/250 samples remain below2 N. The trial still
fails. Candidate016 adds2 mm of axial thumb reserve; the next physical test also
uses a predeclared2.6 N palm target. No acceptance threshold changes.

The streaming contact auditor exactly reproduces every numerical and acceptance
field of qualified native return004 (source/dependency hashes naturally differ).
Both full audits completed sequentially; recorded child peak RSS was1,688,797,184
bytes. The parser and focused tests pass36 checks. See the validation receipt.

Isaac operation003 loses all digit loads after its initial successful lever/latch
and partial opening; its36-second result is not yet final. Our owned L40S deadline
has been safely renewed to15:46:32 UTC with independently acknowledged local and
remote guards. A source-frozen attained-grasp hold comparison004 is queued behind
003. After a continuous half-second of actual qualified partial opening, it
retains coupled finger posture using original capped motors and a one-second
command handoff;32 related tests pass. A fresh native prerequisite and independent
audit precede its Isaac run. Whether this capture actually activates will be
verified from recorded controller epochs; enabling the flag is not proof.

At 14:10 UTC, the incomplete Isaac003 archive is independently byte-verified
(219 files). Its last progress snapshot is35.502 s; the last completed checkpoint
receipt is33.982 s. No final operation report exists. The original coordinator
cutoff interrupted the final periodic export even though the pod guard had been
renewed. The README now discloses this incomplete trial.

Attained-hold004's native prerequisite passed runtime and independent contact
checks with zero invalid patches; its actual hold activated at21.282 s. Its
coordinator was explicitly replaced before Isaac launched, allowing the native
audit child to finish. Replacement005 uses the same tested controller, verifies
actual hold activation, reserves70 minutes plus export time for Isaac, and requests
`stop.request` before process signals on a timeout. Both completed and incomplete
evidence remain distinguishable. It is running the native prerequisite now.

Withdrawal015 still has one invalid thumb interval and higher palm force made
support worse (154/250 final samples unloaded). The next candidate uses the prior
2.25 N force target, greater axial thumb reserve and a12-second withdrawal, avoiding
an unnecessarily long push while the arm retracts. Its unchanged2001-sample
admission passes at1.738 rad/s maximum joint-reference speed, below the2 rad/s
limit. This is a new physical trial016, not a retiming of reported successes.

At 15:04 UTC, withdrawal016 completed its physical trial and independent audit.
Support, clearance (46.009 mm) and mechanical checks passed, but 407 intervals
contained 1,727 invalid right-hand contact patches. Faster withdrawal is therefore
rejected. Trial017 returns to the slower screened route and adds feedback for a
fixed material point on the original distal thumb. Its Cartesian error is mapped
only to capped finger motor commands; no physics state or helper forces are
written. Sixteen focused checks pass, including motor caps and unchanged plant
state. The physical result remains pending.

Isaac005 is still recording. Its opening-stage grasp capture has not activated in
observed controller snapshots, so the flag alone cannot qualify this comparison.
Trial006 is queued to capture a continuous qualified grasp before substantial
lever motion; it must pass a fresh native prerequisite and independent audit
before Isaac runs. Each source bundle and collector has its own receipt. Both
local and remote teardown guards were acknowledged for the owned L40S through
2026-09-09T16:59:23.513286+00:00. This remains a privileged controller experiment, not a learned
vision/tactile policy or a completed standing traversal.

At 15:07 UTC, withdrawal017 passed all29 runtime and8 independent checks across
40,300 actual intervals (80.6 s). No invalid loaded RH patches occurred; final
hand clearance was46.332 mm, with supported left palm. The right-hand release
close-up was personally inspected. Raw successful evidence is retained locally
and hardlinked into the permanent run archive. The thumb's transient tracking
error did not cause an invalid loaded contact; original force caps were unchanged.
The next stage must begin from this actual attained state, screen upright panel
continuation, and qualify another continuous physical rollout.

Isaac005 completed17/19 runtime checks. Its independent recorded-contact audit
passed its scoped checks, but runtime grasp/opening hold failed and the intended
hold never activated. This is explicitly a failed comparison. Isaac006 is now
running its fresh native prerequisite before the earlier-capture GPU test.

At 15:23 UTC, the upright withdrawal remains qualified29+8. Both early-grasp006
and post-latch007 captures failed the destination-native sustained grasp check;
neither was sent into Isaac. The next candidate008 captures a qualified grasp
at an already-open leaf without requiring continued full lever depression.
Actual005 traces show a continuous opposed window from17.022 to23.542 s while
the operator settles below the old capture threshold. Runtime mechanism and
contact gates are unchanged;43 focused tests pass. Owned GPU guards still expire
at16:59:23 UTC, and008 uses the remaining allocation with its own collector.

Stationary full panel screens001/002 collided at the elbow. Screen003 includes
original elbow mesh clearance against the panel plane and removed sampled
collisions, but misses pose tolerances. Shorter004/005 paths fail the independent
dense audit on interpolation/torso and reference-acceleration limits. The solver
now reserves numerical margin inside declared bounds, and the audited reference
speed/acceleration are consumed by the controller instead of a fixed default.
These changes do not qualify a physical continuation. Screen006 starts at79.598 s
of qualified withdrawal017; the preceding250 actual intervals have at least
43.011 mm RH clearance and2.077 N left-palm load. It retains the four-degree
absolute posture bound and screens a smaller contact-height change.

At 15:27 UTC, panel screen006 passed all2,001 dense
samples, including original mesh collisions,40 mm RH clearance, four-degree
absolute torso tilt and derivative bounds. Maximum reference joint speed is
0.0752 rad/s and acceleration0.3296 rad/s² with the explicitly audited0.06 rad/s
aperture cap and0.005 rad/s² aperture acceleration. This remains geometry only.
The source-bound motor composition is now running as native-standing-panel-001
from the original closed-door start for100 s. It carries the qualified withdrawal
prefix, capped force handoff and unchanged hybrid palm support, followed by the
screened body/contact targets. New physical gates require the0.75 rad target to
be held, completed reference progress and <=5 degrees torso tilt throughout the
panel phase, in addition to existing mechanical/contact/clearance checks.

At 15:38 UTC, native panel001 completed100 s and failed only
`standing_panel_aperture_held` (32/33 runtime checks). The door settled at
0.678031 rad, short of the0.75 rad target. Independent contact reduction
found zero invalid loaded RH patches and passed every check except the physical
report's overall failure. Final unintended penetration was only0.016 mm; the
wide final frame was personally inspected. This does not qualify the extension.
Raw failed evidence was uploaded to the draft research archive, independently
downloaded and hash-verified before local raw copies were evicted. Qualified
withdrawal017 remains intact. Panel002 adds an explicit10 N/rad aperture tracking
correction to the palm-load target, bounded to2.05–3.5 N with original motor caps.
It is a new continuous100-second physical trial; all acceptance gates remain.

At 16:22 UTC on September 9, material-contact comparisons009 and010 both
failed the destination-native prerequisite, so neither launched Isaac. Profile
`actual-material-v1` follows measured handle material points through finger
motors while relaxing posture control; it failed joint limits, sustained grasp,
lever release and held opening. Profile `actual-material-v2` retained posture
control and restored the lever-release check, but still failed joint limits,
sustained grasp and held opening (maximum leaf0.03477 rad; final0.01108 rad).
These are failed hypotheses, not benchmark improvements. Immutable source010
was staged by copying725 hash-verified unchanged source files and transferring
ten files into a new directory; the earlier source remains unchanged.

Isaac hold008 was deliberately stopped and exported24 seconds of actual
physics. The capture condition never activated. Inspection of hold005's strict
raw pad qualification found no half-second valid window after16.658 seconds,
although the weaker opposition summary continued later. Opposition alone must
not be used to claim an eligible grasp. The24-second trial fails completion and
sustained grasp. Own pod4jqu6fih3f0cc0 retains acknowledged local and remote
teardown guards for17:56:07 UTC; no other agent's node was changed.

Native panel002 again failed the aperture target. A separate audit of all10,201
panel intervals found zero unexpected external support contacts and at least
44.377 mm RH clearance. Recorded late contact wrenches produced0.3241 Nm opening
moment against0.457638 Nm hinge friction. Panel003 tested bounded integral palm
feedback but aborted at93.424 seconds when a legacy adapter rejected a target
above4 N. Its failed partial result was preserved and remotely verified before
raw eviction. The adapter now accepts the explicit6 N profile limit, covered by
a test passing controller output through the actual adapter. Panel004 is a new
120-second continuous trial using that fix; it is still running. Original motor
caps, joint limits and contact acceptance checks are unchanged.

Native panel004 completed120 s with32/33 runtime checks. It overcame stiction
but overshot to0.834731 rad, failing the unchanged target interval0.73–0.80 rad.
The independent60,000-interval grasp audit found zero invalid RH patches and
matched every qualification label/load. A separate20,201-interval panel audit
found only declared external support and at least44.766 mm RH clearance. Both
audits retain overall failure because the physical target failed. Wide and
left-palm close-ups at119.002 s were personally inspected. Panel005 uses the
separate `bounded-pi-stop-v1` profile: after actual aperture reaches10 mrad below
the screened terminal target, a250 ms smooth ramp removes accumulated excess
pushing force while retaining palm support. No success is claimed before its
new120-second run and independent audits.

The next hand comparison011 uses `measured-pressure-v1`: measured distal-body
contact reaction is projected onto the current inward pad normal, with a
0.25 proportional correction bounded to±1 N per digit. It retains the original
finger posture controller and arm commands, and introduces no tangential
material-point spring. Analytic tests check contact-force sign, tangential
invariance, original motor caps and absence of physical state/force writes.
It is queued on the same guarded node with a mandatory native prerequisite;
neither pressure feedback nor this implementation establishes a sensor-only
policy. Source identity2fab4d108b6395b80a58cf8cd6d15801e33b6beebf7d609bbca292d3cd21d685
is frozen remotely.

The desktop shortcut's clean sparse checkout was updated to9a44e7b8a.
Default launcher and direct preparation now select corrected v2 mechanics;
`runtime-v1.json` preserves explicit historical reproduction. Installed CLI help
and configuration checks pass ([receipt](evidence/isaac-desktop-v2-default.json));
this is not a new cold bootstrap measurement. Pressure011 failed only the
unchanged held-aperture gate at0.074172 rad, with sustained grasp and joint limits
passing and zero invalid loaded pad patches. Pressure012 tests an explicitly
recorded0.085 rad command against the same0.075–0.10 physical acceptance bounds.
The source is197d9ceecc1e9ad1ecc2b5552fe11ac177c3feb6352ccf46745457b2e7d08996.
Both active candidates appear in Run Center.

Panel005 completed120 s and held the leaf at0.741834 rad, passing the aperture
gate. It failed only final left-palm support:74 of250 final intervals were below
2 N (minimum1.738 N, maximum2.749 N), although the119.822 s close-up still shows
palm contact. Independent60,000-interval grasp and20,201-interval external-support
audits match their recordings, with zero invalid RH patches and no unexpected
external support; their overall result remains false. Raw evidence was remotely
uploaded, downloaded and hash-verified before eviction. Panel006 starts from the
closed door again and tests the separate `bounded-pi-stop-v2` profile, retaining
a0.5 N terminal support margin (2.75 N target) inside existing motor limits.

Pressure012 passed physical joint limits, actuation and the held-aperture range
(final0.082539 rad), but failed sustained distal grasp with6,522 intervals
containing invalid distal-profile patches. The index and middle fingers shifted
to middle-segment contacts. Pressure013 reuses exactly the same frozen source
with a0.082 rad command; all existing acceptance gates are unchanged. This is
controller calibration on a development door, not a robustness or generalization
score. The repository also retains a separately declared `volar-phalange-v1`
contact protocol from earlier research; none of these distal trials has been
retroactively promoted under it.

At16:55 UTC, native panel006 passed33/33 runtime checks,8/8 independent
grasp/release checks and4/4 independent external-support checks. It holds the
door at0.742428 rad after a continuous120-second closed-start trial. There are
zero invalid loaded RH patches across60,000 intervals, no unexpected external
support contacts across20,201 panel intervals, and at least44.766 mm RH clearance
throughout that phase. Exact-input body imagery was inspected. Complete raw
evidence is retained in both the worktree and the permanent run archive. This
qualifies the upright partial-opening continuation, not approach or traversal.
Screen007 is exploring1.2 rad from its actual119-second pose with fixed feet.

Pressure013 passed18/18 native runtime and6/6 independent contact checks at
the0.082 rad command (actual final leaf0.076116 rad). Its coordinator stopped
before Isaac because the old guard left less than the required run/export
reserve. The node's local and remote guards were renewed with acknowledged
deadline18:47:58 UTC and an explicit10-hour total ceiling. Pressure014 repeats
the same frozen source and command; its native prerequisite passed and actual
Isaac physics is running. No Isaac result is claimed yet.

Preview verification was tightened: the renderer now rejects robot or door XML
that differs from the recorded manifest even if the model/version name matches.
Three local reconstructions of GPU-native trials were marked unverified-inputs.
The differing robot files share all731 asset hashes but have different paths
and redundant OBJ type declarations; the door files differ in an inertial-frame
representation. No numerical audit was changed. Pressure013's new close-up was
rendered on the original node using its exact robot/door files and inspected at
35.502 s from120-degree azimuth. The imageio2.37.4 diagnostic package was added
without dependencies to the asset environment; simulation packages were retained.

Wider opening screen007 fails pose tolerances despite no sampled collisions.
Its final left-palm position error is2.062 mm and foot error1.156 mm. The left
wrist yaw reaches its lower limit and Shadow wrist deviation its upper limit.
Screen008 separately permits up to0.3 rad of root yaw, keeping a0.05 rad
roll/pitch increment cap, four-degree absolute torso bound and all foot/contact/
rate checks. It still misses pose tolerances; yaw alone does not remove the
wrist constraint. An actual-pose Jacobian diagnostic predicts that lowering
the left palm moves both wrist coordinates away from their stops. Screen009
therefore adds a declared0.15 m contact-height descent. These remain unstepped
geometry candidates and cannot qualify physical opening.

Screen009 failed the dense audit despite successful sampled poses: interpolation
overshot body limits and introduced abrupt target motion. Screen010 uses81 nodes,
a0.10 m palm descent, smooth0.15 rad body yaw preference, continuity regularization
and an interior joint-limit margin. It passes all2,001 dense samples with zero
collisions or increased joint-limit violation, maximum torso tilt3.990 degrees,
root translation19.579 mm and conservative peak joint acceleration0.522 rad/s².
All30 focused planner/schedule tests pass. This qualifies geometry only; panel007
will test the two-segment continuation from a closed start under physical load.

Pressure014 was stopped after its contact recording established failure. The
first excluded loaded patch appeared at17.360 s during opening; the last valid
opposed grasp was17.328 s. At26 s the index had no contact and the thumb loaded
its middle link on an excluded surface. The independent audit completes and
agrees that the trial failed. A new explicit measured-leaf lead limit will test
whether hand targets outrun the door; it caps only the motor controller's goal,
retaining original contact, joint and aperture criteria. Twenty-four focused
controller tests pass; physical success remains unproven.

Pressure014 exported27.4 s and retains17/19 runtime checks: contact hold and
requested-duration completion fail. Independent audit accounting passes; the
trial remains failed. Its raw evidence is retained off-pod. Lead015 tests a
0.012 rad measured-leaf lead bound on frozen source5d4bff03b, under the same
18:47:58 UTC guard, with a45-minute Isaac budget after native qualification.

Panel007 entered segment2 at119 s but stalls near0.76454 rad. Its measured-phase
lead is approximately0.005 rad, so the ordinary integral builds pushing force
very slowly against static hinge friction. A separate opt-in continuation
profile adds a bounded integral increment only with positive requested progress,
measured velocity below0.001 rad/s and remaining travel above0.02 rad. Existing
6 N target cap, motor limits, terminal braking and all physical gates stay fixed.
Fourteen focused force/schedule/contact tests pass; no physical success follows.

Panel007 finishes180 s at0.7645404 rad,32/34 runtime gates. Only reference
completion and final aperture fail. Its independent grasp/release and external
support audits match all90,000 physical intervals and50,201 panel intervals;
their overall reports correctly remain failed. The first segment hands off after
a7.6-second supported hold at119 s. A brief wall-clock SIGSTOP/SIGCONT avoided
disk exhaustion; no physical state changed. Closed failed Isaac005/008 raw
trials were uploaded, independently downloaded and hash-verified before eviction.
Lead015 passes its native prerequisite and is now running actual Isaac physics.

Panel008 is a new180-second closed-start trial enabling stiction assistance
only on the second segment. The first segment and its handoff remain unchanged.
Panel007's exact-input179.502 s body view was inspected and preserved.
Pressure014's collector independently verifies all220 exported files. Future
recorded Isaac operation trials now include the dedicated moving hand camera
as well as the wide camera; the already-running frozen lead015 trial is unchanged.

A measured-body diagnostic renderer now produces close-ups directly from the
archived PhysX hand transforms and contact cylinder. It performs no IK or physics
and labels the result as a native-mesh diagnostic rather than an Isaac camera
image. On pressure014, six original-node views at16.0,17.36 and26.0 s reproduce
archived world contact positions within1.1e-16 m. The qualified16 s grasp and
later slip were personally inspected at hand scale; images and hashes are
retained. Five pose-validation tests pass. Future operation runs also record
the actual hand camera, avoiding dependence on diagnostic reconstruction.

Lead015 reproduces pressure014 exactly through the measured15.002 s prefix,
including motors, body/root joints and door states. Its first invalid contact
again occurs17.360 s. It was stopped for export: the0.012 rad lead setting had
not changed the pre-failure trajectory and therefore does not address slip onset.
A fresh event-based experiment will follow measured handle rotation once actual
leaf travel reaches0.02 rad, blending out the unnecessary continued press over
one second. No physical pose, door force, contact threshold or motor cap changes.
Twenty-nine controller tests pass; actual native/Isaac qualification is required.

Lead015 exports20.0 s,17/19 runtime checks, with its failed contact audit
independently reproduced and all200 files verified off-pod. Every recorded root,
joint, door and motor value matches pressure014 across all10,000 samples. The
lead bound did not intervene; this is not evidence of an effective correction.
Follow016 uses a new source-bound measured-operator follow transition at0.02 rad
leaf clearance, with native qualification before actual Isaac and both diagnostic
cameras enabled. Owned-pod guards are acknowledged through19:32:06 UTC within
the existing10-hour ceiling.

Panel008 completes180 s at1.100228 rad; only reference completion and final
aperture gates fail. Both independent contact audits reproduce the recording.
The actual179 s opening contact moment is0.455712 Nm against the original
0.457638 Nm static-friction limit; tangential palm forces cancel0.231396 Nm of
the normal-force moment. Its raw archive was uploaded, downloaded and verified
before eviction. Outward palm repositioning is now being screened. A4 cm shift
misses posture tolerances;1 cm passes sampled positions but fails the dense foot
rotation check. Increasing the declared foot orientation objective weight keeps
the original acceptance tolerances and exposes the other constrained hand/foot
tradeoffs. No failed geometric or physical trial is promoted.

The [upright experiment and portability guide](UPRIGHT_OPENING_EXPERIMENTS.md)
collects the active controller switches, qualification stages, source/coordinate
contracts, and explicit requirements for another robot or cluster. Follow016
passed18/18 native runtime and6/6 independent contact checks, holding the leaf
at0.078041 rad while its operator-follow transition returned the handle toward
rest. Its actual Isaac trial is running with a dedicated hand camera; the8 s
close-up was inspected.

Planner screens016/017 were stopped as incomplete numerical diagnostics after
repeated iteration-limit stalls. A smooth pose-interior penalty and same-state
numerical warm starts are now supported; neither relaxes the dense audit.
Screen018 still misses late pose tolerances with a1 cm radial shift. Screen019
is testing a3 mm shift, motivated by the measured near-threshold hinge moment.
Thirty-two focused planner/schedule tests pass. No new wider physical pass is
claimed.


### September10: storage admission before resuming physics

The continuation rechecked actual resources: the stopped owned pod still cannot resume because RunPod reports no free GPU on its host. Its temporary resume guard was rolled back; no GPU work was dispatched. Local free space recovered to roughly19–20GiB after cleanup, but native admission found27.14GiB of retained evidence plus2.50GiB reserved for the next160-second episode, exceeding the20GiB retention cap. The cap remains unchanged.

Closed failed recordings are being streamed to the existing private research archive, independently downloaded/hash-verified, then offloaded locally. The first completed archival is `2026-09-09--walked-moving-body-recontact-002` (490,557,440-byte archive); receipts live in `DoorBench-runs/remote-archives`. Current qualified reference dependencies are excluded. Two bounded file batches are documented in `out/archive-older-failed-records-launch.json` and `out/archive-older-failed-records-batch2-launch.json`; inspect their actual PIDs/logs before acting, rather than assuming they finished. Their helper preserves small reports/source inputs and removes only verified matching large recordings.

The next physical comparison is prepared as `out/launch-standing-hub-panel006.py`, reusing the interrupted005 configuration exactly. It now performs both space and retention admission before launching. It has **not** run. Following admission it still requires runtime, pad, whole-handle and external-support audits; neither a larger aperture nor storage work establishes full opening/traversal or sensor-policy success.


The following continuation finished verification/offloading of all13 individual failed-run archives. Retained evidence fell from27.14 to23.71GiB by the latest check, still above the17.5GiB needed to admit the reserved2.5GiB next run under the20GiB cap. Two historical-bundle archivers remain live as of this checkpoint: see `out/archive-old-failed-bundles-resume-launch.json` / `.log` and `out/archive-old-development-bundles-launch.json` / `.log` (PIDs25125 and25810; revalidate before relying on them). The first failed-bundle attempt exited at metadata validation with no deletion; the corrected attempt distinguishes explicit task pass/fail reports from auxiliary reports. Original result files are preserved verbatim, including historical passes, and archive verification is never task qualification. `out/obsolete-standing-retention-candidates.json` inventories additional old non-hub standing episodes if further reduction is needed; it does not authorize claiming those files already archived. No new physical run has been dispatched.


### Native panel006 admitted and running

Verified archival now admits the160-second native comparison without changing the20GiB retention cap or10GiB host reserve. The actual admission receipt records18,577,753,284 retained bytes,31,094,263,808 free bytes and13,421,772,800 required free bytes. `out/native-standing-hub-panel-006` starts at the closed door and reuses the exact interrupted005 configuration:2.4N terminal support target floor, original6N cap, same screened0.75rad path and all physical thresholds. It is running, not yet qualified. PID31058 owns physics;31063 waits for pad/whole-handle audits;31190 waits for the external-support audit. Monitor actual handles and reports before restarting anything. Run Center is back at port5193 (PID31191), without opening a browser.

Historical archival batches finished, including13 failed runs, the4 failed bundles,14 other obsolete development bundles,5 pre-hub standing episodes and7 smaller failures. The complete legacy v1 folder and interrupted005 recording were also independently archived; absence of a task result remains absence of qualification. Current hub002/transfer003/return001/release016 dependencies remain local. Additional receipt-based offloading removed obsolete sensor/trace duplicates. Restores remain available under `DoorBench-runs/remote-archives`. The030 final failed Isaac archive is verified too; local numeric records were offloaded, reports/preview frames retained.

A new native process RSS sample is about2.1GiB on this16GiB host; swap use is about0.74GiB with ample disk headroom. This differs from the earlier restart snapshot and is not proof of a leak or its absence. Bounded RSS/free-space sampling for006 is saved in `out/native-standing-hub-panel-006-resource-monitor.json`. No GPU has been resumed or newly allocated.


### Native panel006 qualified; wider opening next

The160-second closed-reset episode completed34/34 runtime checks and passed independent pad, whole-handle and panel-support audits. It reaches0.7954054rad (45.57 degrees), with no invalid loaded finger patches or extra handle-assembly patches. The2.4N terminal support floor preserves the original6N cap. Recorded body and left-hand closeups were personally inspected: upright torso and panel support at the endpoint. This is a privileged MuJoCo partial opening, not traversal or an Isaac/sensor-policy result. Machine-readable evidence is in [native-standing-hub-panel-006](evidence/native-standing-hub-panel-006.json).

A candidate continuation source is the actual156.722s state (leaf0.7954060rad, positive velocity2.0628e-8rad/s), raw chunk313. The final state had negative velocity and was not substituted into the nonnegative-velocity admission. Screen toward1.2rad with the existing upright/contact/derivative tolerances before full physical replay.

Replacement pod innosemzr7vkef is preparing Isaac Sim5.1.0 with a separate journal; installer and remote teardown guard were confirmed live at05:10UTC September10. Deadline07:58:08UTC remains enforced. No experiment has been dispatched there yet. The old stopped pod is preserved for its unrecovered031 evidence.


### Wider-panel planning diagnostics

Screen wide001, from the qualified006 state at156.722s toward1.2rad, was stopped after repeated pose failures. At1.02805rad it missed both palm orientation (0.01046rad) and foot orientation (0.00253rad); unchanged dense limits are0.001rad. This is a failed numerical path, not proof that the task is infeasible. No physical rollout used it.

Wide002 tests a4cm inward and8cm downward palm route but its optional pose-interior penalty stalled the first samples at800 solver evaluations and missed the same tolerances. It was stopped as incomplete. Wide003 tests the identical geometry without that optional optimizer penalty, retaining the4-degree torso bound, original feet/hand tolerances, original joint ranges and elbow collision constraint. Its launch receipt/log are `out/standing-panel-wide-003-launch.json` and `.log`; inspect the actual PID before restarting. These small geometric screens produce no physics-step archives. If fixed-foot reach remains insufficient, a supported step/reposition is needed rather than relaxing geometric validity.


Wide003 completed the full1.2rad geometric path. Dense2001-sample geometry checks pass, maximum torso tilt1.814degrees; the first conservative rate envelope failed at3.178rad/s² against3.0. Reducing commanded aperture speed to0.09rad/s and acceleration to0.045rad/s² passes the unchanged complete audit (`out/standing-panel-wide-003-audit-slow/report.json`). This is planning evidence only.

A continuation initialization bug would drop the explicit terminal support floor after switching segments; it now preserves that already validated setting.28 panel-force/sequence/path tests pass. The230-second closed-reset physical command is prepared in `out/native-standing-panel-wide-001-command.json`, but was NOT dispatched: storage admission found18.07GiB retained plus3.59GiB reserved, exceeding20GiB. Archive another1.67GiB of obsolete evidence with verification before admission; do not lower the reserve or remove the qualified source chain.


The candidate156.722s handoff has252 recorded support intervals over the preceding half second; minimum palm load2.07295N passes the unchanged2N requirement (`out/panel006-source-support.json`). A bounded private archival batch is processing five obsolete development bundles: pressure-press-replan002, standing-sustain-Isaac003, continuous-opening-traversal003-sensors, own-IMU-grasp-pressure006-Isaac and Isaac-standing-hold004. Original reports are preserved, including historical passes, and upload is independently hash-verified before local removal. See `out/archive-wide-opening-budget-launch.json` and `.log`.

`out/launch-wide001-after-archive.py` waits for that specific archival process, then repeats both storage admissions and verifies the path/support receipts before dispatching the230s closed-reset trial. It refuses to launch if retention remains above budget. Its bounded archival wait is30minutes; launch/audit status is in `out/launch-wide001-after-archive.log`. Physics has not started at this checkpoint. After any emitted runtime report, all three independent audits run, including on physical failures. Revalidate process handles rather than starting duplicates.


### Wider physical replay admitted; replacement Isaac comparison queued

All five historical bundles were uploaded and independently downloaded/hash-verified. The first replay admission still failed at16.53GiB retained plus3.59GiB reserved. Additional exact-hash offloading of537 archived sensor/recording files (273,032,224 logical bytes) then admitted the run without changing budgets. Actual admission:17,539,751,812 retained bytes,32,075,481,088 free bytes,14,596,177,920 required free bytes.

`out/native-standing-panel-wide-001` is now executing230s from the closed reset (physics PID49097; wrapper session40143). It has reached acquisition, not yet the continuation. `out/launch-wide001-after-archive-retry.log` records runtime and independent-audit exits. The original failed admission remains recorded. Do not restart the active trial.

Replacement Isaac comparison032 is queued on innosemzr7vkef under coordinator PID5673, waiting for verified readiness. It uses reference SHA595bc4f4287013d69bd9ac29fb09cfee0ee25aa9e6f653abcff7d35b7bdbacf6, operator-stage attained hold,0.0815rad partial target and0.75rad opening trigger. It must pass destination-native runtime, independent pad and whole-handle prerequisites before Isaac. This is a new run, not a recovered031 result. Local prepared inputs: `out/isaac-standing-hub-hold-032`; remote: `/workspace/doorbench-standing-hub-hold-032`. Collector PID48671 is attached to `DoorBench-runs/2026-09-10/isaac-standing-hub-hold-032`. The07:58:08UTC teardown remains armed; extend guarded collection if the run approaches its current07:53:08UTC collector deadline. Installer process2577 had advancing write counters at the last check; no readiness success yet.


### Wider001 physical horizon reached; contact-moment diagnosis

The continuous replay passed the156.722s measured-state handoff into segment1, retaining the2.4N terminal floor and6N command cap. It reached the230s horizon at1.0164189rad, short of1.2rad; export and independent audits were still running at this checkpoint. No wider-opening qualification is claimed. The complete raw archive allows an actual final-interval contact-moment diagnostic (`out/native-standing-panel-wide-001-final-moment.json`): net palm moment0.436928Nm, normal component0.554739Nm, tangential cancellation0.117810Nm, original hinge frictionloss0.457638Nm, velocity~2e-7rad/s. Frictionloss is the model limit, not an inferred solver multiplier; the evidence supports a force-limited stall hypothesis.

A prospective second comparison changes only the wider segment to the existing `bounded-7N-v1` palm-load profile (7N target cap,4.5N integral cap). The first segment, screened geometry/rates, original robot motor limits and physical/contact criteria remain unchanged. Prepared command: `out/native-standing-panel-wide-002-command.json`. It has NOT been launched; first collect001 audits and rerun storage admission. A future supported reposition remains relevant if greater bounded load cannot progress without violating contacts or posture.


Wider001 export/audits completed:33/35 runtime checks pass, failing only reference completion and held target aperture. Independent pad classification is exact with zero invalid patches; the whole-handle audit passes. All contact-specific external-support checks pass, with no unexpected support and continuous RH clearance. Pad/support overall `passed` fields remain false because they require a passing physical report; these are not promoted to successes. Recorded final body and LH closeups were personally inspected. Compact evidence: [native-standing-panel-wide-001](evidence/native-standing-panel-wide-001.json).

Archival of this completed failure is live (`out/archive-wide001-launch.json`, `.log`). A new bounded waiter `out/launch-wide002-after-archive.py` waits for that specific process, reruns230s storage admission, then starts the prepared7N comparison and all three independent audits. Its receipt/log identify the live handle; no physics dispatch is claimed until `out/native-standing-panel-wide-002-launch.json` exists and its PID is confirmed. The GPU bootstrap has advanced from Isaac Sim installation to Isaac Lab setup; its active run remains separate from these native trials.


Wider001's1,123,409,920-byte private archive was independently downloaded/hash-verified, then462 local raw copies were removed. The first002 admission remained0.05GiB over budget; its additional archived trace (103,944,926 bytes) was rehashed and offloaded with a separate receipt, preserving the trajectory and reports. Repeated admission passed:17,566,971,904 retained bytes and32,037,974,016 free bytes, with14,596,177,920 required.

`out/native-standing-panel-wide-002` now runs from the closed reset under physics PID64059, wrapper session46296. It changes only the wider segment's declared load profile to bounded7N; initial segment and all physical limits/gates remain unchanged. `out/launch-wide002-after-archive-retry.log` owns completion and audit status. Its first observed progress is4.982s acquisition, not opening qualification. Isaac Lab setup and coordinator032 remain active on the guarded replacement pod.


### Replacement GPU deadline renewed after long bootstrap

The first renewal attempt timed out during the read-only RunPod status request, before guard mutation. Retrying the same plan succeeded: both replacement guards acknowledged live before the inventoried old guards were stopped. `out/guard-renewal-replacement-001-applied.json` records local guard74446, remote guard23460 and teardown09:10:56UTC (epoch1789031456.7205608), within the8-hour total ceiling. This extends the previous teardown by~73minutes; no new pod was allocated.

Coordinator032 was still exclusively waiting for readiness with no commands/physics dispatched. It was stopped after verifying that state, and its tiny prior wait directory preserved as `/workspace/doorbench-standing-hub-hold-032/setup-wait-before-renewal`, with an administrative receipt explicitly saying no physical experiment started. Replacement coordinator23726 now waits with deadline09:05:56UTC. Collector74944 was resumed with deadline09:09:56UTC; the earlier collector exited cleanly. These changes do not alter the frozen experiment source or native trial002. Bootstrap750 remains live and has reached the final core-package dependency installation.


Wider002 reached its230s horizon at1.0186536rad; export/audits are pending. The complete raw final interval shows net palm moment0.474706Nm (normal0.626389Nm, tangential cancellation0.151683Nm) versus original hinge frictionloss0.457638Nm. Actual velocity increases from0.0001280 to0.0001314rad/s over that interval. Compared with001's0.436928Nm and~2e-7rad/s, the higher bounded profile is producing additional opening moment, but did not meet the same time horizon. These observations do not qualify the task. Diagnostic: `out/native-standing-panel-wide-002-final-moment.json`.

A300s comparison is prepared as `out/native-standing-panel-wide-003-command.json`, changing duration only relative to002, with no further force/acceptance changes. Do not dispatch before002 audits and storage admission. A six-bundle verified archival batch is live (`out/archive-longer-opening-budget-launch.json`, `.log`) to make room under the same20GiB cap. It covers obsolete complete parent bundles for hold005, pressure014, follow016, standing-operation-Isaac001, hub-pressure026 and hub021, retaining original small reports and preserving the current hub002/transfer003/return001/release016/panel006 chain.


Wider002's final runtime again fails only reference completion and target hold; contact-specific pad/support checks pass and whole-handle passes. The overall pad/support `passed` fields correctly remain false because the physical task failed. Actual endpoint body and LH closeups were inspected. Compact evidence: [native-standing-panel-wide-002](evidence/native-standing-panel-wide-002.json).

Its verified archival is live (`out/archive-wide002-launch.json`, `.log`). `out/launch-wide003-after-archive.py` waits for that specific archive, repeats the300s space/retention checks, then launches the same controller with only a longer duration and all independent audits. The six obsolete-parent archival batch finished; raw current reference-chain files are preserved. The next trial is not dispatched until its launch receipt and live PID confirm it. GPU preparation has moved on to DoorBench/asset-environment installation.


Wider002's1,123,737,600-byte archive was independently downloaded/hash-verified, then462 local raw copies were offloaded. The300s duration comparison passed storage admission and is now live as `out/native-standing-panel-wide-003` (physics79645, waiter78848). Its first observed acquisition progress is13.582s; no wider result is claimed. All three independent audits remain attached via `out/launch-wide003-after-archive.log`. Its controller/configuration is identical to002; only the episode horizon changes.

Replacement GPU bootstrap generated db0055 successfully, with asset check/sign-off passing, and is now restoring the pinned shared Isaac runtime dependencies before its headless startup and physical smoke proof. Coordinator032 remains waiting for the readiness receipt; asset sign-off does not establish robot interaction or backend qualification.
