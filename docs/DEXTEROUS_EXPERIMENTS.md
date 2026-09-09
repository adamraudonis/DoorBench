# Dexterous experiment ledger

All results on this page are development experiments. **No complete door-opening policy has been achieved.** The acceptance target remains in [the approved plan](DEXTEROUS_PLAN.md); platform/reproduction instructions are in [the migration guide](DEXTEROUS_REPRODUCTION.md).

## Body reaching, September 7, 2026

Same development seeds 10000–10029, 0.25 m target-offset curriculum, six-second horizon. A reach requires both forearm target errors below 8 cm, torso tilt below 12 degrees and pelvis height above 0.8 m for 0.5 seconds. A fall or numerical instability fails the trial. These criteria are only a body-skill gate, not the final opening/traversal protocol. Evaluations below ran on local macOS CPU.

| Controller | Training transitions | Upright reaches | Falls | Evidence |
|---|---:|---:|---:|---|
| Frozen upstream body skill | External pretrained checkpoint | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/upstream-body-reach.json) |
| Reach 001, standing reward weight 1 | 430,000 | 0/30 | 9/30 | [Trials](../results/dexterous/2026-09-07/reach-001-body-reach.json) |
| Reach 002, standing reward weight 4, intermediate | 820,000 cumulative | 1/30 | 12/30 | [Trials](../results/dexterous/2026-09-07/reach-002-intermediate.json) |
| Reach 002, later checkpoint | 2,120,000 cumulative | 0/30 | 4/30 | [Trials](../results/dexterous/2026-09-07/reach-002-late.json) |
| Reach 002, final checkpoint | 2,432,944 cumulative | 0/30 | 7/30 | [Trials](../results/dexterous/2026-09-07/reach-002-final.json) |
| Reach 003, continue after training success | 2,936,752 cumulative | 1/30 | 2/30 | [Trials](../results/dexterous/2026-09-07/reach-003-final.json) |

Reach 001 stopped around 432,000 transitions because a fall exhausted the native constraint arena. Its last checkpoint is retained; it did not complete the 500,000-transition budget. Increasing solver memory is an infrastructure repair, not a change to collisions or success thresholds. Reach 002 resumes that checkpoint with 128 MiB arena memory per environment and a stronger standing reward. The intermediate result does not establish improvement in reliability.

The later Reach 002 checkpoint reduced falls but still failed the upright-reaching gate. Final pelvis heights ranged from 0.713 to 0.794 m, with median 0.767 m. Inspection identified an incentive problem: successful episodes terminated while unsuccessful episodes could continue receiving positive dense reward. This is a plausible explanation for settling just below the 0.8 m threshold, not a proven causal result. The v3 configuration tests continued training episodes after reaching, with the original evaluation gate unchanged. A fall after reaching still fails the training episode. Historical v1/v2 configurations preserve their original termination behavior.

Reach 003 completed 503,808 additional transitions. Its final median pelvis height was 0.777 m and it passed only 1/30 trials, with two falls. The termination correction alone has not solved body control, and these small development samples do not establish a reliable improvement. Further posture/target curriculum work is required before connecting this body skill to approach and traversal. All three GPU-run checkpoints and logs are archived; the owned pod was terminated and its absence confirmed after copying them.

A short throughput profile at unchanged physics settings measured approximately 251, 467 and 781 transitions/sec for 4, 8 and 16 environments respectively. [Raw profile](../results/dexterous/2026-09-07/native-profile.json). This profile includes native CPU physics and actor inference, excludes rendering/PPO updates, and establishes the best of those tested counts only. It is not an Isaac Sim or maximum-GPU-capacity measurement.

## Grasp initialization and contact probe

The current grasp spike is specific to the right Shadow Hand and `db0055_swing_single`. It is not an arbitrary-robot or arbitrary-door adapter.

1. Fit finger-pad positions, pad directions and a palm target within joint limits. Seed optimization reserves 4% of each joint range and permits a bounded nearby base pose.
2. Penalize robot collisions explicitly. A pose with accurate fingertips but self-intersection is rejected. The inspected candidate has about 1.1 mm maximum pad-target error and no recorded robot penetrations deeper than 2 mm; this is kinematic evidence only.
3. Execute the pose in native physics with a free body and bounded motor position commands. No runtime pose writes, welds or external supporting forces are used. Starting already near the handle is explicitly non-qualifying for the task.
4. Initial constant targets load four fingers but lose thumb contact. Optimize thumb/finger preload using solved forces and opposing contact directions; retain failures and longer-horizon tests.

The short optimized probe established all five loaded digits on opposing sides for most of its 0.4-second window. Extending one selected command to two seconds retained opposition for only 34 of 100 control frames. That constant command did not provide a reliable grasp, and the lever did not complete its release travel. This motivated the tactile-feedback experiment below. Approach, full release, opening and traversal remain outstanding.

The subsequent tactile PPO experiment trained for 201,216 transitions on six local CPU environments. Development seeds 20000–20029 perturb the initialized hand joints by ±0.002 radians. Each trial requires one continuous second of loaded opposition within a three-second horizon, pelvis height above 0.8 m and torso tilt below 12 degrees.

| Controller | Initialized grasp successes | Falls | Evidence |
|---|---:|---:|---|
| Constant optimized preload | 0/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-constant.json) |
| Tactile PPO, final checkpoint | 30/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-final.json) |

An intermediate checkpoint at 160,032 transitions also passed 30/30 under this narrow validation. Close-up native-state replays were inspected at the beginning, middle and end of a trial, and a wide view of the final checkpoint was inspected. These results establish a local contact-hold skill only: one robot, one door, a supplied near-handle pose, small perturbations, no approach or complete lever release. Integration with approach and body movement remains unvalidated. The final policy uses hand proprioception and touch; it has not yet learned vision-based task execution.

Additional initialization stress tests of the unchanged final checkpoint:

| Uniform per-joint perturbation | Successes | Falls | Evidence |
|---|---:|---:|---|
| ±0.02 rad (about 1.15°) | 30/30 | 0/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-noise-002.json) |
| ±0.1 rad (about 5.73°) | 19/30 | 1/30 | [Trials](../results/dexterous/2026-09-07/grasp-001-noise-010.json) |

These tests clip initialized joint positions to native joint limits. They are separate from the original ±0.002-radian validation, and do not change its result. The larger perturbation exposes limited acquisition tolerance and a body fall; the policy is not generally robust.

A separate three-second continuous-hold test within a five-second horizon, at the original ±0.002-radian perturbation, passed 30/30 with no falls. [Trials](../results/dexterous/2026-09-07/grasp-001-hold-3s.json). Reproduce using `evaluate_grasp.py --hold-steps 150 --horizon 250` with the same checkpoint and input files. This longer initialized hold still does not establish opening or traversal.

`scripts/dexterous/fit_grasp_seed.py`, `optimize_grasp_contact.py` and `probe_grasp_seed.py` save inputs and source manifests for subsequent experiments. Their output directories contain the exact initialized pose, preload, contact traces and native trajectories. Rendered close-ups are diagnostic views; the gold operator color changes visualization only.

## Isaac import and controller development, September 7–8

These are development attempts, not benchmark results. Each attempt uses one initialized H1/Shadow grasp on Door55. The final sensor-only policy and traversal task remain unsolved.

- The pinned Isaac 5.1 / Lab 2.3.2 environment starts successfully on L40S. A full automated readiness invocation passed on September 8 at 01:00:21 UTC; see [one-click setup](ISAAC_ONE_CLICK.md).
- Import attempts exposed nested joint-default errors, an incorrectly converted free joint, and concurrent mesh-conversion races. The corrected native/imported link comparison has micrometre-scale position error. Automated tests cover free-root and nested-axis preservation.
- Opening attempts 006–008 showed door movement, but their recording path inserted extra physics steps. They are retained as invalid timing diagnostics, excluded from any controller score. Later attempts enforce the 2 ms clock explicitly.
- The initial open-loop motor reference did not transfer reliably. Closed-loop task-space control is being tested with actual Isaac body/door state. This is a privileged teacher; MuJoCo is used for analytic FK/IK only, while PhysX alone advances the plant.
- Actuator transmission bounds are authored explicitly after import. Actual torque commands sent to PhysX are recorded alongside the native motor calculation. Hand forces come from the solved rigid-contact tensor view.
- The first correct-clock teacher could depress the handle and retract the latch but reached its arm limit before opening sufficiently. A lower reset and gravity-feedforward stance are being investigated; they are not yet validated successful policies.

The native opening probes and stance-controller prototypes in this development branch include unsuccessful experiments. They are opt-in diagnostic scripts, not default benchmark policies. Retain their source and failed trajectories when reproducing development decisions.

Later development checks separate kinematic reachability, actual palm tracking error, per-digit loaded opposition and physical door movement. Attempt 028 maintained opposing contacts through most of the lever turn, reached 0.870 rad handle travel and 12.59 mm bolt retraction, but finished with the leaf at only 0.154 rad. It fails the opening audit. Attempts before 031 also used the importer's unauthored robot friction defaults and cannot establish contact-material parity.

The revised import binds the native friction contract across 974 robot colliders. A complete setup invocation with those bindings passed at **2026-09-08 03:02:07 UTC**. Current runs additionally read the material coefficients back from the PhysX backend. Controller development now includes the waist joint, collision screening of the unused hand, a contact-directed release, and withdrawal away from the door. These changes remain development work until the saved live opening audit and wide/hand visual review pass.


The contact/standing investigation isolated three further failure classes. Trial 034 opened past 60 degrees but fell after a hand reaction spike above 100 N, so it fails. Trial 035 reduced the push lead and stayed upright but finished at 27.7 degrees. Trial 036 added solver-verified 1 mm offsets and separate panel-contact measurement; its release caught the lever/panel and it finished at 21.6 degrees. Trial 037 failed before useful motion because the diagnostic camera supplied a double-precision tensor to a float-only API; the camera input was corrected, and the failure is retained.

Holding the released hand's own leaf-relative pose instead of following the spring-returning lever fixed the repeated withdrawal catch. Trial 038 passed all 18 physical checks and finished at 72.1 degrees; its close-up camera was occluded during release, so the video alone is inadequate visual evidence. Trial 039 repeated with the packaged one-row numerical configuration and a higher camera, finishing at 60.3 degrees. The one-command demo then passed on a freshly imported robot and regenerated door from the desktop launcher (trial 040), reaching the 95-degree stop. These fixed-start development repetitions do not establish a policy success rate or catalogue coverage. [Reproduction and current evidence](ISAAC_HANDLE_DEMO.md).

The final wide repetition (041) ran from a second freshly prepared desktop environment and also reached 95 degrees, passing all 18 checks. Wide frames through pressing, release and final standing and close-up frames through the working grasp/release were inspected. Final demonstrations remain initialized privileged control, not a trained sensor-only policy.

## September 8: contact-free start and acquisition

The [approved next-step plan](DEXTEROUS_NEXT_STEPS.md) was committed as `9fbd1e387` before new development. [The acquisition record](DEXTEROUS_ACQUISITION.md) documents the geometric screen and seven failed native physical attempts. Candidate 009 passes 1,001 sampled configurations, but the actual robot has not acquired the required opposed grasp. Explicit bounded motor commands fix a local servo-target clipping error and reduce reaching error below 1 cm; adaptive closure still misses digit contacts and pushes `LFJ5` beyond its allowed range. These failures remain visible. The initialized Isaac baseline is unchanged; no new GPU run, opening/traversal claim, or catalogue score was made.

## September 8: continued acquisition, walking and live sensor validation

The [continued acquisition failures](../results/dexterous/2026-09-08/acquisition-continuation.json) include four additional root trials. Motor tracking and a dynamics-based limit filter did not acquire the handle. The filter kept the tested joints within the physical gate but stalled early. Fixed material-pad tracking also failed. A lateral approach and equal-pair geometric prior reached within 2.72 mm, remained upright and respected the physical limits, but the ring finger contacted the wrong side. This is a failed grasp, despite all five digits carrying force.

An independent audit verified the native tendon mapping, demonstrated that the canonical initialized grasp can hold all five opposed contacts, and localized the acquisition problem to the approach itself. The old geometric route permits shallow wrong-side contact during most of the approach. New candidates add positive pre-contact clearance and explicitly represent coupled flexion. These are geometric development changes, not a new physical success. Failed sources, trajectories and zoomed hand views are archived off the temporary worktrees.

The [H1 locomotion report](DEXTEROUS_LOCOMOTION.md) records three successful forward plane trials on the unchanged full robot, including a start from the deep opening posture. An independent integration-worktree repetition reproduced 4.022 m travel and 1.991 degrees peak tilt. The original zero-command stop fails, so forward walking is not a complete approach or settle skill.

The owned L40S passed a fresh complete environment readiness check at **2026-09-08 07:32:25 UTC**. A separate [live finite-sensor fixture](../results/dexterous/2026-09-08/isaac-sensor-fixture.json) passed all ten force, shear, taxel, IMU, RGB timing and physics-clock checks at **07:43:28 UTC**. The fixture uses simple pads; actual H1-mounted capture and student-policy execution require their own evidence. The allocation has both local and remote teardown guards. No complete traversal or sensor-only success is claimed.


## Corrected Shadow mechanics, September 8, 2026

The [v2 hand model](SHADOW_LOOPBACK_MECHANICS.md) adds the documented passive
J1 <= J2 limit while preserving the original mass, geometry and motor strength.
The v1 grasp/opening mechanical qualification is withdrawn; its recordings and
previous failed searches remain reproducible. The stricter verifier checks all
five volar pads and all eight joint-pair differences at every 2 ms step.

| Corrected-model component | Measured outcome | Scope |
|---|---|---|
| Native actual Door55 approach | 9/9 varied starts; worst loopback violation 0.306 mrad | Approach only |
| Native initialized grasp hold | Six seconds; all physical/pad checks pass | Initialized, privileged hold |
| Live PhysX unilateral fixture | 7/7 checks; 0.0143 mrad peak difference | Synthetic joint fixture; not native compliance parity |
| Native acquisition v2-001 | Failed; only little finger loads, individual stop violation 27.4 mrad | Failed complete acquisition attempt |
| Native release v2-001 | 13/13 release checks; 44.74 mm final clearance | Initialized physical release, not acquisition |

The release is executed by `probe_acquisition.py --mode initialized-release
--landed-stance`, then converted to a candidate using `reverse_native_release.py`.
The source audit records actual t=0 and every subsequent physics step. Its generic
acquisition report intentionally fails the contact-free-start and final-grasp
checks; `release-audit.json` defines the separate release result. The reversed
candidate preserves measured root/operator frames as reference targets and never
writes the moving robot pose. Physical reversal remains a separate trial.

All details and retained initial PhysX fixture failure are in the
[integration record](../results/dexterous/2026-09-08/shadow-v2-integration.json).
The earlier v1 bounded searches each completed 192 rollouts with zero qualifying
grasps; [their records](../results/dexterous/2026-09-08/grasp-search-v1.json) are
historical diagnostics and are not being reused as mechanically valid targets.

## Continuous v2 integration, September 8, 10:55 UTC

| Trial | Outcome | Scope |
|---|---|---|
| Frozen native acquisition plus seeds 17/29/43 | 4/4 tested initializations pass | One-door acquisition; no catalogue score |
| [Native walk, lower, prepare, acquire and operate](DOOR55_CONTINUOUS_WALK_GRASP.md) | 55 uninterrupted seconds; all physical/task-stage checks pass; final leaf 0.0846 rad | Partial opening only; old intermediate heading target remains failed |
| [Native left-hand contact](BIMANUAL_OPENING_SCREEN.md) | 18/18 checks; left hand minimum final load 3.788 N; palm-only load verification pending; right pads preserved | Stops before right release and full opening |
| Isaac acquisition 001 / 002 | Both fail the strict final hold; all other checks pass | Four / three isolated middle-finger unload steps in the final 251 samples |
| [Isaac operation 001](../results/dexterous/2026-09-08/isaac-operation-v2-001.json) | Final hold and partial-opening hold fail | Lever crossed release before the press timer ended; transition never started |
| [Native measured-release transition](../results/dexterous/2026-09-08/native-operation-clear-v2.json) | 16/16 checks; final leaf 0.0872 rad | Opens on actual clearance; ten intermediate digit-unload ticks, no invalid pads |
| [v2 one-click readiness](ISAAC_V2_READY.md) | Live regeneration, import and solver verification pass | Environment readiness, not task performance |

Two completed Isaac operation runs now include fixed robot cameras and tactile packets alongside the
privileged teacher; both failed their declared task checks. See the [independent archive review](ISAAC_OPERATION_SENSOR_REVIEW.md). The [sensor learner](DEXTEROUS_SENSOR_IMITATION.md) has passed
synthetic boundary/optimizer tests; no closed-loop learned result exists. Release,
complete opening, traversal, repeated task evaluation and broad coverage remain
work in progress. Every failed physical run remains archived.

### Bounded wrist compensation and continuous Isaac integration — September 8, 11:35 UTC

The native release-frozen compensation trial passes all 16 checks, with final leaf angle 0.08785 rad, zero invalid loaded pad patches and 17 transient digit-unload ticks. [Complete receipt](../results/dexterous/2026-09-08/native-operation-compliance-freeze-001.json). The preceding continuously integrating variant is also retained as a native pass; its corresponding Isaac attempt failed fingertip contact after opening. The compensation changes a bounded controller reference, never the physical operator limits or motor caps.

The shared continuous teacher is wired into `isaac_opening.py` through `--full-sequence-reset`, `--preparation-reference`, `--locomotion-checkpoint` and `--native-door`. Readiness uses actual landed state in an unstepped collision model. A normal-plus-tangential hand-load adapter, actual torque-delivery checks, full-episode contact checks and explicit phase-clock offsets were reviewed before execution. Source and input packages are frozen before each GPU run. The original near-handle path and failed experiments remain available. No full Isaac traversal or learned task success has been established.

### Native loaded opening and fresh Isaac surface-profile trial — September 8, 12:18 UTC

The [loaded bimanual sequence](BIMANUAL_OPENING_SCREEN.md) passes 20/20 native checks over 36.736 seconds, reaching 1.200999 rad (68.8°). It acquires the right-hand grasp, operates the lever, establishes left-palm support, releases axially from the lever and continues opening while supported. Final palm load is 3.373 N, with at least 2.027 N throughout the final half-second. This initialized lowered-stance trial is separate from the earlier continuous walking-to-partial-opening result; neither establishes traversal or an Isaac full-opening pass.

[Fresh Isaac operation-005](../results/dexterous/2026-09-08/isaac-operation-v2-005.json) fails sustained grasp. Its predeclared volar-phalange profile records no wrongly placed contact patches, a final leaf angle of 0.09819 rad, and maximum torque-delivery error 1.91e-6 Nm. Independent inspection of 251 final-hold samples finds 98 opposition failures and four low-load samples. The historical `operation_digit_unload_samples` counter means any invalid grasp, not only unloading; read it accordingly. The earlier distal-only failures are unchanged and remain excluded from opening teacher data. A clear side close-up was personally inspected, in addition to the every-step numeric audit.

Continuous Isaac trial 001 remains failed and incomplete: it walked and lowered but never began arm preparation. Native SIGINT bypassed final numeric export, leaving video, snapshots and provenance but no complete 500 Hz archive. Trial 002 records each readiness condition and distinguishes separated, zero-force speculative contact entries from actual touch. Positive-gap contact entries remain available for diagnosis; physical penetration checks and solver offsets are unchanged. Periodic evidence is explicitly incomplete and cannot qualify a run. A `stop.request` file triggers a graceful export. Actor cold-start observations and the first force command are also retained before stepping, and sensor data readers reject partial or hash-mismatched captures and actor-generated teacher demonstrations.

### Qualified initialized partial opening and continuous integration — September 8, 13:04 UTC

| Actual Isaac trial | Outcome | Limitation |
|---|---|---|
| [Operation006](../results/dexterous/2026-09-08/isaac-operation-v2-006.json), 24 s | 19/19 declared checks; final leaf 0.0977786 rad | Near-handle contact-free start; transient contact failures disclosed; no traversal |
| [Continuous002](../results/dexterous/2026-09-08/isaac-full-v2-002.json), 55 s | 26/27 checks; approach, preparation and mechanism operation pass | Final sustained grasp fails; full episode remains failed |
| Sensor actor001, 0.522 s | Failed: fell before acquisition | First absolute motor commands have wrong-sign knee forces |
| [Sensor actor002](../results/dexterous/2026-09-08/sensor-actor-acquisition-v2-002.json), 0.762 s | Failed: fell before acquisition | Startup supervision improves prediction, but feedback drift remains |

The original distal-only grasp result remains false in the successful volar-phalange trial. All original motor limits, passive couplings and collision checks remain in effect. [Independent operation006 review](ISAAC_OPERATION006_REVIEW.md). Complete recordings, source manifests, reports, videos and failed actor prefixes are archived off-pod under `DoorBench-runs/2026-09-08-shadow-loopback`. The new teacher-query recorder stores the failed actor's exact pre-action states for counterfactual offline corrections; these are never runtime actor observations or failed-student actions relabeled as expert.


### Actual-step passage and continued Isaac development — September 8, 13:56 UTC

 [continuous Isaac trial003](../results/dexterous/2026-09-08/isaac-full-v2-003.json) finishes 55 seconds with 26/27 checks; its increased index preload does not fix the final grip (seven isolated 2 ms index unloads in the final 251 samples). All independent archive checks pass, and no loaded pad patches are misplaced. The unchanged trial002 failure is retained. [Native initialized continuation009](POST_OPENING_CONTINUATION.md) now passes 24/24 physical checks and 5/5 independent archive checks over 65 seconds: arm stow, rise, actual walking, whole-body passage and quiet stop. It starts from an earlier attained open-door state with leaf momentum; it does not establish uninterrupted opening or a difficult loaded release. Earlier native contact-force qualifications that used an extra forward solve are explicitly withdrawn in the timing review. The corrected actual-step recorder retains force/pose epochs separately. Four sensor-only Isaac candidates have failed; the longest lasts 1.042 seconds, with incidental hand collisions but no acquired grasp or operated handle. Offline corrections and a longer recurrent history are being trained. Full bimanual Isaac opening is running; full-sequence composition, robustness and broad coverage remain unfinished.


### Walking/full-opening integration — September 8, 14:54 UTC

[New native trial table, retained failures and exact scene scope](DOOR55_WALKING_FULL_OPENING.md). Trial 002 passes 28/29; neither it nor the initialized passage components establishes an uninterrupted qualified task. Isaac full-opening 002 fails transfer;003 is a pre-physics setup failure;004 tests an independently screened actual-state LH path.

The five physical sensor-only attempts all fail. Longer CUDA fit 006 completes 5,000 steps in 444.61 s, with 4/6 offline readiness criteria passing; corrected-sampler 007 also completes 5,000 steps but still misses nominal/startup criteria. [Data, checkpoint and fitting receipts](SENSOR_DAGGER_STARTUP.md). The newly discovered sampler gap and autoregressive previous-command drift are documented separately; no failed candidate is presented as a policy success.

- **September 8, 15:15 UTC — current-door continuation and active runs.** The revised Door55 initialized continuation012 passes 26/26 physical checks and 6/6 independent checks, with all 34,500 actual 2 ms transitions retained and zero solver failures or warning-counter increments. [Exact scene, rejected routes, and reproduction](POST_OPENING_REVISED_DOOR.md). Its source walking/opening002 still fails the right-pad release gate; there is no qualified uninterrupted task. Isaac full-opening004 is running with a newly screened actual-state left-hand route and original release controller. Sensor training008 separately tests actor-owned command history; no checkpoint has qualified a new physical actor trial.

- **September 8, 15:55 UTC — continuous adapters and retained failures.** Native005 reaches 1.200903 rad with 9.092 N palm support and passes 29/30 checks; the sole failed gate remains RH pad validity during release. All actual warning counters are retained. Isaac004 is stopped at 43.1 s and fails 7/24 checks; independent reduction finds 3,308 invalid pad intervals, first at35.142 s. Late LH support cannot cure earlier invalid contact. The full152-file run and476-file frozen source archives match remote SHA256 manifests. The same recorded joint states were rendered without stepping for frontal and oblique hand inspection, explicitly labeled geometry replay. [Trial details](DOOR55_WALKING_FULL_OPENING.md).

  The native `--traverse` interface has500 actual steps bit-identical to the existing walking prefix and5/5 interface checks; its one-second full-task report is deliberately failed. The new Isaac traversal adapter preserves global time, freezes opening qualification before continuation and requires independent final passage/quiet evidence; live validation remains pending. Sensor fits008/009 retain all failed checkpoints. The [frozen-history diagnostic](SENSOR_DAGGER_STARTUP.md) reproduces2–3 Nm differences from truncated recurrent history; it does not establish that direct torque control itself causes the falls. Coordinated body motion for lever return and continuous-history training are the next controlled experiments.


### Whole-body return, adapter audit and first fitting-qualified actor — September 8, 16:38 UTC

| Experiment | Measured outcome | Scope |
|---|---|---|
| [Native walking/opening006](../results/dexterous/2026-09-08/native-walking-opening-006.json) | 28/30 checks; all 30,000 actual transitions reproduce the original component exactly | Returns the lever through robot motors after real walking; intentional stop before release/full aperture |
| [Isaac full-opening005](../results/dexterous/2026-09-08/isaac-full-opening-v2-005.json) | 17/24 checks; first invalid right pad at 35.346 s; failed transfer | 6 N left support target; no continuous half-second qualified support/grip overlap |
| [Executed Isaac traversal smoke](ISAAC_TRAVERSAL_SMOKE_RESULT.md) | 26/26 independent interface checks, 500 actual 2 ms steps | One-second interface test; original full-task report remains incomplete/false |
| [Whole-body withdrawal development](DEXTEROUS_RIGHT_RELEASE.md) | Route completes in second trial, but ten invalid distal contacts and loss of left-palm support remain | Failed actual-state-specific native development, not a qualified release |
| [Sensor training010 candidate56](evidence/sensor-first-qualified-candidate-010.json) | 6/6 frozen fitting gates and 16/16 runtime-boundary checks | Offline admission only; actual Isaac acquisition006 launched separately |

The integrated native lever return preserves all original motor limits, passive constraints and authored collision geometry. The independent reproduction compares every common state, control, body and actual solved-contact array, with no differences. All runtime warning counters remain zero. Actual Isaac005 source and outputs total 259,159,958 verified bytes; the comparison with004 is descriptive, because their initial acquisition timings already differ before the load intervention.

Training010 holds weights fixed through complete source histories and only then applies each accumulated optimizer update. Candidate56 is the first qualifying checkpoint, selected before the physical run; the also-qualifying final64 checkpoint does not replace it. Actual acquisition006 preserves all 304 runtime/input files from 005, including reset, sensor layout and plant paths. It uses only RGB, local touch, proprioception, previous actions and observation timing/validity, without a teacher fallback. Its result is not inferred from offline fitting.

Actual [actor006](../results/dexterous/2026-09-08/sensor-actor-acquisition-v2-006.json) failed at 0.702 seconds:14/17 checks passed, with upright posture, duration and sustained grasp failing. No loaded handle contact occurred. Maximum submitted motor-command error was 3.815e-6 Nm. All 65 output files and 309 source files are independently hash-verified off-pod. The model's initial torque error is much smaller than earlier candidates, but its feedback trajectory still diverges; neither its offline fit nor its longer duration than 005 is a stability qualification. Six sensor actors have now failed, and their outcomes remain in the development ledger.

[Sensor-only stationary balance](SENSOR_BALANCE.md), final-source native005 and006: both 5-second, 2,500-step trials pass 19/19 checks. Nominal peak tilt is 0.3114 degrees; a separate 0.05 m/s reset-velocity trial also passes. The controller receives only calibrated joint encoders, local IMU, foot tactile readings and previous command, with a fixed joint-only desired posture. Root/support coordinates are estimated in an arbitrary local frame; actual simulator state remains evaluator-only. RGB is unused. These component passes do not establish reaching, door opening, Isaac behavior or a learned policy.

### Sensor-only balance reaches Isaac — September 8, 17:26 UTC

[Isaac balance001](../results/dexterous/2026-09-08/sensor-balance-isaac-001.json)
passes 12/12 stationary checks and 14/14 original physical checks over exactly
5 seconds. Maximum tilt is 0.3566 degrees; no hand contact or stance-QP failure
occurs. This is analytical balance with encoders, IMU and tactile input, not a
learned policy or door interaction. Source `f7c07f111`, calibration, reset and
all 81 outputs/509 frozen source files are hash-verified off-pod. Independent
raw-contact audit reproduces every interval with zero error. Strict packet
replay fails at the second command after a 0.125525 Nm initial leg-QP difference,
including in the original runtime; this discrepancy remains under investigation.
Root inspected the
actual wide view and handle close-up; the latter misses the raised hand.

The [independent native reproduction](../results/dexterous/2026-09-08/sensor-balance-native-root-001.json)
passes 19/19 checks with every state, motor command and sensor packet exactly
matching the earlier nominal native005. Modest arm movement has a separate
native component result; its actual Isaac test is next. Full opening, safe
release and uninterrupted traversal remain unresolved.

### Scripted-arm balance in Isaac — September 8, 18:02 UTC

[Actual arm001](../results/dexterous/2026-09-08/sensor-arm-balance-isaac-001.json)
passes16/16 grouped and14/14 original physical checks across3,000 steps.
Peak tilt is0.3577degrees; arm tracking error stays below0.002773rad.
This is the same sensor-only balance architecture with a declared numeric
arm schedule; vision is unused and no door contact is attempted. All95 outputs
and528 source/input files are hash-verified off-pod. Actual wide and hand views
show the supported robot and clear hand at maximum excursion and after return.

The native shortened release still fails three LF/FF contacts under the original
pad gate. Fixing post-release target ownership produces exactly the same143 raw
chunks and physical commands as its failed predecessor, so that cleanup does
not explain the panel collapse. The next controlled comparison projects the
normal contact acceleration across all eight actuated waist/left-arm joints.
The older projection covers only seven arm joints while the waist servo remains
active. This is a hypothesis being physically tested, not a successful result.

### Coordinated sensor-feedback reach and grasp — September 8, 19:00 UTC
[actual Isaac coordinated reach001](SENSOR_REACH_RUNTIME.md) passes 20/20 reach and 14/14 physical checks over 5,500 steps, with 5.05 mm palm endpoint error and peak torso tilt 0.384 degrees. All 111 outputs and 551 source/input files are hash-verified off-pod; wide and hand views were personally inspected. The passive finger split differs from native, so grasp transfer is tested separately. [Native full acquisition001](SENSOR_SCRIPTED_ACQUISITION.md) passes 25/25 with a 2.628-second opposed five-pad hold and all 9,500 contact intervals independently reproduced. Its first uninterrupted lever press fails when fingers unload and the thumb middle segment takes load. Tactile preload/progression and actual Isaac acquisition are next. The new bounded whole-body panel path completes 95 seconds without the prior collapse, but still fails aperture, palm-specific support and inherited RH release contact. Full sequence, repeatability, learned sensor-only task success and catalogue coverage remain open.

- September 8, 19:45 UTC: [actual Isaac acquisition001](SENSOR_ACQUISITION_ISAAC_001.md) completes all 9,500 steps but fails the original distal-contact anatomy and opposed-hold gates. Independent raw reduction finds 1,549 invalid little-finger patches, including middle-segment loading; FF/MF/RF end unloaded. An evaluator arithmetic correction reproduces the producer's float32 sum while retaining the independent force threshold; it corrects the reported duration to 19 seconds and leaves task failure intact. All 138 outputs and 566 source/input files are verified off-pod. Hand and wide views were personally inspected. Native tactile feedback advances the lever to 0.3265 rad with valid opposed pads, but remains below the release threshold. The flat-palm reference path now sustains support without collapse, while insufficient hinge moment and three earlier RH withdrawal contacts still prevent qualification. Run Center now includes wide and hand previews. Full-task completion, repeatability, learned sensor-only success and catalogue expansion remain open.

- September 8, 20:15 UTC: the [corrected passive-joint adapter](ISAAC_PASSIVE_JOINT_PROFILE.md) passes its first [actual Isaac reach qualification](../results/dexterous/2026-09-08/sensor-reach-dry-isaac-001.json), launched at 20:04:51 UTC: 20/20 reach and 14/14 physical checks over 5,500 steps. All 69 original passive coefficients and armatures remain exact throughout; the initial free-space finger split drift is greatly reduced. Peak tilt is 0.3785 degrees and palm endpoint error 5.164 mm. The old explicit friction approximation is retained under its historical profile; a native/PhysX isolated fixture independently exposes its persistent velocity oscillation. All 114 outputs and 588 source/input files are verified off-pod, with hand and wide views personally inspected. A separate corrected-profile 19-second grasp test is starting; the original grasp failure remains failed. Native index coordination reaches 0.5206 rad of lever rotation but still fails release, and a 0.25 mm palm shift does not improve it. Bounded local force feedback and the full-sequence reference remain in development. Stages 1–5 remain unfinished.

- September 8, 20:42 UTC: the corrected-friction [actual Isaac grasp](SENSOR_ACQUISITION_DRY_ISAAC_001.md), launched at 20:14:28 UTC, completes 9,500 steps with 21/23 task and 14/14 physical checks, but fails distal-contact anatomy and sustained opposition. All 141 outputs and 594 source/input files are hash-verified off-pod; independent raw-contact and passive-property audits reproduce the failure exactly. Body placement accounts for most of the remaining hand-position difference, motivating bounded Cartesian compensation before further GPU grasp trials. Native direct tactile force feedback retains a qualified grasp but stalls at 0.4426 rad; exact motor decomposition shows competing posture effort. Contact-normal force/posture separation and actual-base panel tracking are in development. A portable ready-host sensor preparation command passes its real-host dry runs; full preparation execution is next. Full sequence, repeatability, sensor-only learning and catalogue coverage remain unfinished.

- September 9, 04:31 UTC: the corrected native teacher has a complete 127.99-second door-opening and traversal replay with 47/47 task and 16/16 independent checks, but is privileged and slow. The first fresh sensor student falls at 0.296 s (0/1 full tasks). Actual Isaac grasp reaches 22/23 checks; the first local tactile preload improves the opposed hold from 2 to 32 ms but still fails the required 500 ms. Independent contact and gyro audits reproduce it, with zero invalid loaded patches. A 1.5 N follow-up fails the native 0.04 rad motor-tracking gate and is not launched in Isaac; the 0.8 N retry is undergoing the unchanged checks. See [the measured grasp comparison](ISAAC_TACTILE_GRASP.md) and [continuous native evidence](CONTINUOUS_NATIVE_TRAVERSAL.md).
