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
