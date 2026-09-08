# DoorBench: simulated dexterous humanoid policy

Revised September 7, 2026, following the owner's clarification: no real hardware is available. This plan supersedes the hardware-dependent stages in the earlier vision/tactile proposal. All targets below are proposed acceptance criteria, not completed results.

The [September 8 execution plan](DEXTEROUS_NEXT_STEPS.md) records the next milestones from the verified initialized Isaac opening to complete traversal and sensor-only control, with current limitations and progress requirements.

**Objective: a freely moving, physically simulated humanoid with two high-DOF, five-finger hands that reliably opens almost every DoorBench door a human could open under the same task conditions, and traverses the openings intended for passage. The final policy uses simulated vision, tactile sensing, and its own body state.**

Real data collection, hardware procurement, sensor calibration against a physical device, and real-world deployment are outside this project milestone. Physical plausibility remains essential: success must come from the robot's actuators and valid interactions with the mechanism.

**Make “almost all” measurable**

I propose the following final target:

| Measure | Proposed target and interpretation |
|---|---|
| Reliable door-opening coverage | At least **95% of the frozen human-openable door set**. A door counts as reliably solved only if the policy passes at least **9/10 independently seeded trials for each required opening scenario** on that door. Report episode success as well. |
| Mechanism coverage | At least **90% reliable door coverage in each populated mechanism family**. Report small families as counts so rounding does not hide omissions. |
| Full traversal | At least **95% reliable coverage of the human-traversable doorway subset**, using the same 9/10 scenario rule, including controlled release and a stable finish. Report separately from opening. |
| Unseen-door generalization | Apply the same targets to a reserved test split, grouped by mechanism construction/CAD lineage. A separate catalogue-fit score may include trained-on doors, but cannot stand in for unseen-door performance. |
| Physical validity | Every counted successful trial passes actuator, contact, collision, balance, clearance, and no-assistance checks. Invalid trials count as failures. |

The 95% target formalizes “almost all”; it does not guarantee the result is achievable with the initial model or compute allowance. Development evaluations can use fewer trials, but cannot be labeled as the final ten-seed result. Numerical passage and force thresholds, time budgets, randomization ranges, and splits must be frozen before the final run. Do not use the final seeds for curriculum selection.

**Define human feasibility independently of robot success.** Begin with the 985 non-pet assets and their actual starting conditions, then review the mechanism, available release, required equipment, reach, force, and clearance. Record the intended procedure and evidence. A physically valid simulated human rollout is strong evidence; lack of one is not proof of impossibility. Existing scripted-hand results are useful diagnostic leads but are not equivalent to human feasibility.

A door locked without an available release or credential is not an opening task for either actor. Keep it in a separate abstention/task-understanding score. Where a key, badge, code, or other aid is necessary, define identical access for the human and robot. A supplied credential may be task context, but it must be used through the modeled interaction: no privileged unlock API. If required equipment is not modeled, record the scenario as unsupported/pending and show that coverage gap.

Hatches and other openings a person operates without walking through belong in the opening score. Their geometry does not justify requiring upright traversal. Pet doors remain supplementary under the existing project rule. Robot reach, strength, or learning failures do not justify removing a human-feasible door. Unresolved feasibility and asset defects remain visible in the complete inventory; freeze documented exclusions before the scoring run and rerun after fixes. No claim of full catalogue coverage while a hidden pending set remains.

The existing `human` benchmark suite concerns interactions with another person, such as yielding. It is not the human-feasibility reference proposed here. Give the new protocol its own versioned identity.

**Start from an existing dexterous humanoid**

My preferred starting embodiment is the **HumanoidBench H1 with two Shadow Hands**, subject to a model audit. HumanoidBench already combines that robot with egocentric cameras and tactile sensing, and supplies simulation/training code. This directly reduces the amount of custom body and hand integration needed. [HumanoidBench project](https://humanoid-bench.github.io/).

Shadow's hand design has five fingers, 24 movements, and 20 motor units; the chosen simulation's precise joints, actuators, coupling, and wrist definition must be enumerated rather than inferred from that headline. [Official Shadow documentation](https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/stable/index.html).

Audit both hands for thumb opposition, range limits, tendon/coupled-joint behavior, collision geometry, realistic mass, bounded force/velocity, and usable contact surfaces. Audit the body for reach, shoulder/wrist maneuverability, crouching, and clearance across the catalogue before freezing it. Keep the body free-floating, with actuated legs and contact-supported balance in qualifying trials.

If anatomy or arm range makes otherwise ordinary human actions structurally impossible, change the benchmark embodiment explicitly during development. Freeze its dimensions and limits before evaluation. Do not silently increase strength, resize fingers, remove collisions, or switch robot geometry per door. Treat this as a documented simulated research embodiment; a composite H1/Shadow model does not establish compatibility with a purchasable robot.

The existing G1 locomotion checkpoint remains a useful separate baseline. It does not need to constrain the primary high-dexterity embodiment.

**Separate physical capability from sensor-policy learning**

The project should produce two independently scored systems:

1. **A privileged physical teacher.** During training and diagnostic evaluation it can see mechanism state and geometry. It must still solve the task through the full humanoid's actuators and contact. This asks whether the robot, mechanics, and controller can perform the action at all.
2. **The final vision/touch policy.** It sees rendered head/optional wrist images, finite tactile arrays, joint encoders, IMU, previous actions, and the task instruction. It must infer handle position, engagement, release, motion direction, and progress from those observations. This is the system that must meet the final success criterion.

Exact door coordinates, latch flags, simulator object IDs, world-space contact lists, and door-specific policy lookup are forbidden in the final actor and its runtime controller stack. A privileged critic or reward function is allowed during training. Internal expert selection can be learned from permitted observations, but a door ID cannot select a memorized controller at evaluation.

Current DoorBench observations contain several privileged values, so build a separate sensor-only API rather than changing the meaning of the existing baseline. The upstream HumanoidBench environment also defaults to privileged observations and documents a separate visual/tactile wrapper; audit it before reuse. [HumanoidBench implementation](https://github.com/carlosferrazza/humanoid-bench).

The human reference contributes contact strategies, posture priors, and candidate trajectories. Robot teachers re-solve those examples for their own embodiment. We do not need to finish a perfect synthetic human animation for every door before training.

**Training strategy**

My hypothesis is that compositional skills and a curriculum will be more effective than asking one policy to discover locomotion, dexterous grasping, lock sequencing, and visual perception simultaneously. HumanoidBench's findings support hierarchical control with useful low-level skills, while DoorMan provides a relevant teacher-to-visual-policy recipe. These precedents do not demonstrate near-complete DoorBench coverage. [HumanoidBench paper](https://arxiv.org/abs/2403.10506), [DoorMan paper](https://arxiv.org/abs/2512.01061).

Use the following progression:

| Stage | Training work | Reviewable result |
|---|---|---|
| 0. Lock the task and model | Feasibility inventory, robot/hand audit, sensor contract, simulator compatibility tests, training/development/test split. | Frozen initial protocol and native videos of hands and body operating correctly. |
| 1. One complete doorway | Bootstrap reaching and balance; optimize grasp/press/pull or push; hold, traverse, release, settle. Train in simulation with full state first. | One complete unsupported-body rollout with close-up hands and physical validity checks, then high repeated success. |
| 2. Mechanism skills | Learn on representative lever, knob, pull, push-bar, bolt/thumbturn, sliding, folding, and other distinct mechanisms. | Approximately 20–30 representative configurations with per-skill success, force, and failure reports. This is a curriculum seed, not the final scope. |
| 3. Expand physical capability | Combine learned skills, automatic task sampling, trajectory optimization, and family specialists. Include both hands, regrasping, changed approach side, and multi-step releases. | Privileged full-humanoid coverage progressing toward 99% on the declared scope; every failure triaged as controller, embodiment, mechanism, or unresolved. |
| 4. Train the sensor student | Generate robot-view/tactile trajectories from teachers, imitate, collect corrections on student states, then fine-tune with recurrent RL. Start on stage-1 data while the curriculum grows. | A sensor-only policy whose coverage approaches the teacher; separately measured perception/contact gap. |
| 5. Robustness and generalization | Train with appearance and dynamics variation, contact interruptions, approach errors, and student-generated failures. Preserve holdouts. | Reliable recovery on development conditions and a checkpoint selected without final-test feedback. |
| 6. Final catalogue and holdout runs | Freeze source, geometry, robot, observations, checkpoint, seeds, and score logic. Run complete trials. | Coverage tables, family results, confidence intervals, failure inventory, videos, and reproducible artifacts. |

Family specialists are a training tool. Distill them into one shared policy or a fixed ensemble whose routing uses permitted sensors. The final system must switch among needed skills autonomously within an episode.

Practice difficult later stages from physically reached reset snapshots, then gradually increase complete start-to-finish rollouts. Temporary fixed-base hand training is acceptable for skill acquisition; it cannot count as a qualifying full-humanoid episode. Every final trial starts with the specified door state and the humanoid in the starting region, not already grasping the handle.

Sample training configurations by mechanism and development failures, not their raw catalogue frequency. Otherwise common lever doors could dominate while whole rare families remain unsolved. Batch similar articulation/contact structures for throughput, but never remove required mechanism geometry to make training faster.

Use grasp/contact constraints for initialization, then optimize trajectories and policy behavior in physics. Keep finger commands independently adaptable; a compact grasp synergy may initialize the hand but should not permanently eliminate dexterity. Let the policy use palm contact, precision pinches, power grasps, hooks, two hands, and handoffs when the operator calls for them.

**Sensor and controller design**

A recurrent encoder combines visual context, touch, and body state. A shared skill controller produces wrist targets, finger commands, and compliant contact adjustments. A low-level controller executes arm and leg motion under joint and force limits, with balance trained under door-induced loads. Phase transitions depend on observations rather than a fixed animation clock.

Simulated touch should be finite normal/shear pressure arrays on fingertips, finger pads, and palms, with optional disclosed lower-resolution body/foot sensing. Avoid filtering contacts by privileged object identity: a sensor reports what touches its surface, not an oracle label saying “correct handle.” Add configurable resolution, clipping, noise, dropout, and delay, and publish the chosen sensor model.

MuJoCo's touch-grid sensor aggregates contact forces into spatial bins in a local sensor frame, providing an appropriate starting representation. It is an idealized simulated sensor, not a physically calibrated tactile device. [Official sensor implementation](https://github.com/google-deepmind/mujoco/blob/main/plugin/sensor/README.md).

Tactile updates and contact corrections should run faster than visual planning. Benchmark proposed camera rates around 20–30 Hz and tactile/control updates around 100 Hz against the fixed simulation timestep, compute load, and latency model. Faster motor integration must not be confused with fresh camera information.

In simulation we can collect unlimited procedural variations, but not unlimited useful computation. Favor a compact recurrent student first, then investigate larger action models only when a controlled comparison justifies them. Compare vision-only, vision-plus-touch, and vision-plus-touch-with-history/probing using matching training budgets. The main objective is near-complete task success; proving a tactile improvement is supporting evidence, not a gate that halts a successful policy.

**Physical and visual quality gates**

Count success only when the intended mechanism is actually operated and produces the required usable opening. Partial joint motion alone is insufficient. For paired leaves or bypass panels, score actual usable aperture and task requirements rather than blindly requiring every panel to move or accepting an obstructed passage. An appropriate normal procedure may open only one leaf if it provides sufficient clearance.

Manual doors move through robot contact and the modeled mechanisms. Legitimate automatic motors can respond to modeled sensors/buttons; the actor cannot directly command a door joint or an unlock helper. No runtime pose writes, hand welds, foot anchors, unbounded actuator effort, invisible support force, or disabled required collisions in qualifying rollouts.

Rewards combine task completion and progress with sensible posture, smoothness, low unnecessary force, and recovery. Keep an upright default and a stronger penalty on pointless backward lean, while allowing useful crouching, forward reach, and balance adjustments. Use anatomically valid thumb/finger opposition for grasps, without requiring every task to use all five fingers. Enforce joint and force bounds independently of the reward.

Validate that contact caused operator movement, locks released through the modeled procedure, hands do not penetrate hardware, feet support the body, and the whole body clears the opening where traversal is required. Record uncut wide views, robot views, and hand close-ups with joint/force plots. A training reward and a visually smooth clip are not sufficient validity evidence.

Without real measurements, choose documented physical parameter ranges from specifications and mechanical models, publish assumptions, and test sensitivity. Rerun representative cases with a smaller timestep and tighter solver settings. Selected cross-engine checks can identify simulator-specific exploits; neither engine agreement nor parameter randomization proves real-world transfer.

**Simulation engines and GPU execution**

Start robot/hand integration and native contact inspection in MuJoCo, where the proposed embodiment and DoorBench's current physical-reference tooling already exist. The intended scaling path is Isaac Lab for GPU simulation and dynamic camera rendering, with MuJoCo retained for focused independent checks.

Before committing to large training runs, test the assembled hands, coupling, contact sensing, and several door mechanisms on the chosen GPU backend. MuJoCo plugins, tendon behavior, and Python callbacks are not automatically portable to an accelerator or PhysX. If the port is incomplete, compare a compatible MuJoCo/MJX path against fixing the Isaac integration, then freeze one production training backend. Do not maintain two unverified training implementations or silently simplify the robot to pass the port.

Use Blender for reusable visual assets/materials and reviewed presentation, with dynamically rendered camera observations synchronized to the current simulation state. Keep mechanism geometry physically identical across appearance variants. Broad visual diversity becomes a training stage after the interaction works; no catalogue-wide render redo is a prerequisite.

Profile state-only, RGB, and RGB+tactile modes separately on an isolated RunPod node. Auto-tune simultaneous environments at fixed physics settings, maximizing useful transitions per second with memory headroom. A 4x4 video layout is independent of the training batch. Use a capped profiling allocation before extrapolating training cost, then stage budgets based on learning curves and coverage gains. Preserve the existing unrelated GPU work, arm teardown protection for allocated nodes, and expose stage, heartbeat, GPU-hours, throughput, checkpoint, coverage, and recent failure videos in the local dashboard.

**First work package**

The first implementation package should be: a reviewed dexterous humanoid model; the sensor-only interface; one verified lever door; a complete contact-driven grasp/open/traverse sequence; and a measured training/backend report. It should also include the initial full-catalogue feasibility inventory so a successful one-door demo cannot become the entire scope.

The resulting research contribution is a simulation policy and benchmark with reproducible high-dexterity interactions, broad mechanism coverage, synchronized visual/tactile experience, and honest physical validity checks. Real-hardware transfer is a possible later project, not a dependency or a claim of this milestone.
