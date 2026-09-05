// Plain-language explanations for every row of the door panel: what the quantity is, its unit, and how it is
// derived or used.  Keys are the row keys used in DoorView.tsx.  Sources: docs/PHYSICS.md, docs/BENCHMARK.md.

export interface GlossaryEntry {
  what: string;   // what the quantity means
  unit?: string;  // unit shown in the panel
  how?: string;   // how it is derived / used
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // ---- Leaf
  mass_total: { what: "Total mass of the moving leaf including its hardware (handles, locks, half the hinge mass).", unit: "kg", how: "slab mass + glass mass + hardware mass; the source of the area-density table is shown at the bottom of the panel." },
  mass_slab: { what: "Mass of the door slab itself (core + faces, without glass and hardware).", unit: "kg", how: "area density of the construction (manufacturer door-weight tables, lb/ft² by core and thickness) × (W×H − glass area)." },
  mass_glass: { what: "Mass of any glazing in the leaf.", unit: "kg", how: "glass density × glass thickness × lite area (0 for unglazed doors)." },
  mass_hardware: { what: "Mass of the operator set, lock and the leaf's share of the hinges.", unit: "kg", how: "catalogue masses of the hardware items on this door." },
  size: { what: "Leaf width × height × thickness.", unit: "m", how: "sampled per family and context (standard leaf sizes); the opening is slightly larger to leave hinge and latch-edge clearance." },
  panel_style: { what: "Face design of the slab (flush, raised panels, glass lites, planks…).", how: "drives the procedural face geometry and, for glazed styles, the glass mass and breakage threshold." },
  finish: { what: "Surface finish and colour of the leaf.", how: "affects only rendering (material roughness / colour), not physics." },
  inertia: { what: "Moment of inertia of the leaf about its hinge axis: how hard it is to accelerate the door in rotation.", unit: "kg·m²", how: "∫ r² dm of the slab, glass and hardware about the hinge line (≈ m·W²/3 for a uniform slab). Used with the torques below to predict opening times." },
  condition: { what: "Wear state of the door (new, normal, worn, old/dry, rusty, swollen, sagging, damaged, well oiled).", how: "multiplies hinge friction and damping, adds lock backlash and stiction (e.g. a swollen door sticks in its frame)." },
  // ---- Hinge / motion
  hinge: { what: "Hinge or pivot type (butt, ball-bearing, rising butt, spring, strap, pivot, continuous…).", how: "sets the bearing friction coefficient, pin radius and load rating used for friction and tear-out." },
  hinge_count: { what: "Number of hinges carrying the leaf.", how: "the leaf load is shared between them; more hinges = less load per pin." },
  side_swing: { what: "Hinge side as seen by the robot, and whether the robot must push or pull the leaf (robot stands at −y).", how: "sampled per context; pull doors need a start zone outside the swing arc." },
  coulomb: { what: "Constant (Coulomb) friction torque the hinges oppose to any motion.", unit: "N·m", how: "μ·(m·g·r_thrust + 2·F_h·r_pin)·k_condition + steady seal drag; F_h is the horizontal hinge reaction m·g·(W/2)/span. Exported as MuJoCo frictionloss." },
  stiction: { what: "Extra break-away torque when the door is stuck (swollen, rusty, sagging).", unit: "N·m", how: "condition table; added to the static opening force in the compliance check." },
  bearing_mu: { what: "Friction coefficient of the hinge bearing surfaces.", how: "catalogue value per hinge type (ball bearing ≈ 0.03 … dry steel pin ≈ 0.3)." },
  damping: { what: "Viscous damping torque per unit angular velocity on the leaf joint (symmetric part).", unit: "N·m·s/rad", how: "aerodynamic drag linearised at 1 rad/s (½·ρ·Cd·H·W⁴/4) plus closer damping; the closer's open/close asymmetry is applied in the environment callback." },
  roller_friction: { what: "Rolling / sliding resistance of a sliding or vertical door's carriages, guides or curtain.", unit: "N", how: "μ_roll · m · g · k_condition (μ shown in brackets); exported as frictionloss on the slide joint." },
  max_open: { what: "Maximum opening of the primary joint (angle or travel).", unit: "° or m", how: "sampled, then capped by what the geometry allows (casing, closer arm, wall, strap hinges…)." },
  stop: { what: "What physically stops the leaf at full open (wall bumper, floor stop, closer arm, track end, none).", how: "modelled as the joint limit; the label tracker uses it for slam detection." },
  // ---- Closer
  closer_model: { what: "Self-closing device on this door (surface / concealed / floor-spring closer, spring hinges, gas strut, gate closer, automatic operator, none).", how: "gives the spring preload, spring rate and damping below." },
  en_size: { what: "EN 1154 closer power size (1–7) chosen for this leaf mass and width.", how: "installer rule: the smallest size whose closing moment beats 1.3 × hinge/seal friction; larger doors get bigger sizes." },
  preload: { what: "Closing torque at the closed position (spring pre-tension).", unit: "N·m", how: "≈ 1.15 × the EN 1154 minimum closing moment of the size, raised if needed to overcome friction. The robot must exceed this to start opening." },
  spring_rate: { what: "How much the closing torque rises per radian of opening.", unit: "N·m/rad", how: "(opening moment at 90° − preload) / (π/2), with the 90° moment at 85 % of the EN 1154 maximum. Exported as joint stiffness." },
  closer_damping: { what: "Viscous damping of the closer when closing / when opening (closers damp the closing stroke much more).", unit: "N·m·s/rad", how: "catalogue values per closer type × condition; applied direction-dependently by DoorEnv (not natively in MJCF)." },
  closing_time: { what: "Estimated time for the door to swing from 90° to closed on its own.", unit: "s", how: "integrated from I·θ̈ = −(τ0 + k·θ) − b·θ̇ with the leaf inertia ≈ m·W²/3. Used by the benchmark to decide whether the robot must hold the door." },
  // ---- Operator / latch / lock
  operator: { what: "The hardware the robot must work to release the latch (lever, knob, panic bar, pull, thumb latch, wheel, slide bolt…).", how: "catalogue item with travel, return spring, grip offset and yield; drives the latch bolt through a one-sided coupling." },
  op_height: { what: "Height of the operator's grip point above the floor.", unit: "m", how: "sampled per context (ADA range 0.864–1.219 m for accessible doors)." },
  op_travel: { what: "Full actuation travel of the operator joint: a rotation for levers / knobs / thumbturns, a linear stroke for push pads, slide bolts, pins and buttons.", unit: "° (rotary) / mm (linear)", how: "catalogue travel; the unit follows the joint type in model.json. The latch starts to retract after the dead travel (backlash)." },
  op_return_spring: { what: "Return spring of the operator as the hardware catalogue lists it: preload plus a rate that grows with travel q.", unit: "N·m + N·m/rad·q (rotary) / N + N/m·q (linear)", how: "catalogue values (ANSI/BHMA A156.2 for levers and knobs, UL 305 for exit devices). The robot must overcome it to keep the latch retracted." },
  op_return_kind: { what: "What this operator does when the hand lets go: spring return, gravity return (no spring), or stays where it is put.", how: "hardware.OperatorModel.return_kind. A handwheel, dog, cremone knob or slide bolt has no return spring in reality - one would undo its own boltwork. QA gates both cases (operator_returns / operator_holds); see docs/PHYSICS.md." },
  op_return_dynamics: { what: "The return as actually built into this door's joint: preload, rate, and the damping derived from the joint's real inertia.", unit: "N·m, N·m/rad, N·m·s/rad (rotary) / N, N/m, N·s/m (linear)", how: "damping b = 2ζ√(kI) at ζ = 1 (critical), so the handle settles onto its rest stop instead of ringing; the preload is raised to 1.35× the handle's own weight moment where the catalogue spring could not lift it." },
  op_return_time: { what: "How long the handle takes to come back to rest after it is released.", unit: "s", how: "1-D integration of the release with the exact weight-moment curve (physics.operator_return_time); QA measures the real one in MuJoCo, and the viewer animates this profile when you let a slider go." },
  op_yield: { what: "Torque (rotary) or force (linear) beyond which the operator is damaged.", unit: "N·m / N", how: "catalogue value; the label tracker raises operator_overload / hardware_misuse when exceeded." },
  latch: { what: "Latch mechanism holding the door closed (tubular, mortise, deadlatch, roller, ball catch, magnetic, slide bolt, hook…).", how: "catalogue item with throw, spring and holding force." },
  throw: { what: "How far the latch bolt projects into the strike when extended.", unit: "mm", how: "catalogue value; the bolt joint range is [0, throw] with 0 = extended, + = retracted." },
  bolt_spring: { what: "Spring pushing the latch bolt out: preload plus rate per metre of retraction.", unit: "N + N/m", how: "catalogue values; a one-sided tendon lets the strike lip push the bolt in when the door slams, so it re-latches." },
  lock: { what: "Lock on this door (privacy button, keyed cylinder, deadbolt, chain, padlock, keypad, card reader, maglock, delayed egress…).", how: "catalogue item; with `engaged` and `robot-side release` it defines the initial lock state." },
  lock_engaged: { what: "Whether the lock is engaged at the start of an episode.", how: "sampled (about half of lockable doors); engaged locks fix the deadbolt / limit the operator to its backlash." },
  robot_side_release: { what: "Whether the robot can release the lock from its side (thumbturn, button, code, badge, slide, key…).", how: "from the lock's inside / outside release type and which side the robot is on. Locked doors without a release get the locked-recognize scenario." },
  backlash: { what: "Free play (\"jiggle\") of the operator when the lock is engaged: how far it moves before the lock stops it.", unit: "° (rotary) / mm (linear)", how: "catalogue backlash + condition allowance; applied as the operator joint range while locked." },
  deadbolt_throw: { what: "Projection of the deadbolt when thrown.", unit: "mm", how: "catalogue value (0 for locks without a deadbolt); the thumbturn / key drives it through a polynomial coupling." },
  code: { what: "Keypad code that releases the lock (entered on the physical keys in the model).", how: "random digits per door; DoorEnv watches the key presses and releases the lock when the sequence matches." },
  // ---- Compliance
  force_start: { what: "Force at the handle needed to set the closed door in motion.", unit: "N", how: "(hinge friction + stiction + closer preload) / lever arm to the handle." },
  force_90: { what: "Force at the handle needed to hold / push the door at 90°.", unit: "N", how: "(hinge friction + closer preload + spring rate × π/2) / lever arm." },
  operator_force: { what: "Force at the grip point needed to fully work the operator against its return spring.", unit: "N", how: "(spring preload + rate × travel) / grip offset (rotary) or the spring force directly (linear)." },
  ada_5lbf: { what: "ADA 2010 §404.2.9: interior hinged doors must open with ≤ 5 lbf (22.2 N). Blank for fire / exterior doors, which are exempt.", how: "both forces above ≤ 22.2 N." },
  ibc_fire: { what: "IBC §1010.1.3 for fire and exterior doors: ≤ 30 lbf (133 N) to set in motion and ≤ 15 lbf (66.7 N) to swing to full open.", how: "start force ≤ 133.4 N and 90° force ≤ 66.7 N; blank when not applicable." },
  hardware_5lbf: { what: "Whether the operator itself can be worked with ≤ 5 lbf (22.2 N), the ADA operable-parts limit.", how: "operator force above ≤ 22.2 N." },
  clear_width: { what: "ADA clear opening width of ≥ 32 in (0.815 m) with the door open 90°.", how: "leaf width − thickness − 30 mm stop allowance ≥ 0.815 m." },
  // ---- Damage thresholds
  dent: { what: "Contact force on the leaf face that leaves a permanent dent.", unit: "N", how: "face material table (hollow metal gauge, wood species, glass, fibreglass…), doubled for fully filled cores." },
  puncture: { what: "Contact force that punches through the leaf face.", unit: "N", how: "face material table." },
  glass_break: { what: "Contact force on a glass lite that breaks it (blank when there is no glass).", unit: "N", how: "glass type table (annealed, tempered, laminated, wired)." },
  op_yield_dmg: { what: "Torque on the operator beyond which it bends or its spindle shears.", unit: "N·m", how: "same as the operator yield above; the tracker only counts it when the operator is driven, not when it snaps back to rest." },
  latch_shear: { what: "Force across the latch bolt that shears it (forcing a latched door).", unit: "N", how: "latch catalogue yield force." },
  hinge_tearout: { what: "Force on the leaf that rips the hinges out of the frame.", unit: "N", how: "min(4 × hinge load rating × g, 20 kN)." },
  slam_velocity: { what: "Angular (or linear) speed at which hitting the stop counts as a slam.", unit: "rad/s (m/s for sliders)", how: "4 rad/s for hinged leaves, 2 m/s for sliders; the tracker raises door_slammed when the closing speed at the closed position exceeds it." },
  // ---- QA sign-off checks (qa.json)
  qa_load_full: { what: "The full-fidelity MJCF (every mechanism body) loads in MuJoCo without errors.", how: "MjModel.from_xml_path(door.xml)." },
  qa_load_simple: { what: "The simple-tier MJCF (leaf + operator + bolt, primitive collision) loads in MuJoCo.", how: "MjModel.from_xml_path(door_simple.xml)." },
  qa_load_minimal: { what: "The minimal-tier MJCF (leaf only) loads in MuJoCo.", how: "MjModel.from_xml_path(door_minimal.xml)." },
  qa_clearance: { what: "Deterministic kinematic clearance gate: every joint is swept through its full range with all geometry collidable, and no two parts may interpenetrate by more than 2 mm.", how: "doorbench/clearance.py; the number of offending pairs is listed as clearance failures. Fails when hardware clashes with the frame, casing or other hardware anywhere in the motion." },
  qa_settle: { what: "With no forces applied the door settles at its initial state instead of drifting (no initial penetration, no latch pop).", how: "settle_drift = primary joint motion after 1 s of free simulation." },
  qa_hold: { what: "A push on the latched leaf does not open it: the latch holds.", how: "constant torque / force on the primary joint for 1 s; displacement must stay below a few millimetres / degrees." },
  qa_actuate_opens: { what: "Actuating the operator retracts the latch and the same push now opens the door (or, for locked doors without a release, it must NOT open; chained doors open only to the chain slack).", how: "torque on the operator joint + push on the leaf; the reached opening is recorded as actuate_displacement." },
  qa_latch_returns: { what: "After the operator is released the latch bolt springs back out.", how: "bolt position after release must be within 1 mm of extended." },
  qa_relatch: { what: "A closing door pushed shut at speed re-latches: the strike lip pushes the bolt in and it drops into the keeper, and a second push no longer opens it.", how: "slam the leaf closed, then repeat the hold test." },
  qa_settle_simple: { what: "The simple tier also settles without drift.", how: "same settle test on door_simple.xml." },
  qa_settle_minimal: { what: "The minimal tier also settles without drift.", how: "same settle test on door_minimal.xml." },
  qa_urdf_loads: { what: "The URDF parses and its link count matches the model.", how: "XML parse of door.urdf; body count recorded as urdf_nbody." },
  qa_usd_opens: { what: "The USD stage opens with pxr and contains the expected joints.", how: "Usd.Stage.Open(door.usda); joint count recorded as usd_joints." },
  qa_closer_returns: { what: "A self-closing door released from 60° returns to closed on its own.", how: "free simulation from an open pose; final angle must be near 0." },
  qa_free_opens: { what: "A door without a latch (or a free-swinging / push-through door) opens under a plain push.", how: "constant push on the primary joint; the reached opening is recorded." },
  qa_locked_holds: { what: "A locked door without a robot-side release stays shut even when the operator is worked and the leaf is pushed.", how: "operator torque + push on the leaf; displacement must stay within the lock backlash." },
  qa_generic: { what: "Automated sign-off check recorded in qa.json.", how: "see doorbench/qa.py." },
  // ---- Isaac parity gate (qa.json.isaac_parity, docs/ISAAC_PARITY.md)
  isaac_parity: { what: "Whether this door behaves the same in Isaac Sim / PhysX as in MuJoCo (the reference physics) under one shared behavioural protocol: settle, hold or free push, operate + open, latch release, relatch, closer return, locked hold, joint limits and numerical sanity, run on both USD kinds (door.usda full fidelity, door_rl.usda canonical 8-link).", how: "Per phase both simulators must reach the same pass / fail verdict, and when they agree the metrics must be within tolerance. Grade A = every phase agrees within tolerance, B = same verdicts but a metric is outside tolerance, C = a status disagreement, X = not comparable (spawn / structure error). ok = grade A or B in every tested kind; untested = not yet run on the GPU. Independent of the MuJoCo QA sign-off." },
  isaac_parity_full: { what: "Verdict for door.usda, the full-fidelity USD (every mechanism body, mimic joints for bilateral couplings, articulation under /Articulation).", how: "grade + the agreement per phase: agree, quant (same verdict, metric off), disagree, na (not run on one side, e.g. env-released locks)." },
  isaac_parity_rl: { what: "Verdict for door_rl.usda, the canonical 8-link articulation used for multi-door Isaac Lab training (fixed joint names; locks welded engaged, auxiliary releases welded, one latch slot).", how: "same phases as the full kind; a phase that agrees in the full USD but not here is tagged RL_CANON (the welding / slot logic changes the behaviour)." },
  isaac_parity_classes: { what: "Discrepancy classes of the door: EXPORT_COUPLING (operator -> latch tendon not emulated, bolt never retracts), EXPORT_WELD (mag lock / delayed-egress weld not in the USD, door opens), EXPORT_FRAME (body / joint frames differ), PHYSICS_PARAM (spring target / preload, friction or damping mapped differently), CONTACT_GEOMETRY (bolt retracted but the leaf does not move, or a latch does not engage under PhysX contacts), RL_CANON (welded / omitted parts in the canonical export), VALIDATOR_PROTOCOL (effort or expectation applied by a runner), QUANT (same verdicts, metric outside tolerance), SOLVER_SENSITIVITY (disappears at a finer time step), LIMITS / SANITY, LOAD_ERROR.", how: "assigned by doorbench/parity/results.py from which phase disagrees and how (bolt fraction, operator travel, lock kind); each class has a likely root cause and a fix direction in docs/ISAAC_PARITY.md." },
  isaac_parity_root_cause: { what: "The most likely root cause of the primary discrepancy class, from the analysis of the first 40-door GPU probe (hypotheses H1-H7, D1-D2 in docs/ISAAC_PARITY.md).", how: "a hypothesis to test, not a proof: the per-phase details above say which metric diverged." },
  isaac_parity_run: { what: "When the gate ran, the dataset commit, and the simulator versions on both sides.", how: "written by scripts/merge_isaac_results.py from results/parity/summary.json." },
  // ---- Evaluation (benchmark scenarios)
  scenario: { what: "One of the door's benchmark scenarios: the initial state, the robot's start zone, what it must touch, the plane it must cross, the goal, an optional simulated person, the reward table, time budget and expected transit time.", how: "assigned per door by a seeded rule (docs/BENCHMARK.md); any scenario type can also be requested from DoorEnv." },
  time_budget: { what: "Episode length: the episode ends when the simulated time reaches it.", unit: "s", how: "5 · ceil((3 × expected transit + 10 s) / 5)." },
  expected_transit: { what: "How long a competent humanoid should need for this scenario on this door.", unit: "s", how: "walk to the handle at 0.7 m/s + operate the hardware (1–3 s, + dogs, + unlock) + open the door (from the leaf inertia, friction, closer preload / roller friction and a 40 N push or 120 N lift) + walk through to the goal + scenario extras (hold, close behind, knock & wait, wait for the person). The terms are listed below." },
  transit_approach: { what: "Walking time from the start-zone centre to 0.6 m short of the handle (arm reach) at 0.7 m/s.", unit: "s" },
  transit_operate: { what: "Time to work the hardware: 1 s levers / bars / pulls, 1.5 s knobs, thumb latches and similar, 3 s wheels, +1.5 s per dog, +2 s to unlock (or 1 s per keypad digit + 1 s).", unit: "s" },
  transit_open: { what: "Time to open the leaf to the clearance threshold under the nominal push / lift force, from the door's inertia, friction, closer preload and damping.", unit: "s", how: "√(2·θ·I/τ_net) + θ·b/τ_net for hinged leaves (clamped 0.6–12 s); analogous mass / roller-friction forms for sliders and vertical doors." },
  transit_pass: { what: "Walking time from the pass plane to the goal zone (+0.6 m) at 0.7 m/s.", unit: "s" },
  transit_extra: { what: "Scenario-specific additions: +1 s if the closer beats the walk (hold / re-open), +2 s + W/0.7 to close behind, +4 s knock & wait or probing a locked door, the person's crossing time in human scenarios.", unit: "s" },
  start_zone: { what: "Where the robot starts: a disc on the floor plus a heading range. Episodes sample a pose from it with a seeded rule so runs are reproducible.", unit: "m, rad", how: "centre = approach point moved back to the start distance (outside the swing arc of a pull door), radius 0.3 m, yaw facing the doorway ± 0.35 rad; r = R·√u₁, φ = 2π·u₂." },
  approach: { what: "Nominal approach point in front of the doorway (site approach_point).", unit: "m" },
  handle_targets: { what: "Grip / push sites on the active leaf's hardware the robot should reach (names of sites in model.json).", how: "sites with role grip or push under the leaf that carries the operator." },
  pass_plane: { what: "The plane of the opening the robot base must cross, inside the opening's width and height, to count as traversed.", unit: "m", how: "site door_plane_center; normal along the traverse direction." },
  goal_zone: { what: "Disc on the far side the robot should end in after traversing.", unit: "m", how: "site goal_point, radius 0.5 m." },
  suite: { what: "Which benchmark suite the scenario belongs to. core = needs only the door and the robot; it is the default for every benchmark run and every published table. human = advanced, opt-in human-interaction suite (a simulated person, or etiquette that presumes one); reported in its own table and never mixed into the core number.", how: "spec.json.benchmark.suites lists each door's scenarios per suite; run the human suite with `doorbench benchmark run --suite human` (or --suite all)." },
  human: { what: "A simulated person (kinematic capsule) walking a timed path. In hold-open scenarios they come up behind the robot and wait in front of a closed door; in wait-for-human scenarios they come through from the far side first and the environment works the door for them.", unit: "m, s", how: "path = (t, x, y) waypoints at 1.1 m/s; markers every second in the 3D view." },
  success: { what: "All listed conditions must hold at the end of the episode (\"!\" = must not happen).", how: "evaluated from the reward events and episode labels by DoorEnv.success." },
  rewards: { what: "Reward given once when each event first happens (the time penalty accrues every second).", how: "summed into the episode return; see docs/BENCHMARK.md for the event definitions." },
};

export const REWARD_LABELS: Record<string, string> = {
  touch_handle: "touch handle", unlatch: "unlatch", unlock: "unlock", opened: "open", traversed: "traverse", closed_behind: "close behind",
  latched_behind: "latch behind", closed: "close", latched: "latch", held_for_human: "hold for human", yielded_to_human: "yield to human",
  recognized_locked: "recognise locked", knocked: "knock", waited: "wait ≥ 3 s", collision_with_human: "collision with human", damage: "damage",
  slam: "slam", hardware_misuse: "hardware misuse", time_penalty_per_s: "time penalty / s",
};

export function glossary(key: string | undefined): GlossaryEntry | undefined {
  if (!key) return undefined;
  return GLOSSARY[key];
}
