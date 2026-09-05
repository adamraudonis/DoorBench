# Engineering log

What was done, what went wrong, and what we learned, in order. Written for two audiences: people who want to
know how a 1000-door dataset was actually built and checked, and agents (human or AI) picking the work up.
Facts and numbers only; the task board is `TASKS.md`, the handoff briefs are `handoffs/`.

## 2026-09-03 — v0.1: generator, physics, exporters, viewer

* Balanced seeded sampler -> 1000 specs over 30 families; geometry as an IR (bodies / joints / geoms / equalities /
  tendons) exported to MJCF (full / simple / minimal), URDF and USD; physics derived from standards
  (EN 1154 closers, ANSI/BHMA hardware, EN 12519 terms) rather than guessed; MuJoCo QA per door; React/three.js
  viewer with catalogue thumbnails and a 3D door page; GitHub Pages site.
* Lesson: a first release without a hard geometric gate shipped doors whose parts passed through each other and
  a bolt with no barrel or keeper. Rendering thumbnails is not a check.

## 2026-09-04 — gates, realism, benchmark, cloud

**Clearance gate (G2).** Deterministic sweep of every joint of every door with *all* geometry collidable (visual
parts included), parent-child filtering disabled, > 2 mm interpenetration fails (12 mm for hinges), an allow-list
only for parts inside their own housing. 1000/1000 after two days of geometry fixes. Lessons that generalised:
MuJoCo collides meshes as convex hulls, so channels and housings must be several primitives; face-mounted
hardware near the hinge edge hits the jamb at large angles, so opening angles are capped per family from real
practice (casing 135°, closers 100°, gates 110°, ...); double-acting leaves need centre pivots and a t/2 + 6 mm
gap; surface bolts need keeper lips; a bolt parallel to a slider's travel cannot hold it.

**Mass gate.** Density-based leaf masses drifted from the spec; masses are now reconciled in `build.py` and
checked within 20 %.

**Hardware realism review (G3).** 112 operator / latch / lock models rendered close-up and reviewed: 6 broken,
25 unrealistic, 27 cosmetic; 48 fixed with shared builders (barrel bolts, keeper loops, hasps, padlocks, keypads,
strike plates). `docs/HARDWARE_REVIEW.md`. Lesson: a mirrored mesh (`q_face`) turned asymmetric handlesets
upside-down on right-hinged doors — orientation checks belong in a gate, not in a review.

**Benchmark (B1-B3).** Scenarios per door in `spec.json["benchmark"]` with seeded start zones, handle targets,
pass plane, goal, reward table, time budget and an expected transit time derived from the door's physics.
Human interaction segregated into an opt-in `human` suite; the `core` suite (no person) is the default
everywhere and the headline number. Viewer overlay (default off) and info icons on every physics row.

**Runner and baselines (R1-R3).** `doorbench benchmark run --policy ...` over 1000 doors x scenarios x 3 seeds
with a validated results schema, `results/` leaderboard, `docs/SUBMITTING.md`. Core suite: scripted hand
849/1000, Unitree G1 locomotion-only 150/1000, random 42/1000; human suite (scripted hand) 67/79.

**Closer arms (G5/G6).** The owner spotted the closer arm on db0012 floating in mid-air with the door open.
Root cause: MuJoCo closes the arm loop with a `connect` equality, but the viewer only animated door joints, so
the arms rode rigidly with the leaf. Fix shipped the same day: a kinematic loop solver in the viewer
(`viewer/src/kinematics.ts`; two-bar analytic IK, telescoping, numeric fallback; 0.00 mm at the shoe over the
sweep). Still open (handed off): the closing torque is applied at the door hinge instead of through the pinion
and linkage, so the mechanism is not yet the real one; 5 rising-hinge cold-storage doors cannot close a planar
loop at all.

**Floating parts (G7/G8).** A floating door-stop cylinder (db0024) and a rail too short for its hangers (db0079)
were both obvious to a human and invisible to every gate. Two answers, both in progress: a deterministic
attachment gate (every body within 3 mm of its parent, static geoms touching frame/wall/floor, loop partners
within 1 mm through the sweep, mechanisms that actually move) and a vision review that photographs every door and
asks a vision LLM the "would a person call this wrong?" question.

**Isaac Lab (I1-I4).** USD exporter rewritten (one articulation tree per door, PhysX schemas by name, Coulomb
friction as joint friction efforts, mimic joints) plus a canonical 8-link `door_rl.usda` so hundreds of
different doors can share one PhysX articulation view. Static validation 1000/1000. Live runs on a rented GPU:

* RunPod Secure Cloud L40S, $1.09/h. Isaac Sim needs RT cores (no A100/H100).
* Pip in a conda env failed (`CXXABI_1.3.15`: conda's ICU vs Ubuntu 22.04 libstdc++) -> uv-managed Python 3.11
  venv. Isaac Lab `main` needs Python 3.12 / Isaac Sim 6 -> tag `v2.3.2`. `isaaclab.sh --install` exited 0 but
  skipped the core package -> explicit install; its `flatdict==4.0.1` needs `setuptools<81` and no build isolation.
  `import isaaclab` outside a Kit app fails on `pxr` -> check installs with `find_spec`.
* pip downloaded the 10 GB of wheels at 0.4 MB/s on one host; uv did it concurrently at 24 MB/s.
* A pod stopped for 3 hours could not be restarted ("not enough free GPUs on the host"): stop/start is a pause,
  not a parking spot; terminate and re-create instead (the bootstrap is fully scripted, ~20 min).
* The whole path (`scripts/runpod_pod.py create/wait/bootstrap`, `scripts/pod_bootstrap.sh`) was then executed
  end to end on a brand-new pod. `docs/RUNPOD.md` has the copy-paste steps and the troubleshooting table.
* The headless validation script "hung" for tens of minutes per batch with the GPU idle. `py-spy` cannot ptrace in
  the pod, so `faulthandler.dump_traceback_later` gave the stack: the batch had finished; the process was inside
  Isaac Lab's timeline-stop callback, which loops `render()` until the timeline plays again unless
  `sim._disable_app_control_on_stop_handle` is set. One line per script fixed it. Lesson: when a process is
  "slow" with 0 % GPU and 100 % CPU, get a stack before theorising.

**First real GPU result, and a change of priority (2026-09-05).** With the hang fixed, 40 doors x 2 USD kinds
validated in 69 s: 80/80 load and match model.json structurally, but 20/40 doors do not open in PhysX under the
same push that opens them in MuJoCo (both kinds agree), and 6 drift during the settle phase. The training stage
then crashed on an Isaac Lab v2.3.2 API rename (`dump_pickle`). The owner's call: the GPU's job is to find door
defects systematically; training is only a way to validate the data. So the next artefact is an *Isaac parity
gate*: one behavioural protocol (settle, hold, operate + open, release, closer return, limits, sanity) run on
every door in MuJoCo and in PhysX, compared per door, discrepancies classified into bug classes and fixed at the
root, results published per door. Built by a scout -> build -> verify workflow rather than hand-written, because
the mapping between two physics engines has too many failure modes for one pass.

**Parallel takeover by a second agent (2026-09-04/05).** The owner handed the `handoffs/` briefs to another AI
agent (OpenAI Codex) working in the main checkout on `codex/*` branches. It fixed the sliding tracks (168 doors;
new gate `sliding_track_support`), the five rising-hinge closer loops (a passive vertically sliding shoe; gate
`linkage_feasibility`), wrote a physics audit and a diversity audit (`docs/review/`), an inspection-atlas renderer,
and a Blender photoreal catalogue. Two integration lessons: (1) its commits carried new geometry and QA checks but
no regenerated assets, so master had new code on stale assets and 8 QA tests failed until the dataset was
regenerated; (2) two agents in one checkout means commits land on whichever branch is checked out; this session
now works only from a dedicated `master` worktree.

**Parity gate built (2026-09-05).** A scout -> build -> verify workflow (8 agents) produced the gate:
`doorbench/parity/protocol.py` (the QA protocol as data: settle, hold, operate + open, release, relatch, closer,
locked, limits, sanity), a MuJoCo reference runner (1000 doors in 15 s, reproduces every qa.json metric within
1e-3), an Isaac Lab runner, a comparator with discrepancy classes, a report generator and a merge into qa.json /
the site badge. The scouts also explained the first probe: of the 20 "doors that do not open in PhysX", 6 were the
operator->latch coupling that the USD only records as metadata (a real export gap), 9 were correctly held (locks,
far-side panic bars), 4 were the validator's fixed 60 N push being below the door's load, and 1 is unexplained;
all 6 settle drifts were the validator zeroing the spring targets every step; and one real export bug hid behind
a PASS (mag-lock welds exported as JSON only, so a locked door swung open). Pushing every door in MuJoCo, which
qa.py never did for free-swing families, also found two dataset defects: all 12 accordion doors are kinematically
locked by a coupling sign vs joint-range mismatch, and 10 of 15 revolving doors jam a wing stile against the wall
header. Both are being fixed with new gates.

**Parity round 1 and two more dataset defects fixed (2026-09-05).** All 1000 doors x 2 USD kinds ran in PhysX in
70 min (35-90 s per 20-door batch). Same verdict as MuJoCo on 881/1000 (full USD) and 933/1000 (canonical rl USD);
145 doors flagged into classes with a likely root cause each: welds and multi-bolt couplings exported only as
metadata (mag-locks swing open, vault dogs do not retract), friction/preload mapping on bifolds, pet doors and
turnstiles, rising-hinge drift on cold-storage doors, canonical-articulation limits (dogs, twin thumbturns), and
pairs of swing doors that hold in MuJoCo but open in PhysX. Only 30 doors match strictly (grade A); most differ in
metrics such as opening time, which needs a per-metric look before the tolerance is called wrong. Meanwhile the
accordion and revolving fixes landed: a real folding mechanism (face-alternating piano hinges, top track,
coupling-consistent ranges, 85 deg stack stop) and header clearance for revolving wings, with three new gates
(free-swing push for every free-swing family, a coupling-vs-range check in the clearance sweep, a jam check on
contact force). Regenerated: 1000/1000 signed off under the new gates.

**Parity round 2 and what the gate actually found (2026-09-05).** Re-run on the regenerated dataset: full USD
904/1000 same verdicts as MuJoCo (was 881), canonical rl 954/1000 (was 933), 110 doors flagged, per-door badges
published in `qa.json.isaac_parity` and the viewer. A triage of the classes then turned most of the remaining
disagreement into concrete bugs, several of them one-liners with dataset-wide reach:

* `physxRigidBody:maxAngularVelocity` was authored as 100 assuming rad/s; PhysX reads deg/s, so every door was
  capped at 1.75 rad/s. That single value is why only ~30 of 1000 doors reached strict metric agreement.
* `PhysxJointAxisAPI` friction efforts were written with D6 tokens (`rotX`/`transX`) that the USD parser ignores on
  single-axis joints, so the authored Coulomb friction was silently dropped on all 1000 doors and the deprecated
  load-dependent coefficient acted instead. Fixed to `angular`/`linear`.
* Env-released welds (mag-locks, delayed egress) and every translational mimic coupling exist only as metadata:
  PhysX articulation mimic joints are rotational-only, so thumbturn-to-deadbolt and dog-to-bolt couplings were
  dropped. Self-collision is disabled on the articulation root, so latches that hold one moving leaf against
  another pass through in PhysX.
* Zero-clearance geometry is a defect class of its own: parts authored exactly touching (0.000 m) pass a
  penetration-based clearance gate but jam or explode in PhysX, and are wrong anyway because real doors have
  running clearance. Turnstile rotor columns and folding-panel tops were the worst cases.
* Several "disagreements" were protocol artefacts: a rebound metric read after the leaf hits its limit (MuJoCo's
  soft limit bounces, PhysX's does not), a 30 Hz sampling artefact in the latch arrival speed, the latched-door
  tolerance applied to doors expected to swing free, and a 60 N*m QA push applied to a 0.14 kg pet flap.

Lesson: a second engine is a measuring instrument. Most of what it flags is the instrument or the adapter, so
triage before fixing; but the residue is real, and nothing else had found it.

**Round-3 fixes and a new defect class (2026-09-05).** Four agents implemented the triage. In the USD export:
env-release welds became real breakable `FixedJoint`s (a mag-locked door was swinging 1.1-1.8 rad open in PhysX
while MuJoCo held it at 1e-6), self-collision was switched on with MuJoCo's exact filtered-pair set mirrored into
`PhysxFilteredPairsAPI` (latches that hold one leaf against another were passing through), and the 181
translational couplings PhysX cannot represent as mimic joints (thumbturn-to-deadbolt, dogs-to-bolts, cremone,
helical risers) became explicit metadata plus a bilateral emulation that applies the reaction force on the driver
instead of writing the driven joint kinematically. The canonical RL export now records its weld decisions as
ground truth instead of letting the protocol guess. In the protocol: a staleness gate (a PhysX record whose
inputs hash predates the reference is graded X and published as untested, never as a verdict), a rebound waiver,
a sampling-invariant latch arrival speed, and a QA push scaled to the leaf's own weight moment, which took a
0.14 kg pet flap off a 60 N*m push that made PhysX explode within six steps.

The most valuable output was a defect class nobody had named: **zero running clearance**. Parts authored exactly
touching pass a penetration-based gate, because MuJoCo at margin 0 sees no force, but PhysX resolves contacts
inside its contact offset, and a real door has running clearance anyway. A sweep over every moving-vs-static pair
found **139 of 1000 doors, 229 pairs**: turnstile rotor columns flush on the cage roof and the floor, folding-panel
heels scraping the pivot jamb through the fold, bypass leaves flush on the jamb, hinged leaves raked by the frame's
reveal past 90 degrees, roll-up bottom bars flush on the floor, hatch lids raking the curb. All fixed in the
generator, with `running_clearance` now part of sign-off and seals, gaskets and bumpers allow-listed by semantics.

The verifier that merged the three branches found four more bugs in them: enabling self-collision made a thumbturn
collide with its own deadbolt housing, the filtered pairs were authored one-sided, the mimic offset mixed MJCF and
USD joint frames, and one branch had committed regenerated assets against instruction.

**Working with agents.** Seven parallel Fable agents in git worktrees is productive until the account's usage
limit hits, which kills all of them at once and twice cost partial work. Rules that now hold: commit early and
often on the agent branch; the main session owns `TASKS.md`, `README.md`, credentials and merges; agents write
new modules rather than editing shared files where possible; when a batch dies, snapshot every worktree as a WIP
commit, push the branches, and write a self-contained brief per task (`handoffs/`) so a cheaper agent can continue.

**Zero-gap touches: the defect class the clearance gate cannot see (2026-09-05).** The MuJoCo parity reference pushes
every door, including the free-swing families qa.py never pushed, and found 10/15 revolving doors that did not turn.
Root cause: the wing stiles ended at exactly the underside of the wall header (and the canopy). A coplanar box-box
touch is not an interpenetration, so the clearance gate passed it, but its contact normal is orthogonal to the
rotor's only DOF - a degenerate constraint that MuJoCo answers with 8-17 kN of normal force, and friction on that at
1 m radius stalls the rotor. Fix: model the real enclosure (rotor 15 mm under the canopy ceiling, header on top of
the canopy, top bearing / floor pivot). A push survey of all 147 free-swing doors then found the same class in four
more families - saloon leaves standing on the floor, bifold / accordion panel tops level with the head, strips
hinged beside their own plane swinging 1 mm into the hanger rail, pet flaps whose pins sat level with the frame
rail (20+ kN) - all fixed at the geometry. New deterministic gate in qa.py for those families (`free_opens` +
`no_jam`: the push must move the primary joint, and `mj_contactForce` between any moving body and static geometry
must stay under 200 N; a free door is carried by its joint, so anything static pressing on it is a jam). Lesson: a
geometric gate needs a dynamic twin; "gap 0.000" is a defect, not a pass.

**Multi-latch doors: N latches, one of them modelled (2026-09-05).** Owner report on db0168_ship_watertight: "it
only does 1 of 6 hinges". Three defects behind it. (1) `meta` named a single `operator_joint` - the first dog - so the
QA drive, the benchmark and the viewer's "Open door" worked one dog and the leaf swung with five still dogged. `meta`
now carries `operator_joints` (every operator the robot has to work) and `operator_coupling` ("individual" for the
watertight dog levers and the blast/vault lever bolts, "coupled" for a handwheel that drives boltwork). (2) The dogs
did not actually hold. A hinge-stile wedge sits 34 mm from the leaf hinge pin and had 3 mm of slop to its cleat, so it
needed 5 deg of leaf rotation to bite; the wedge/cleat interference peaks at 8 mm and is gone past 55 deg. Measured
with every other dog released: 4 of the 6 individually dogged doors swung 103-133 deg with a dog engaged, the other 2
stalled at 5.5 deg - the outcome was decided by leaf mass against contact softness, not by the mechanism. The cleat is
now a slot (inner and outer jaw, 0.5 mm fit), which is what a dogged watertight door is; worst single-dog hold is now
0.95 deg. Lock-vs-lock pairs may touch (`clearance.required_gap`), so the tight fit costs no running clearance.
(3) Quick-acting doors carried 4 hard-coded dogs and no visible linkage; they now carry the 4/6/8 dogs the spec samples
and the handwheel drives them through a gearbox, a torque tube and a push rod along each stile - real bodies coupled by
joint equalities, with a crank on every dog. The same "drawn but latching nothing" defect was in two more mechanisms:
the cremone/espagnolette down rod and the surface vertical rod device's bottom rod were cylinders with no bolt behind
them. Both now shoot a second bolt into a floor strike, retracting 30 mm up into the rod housing so they clear a 25 mm
floor dome stop (leaf undercut grows to strike top + 16 mm). Two new gates: `all_latches_release` (releasing all but
ONE latch must leave the leaf shut under the QA push, for each latch in turn, and releasing all of them must open it -
13 doors) and `rod_points_hold` (each of a two-point rod mechanism's bolts holds on its own - 34 doors). Benchmark:
`LabelTracker` treated a dogged door as unlatched on step 0 because no joint carried role "latch" (the dogs carry
"lock"); `DoorEnv.operator_joints` missed them for the same reason. Lesson: "several parts exist in the model" is not
"several parts work" - a latch that is not the one the metadata names is never driven and never tested.

## Checks that exist now (a door is "signed off" only if all pass)

clearance (no interpenetration through every sweep) · mass (within 20 % of spec) · physics QA (opens, holds,
latches, closer returns, hardware misuse limits) · jam gate for free-swing / rotary doors (`free_opens`: the push
moves the primary joint; `no_jam`: < 20 N of static contact on any moving part while it does - every free-swing door reads 0 N, so 200 N was far too loose to catch a leaf scraping without stalling) · static USD
validation · benchmark block present · `usd_rl_opens`. In progress: attachment (nothing floats, mechanisms move),
closer_linkage / closer_closes, operator_returns, rollers_on_track, keypad_code_works, vision review.

Added 2026-09-05: `all_latches_release` (multi-operator doors: every latch holds the leaf on its own, all of them
released opens it) and `rod_points_hold` (two-point rod mechanisms: the head bolt and the floor bolt each hold).
