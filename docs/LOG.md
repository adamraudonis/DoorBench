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

## Checks that exist now (a door is "signed off" only if all pass)

clearance (no interpenetration through every sweep) · mass (within 20 % of spec) · physics QA (opens, holds,
latches, closer returns, hardware misuse limits) · jam gate for free-swing / rotary doors (`free_opens`: the push
moves the primary joint; `no_jam`: < 200 N of static contact on any moving part while it does) · static USD
validation · benchmark block present · `usd_rl_opens`. In progress: attachment (nothing floats, mechanisms move),
closer_linkage / closer_closes, operator_returns, rollers_on_track, keypad_code_works, vision review.
