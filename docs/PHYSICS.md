# Physics model

Every number in `spec.json → physics` is derived from first principles plus published catalogue data and carries
its formula and source.  This page summarises the models.

## Mass and inertia

`m_leaf = ρ_area(t) · (W·H − A_glass) + ρ_glass · t_glass · A_glass + Σ hardware`

`ρ_area(t)` comes from a **slab construction** (`materials.SLABS`): two skins + core (+ stile/rail frame) or a
monolithic material.  Constructions are calibrated against manufacturer door-weight tables
(Knape & Vogt, VT Industries, Steel Door Institute): e.g. a 3'0"×6'8" hollow-core interior door ≈ 12–14 kg,
a 3'0"×7'0" 18-gauge hollow-metal door ≈ 47 kg, a 12 mm frameless tempered glass door ≈ 60 kg, a vault door ≈ 1 t.
Hardware masses (lever sets 1–2 kg, exit devices 5–8 kg, closers 3–4 kg, hinges 0.2–0.6 kg each) are added.
Inertia tensors are computed analytically per geom (boxes, cylinders, capsules, spheres; trimesh for meshes) and
combined with the parallel-axis theorem; MJCF/URDF/USD carry explicit inertials.

## Hinge friction

Coulomb torque about the hinge line from a bearing-load model:

`τ_f = μ · (m·g·r_thrust + 2·F_h·r_pin) · k_condition + τ_seal`,  `F_h = m·g·(W/2) / L_hinge_span`

with `μ` per bearing type (ball-bearing 0.04 … rusty pin 0.55), pin/thrust radii per hinge model, and a condition
multiplier (new 1.0 … rusty 6.0). Doors flagged *swollen* or *sagging* add a stiction term.  Rolling friction for
sliding doors: `F = μ_roll · m · g · k_condition` (sealed bearings 0.012 … dirty track 0.12 … wood-on-wood 0.30).
Aerodynamic damping is linearised at 1 rad/s.

## Closers (EN 1154)

Power size is chosen from leaf mass and width (EN 1154 Table 1) unless the closer model fixes it.  The spring is
`τ(θ) = τ₀ + k·θ` with `τ₀ = 1.15 × closing moment (0–4°)` and `τ(90°) = 0.85 × max opening moment`.  Hydraulic
damping is asymmetric (closing sweep ≈ 40–120 N·m·s/rad, opening 4–15) with backcheck beyond ~70°; MJCF/USD carry
the symmetric opening value natively and the environment applies the asymmetric law in a passive-force callback.
Spring hinges, floor springs, pneumatic screen closers, gate springs, gas struts and torsion-spring counterbalances
have their own models.

Cam-lift cold-room doors with an overhead arm closer use a passive vertically sliding shoe in a frame-mounted
guide, with 2 mm travel allowance beyond the specified cam rise. The shoe follows the forearm through the point
connection; it adds no spring and preserves the hinge's gravitational return work. The full-tier linkage gate
checks endpoint separation below 1 mm while prescribing the rise coupling. This corrects the previously incompatible
fixed planar shoe; the closer torque itself remains the joint-level approximation described above. No manufacturer's
installation or full-mechanism torque calibration is claimed for this generic slotted-shoe assembly.

## Latches and locks

* Spring latches (BHMA A156.2): 12.7 mm (Grade 3) or 19 mm (Grade 1 deadlatch) throw, 3.5–8 N spring preload;
  the bolt is a capsule that rides over a beveled strike lip and drops into a pocket in the jamb + stud.
* Operators: levers (ADA-compliant return springs 0.3–0.5 N·m preload, ≤ 22 N at the grip), knobs (45–60° travel),
  paddles, exit devices (UL 305: 16 mm pad travel, 18–45 N preload, ≤ 67 N to unlatch), thumb latches, wheels,
  dogs, slide bolts, hooks, lift pins, keypads (12 physical keys, 1.5 mm travel).
* Deadbolts (A156.5, 25.4 mm throw) driven by thumbturns; keyed-only sides are fixed.
* Locked operators keep a small backlash range so the robot feels the "jiggle" of a locked handle.
* Maglocks are welds with a holding-force breakaway (600/1200 lbf), electric strikes/bolts and interlocks are
  released by the environment (REX button, badge, call button, timers).
* Armature (reflected inertia of internal mechanisms) is added to mechanism joints so the coupled constraints are
  well-conditioned in MuJoCo.

## Compliance checks

For each hinged door the simulated opening force at the handle is compared with ADA 2010 §404.2.9 (5 lbf interior),
IBC §1010.1.3 (30 lbf to set in motion / 15 lbf to full open for fire & exterior doors) and §1010.1.10 (panic
hardware ≤ 15 lbf), plus ADA handle height (34–48 in) and clear width (32 in).

## Damage thresholds

Per material (dent / puncture force), glazing (breakage), operator (yield torque), latch (shear), hinge (tear-out) and
slam velocity; the benchmark labeller compares contact and constraint forces against them each step.
