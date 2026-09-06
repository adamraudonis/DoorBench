# Surface closer mechanical rebuild

The source now builds **233 physical closer instances across196 doors**:
213 regular two-arm instances on183 doors, and20 electromagnetic track
instances on13 doors. The37 paired-door cases previously used body-only
substitutes. Every MuJoCo tier now retains the native force-transmitting
joints, arm mass, shaft supports, loop constraint and pinion spring. Published
assets and recordings were not regenerated; these working fixtures live only
under `out/mechanical-foundations/`.

## Connected mounting and force transmission

The old pinion shafts missed their housings by2–75mm. Ordinary shoes were
40.5–133mm from actual frame material because mounting calculations omitted
authored wall offsets. Corrected housings have physical spacers where needed;
the rotating shafts enter their cases; frame backplates and brackets meet
structural material. Four shoe solids surround a14mm bore for the10mm pivot,
and the forearm neck reaches that pivot. Rising-hinge variants have a passive
vertical shoe and connected backing. That vertical guide still idealizes its
internal bearing and axial retention; its retaining end parts are not modeled.

[LCN's4040XP pull-side instructions, page1](https://allegion.ca/content/dam/allegion-us-2/web-documents-2/InstallInstructions/LCN_4040XP_Series_Pull_Side_Mount_Installation_Instructions_107160.pdf)
were visually inspected. The generic250mm pinion/300mm shoe offsets and
280/260mm arms follow its mounting ordering. These are authored metric
components, not reconstructed LCN/Norton product CAD.

The native torsional spring is on the rotating **pinion**, and door torque
travels through its arms by virtual work. Direct leaf closer stiffness and
hydraulic terms are removed. Hinge air drag, bearing friction, gravity and
independent locks remain. `meta.closer_pinion_calibration` records the achieved
nonlinear torque and the original linear sizing target separately. Relative
mismatch from that old target reaches about101% in the original159-door
subset; equivalence to that curve or EN1154 product compliance is not claimed.

Sweep, latch, opening, backcheck and delayed hydraulic resistance act on that
same pinion. Native implicit damping stores the largest valve coefficient;
the passive callback subtracts its unused portion, leaving exactly the
authored dissipative force. The callback never writes native positions,
velocities, limits or model parameters. Loop configuration is resolved only
for initialization and geometric inspection.

## Delayed action

All five declared delayed-action doors now use an angle-zone hydraulic valve,
not a timer or frozen pose. This follows the continuous90°→70° delay region
in [Allegion's closer adjustment guidance](https://kc.allegion.com/kb/article/how-do-you-adjust-a-door-closer/).
Their high-resistance closed loops require a0.25ms native step. The unchanged
law was checked at0.25/0.125/0.0625ms on DB0171: reaching12° took
16.25175/16.251625/16.2514375s; reaching1° took
17.17625/17.18425/17.1885s. Maximum sampled connect errors were
0.11755/0.04749/0.01889mm, with no native warnings. The earlier2/1/0.5ms
oscillation/stall trials are retained as rejected numerical configurations.
All five source IDs have focused native delayed-versus-normal-valve controls.

## Electromagnetic single-arm tracks

[LCN's4040SEC installation instructions, pages2–3](https://www.lcnclosers.com/content/dam/allegion-us-2/web-documents-2/InstallInstructions/LCN_4040SEC_Series_IS_109730.pdf)
were visually inspected. The new13-door/20-instance layout uses an actual
single arm below a retained frame track, vertical arm-tip pin, sliding
carriage, rolling follower and solenoid detent. The fixed track has physical
backing, lips and end caps. An axle seated in the outer carriage wall carries
the roller without crossing the independent vertical pin.

The spring-loaded cam is a native sliding body with two actual contact flanks.
Coil attraction acts only on its armature and decreases with seat gap. A
separate physical momentary button interrupts current at the authored travel
threshold. Cam/roller contact carries the door's holding load. There is no
leaf detent torque, weld, threshold pose hold or runtime coordinate reset.
Power loss withdraws the cam through its return spring; finite manual effort
can overcome the roller detent. Electrical behavior and the magnetic field
are idealized, and dimensions/forces are generic rather than OEM calibration.

The validated hold point is90°. Recapture after manually overrunning that
point is **not guaranteed**: the retained105° DB0773 experiment fails recapture.
The metadata explicitly reports this limitation. It is not a universal
hold-open or fire/egress certification.

## Evidence and limits

Current dedicated evidence is under `closers-track/`:

- `native-final/native-qa.json`:39/39 source-door/tier combinations pass mounting
  and native holder QA.120 power-loss/test-button actuator probes run even
  when a separate credential lock prevents opening.48 holding-load probes
  cover unlocked instances; three locked doors retain their credential locks.
- `clearance-final/receipt.json`: all13 integrated penetration sweeps and native
  holder gates pass.12/13 running-clearance checks pass. DB0396 retains a
  separate1.5mm slab-to-mullion-lip gap versus the3mm required clearance.
- `tests-combined-final.log`:20 focused closer/holder/environment tests pass.
  They include detached mounting, removed spring, obstructed cam, authored
  switch thresholds, real finite-force entry/manual release, callback isolation,
  and caller-state preservation. The tests regenerate in isolated directories.

Track geometric inspection uses a private native submodel: all surrounding
native colliders are fixed at the requested inspection pose, while only the
original plunger mass/inertia/joint/spring/coil responds. Reconstructed native
collider vertices and frames must agree within2µm. The original input states
and selected driven joint remain unchanged. More than1mm cam penetration,
0.1mm travel-limit excess,2mm/s residual motion or a native warning rejects the
pose. The full geometric gate then checks all source geometry, including
visual-only parts. This is a prescribed-boundary inspection calculation, not
proof that an energized holder freely traverses its range.

The original159-door force study remains frozen at
`closers/pinion-stage-receipt.json` and `closers/native-final/native-cycles.json`:
666 unlocked closing trials,632 reaching1°,34 arrests on six doors, no native
warnings and maximum sampled loop error0.958mm. That older generic signoff
predates declared-function and strict relatching gates. Four top-rod contact
arrests were subsequently repaired through real bolt bevels and steel strike
plates. Subsequent flush-strike stock correction also removed DB0280's arrest,
which had been incorrectly attributed to rust. DB0508 still fails to self-latch;
its otherwise identical normal-condition control closes. No spring boost masks
that remaining loss. These later findings supersede the earlier two-door
condition attribution, without rewriting the historical receipt.

The complete196-door generic rerun is recorded separately in
`closers-final-fullqa/full-qa.json`:196/196 mount checks and172/196 complete
generic QA pass;24 doors retain explicit failures. The exact source-module,
fixture and evidence hashes are in `closers-track/stage-receipt.json`. Shared
unrelated hardware remained in development during this sweep. Mount correctness does not imply whole-door
signoff: independent keeper/condition, paired-mullion and in-progress security
hardware failures remain visible there. Native closed-loop load transfer does
not establish structural strength, hydraulic fluid dynamics, causal humanoid
success, or URDF/Isaac parity. Those exporters require their own capability
checks for loop constraints and passive electrical/hydraulic fields.
