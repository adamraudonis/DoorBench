# Preserve thumb opposition while controlling pressure

The failed whole-thumb hierarchy allowed THJ5 to drift40.091mrad from its
unchanged target at25.010s, before the index lost touch at25.076s. Its pressure
projection removed THJ5's normal-direction posture effort. A separate contact
mode repair therefore cannot establish that the thumb tracking issue is fixed.

`RobotThumbFlexionForce` is a detached own-robot virtual-work mapper. It keeps
the original THJ1/THJ2 motor transmission and leaves THJ3/4/5 out of the added
pressure effort. Those three joints can retain ordinary posture feedback.
Other fingers retain their original coupled J1/J2 projection. This module does
not step a plant, alter a joint or change any motor cap.

The authored model has THJ2 axis−Y in its middle body and THJ1 axis+X in its
distal body. The intervening fixed rotation maps those axes into the same
direction. Independent FK at the actual19,21,23 and25.008s attained states gave
axis dot products within5e−16 of1. THJ5 is the separate base-axis rotation.
These are model-derived axis facts; this screen does not assume every thumb
joint is interchangeable merely because all five are independently actuated.

At19s a unit pad-normal request produces THJ2/THJ1 moments of0.0514342 and
0.0200000Nm. The full thumb map additionally requested0.0674860Nm at THJ5 and
−0.0043232Nm at THJ4. The restricted map retains39.97% of the full squared
normal-moment norm. This number describes this effort allocation, **not** a
measured pressure efficiency or a guarantee of force controllability. A finite
difference test independently verifies virtual work through the original two
motors; all other digits' computed efforts remain identical.

## Retained static screen

`out/continuous/sensor-thumb-flexion-plan-001/screen.json` contains648 whole-scene
FK/collision candidates, with the actual root, door and remaining joints fixed.
For each of the four attained states it sweeps a9×9 grid of THJ1/THJ2 offsets
within±35mrad. It separately evaluates actual opposition posture and a hypothetical
instant restoration of nominal THJ3/4/5 targets.

All324 candidates retaining the actual opposition passed the original static
joint, hand-surface and3mm penetration checks. Restoring nominal opposition
rejected29 of324 candidates at later states; some penetrate the lever by more
than3mm and some put the distal thumb on an unqualified surface. The complete
648-candidate envelope is therefore **rejected**, with every failed candidate
retained. All162 candidates at the initial19s state passed geometry.

This supports testing a separately declared smooth pressure-subspace handover
from the already attained19s grasp. It rejects a broad late corrective snap to
nominal opposition. No physical restricted-thumb trial has been qualified, and
static overlap is never substituted for the original loaded opposed-pad gate.

Reproduce with:

```sh
python scripts/dexterous/screen_thumb_flexion_pressure.py \
  --trial out/continuous/sensor-hierarchical-force-001 \
  --output out/continuous/sensor-thumb-flexion-plan-002/screen.json
```

The receipt binds the original robot/door XML, actual trajectory, controller log,
mapper and screen source. It records per-geometry failures and the unstepped
calculator clocks. The original robot and failed hierarchy result stay intact.

## Separately frozen physical candidate

`SensorThumbFlexionForceController` layers the restricted map over the tested
smooth contact-mode repair. Its explicit
[pressure profile](../configs/dexterous/sensor-thumb-flexion-pressure-v1.json)
and [mode profile](../configs/dexterous/sensor-contact-mode-thumb-flexion-v1.json)
declare THJ1/THJ2 allocation. The original whole-thumb mode is rejected by this
controller, and the original whole-thumb controller rejects the new mode scope.

The handover begins at19s through the existing two-second quintic force ramp.
It does not snap any opposition target to a new angle. THJ3/4/5 keep their
original goals, feedback and motor caps throughout. Local force targets, PI
bounds, closing offsets, loss recovery and all physical scoring remain unchanged.
The first19s has no additional force term.

The fresh force screen admits an initial thumb normal-equivalent posture effort
of0.516292N within the unchanged0–2N transfer range. This scalar is an effort
coordinate, not measured pressure. The recorded-pose force screen passes6/6;
the separate geometry receipt admits only the162 candidates at the attained19s
grasp. Its full later-pose envelope remains rejected29/648. Neither receipt
qualifies a new loaded trajectory.

The probe requires both profiles, their source-bound force screen and the
initial-grasp geometry screen. The latter additionally binds the reference,
calibration, schedule, motor contract and same actual source trajectory. Run
with the prior force/hierarchy flags plus:

```sh
--contact-mode-protocol configs/dexterous/sensor-contact-mode-thumb-flexion-v1.json \
--contact-mode-screen out/continuous/sensor-thumb-flexion-plan-001/force-screen.json \
--thumb-flexion-protocol configs/dexterous/sensor-thumb-flexion-pressure-v1.json \
--thumb-flexion-screen out/continuous/sensor-thumb-flexion-plan-001/screen-003.json
```

Thirty-one focused tests verify scope admission, direct original virtual work,
preserved opposition effort, legacy defaults, original capped command ownership
and sensor chronology. Physical results must be reported separately.

## First physical trial: continuous grasp, incomplete operation

`out/continuous/sensor-thumb-flexion-001`, sourcee23de8999, completed36s and
18,000 actual2ms steps: **failed26/28**. The first19s matches the retained
acquisition bitwise. All original loaded distal-patch and continuous opposed
grasp gates pass throughout the force phase; no loaded middle/endcap patch or
unintended hand contact occurred. Caps, loopbacks, balance, original physical
tolerances and all warning gates pass. The route clock reached27s and settled.

The independent remaining failures are motor-coordinate tracking and actual
operation. The lever reaches0.683547rad and the latch9.89183mm, short of the
unchanged0.8rad/11mm requirement. THJ2 first exceeds40mrad tracking at24.122s,
peaks at71.137mrad at28.718s, and physically crosses its authored lower soft
stop at24.494s. Maximum stop penetration is16.684mrad, within the existing20mrad
numerical tolerance but unsuitable as an intended interior hand configuration.
The failed report remains unchanged; this is not reclassified as an operation
success because the controlled axes now use force feedback.

The diagnostic uses the **effort-space** projector actually applied by the
controller. With position stiffnessK, it separatesPKe and(I−P)Ke before mapping
each component back throughK⁻¹. Final mean THJ2/THJ1 errors are−69.817/−45.679mrad;
the retained tangential equivalents are only+1.067/−4.094mrad. Across the whole
force phase, retained tangential error≤4.171mrad and THJ3/4/5 opposition
error≤17.257mrad. Most tracking error is therefore in the removed normal
posture direction, yet its resulting contact with the soft stop remains a
physical limitation.

In the final second, local thumb pressure is6.586N and independently qualified
normal load8.945N even though the clipped virtual-normal request is0N. The
actual saved contact wrenches, transformed through matching pre-integration
Jacobians, apply THJ2/THJ1 moments of−0.18998/−0.05626Nm. Normal-only components
are−0.34347/−0.12552Nm; friction changes the total. THJ2's actual motor also
commands−0.02151Nm toward its lower stop. No actual joint-limit constraint
torque was saved, so its magnitude is not invented by a fresh dynamics solve.

The geometry explains why stronger interior posture alone is insufficient:
at the final fixed palm/opposition pose, THJ2+2mrad increases distal overlap
by76.2µm. THJ2+30mrad, restoring interior margin, changes overlap from0.211 to
1.354mm. THJ1−20mrad compensation still leaves1.071mm overlap. Thus the fixed
palm, fixed opposition targets and current lever position constrain the two
flexion axes to a compressed configuration. Thumb opposition coordination must
be screened before expanding to wrist/palm relief. No gain or strength increase
is justified by this evidence.

Independent contact classification and qualified loads match the lossless raw
archive exactly; the contact auditor rejects the task solely through its
unchanged physical report. All recorded finger commands reconstruct within
4.64e−15Nm with zero motor saturation. Full evidence and the personally inspected
final close-up remain under the run directory; see `pressure-limit-diagnostic.json`
and `touch-operation-hand-az150-0002.png`.
