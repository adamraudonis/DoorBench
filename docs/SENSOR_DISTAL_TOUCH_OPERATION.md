# Local distal touch prevented slip, but the first press stalled

The first local-touch controller completed36 seconds /18000 physical2ms steps
without losing the opposed grasp after acquisition, loading an invalid distal
patch, touching the palm to the lever, falling, or exceeding original motor and
anatomy limits. It nevertheless **failed operation**: the press progression paused
at0.680 of8 seconds because several fingers could not develop the required load.
No thresholds were changed to count this as success.

The first19 seconds of actual positions, velocities and61 motor forces are
bitwise identical to the qualified scripted acquisition. The final original
12-check opposed-pad grasp audit still passes. The full operation result is
24/27: requested operation excursion, route completion and lever release fail.

## Frozen control experiment

`SensorDistalTouchController` wraps the existing sensor-balanced joint controller
and a pure static press schedule. Its runtime inputs are the numeric sensor
packet and local clock. It uses no real lever angle, world/root pose, contact
object ID, anatomical label, grasp score, or active plant handle. Robot-model
metadata is used only at construction to validate the fixed tactile layout.

The five distal sensors use4×2 angular bins with three force channels. Their
calibrated rotation maps sensor-Z to body+Y; force into the palmar body-Y surface
therefore has positive sensor-Z projection. The middle two horizontal columns
cover the palmar hemisphere. Both vertical rows are summed; the other columns
and negative projection cannot qualify a preload. These coarse measurements do
not identify the contacted object or certify the original axial pad band.
Independent actual-contact evaluation retains those requirements.

After the unchanged19-second acquisition, the declared controller:

- Waits at least2 seconds while preloading toward2N per finger and3N for the thumb.
- Integrates closing-goal offsets with gain0.015rad/(N*s), maximum rate0.03rad/s.
- Caps each four-finger J1+J2 offset at0.06rad, divided equally between the two
  nominal joint targets; the original passive inequality is unchanged.
- Caps thumb THJ1 closing offset at0.08rad.
- Filters local loads with a0.02-second time constant.
- Advances the static press clock at most2ms per real step only after all local
  loads stay above0.8N per finger /1.2N thumb for0.1 seconds, and the8 commanded
  arm/torso/wrist encoder errors are below0.03rad.
- Keeps the original finger feedback profile, motor caps and physical model.

A3872-pose offline screen tested32 closing-offset corners at each of121 rigid
press poses. Maximum nominal penetration was1.928mm; only original distal lever
surfaces were admitted. This geometric hypothesis did not qualify loaded grasp
or pressure, and it did not override any actual2ms checks.

## Reproduce

```sh
python scripts/dexterous/screen_sensor_touch_preload.py \
  --acquisition out/qualified-sensor-acquisition \
  --press-plan out/press-plan/plan.json \
  --output out/preload-screen.json
python scripts/dexterous/probe_sensor_touch_operation.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --press-plan out/press-plan/plan.json \
  --preload-screen out/preload-screen.json \
  --output out/touch-operation-fresh
python scripts/dexterous/audit_sensor_acquisition_contacts.py \
  --trial out/touch-operation-fresh \
  --output out/touch-operation-fresh/independent-contact-audit.json
python scripts/dexterous/diagnose_sensor_touch_stall.py \
  --trial out/touch-operation-fresh \
  --output out/touch-operation-fresh/stall-diagnostic.json
python scripts/dexterous/render_sensor_touch_operation.py \
  --trial out/touch-operation-fresh --hand
```

The press plan comes from [SENSOR_SCRIPTED_HANDLE_OPERATION.md](SENSOR_SCRIPTED_HANDLE_OPERATION.md).
The native trial is `/tmp/doorbench-continuous/out/continuous/sensor-touch-operation-001`.
Its source, reset, model, layout and static plans are frozen with the evidence.

## Why it stalled

The diagnostic aligns each input's tactile timestamp with the **previous actual
physical interval**, then compares the local projection with independently
qualified contact loads. Encoder readiness was100% after acquisition. All five
closing offsets reached their declared bounds.

| Final-second mean | Local palmar projection | Qualified distal normal load |
|---|---:|---:|
| Index |0.520N|0.571N|
| Middle |1.134N|0.997N|
| Ring |0.700N|0.728N|
| Little |0.695N|0.670N|
| Thumb |1.618N|2.170N|

Projection and normal load differ because of frame orientation and shear. The
index, ring and little finger loads are actually weak; the paused progression
is not explained by sensor under-reading alone. The index/middle/ring/little
coupled motors deliver only0.0206 /0.0397 /0.0224 /0.0235Nm against their original
±1Nm caps. Their final mean tracking errors are0.0103 /0.0199 /0.0112 /0.0117rad.
The index proximal motor delivers0.0542Nm; thumb THJ1 delivers0.0405Nm. These
figures come from the **actual step's motor-force buffer**, which matches every
submitted force exactly. The motor caps are not the active limitation.

A separately versioned, smooth post-acquisition stiffness increase is therefore
a reasonable next comparison. Its thresholds, offset bounds, original forces
and anatomical checks must remain unchanged. The current result remains failed.

## Evidence

- Peak tilt0.33247 degrees; joint-stop2.040mrad; loopback0.188mrad.
- Maximum nonfoot penetration0.105mm; no unintended or invalid loaded contact.
- Opposed grasp remains continuously valid for19.628 seconds through the end.
- Maximum actual lever angle0.001105rad; the latch did not retract.
- Original model and every motor cap retained; zero warnings, QP failures or
  external assistance.
- Independent reconstruction matches all18000 pad classifications and load
  totals exactly, without recomputing contact dynamics.

Raw transitions use the existing float64 lossless `NativeTransitionArchive`,
with every source field checked before writing. A250-row exact round trip
verified all numeric, contact, frame, control and warning values. The new reader
supports both this compact format and the older immutable JSONL archives.

Trial001 provenance SHA256:
`8889f8c72a58c7162c0d8003be3802d1324c408a59034fc9b66c15351cc5f295`.
Its compact manifest SHA256:
`79990a50d6123e57436dfdfc2c95768e97c341e56ca2ed4bdf46ff3f921875bd`.
The finalized wrapper additionally rejects boolean clocks; that boundary
hardening does not alter this recorded run's numeric-clock behavior.
