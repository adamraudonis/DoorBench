# Index-finger coordination comparison

This opt-in instance-specific scripted press retains sensor-only balance and
local tactile feedback. No world pose, contact object ID, geometric oracle or
active simulator enters inference. It is not a learned or vision policy.

The preceding [smooth-stiffness trial](SENSOR_DISTAL_TOUCH_IMPEDANCE.md) remained
failed with a valid grasp: index local load 0.706 N was below its unchanged 0.8 N
progression threshold, despite small motor-coordinate tracking error. The other
four local loads exceeded their thresholds. Raising stiffness again was not the
chosen next experiment.

## Measured geometry and rejected proposal

The final actual index contacts were on the original qualified distal pad at
approximately `[2.941,-6.439,20.432]` mm in its body frame, with outward normal
`[0.153,-0.970,0.189]` and 9.49 mm axial clearance. The three actual contact
points carried about 0.273 N each in the final 2 ms interval. These measurements
come from the actual interval's force buffer and its matching pre-step geometry.

A detached 0.1 mrad joint perturbation found the closest index/lever gap changed
by -75.82 mm/rad for proximal FFJ3 flexion, compared with -30.53 mm/rad for an
equal-split coupled J0 closure. Proximal flexion therefore has approximately
2.48 times the closing effect at this attained state. Abduction moved the pad
away. This is a geometric sensitivity, not a force or stiffness prediction.

A 0.02 rad proximal bound passed the attained-state screen but **failed** the
complete nominal press-envelope screen: 1,936 sampled configurations exceeded
the original 3 mm penetration limit, reaching 3.412 mm. This rejected screen is
preserved at `out/continuous/sensor-index-plan-001/screen.json`; it was never
executed physically. The smaller 0.01 rad proposal is separately screened at
`out/continuous/sensor-index-plan-002/screen.json`, including 101 attained-state
increments and 15,488 nominal full-route/five-digit-offset combinations. Static
maximum nonfoot penetration was 2.645 mm with no forbidden hand surfaces.
This
sampling is only an admission screen; all live physical checks remain required.

## Frozen finite comparison

The new `sensor-index-touch-v1.json` profile adds only an FFJ3 offset after
19 seconds. The offset integrates the **preceding accepted decision's** filtered
index distal projection toward the existing 2 N target, with gain
0.005 rad/(N s), rate limited to ±0.01 rad/s and offset limited to [0,0.01] rad.
An overloaded index reverses that offset; no other finger goal formula changes.
The extra one-decision feedback delay is explicit and source-bound.

Acquisition through 19 seconds, all other finger settings, the 4→16× smooth
stiffness ramp, damping, original motor caps, anatomy, closure bounds, local
load thresholds, encoder readiness and press progression are unchanged. In
particular, the index 0.8 N progression gate is preserved. Motor force output
still comes from the single original capped adapter and owns previous-action
history. Only numeric local measurements, fixed joint goals and clock are used.

One new 36-second native trial is planned after the screen and focused tests.
Add these flags to the preceding tactile probe's reproducible command:

```sh
--impedance-protocol configs/dexterous/sensor-distal-touch-impedance-v1.json \
--index-protocol configs/dexterous/sensor-index-touch-v1.json \
--index-screen out/continuous/sensor-index-plan-002/screen.json
```

The runner requires a fresh output directory and matching source-bound screen,
and snapshots both profile and screen. The unchanged actual 500 Hz gates decide
whether balance, original anatomy, grasp, lever and latch motion pass. The
rejected geometry and prior failed physical trials remain available separately.
