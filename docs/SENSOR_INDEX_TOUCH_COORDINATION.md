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
increments and 15,488 nominal full-route/five-digit-offset combinations. Maximum
nominal nonfoot penetration was 2.645 mm with no forbidden hand surfaces. This
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

One new 36-second native trial was executed after the screen and focused tests.
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

## Actual result: improvement, still below latch release

`out/continuous/sensor-index-touch-001` completed all 18,000 steps in 121.8 CPU
seconds, retaining **24/27 passed checks and an overall failure**. The lever
reached 0.520626 rad (29.8 degrees), compared with 18.7 degrees before index
coordination. Latch movement reached 7.525 mm; the unchanged requirement is
0.8 rad and 11 mm. The virtual press clock advanced 4.794 of eight seconds.

Every first-19-second position, velocity, motor force, timestamp and joint goal
is bitwise identical to the original acquisition. The index offset reached its
0.01 rad bound. Original opposed grasp remained continuously valid for the
last 19.628 seconds. All anatomy, original caps, joint/loopback bounds, contacts,
balance and warning checks passed, with no invalid loaded distal patches or
unintended hand contacts. The independent raw-contact auditor reproduced all
18,000 classifications and qualified force sums exactly. Actual motor-force
buffers again match submitted commands with 0 Nm error.

The index remains the stopping condition at this later pose:

| Final-second mean | Local projection (N) | Qualified normal load (N) |
|---|---:|---:|
| Index |0.687|0.814|
| Middle |0.886|1.067|
| Ring |1.915|1.671|
| Little |2.392|2.083|
| Thumb |4.831|6.750|

Encoder readiness stayed 100%. Mean actual FFJ3 tracking error was 0.004661 rad,
delivering 0.07470 Nm; the coupled FFJ0 error was 0.004018 rad, delivering
0.03214 Nm. Both retain their ±1 Nm caps. Strong thumb loading coexists with a
weak index projection, and thumb's existing extra-closing offset is already
zero. This suggests a subsequent coordination study should examine load
redistribution and the attained palm/finger geometry. It does not establish
that threshold reduction or another gain increase would work.

Peak tilt remained 0.33247 degrees. Maximum finger command change was
0.002747 Nm per 2 ms during the smooth stiffness ramp and 0.005398 Nm after it.
The final attained-state hand close-up was personally inspected and retains its
FAILED overlay. Only three small snapshots were written; no additional video
or duplicate raw archive was created.

Frozen physical source revision: `366441e77`. Evidence hashes:

- Provenance: `d58b24db103a8c541601440319b8999f52600aa8a851c28f61f79e7f41255290`
- Report: `4c9001be301567213b5703d47d432cbc0e655939f634fb1b7c567bd8fc86767a`
- Trajectory: `e8f1340972f4649eb326ac8d6860d25d8cd62446d440172038ac80e5fd5f836b`
- Raw manifest: `c6ad46628bec70557d24a88f35fd81b0bcd2f5e6d6ff820315f1ad9bc0632fb6`

The subsequent defensive dictionary copy in the schedule proxy protects against
future aliasing schedules. The frozen physical run used the existing schedule,
which already returns fresh dictionaries; this hardening changes no values or
timing for that caller. Original failed traces and source snapshots are retained.
