# Smooth finger stiffness comparison

This is an instance-specific scripted motor-route experiment with sensor-only
balance and local distal tactile feedback. It is not a learned or vision policy.
The previous [36-second tactile attempt](SENSOR_DISTAL_TOUCH_OPERATION.md) is
preserved as a safe but stalled failure. Its independently archived actual motor
forces exactly equal all submitted commands. Three fingertips remain below the
unchanged 0.8 N progression threshold, using only approximately 0.02–0.05 Nm in
their principal closing motors despite the original ±1 Nm limits.

The separately frozen `sensor-distal-touch-impedance-v1.json` comparison preserves
the original acquisition through 19 seconds, then raises right-finger position
feedback from four to sixteen times the authored coefficients over two seconds.
The ramp is `4 + 12*(10u^3 - 15u^4 + 6u^5)`, with `u=clip((t-19)/2,0,1)`.
Its first and second derivatives are zero at both ends. Only copied controller
stiffness arrays change; the native actuator coefficients, caps, passive
loopbacks, collisions and masses stay fixed. The maximum gain increment per
2 ms is below 0.022501 times authored stiffness. Actual command and force
continuity remain measured evidence; smooth coefficients alone cannot guarantee
smooth forces under changing contact.

Velocity damping stays 0.05 Nm s/rad with the same target-velocity term. The
five preload targets (2/2/2/2/3 N), progression minima (0.8/0.8/0.8/0.8/1.2 N),
filter, two-second preload, 0.1-second readiness requirement, offsets
(0.06/0.06/0.06/0.06/0.08 rad), rate bounds and eight-second press route are
unchanged. Most finger motors retain ±1 Nm; THJ4 and THJ5 retain their original
±2 and ±3 Nm respectively. No torque is added after the existing capped
controller, which remains the authority for previous-action history.

The offline joint-offset envelope is identical to the previous trial and is
only a nominal geometry screen. Qualification still requires actual 500 Hz
joint/loopback/collision/cap checks, no invalid loaded distal patches throughout,
the original opposed distal-pad hold, and measured handle/latch depression.
One fresh finite 36-second trial was executed. The original stationary, modest
arm, contact-free reach, acquisition, and constant-stiffness tactile profiles
remain unchanged. A rejection requires an explicit episode reset.

Run the existing `probe_sensor_touch_operation.py` command documented in the
prior trial with a fresh output directory and add:

```sh
--impedance-protocol configs/dexterous/sensor-distal-touch-impedance-v1.json
```

The runner snapshots the actual imported source closure and protocol bytes.
Runtime accepts the numeric actor packet plus local time only. The wrapper
changes no joint goals beyond the existing local-touch controller and accepts
no world pose, object identity, geometry state, or active-plant handle.

## Measured result: safe but still incomplete

`out/continuous/sensor-touch-impedance-001` completed all 18,000 physical steps
in 113.3 seconds on CPU. It remains **failed, 24/27 checks**. The lever reached
0.326506 rad (18.7 degrees) and the latch moved 4.705 mm; qualification requires
0.8 rad and 11 mm. The tactile controller advanced 3.616 of its eight virtual
press seconds before pausing. This is partial handle actuation, not door opening.

The first 19 seconds of actual positions, velocities, timestamps, all 61 forces,
and scripted joint targets are bitwise identical to the preserved tactile001
acquisition. The maximum finger command change during the two-second gain ramp
was 0.002607 Nm per 2 ms, with no abrupt gain or force change at the boundary.

All independent anatomy, collision, original cap, loopback, joint-stop, warning,
upright and balance checks passed. The original opposed distal-pad grasp remained
continuous for the final 19.628 seconds. There were zero invalid loaded distal
patches and zero unintended hand contacts. Independent raw-force/frame reduction
reproduced all 18,000 classifications and qualified pad loads exactly. Actual
motor force buffers equal every submitted command exactly, with 0 Nm error.

The index finger now limits progression:

| Final-second mean | Local projection (N) | Qualified normal load (N) |
|---|---:|---:|
| Index |0.706|0.798|
| Middle |1.972|1.937|
| Ring |2.045|1.795|
| Little |1.998|1.737|
| Thumb |3.740|5.348|

Encoder readiness was 100%. The index motor closure offset reached its unchanged
0.06 rad bound, but its actual coupled coordinate tracked within a mean
0.003904 rad. Its mean force was 0.03122 Nm under the original ±1 Nm cap. Further
stiffness alone has limited remaining angle error to correct. A future experiment
would need an explicitly screened finger-coordination or preload change, while
retaining this failed result and the frozen progression thresholds.

The final offsets were `[0.06, 0.06, 0.03187, 0.02673, 0.0]` rad. Thumb projection
exceeded its 3 N target even at zero extra closure; the original controller only
adds closure and cannot unload below the source posture. This is another concrete
limitation for subsequent coordination design, not a reason to relabel the run.

The frozen runtime revision is `165aa4a1d`. Evidence hashes:

- Provenance: `755da15ae30ace34ea2dfe5a07eb9a7a78fb30381a1111a5a011639c527d0f53`
- Report: `649c95e51fafbd3bafc3b845946968523787709db330647a7d9fc4bfebc31412`
- Raw manifest: `58a6603bd63400ad7d632cff249a7a1eed5bf0007aba11d4af8444d9797342e7`
- Trajectory: `77f6f3071fa57bd0f5395806361a0b0045501c749afaaaa306b0461eedb7df5d`

Use `reduce_sensor_touch_impedance.py --baseline ... --trial ... --output ...`
to reproduce the exact prefix and force-continuity comparison. The raw-contact
auditor and stall diagnostic documented in the preceding experiment apply
unchanged. Final hand snapshots are geometry replays of the attained state and
remain labeled FAILED. They have been inspected; no additional video was written
in this disk-bounded experiment.
