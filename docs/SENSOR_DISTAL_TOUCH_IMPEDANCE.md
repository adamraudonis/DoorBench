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
One fresh finite 36-second trial is planned. The original stationary, modest
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
