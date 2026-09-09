# Isaac grasp: contact maintenance remains unresolved

The fresh `own-imu-grasp-003` trial finished on September 9, 2026 at approximately 03:33 UTC. The corrected H1/Shadow robot passes **22/23 task checks** over 19 seconds. Its independent contact/physics audit reproduces the failed result, and all 16 own-IMU audit checks pass over 9,500 actual intervals. [Machine-readable evidence](evidence/own-imu-grasp-003.json).

The sole failed task check is the sustained opposed distal-pad hold. There are zero invalid loaded contact patches. Maximum torso tilt is 0.478°, maximum motor-coordinate tracking error is 0.01545 rad, and final palm error is 3.394 mm. Several fingers briefly load the lever, then unload. This is an initialized grasp component, not opening or traversal. Corrected gyro synchronization has not solved contact maintenance.

## Next physical comparison

The opt-in `grasp-pressure` preparation uses `four-finger-preload-v1`. After the 17-second approach ends, local distal tactile cells adjust the four coupled finger flexion targets toward 0.4 N normal load. Additional coupled flexion is bounded to 0.08 rad, with a 0.08 rad/s slew limit; each paired joint receives half. Thumb, torso and arm targets are unchanged. Missing or stale tactile observations cannot trigger further closure.

This is a development reflex over a scripted route, not a learned visual policy. Original authored joint bounds, passive loopbacks, force caps, contact anatomy and half-second sustained hold remain required. The evaluator reconstructs feedback goals from the actual causal touch packets. Fresh native physics, an independent contact audit, complete force replay and reset preflight must pass before the actual Isaac comparison.

```bash
PYTHONPATH=. python scripts/isaac/prepare_sensor_demo.py --receipt "$READY" \
  --output "$NEW_OUTPUT" --task grasp-pressure \
  --sensor-gyro-profile pose-delta-angle-v1 --check-only
```

The complete previous trial and its 1,317 source/asset inputs have been verified off-pod. Failed trials remain in the record.

## First tactile comparison and the next force target

`own-imu-grasp-pressure-001` also fails the sustained-hold gate. Its longest qualified hold rises from 0.002 to 0.032 seconds, and every digit has a positive minimum pad force over the final half-second. The minima are 0.134, 0.115, 0.058, 0.054 and 0.176 N for index, middle, ring, little and thumb respectively. No invalid loaded patches are reported. Independent reconstruction is pending; this is still a failed component.

The next declared candidate, `grasp-pressure-v2`, raises the local normal-force target from 0.4 to 1.5 N. Additional coupled flexion remains limited to 0.08 rad at 0.08 rad/s, with unchanged original motor caps and unchanged thumb targets. It must undergo the same fresh preparation and actual physical checks. The v1 native trial applied zero additional flexion because its existing grasp already exceeded the preload deadband; its pass therefore established compatibility, not corrective effectiveness.
