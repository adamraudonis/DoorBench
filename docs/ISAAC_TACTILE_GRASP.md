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

`own-imu-grasp-pressure-001` also fails the sustained-hold gate. Its longest qualified hold rises from 0.002 to 0.032 seconds, and every digit has a positive minimum pad force over the final half-second. The minima are 0.134, 0.115, 0.058, 0.054 and 0.176 N for index, middle, ring, little and thumb respectively. No invalid loaded patches are reported. Both independent physical and own-gyro reconstructions passed. All 8,500 pre-reflex physical intervals are exactly identical to the baseline; additional coupled flexion reaches 0.050–0.063 rad. This remains a failed component.

The next declared candidate, `grasp-pressure-v2`, raises the local normal-force target from 0.4 to 1.5 N. Additional coupled flexion remains limited to 0.08 rad at 0.08 rad/s, with unchanged original motor caps and unchanged thumb targets. It must undergo the same fresh preparation and actual physical checks. The v1 native trial applied zero additional flexion because its existing grasp already exceeded the preload deadband; its pass therefore established compatibility, not corrective effectiveness.

The v2 native run completed all 9,500 intervals and passed sustained grasp, contact anatomy, joint limits and upright balance, but failed the unchanged 0.04 rad motor-tracking gate (maximum 0.06706 rad). Isaac was correctly withheld. Candidate v3 targets 0.8 N with the same bounds: the hypothesis is that a smaller force demand will avoid excessive target error against the loaded handle while providing more pressure than v1. This is an unqualified experiment until its fresh native and Isaac checks pass.

The [complete v1 comparison receipt](evidence/isaac-grasp-preload-comparison-001.json) includes matching configuration checks, pre-reflex state/force comparisons and hashes of every source report. The first v3 preparation stopped before physics because its feedback label did not match its declared profile. Retry `own-imu-grasp-pressure-004` corrects that label and starts from a new frozen source; all three checked-in pressure schedules now pass explicit runtime-admission tests.

## Completed 0.8 N trial and reproducibility finding

`own-imu-grasp-pressure-004` completes all 9,500 Isaac intervals, with both independent audits verified. It reaches a **0.822 s** opposed hold but still fails the final hold: minimum index pad load over the last half-second is **0.171 N**, below the unchanged 0.2 N threshold. The other minima are 0.294, 0.308, 0.347 and 0.306 N; there are zero invalid loaded patches. Maximum motor-coordinate tracking error is 0.01746 rad. All four pressure offsets reach their 0.08 rad cap.

This comparison is not an identical-prefix causal test. Initial reset states are exact matches, but the first balance QP takes 50 versus 75 iterations. The existing solver leaves its adaptive-rho interval to a wall-time heuristic. Pre-reflex state/force differences are retained in the comparison receipt. The new opt-in `fixed-rho-interval25-v1` uses an explicit 25-iteration interval, with unchanged residual tolerances, force caps and objective. A paired test uses that solver for both the 0.08 rad and 0.12 rad pressure limits, holding the 0.8 N target and 0.08 rad/s slew constant. Both require fresh native and actual Isaac qualification; historical solver behavior remains reproducible by omitting the profile.

The first uninterrupted native acquisition-to-press test with v3 reaches 0.53793 rad (30.82 degrees) before stopping at 25.81 s for unintended hand contact. The acquisition handoff passes; the complete operation does not. A qualified grasp is not a qualified lever press or door opening.

## Fixed-solver paired result: qualified grasp

Pressure006 passes all23 original grasp checks. Both independent actual-physics
and own-gyro audits reproduce the result; final files were hash-verified off-pod
on September9 at05:50UTC. It achieves a0.99-second opposed hold, and the final
half-second minimum pad loads are0.324,0.342,0.273,0.329 and0.994N for index,
middle, ring, little and thumb respectively. No loaded patch violates the
original distal anatomy. Maximum motor-coordinate tracking error is0.01742rad.

The matching0.08rad-limit baseline005 still fails its final index-pad hold. All
8,500 pre-reflex physical intervals are exactly identical for root state, all
joints, motor forces, door positions and velocities. Both native preparations
also have identical complete trajectories. The candidate's maximum coupled
preloads are0.10068,0.09566,0.08736 and0.08342rad; its0.12rad limit is sufficient
without changing force caps, thumb targets, anatomy or physical gates.
[Full paired comparison](evidence/isaac-grasp-preload-comparison-006.json).

This is one same-reset grasp component in actual Isaac, using a scripted joint
route over sensor-feedback balance and fingertip preload. It is not vision
learning, mechanism operation, traversal, varied-start reliability or catalogue
coverage. Those remain separate required milestones.
