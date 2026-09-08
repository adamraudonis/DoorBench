# Isaac operation 005: honest failed hold, with a recoverable contact transition

The fresh `volar-phalange-v1` trial passed 28/28 independent archive-consistency checks and every mechanical/motor check. It opened and held Door55 at 0.09819 rad, but **failed the frozen final 0.5-second grasp criterion**. The original report remains unchanged; this is not qualified operation training data.

The misleading legacy field `operation_digit_unload_samples=976` counts every failed aggregate grasp score. Only 77 of those steps had a digit below 0.2 N; 899 had all five digits loaded but failed opposition geometry. The final 251 physics samples contain 102 failures: four low-load middle-finger samples at 21.526, 21.558, 21.614 and 21.654 s, plus 98 opposition failures. No loaded patch violated its selected local volar-surface contract.

The index finger changes from distal to middle contact first. The middle finger then continues rolling around the lever on its distal tip. At 21.702 s, its radial contact direction is `[0.630, 0.328, -0.704]`, versus index `[-0.140, 0.989, 0.045]` and thumb `[0.133, -0.990, -0.038]`. The index–middle dot product is 0.205 and thumb–middle is -0.215; both fail the unchanged opposition bounds. At 21.704 s, the middle finger transfers to its middle phalanx, with radial direction `[-0.117, 0.993, 0.020]`. The grasp immediately satisfies every opposition check and remains valid to 22 s, a 0.296-second hold. Directions were independently reconstructed from the recorded joint/root/door state in a never-stepped FK model.

A bounded next experiment can preserve the exact control law and extend the physical hold to 24 seconds, retaining the final 0.5-second requirement. This tests whether the newly seated branch remains stable; it must not be reported as a pass before execution. A controller improvement should encourage earlier middle-phalange seating through bounded proximal-finger preload or a contact-aware transition, instead of increasing distal curl and rolling the tip farther around the cylinder. All load, opposition, collision, joint and passive-tendon gates stay unchanged.

Reproduction:

```sh
python scripts/dexterous/audit_isaac_operation_archive.py "$RUN" --robot "$ROBOT" --out archive.json
python scripts/dexterous/analyze_grasp_hold.py "$RUN" --out grasp-hold.json
```

Here `RUN` is the durable `isaac-operation-v2-005` archive and `ROBOT` is the versioned Shadow loopback-v2 XML. The compact receipts are [archive consistency](evidence/isaac-operation005-archive.json) and [complete contact diagnosis](evidence/isaac-operation005-grasp-hold.json). No render, original report, physics result or dataset asset was overwritten.
