# Isaac operation and sensor review — 8 September 2026

Both recordings are internally consistent. **Neither is a qualified opening
demonstration.** They use a privileged teacher; recording vision and touch beside
it does not make it a sensor-only policy.

| Run (UTC start) | Actual lever maximum | Final door angle | Result |
| --- | ---: | ---: | --- |
| operation-v2-002, 10:48:04 | 0.7967 rad | 0.01684 rad | Valid final opposed fingertip grip; did not reach release or partial-opening criteria |
| operation-v2-003, 11:05:07 | 0.8510 rad | 0.09834 rad | Reached release and held partial opening; failed the final fingertip contract |

The independent [receipt](evidence/isaac-operation-sensor-review-2026-09-08.json)
passes 27 archive checks per run. Each has 11,000 physical and actor samples at
2 ms, plus 550 distinct 128×128 RGB frames per eye at 25 Hz. Joint ordering agrees
exactly with the physical record; normalized previous actions agree with the
bounded applied motor forces within 3×10⁻⁸. Camera timestamps describe held
frames with ages from 0 to 38 ms. Native joint bounds and all eight corrected
loopbacks pass when recomputed at every physical sample. The source package and
all source hashes agree, including the pad verifier omitted from the shorter
runner-local source list.

In 003, the first disallowed patch occurs at **20.744 s**: the index finger's
middle segment takes 0.292 N while the contact remains on the volar side and
5.98 mm inside the lever's cylindrical section. That patch persists through the
remaining 629 physical samples. At the end, the index fingertip carries zero
qualified load and its middle segment carries 2.44 N. The thumb remains on its
opposing volar surface. This is a failure of this experiment's frozen **distal-pad
contract**, not evidence that a middle-segment power grasp is universally
anatomically invalid. No criterion was changed to accept the run.

The recorded palm compensation continues increasing after latch release:
approximately 0.0337 rad at 15.696 s, 0.095 rad near the first bad patch, and
0.11199 rad at the end. This supports testing compensation freeze after measured
release; it is a diagnosis, not proof that the proposed correction succeeds.
Index-middle touch independently appears in the actor recording at 20.732 s.

I inspected the original close views at 14, 16, 18 and 20 seconds in 003, the
20-second close view in 002, and both robot eye views. The initial opposed grip
and lever depression are visible. **The moving door blocks the close camera by
18 seconds in 003**, so its final camera frames cannot verify the contact change.
The robot cameras still show the door and working arm, but their 128-pixel view
does not resolve pad anatomy. A second approach-side diagnostic view is needed
for visual qualification. All five fingertip sensors are unsaturated; the ankle
sensors saturate at the declared 100 N per-channel limit on about 99.8% of steps.

The receipts cross-check the recorded local pad geometry and its qualification
formula. These historical archives do not separately retain raw synchronized
patch normals and body transforms for an independent transform recomputation.
Sent torque comparisons are available at 50 Hz; joint, motor force, pad and
mechanical checks cover every 2 ms step. Camera calibration uses the disclosed
fixed manipulation profile (45° downward pitch, 100° field of view), rather than
the default upstream view. Cross-engine rendered camera parity is not certified
by this archive audit.

The durable archives are `DoorBench-runs/2026-09-08-shadow-loopback/` under the
owner's Desktop projects directory. Each now includes the original prelaunch
`launch-source/manifest.json` and `source.tar.gz`, copied with digest verification.
No generated assets are committed. Reproduce the audit with:

```sh
python scripts/dexterous/audit_isaac_operation_archive.py \
  /path/to/isaac-operation-v2-002 /path/to/isaac-operation-v2-003 \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --out /path/to/review.json
```
