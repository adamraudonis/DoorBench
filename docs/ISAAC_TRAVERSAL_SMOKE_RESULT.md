# Executed traversal adapter smoke

The frozen `7016c2a80` fixture executed one second in Isaac Sim 5.1 on
2026-09-08. Its independent audit passed **26/26 checks**. The whole door task
remains incomplete and its original result remains false. There was no opening,
hand acquisition, traversal, or sensor-only policy qualification in this smoke.

The independent auditor is `scripts/dexterous/audit_traversal_smoke.py`. It reads
saved evidence without starting or stepping a simulator. Native MuJoCo is used
only to read the original authored joint limits.

| Independently reduced evidence | Result |
| --- | ---: |
| Integrated 2 ms steps | 500 |
| Reset plus completed contact epochs | 501 |
| Causal controller decisions | 500 |
| Actor-origin velocity conversion maximum error | 1.525e-8 m/s |
| Submitted generalized input maximum error | 3.696e-6 Nm |
| Original transmission residual | 7.719e-10 Nm |
| Maximum authored joint-stop penetration | 3.227e-6 rad |
| Maximum passive loopback violation | 7.532e-7 rad |
| Self/nonfoot/working-hand penetration | 0 m |
| Maximum torso tilt | 1.976 degrees |
| Minimum root height | 1.0066 m |

The actual imported pelvis COM offset was
`[-0.000199999995, 0.000039999999, -0.045219998807]` m, confirming the
source-derived estimate in [the API review](ISAAC_MEASUREMENT_API_REVIEW.md).
The maximum COM-versus-actor velocity difference during the smoke was
0.0244431 m/s. Poses and world angular velocities agree exactly between the two
archived SDK fields. Backend input remains submitted actuation input, not an
independent motor-torque measurement.

All 69 joints and 61 motor names/orders, original caps, captured source modules,
13 explicit frozen inputs, array clocks, preceding motor commands, contact
buffer slices and floor-only support reductions passed. Normal and friction
buffers peaked at 381 and 229 occupied slots, below their separate 16,384-slot
capacity. The final incomplete report was checked explicitly against accidental
promotion to a full-task pass.

Controller decisions span 0 through 0.998 seconds. The integrated state and raw
contact archives extend through 1 second. The reset contact interval `[0,0]`
remains unstepped initialization evidence. No additional controller command was
needed to classify this short timeout; a successful complete task still needs
its actual final quiet-stop qualification.

## Retained evidence and reproduction

Local evidence is in
`/tmp/doorbench-continuous/out/continuous/traversal-smoke-live-001/`. `raw/`
contains the original run arrays, source copies, reports and both camera videos;
`evidence.tar` preserves the received remote files. The original remote
`run.log` symlink was recorded in `original-links.json`; its referenced log was
copied as a regular local file. No frozen runtime source or result was changed.

The compact `audit.json` SHA-256 is
`a93c35108865fe15108922a6ad6207b587ed0884ae09868ba631ea9c7e049273`.
It binds its auditor and all top-level raw evidence files by hash. The source
package and exact invocation are described in
[the frozen specification](ISAAC_TRAVERSAL_SMOKE_SPEC.md).

Run the auditor with the dexterous asset Python:

```sh
python scripts/dexterous/audit_traversal_smoke.py \
  --run /path/to/raw \
  --spec /path/to/traversal-smoke-spec-001/spec.json \
  --motors /path/to/h1-import.motors.json \
  --native-robot /path/to/h1-shadow-loopback-v2.xml \
  --output /path/to/new-audit.json
```

Fourteen self-contained contact-buffer tests pass. Three additional negative
controls against disposable evidence copies rejected a wrong COM offset, a
future contact interval and a falsely promoted task result. Original evidence
was retained. I inspected the wide and close-up reset images: the robot is
upright with hands separated from the handle. The wide view is overexposed;
that limits presentation quality and is separate from the verified physics.
