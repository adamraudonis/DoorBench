# Motor test of the23s thumb configuration

This is an explicitly scripted, instance-specific extension of the retained
sensor-balanced pressure controller. It uses the passed23.0s candidate from
`sensor-thumb-interior-plan-002`. Runtime receives only its five frozen named
thumb goals, local clock and the existing permitted sensor packet. The offline
contact centroid, normal, door and root state are absent from its configuration
and runtime interface.

The [frozen profile](../configs/dexterous/sensor-thumb-coordination-v1.json)
preserves every command before23s. From23–24s it applies a quintic blend from
the current pre-cap thumb goals to the candidate goals, then holds them. THJ3/4/5
remain posture-controlled; THJ1/THJ2 retain the same local tactile force mode.
The modifier runs **before** the pressure projector, original force caps and
previous-action owner. Other25 goals and the original motors/mechanics remain.

The source's measured thumb configuration differs from its command targets
because pressure feedback removes normal posture effort. Consequently the
largest command change is26.528mrad, although the geometric candidate moved
the measured thumb by at most1.663mrad. The nominal connected configuration
path is not assumed to be the physically realized motion. The unchanged
tracking, original joint/loopback/collision, distal anatomy and operation gates
must decide the new result. Pre-coordination goals are recorded too, so their
tracking can be reported alongside tracking of the explicitly changed command.

The preparer checks the source candidate/trajectory hashes, all original joint
bounds, complete first23s goal identity and other-joint identity. On the retained
source history the post23 maximum thumb target speed is0.049739rad/s and
acceleration0.153154rad/s². The required1.5rad/s command bound remains; no force
or stop tolerance increases. Its receipt is
`out/continuous/sensor-thumb-coordination-plan-002/receipt.json`.

The same physical probe takes the existing thumb-flexion profiles plus:

```sh
--thumb-coordination-protocol configs/dexterous/sensor-thumb-coordination-v1.json \
--thumb-coordination-screen out/continuous/sensor-thumb-coordination-plan-002/receipt.json
```

At23s the actual incoming thumb goals must match the frozen source within1e−8rad;
otherwise the controller rejects the run for requalification. Every physical
comparison gets a fresh directory and retains its complete original score.
These source/rate checks are preparation, not a learned policy or an operation
success. The initial finite comparison is36s with the unchanged task gates.

## Actual native comparison: failed

`out/continuous/sensor-thumb-coordination-001` completed all18,000 original2ms
steps in118.75wall seconds using frozen source`949b25f62`. It passed25 of the
28unchanged checks. It did **not** release the latch. The complete first23s
qpos, qvel, motor forces, time and joint goals are bitwise identical to
`sensor-thumb-flexion-001`.

| Actual measurement | Thumb-flexion source | Scripted coordination001 |
| --- | ---: | ---: |
| Maximum lever depression |0.68355rad|0.67337rad|
| Maximum latch retraction |9.892mm|9.743mm|
| Maximum declared motor-coordinate error |71.137mrad|47.317mrad|
| Error against the recorded pre-coordination goals |same as declared|73.844mrad|
| First THJ2 authored-stop crossing |24.494s|24.300s|
| Maximum THJ2 stop excursion |16.684mrad|19.391mrad|
| Continuous opposed grasp during pressure phase |yes|no: one2ms FF unload|

The original joint-tolerance gate remains20mrad and therefore passes this run,
but the intended interior thumb configuration was not attained. The commanded
THJ2 endpoint is27.925mrad inside its stop; the actual final-second mean lies
18.096mrad beyond it. Reporting the smaller error relative to the moved target
alone would conceal this worsening. The separate unchanged release criterion
also fails: it needs at least0.8rad lever depression and11mm latch retraction.

The first original opposed-grip failure is the actual interval27.936–27.938s:
the index unloads, then physically recontacts in the next2ms interval. Smooth
mode retention and progression freeze execute as declared; retained mode is
not counted as physical touch. Every loaded distal patch remains anatomically
qualified. There are no unintended contacts, external assistance, cap changes,
native warnings or QP failures; the pelvis remains upright, supported and quiet.

At the final hold the thumb has9.090N actual qualified normal force and6.657N
local projected touch despite a clipped virtual-normal request of zero. Saved
actual contact wrenches produce stopward THJ2/THJ1 moments of
`[-0.201079,-0.056252]Nm`. No actual joint-limit reaction buffer was recorded,
so its magnitude is not invented from a new dynamics solve.

The frozen five-joint configuration is geometrically feasible at the23s source
state. Its motor goals do not directly enforce that configuration during the
continuing wrist route: the existing pressure projector removes the normal
component of THJ1/THJ2 posture effort. At the final measured state, changing
the thumb goals contributes a position-effort difference of
`[-0.636662,+0.097447]Nm`; the projector removes
`[-0.519614,-0.202660]Nm` and retains the tangential difference
`[-0.117048,+0.300107]Nm`. These are same-state algebraic differences, not extra
forces added outside the controller. All delivered commands reconstruct within
4.64e−15Nm, and original caps remain in force. A one-state geometric candidate
therefore does not establish a loaded thumb path through the remaining press.

The independent evidence is retained in that trial directory:

- `coordination-execution-audit-002.json`:8/8 execution/prefix/rate checks,
  exact goal and blend reconstruction, and both tracking definitions.
- `independent-contact-audit.json`:all18,000 actual interval classifications
  and qualified loads agree exactly; original overall task failure retained.
- `force-mapping-audit.json`, `regulation-decomposition.json` and
  `mode-execution-audit.json`:source-bound algebra, actual force and mode checks.
- `pressure-limit-diagnostic.json`:signed actual contact moments, stop timings
  and detached geometric derivatives, with their scopes separated.
- `touch-operation-hand-az150-0002.png`:personally inspected36s hand close-up
  rendered from the actual saved state and clearly labeled failed.

The first execution-audit receipt is also retained. It incorrectly replayed the
quintic with nominal integer-tick time, giving a4.04e−12 blend discrepancy. The
second receipt uses the archived actual interval start, which is exactly the
frozen controller's accumulated `d.time` input. Both goals and blend then match
exactly; no tolerance or physical gate was loosened.

Reproduce the independent coordination reduction with:

```sh
python scripts/dexterous/audit_scripted_thumb_coordination.py \
  --baseline out/continuous/sensor-thumb-flexion-001 \
  --trial out/continuous/sensor-thumb-coordination-001 \
  --output /fresh/path/coordination-audit.json
```

The physical report SHA256 is
`e7adbd884de51f78d07a941584d70299b220e77957895faa8534b372efbf67dc`,
trajectory SHA256 is
`ca27765e6436197cf4793039428354f0bf81518be9ce5e2aeb84085c891cebc0`,
and source provenance SHA256 is
`1fdb3511157696cfd0dbd912401e13fd8be30e97e01a96b014ddbca5383e8ee8`.
