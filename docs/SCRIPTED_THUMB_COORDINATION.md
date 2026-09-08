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
