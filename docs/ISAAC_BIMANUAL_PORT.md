# Bimanual opening in Isaac: development adapter

The `--full-opening` mode runs the shared privileged controller from a contact-free,
near-handle reset through acquisition, lever operation, left-hand support, right-hand
release and loaded panel opening. **This adapter has not yet passed a live Isaac
trial.** It does not include approach, passage or a sensor-only policy.

The existing partial-opening and actor modes keep their separate contracts. This
mode uses only the original 61 bounded motor efforts, actual PhysX contact loads
and synchronized measured body/joint state. Its unstepped native geometry provides
a clearance estimate; independent PhysX collision and penetration checks remain
required. A recorded Isaac pose check agrees within 0.82 micrometres at the initial
sample. That is a frame-consistency check, not evidence of successful control or
complete collision-shape equivalence.

Actual left-palm load is separate from finger load and from other scene contacts.
Normal forces and friction patches retain their actual counterpart and are summed
without pretending their contact-point indices correspond. These privileged
measurements never enter robot sensor packets. The sensor actor rejects this mode
and its geometry inputs.

The declared terminal event is the first measured target-aperture crossing, with a
fixed maximum duration. The independent audit requires 0.5 seconds of opposed
right-hand grasp before operation, grasp and left-panel support before release,
and uninterrupted left-palm support throughout the final 0.5 seconds. It also
requires measured hand clearance, original mechanical checks and complete numeric
coverage of every 2 ms step. An attractive terminal frame cannot replace a failed
contact interval.

On a [ready corrected-hand environment](ISAAC_V2_READY.md), extend the normal
acquisition invocation with:

```bash
--acquisition --operate-after-acquisition --full-opening \
--open-on-latch-clear --operator-compliance-gain .5 \
--native-door "$DOOR_XML" --left-palm-targets "$LEFT_TARGETS" \
--right-release-screen "$RELEASE_SCREEN" \
--bimanual-runtime-screen "$RUNTIME_SCREEN" \
--target-aperture 1.2 --seconds 55 \
--grasp-profile volar-phalange-v1 \
--grasp-profile-definition configs/dexterous/grasp-profiles/volar-phalange-v1.json
```

The source-bound target and release files must match the robot and door. A new
platform may need the explicit runtime geometry re-screen; passing that screen
does not assert physical transfer. Preserve the run's source archive, input hashes,
`full-opening-protocol.json`, actual contact/physics arrays, wide and hand video,
and `full-opening-report.json`. No full-opening benchmark score is established by
the adapter or its unit tests.
