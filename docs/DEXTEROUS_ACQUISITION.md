# Open-hand acquisition development

September 8, 2026 UTC. This implements the first part of [the approved next-step plan](DEXTEROUS_NEXT_STEPS.md). **Acquisition is not solved.** The verified initialized Isaac opening remains separate and unchanged. This work is a privileged, standstill MuJoCo development probe, not approach locomotion, Isaac validation, traversal, or a vision/tactile policy.

## What now runs

`plan_acquisition.py` works backwards from the verified Door55 grasp, optimizing wrist and finger configurations while screening robot/door and self collisions. It retains the original final grasp, keeps the free-root starting pose fixed during this geometric search, and requires the starting hand to be at least 2 cm clear of the lever. Interior wrist poses can detour; their tracking error is diagnostic. The optimized mode allows up to 3 cm / 0.4 rad departure from the straight-line orientation/position proposal. Collision rejection remains 1 mm for non-foot intersections and 3 mm for foot/floor intersections.

Candidate `acquisition-009` passed 201 planned configurations plus 800 interpolated checks. The initial hand/lever gap was **44.40 mm** and the maximum palm detour was **3.32 mm**. Sampled collision screening does not prove continuous collision-free motion or dynamic feasibility. Earlier candidates were rejected for panel, lever and upper-arm/torso intersections. Independent valid waypoints initially hid collisions between waypoints; the denser check caught them.

`probe_acquisition.py` writes the robot starting state once, then executes only robot motor commands in free-base physics. It records a trace, motor forces, grasp opposition, balance, the path phase, and a failing exit status when acquisition checks fail. The final probe also rejects non-foot penetration over 3 mm and joint-limit violation over 0.02 rad. Those are development tolerances, not a final benchmark protocol.

## Measured failures and what they imply

These are different controller-development attempts, not seven independent benchmark seeds. All failed acquisition.

| Trial | Change / result | Next implication |
|---|---|---|
| 001 | Fixed path; upright, four fingers loaded, thumb unloaded; final palm error 24.79 mm. | Contact-free geometric arrival does not guarantee force closure. |
| 002 | Contact-directed finger force and stiffer torso servo; upright, error 23.75 mm, thumb still unloaded. | More force alone does not fix the approach. |
| 003 | Added measured-root Cartesian correction; destabilized and fell (42.93 degrees torso tilt). | Uncoordinated arm correction can overload balance. |
| 004 | Added progress gating on measured hand error; remained upright but stopped at the starting waypoint. | The gate correctly exposed a persistent servo offset. |
| 005 | Explicit bounded motor torque, matching the Isaac servo calculation; reached within 9.71 mm, max tilt 0.265 degrees, thumb still unloaded. | The earlier local wrapper clipped the added servo correction at the elbow position-target bound. Command-unit conversion is now explicit; native torque caps and joint limits remain unchanged. This is not evidence of full Isaac parity for the new controller. |
| 006 | Adaptive thumb target; reached within 8.83 mm but thumb/index contacts were missing. | Digit coordination still needs work. |
| 007 | Repeated with explicit runtime joint/penetration checks. Same reaching result; little-finger `LFJ5` exceeded its lower joint limit by 0.08974 rad (5.14 degrees). Maximum non-foot penetration was 0.446 mm. | Reject the trial. Bound-respecting targets and torque limits are insufficient to guarantee that simulated joints stay within their physical range. Add limit-aware contact control before connecting to opening. |

The best reaching error is not an acquisition success. No opening, traversal or sensor-only score was added. See the [machine-readable results](../results/dexterous/2026-09-08/acquisition-development.json).

## Reproduce the current development probe

Use the native environment and audited H1/Shadow assets described in [the execution record](DEXTEROUS_HUMANOID.md). The existing Isaac fixture configuration supplies the numeric grasp; no private checkpoint is needed. Use new output directories so failed evidence is not overwritten.

```bash
python scripts/dexterous/plan_acquisition.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --output out/dexterous/acquisition-plan \
  --offset 0 -.16 0 --optimize-collisions --samples 201

python scripts/dexterous/probe_acquisition.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --reference out/dexterous/acquisition-plan/reference.json \
  --output out/dexterous/acquisition-trial \
  --grip-force 6 --torso-impedance 10 --cartesian-tracking \
  --tracking-gate --hold-seconds 8 --explicit-motors --adaptive-thumb

python scripts/dexterous/render_acquisition_probe.py \
  --robot out/dexterous/robot/h1-shadow.xml \
  --door out/dexterous/assets/doors/db0055_swing_single \
  --trial out/dexterous/acquisition-trial --view hand
```

The second command currently exits nonzero because acquisition fails; preserve its report and render it anyway. Use `--view body` for an accompanying wide video. Videos replay recorded states with time/outcome labels and a blue diagnostic thumb; they do not invent intermediate motion or qualify as Isaac footage. Source snapshots, dependencies, numeric input, and failed trajectories are retained with the experiments. The seven physical attempts have per-run source snapshots; the first exploratory geometry attempts predate that capture, so use candidate 009 for reproduction.

## Immediate continuation

1. Add limit-aware contact control; keep `LFJ5` and every other joint inside its physical range without increasing robot strength or relaxing the gate.
2. Coordinate thumb and finger closure with actual hand position and contact load. Preserve the established balance and explicit motor contract. Detect missing opposition and recover rather than pushing forward.
3. Require a sustained loaded opposed grasp from the separated start, then connect to the existing opening teacher and run it in the prepared Isaac environment. Only then add approach walking and traversal.

All work here ran locally on CPU. No new GPU allocation was created. Full trials and previews are retained in the owner's `DoorBench-runs/2026-09-08-acquisition` archive and registered in Run Center. Future GPU runs must retain the existing deadline and isolation protections.

## Continued investigation, September 8 after 07:00 UTC

The preceding reproduction command preserves the original failed experiment; adaptive thumb closure is not a recommended acquisition controller. An independent audit found that a finger's distal and middle joints share a single tendon motor. Their sum is controllable, but the two angles cannot be commanded independently. Native transmission lengths match the implemented matrix exactly. Contact can straighten one joint while folding the other to its limit, even when the planned equal-angle pose looks correct.

Fifteen independent diagnostic trials are retained, including unsuccessful slower approaches, pre-curling and retreat directions. The canonical initialized grasp with six-newton motor-mapped preload sustains all five opposed contacts for six seconds. This is useful evidence that the grasp is mechanically possible, but its initialized start deliberately fails the acquisition requirement.

The best lateral acquisition reaches within 2.54 mm, stays within 0.277 degrees torso tilt and respects all existing physical checks. It still fails: the ring finger loads the wrong side while the other digits and thumb oppose correctly. Its first wrong-side contact occurs at 4.562 seconds, path fraction 0.398355. The nominal geometric path already allows about 0.1 mm penetration there, so the old shallow-penetration screen cannot detect the incorrect contact route. Reopening the trapped finger increases force rather than freeing it.

Root trials 008–011 are recorded in the [continuation results](../results/dexterous/2026-09-08/acquisition-continuation.json). All failed. The optional `--limit-filter`, `--motor-tracking-gate` and `--track-pads` controls are retained diagnostics, not validated solutions. The limit filter exposes infeasibility and cannot guarantee safety on an underactuated contact system.

Current geometric development uses `--coupled-flexion --approach-clearance .006` with `--optimize-collisions`. The first option reduces each distal/middle pair to a shared geometric coordinate; it remains a free-space prior rather than a simulated tendon constraint. The clearance requirement tapers continuously from six millimetres to the measured small soft-contact overlap of the original terminal grasp. Interpolated states are checked as well. Candidate 011 was rejected because its interpolation breached this clearance; it is not sent to physics. Candidate refinement continues without changing the native physical acceptance gates.

Use `render_acquisition_probe.py --snapshots-only --azimuth 150` and additional azimuths for recorded hand close-ups. These snapshots remain labeled with the trial's actual outcome. The additional evidence is preserved in `DoorBench-runs/2026-09-08-acquisition-continuation`. Acquisition trials remain CPU work; the concurrently allocated guarded GPU is validating actual Isaac sensors and subsequent integration.

Candidate 012 passed 401 configurations plus 1,600 interpolated clearance checks. Native execution 012 nevertheless trapped the little finger: the underactuated joint configuration differed from the geometric prior. A physically executed release from the canonical grasp then succeeded. `reverse_native_release.py` preserves that source's actual states, forces, lever frames, target clipping and static checks, and constructs a reverse acquisition proposal. It checks that the source begins with opposition and ends contact-free. Reversing a physical trajectory still does not prove that the reverse commands will execute successfully.

The reversed path brings all four fingers to the correct side. Trial 013's bounded Cartesian integral reduced final palm error to 0.264 mm, but the thumb stayed unloaded. Inspection identified a moving-operator error: the source grasp's lever angle was about 0.002 rad, while acquisition depressed it to 0.278 rad. The static world-space hand target therefore remained above the moved lever. Trials 014–015 tested operator-relative tracking; incidental contact then opened the leaf before a secure grasp, and the controller failed. Trial 016 tested measured finger-force feedforward and also failed. Trial 017 tried resisting unintended leaf swing through hand motor targets; it failed physical checks. These are retained unsuccessful approaches, not demonstrations or recommended controller defaults.

The next contact controller must coordinate actual operator motion, gentle seating and force balance before initiating opening. Existing native and Isaac initialized holds remain useful diagnostics. They cannot replace contact-free acquisition, and none of these failures is removed from the record.
