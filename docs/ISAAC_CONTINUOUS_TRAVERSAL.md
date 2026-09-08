# Opt-in continuous Isaac traversal

`scripts/dexterous/isaac_opening.py --traverse` composes the continuous
walking/opening teacher and the measured post-opening controller. It requires
`--full-sequence-reset` and `--full-opening`, along with their existing source,
checkpoint, preparation and screened-geometry inputs. It cannot be combined
with a sensor actor, direct mechanism forces or the older panel-push diagnostic.

This adapter has CPU contract tests and source compilation checks. No GPU run
was launched while implementing it. It does not turn the separate qualified
components into a qualified full episode. The first live run must still satisfy
all mechanical, contact, opening and passage checks.

Add these options to the existing source-bound walking/full-opening command:

```bash
--traverse --seconds 150 --target-aperture 1.2 \
--transfer-load-target 4 --record --view wide
```

Use a fresh output directory and the original H1 checkpoint. The post-opening
profile is fixed to `sequential-v2`, five seconds per stow phase. A 150-second
value is an episode timeout, not a completion claim. The transfer load defaults
to the original 4 N; an explicit 6 N trial is a different declared controller
experiment. Values must be finite, above 2 N and at most 10 N. The native motor
caps and imported plant remain unchanged.

## State and force continuity

The wrapper receives the actual integrated root, all 69 joint positions and
velocities, body origins for both feet/palms and both mechanism bodies, actual
door velocities, the separate palm touch-site pose, and the preceding actual
normal/friction contact interval. It starts once at the walking reset. The
handoff does not reset the active plant or restart the global clock.

PhysX joint-effort readback includes the runner's native passive damping and
friction terms. The adapter restores those terms using the exact velocity used
when sending that command, then inverts the original full-rank 61-motor
transmission. Both the residual in unactuated joint directions and the next
command's agreement with the preceding delivered motor forces are checked. It
does not pass a requested command off as actual delivery.

At the first qualified aperture crossing, the independent opening audit freezes
before applying another physical step. A failed opening prefix stops traversal.
The post-opening planner uses the attained state and measured outward panel
normal; it can reject a newly attained endpoint whose static stow path collides.
Intentional left-hand unloading after this boundary does not erase or invalidate
the frozen opening result.

The completed episode requires actual whole-body passage and one quiet second:
the trailing robot geometry is past world y=0.2 m, horizontal root speed is below
0.03 m/s, both feet support at least 10 N, the hands are clear, left-hand normal
load is below 0.1 N, and XY excursion is below 3 cm. Original limits, loopbacks,
contact penetration, force caps and all right-pad validity gates remain active
through the final physical step. The runner checks these endpoint records
independently of the wrapper's completion flag and stops before applying an
extra command after the qualified finish.

## Evidence

| File | Meaning |
| --- | --- |
| `full-opening-report.json` | Frozen independent opening-prefix result |
| `opening-qualification.json` | Wrapper checks, 0.5-second contact window and exact attained-state hash |
| `continuation-handoff.json` | Actual handoff state and new post-controller initialization |
| `traversal-report.json`, `report.json` | Whole-episode result; success requires the opening prefix and final traversal checks |
| `traversal-steps.json.gz` | Every controller measurement on the unchanged global clock |
| `traversal-contacts.jsonl.gz` | Every occupied actual normal-contact and independent friction-patch slot |
| `traversal-contact-layout.json` | Body/filter names, motor order and slot interpretation |
| `acquisition-physics.npz` | Existing physics arrays plus actual motor/joint delivery, joint velocity, body origins and foot support |
| `full-sequence-steps.json.gz` | Existing approach/preparation contact and foot-motion evidence |
| `continuous-controller.json` | Detached handoffs, current state, opening receipt and any sticky failure |

Unused contact-buffer capacity is omitted from the streaming archive; all
occupied slots, source pair indices and original slot indices are retained.
Normal contacts and friction patches are separate arrays and are never paired
by buffer index. Their explicit time interval ends at the recorded integrated
pose. All prior pad, mechanism, trace, source and input files remain available.

Periodic `.partial` files remain explicitly incomplete. Graceful stop requests,
timeouts, failed delivery or controller/backend exceptions cannot create a
successful traversal report. A qualified opening can remain a valid prefix when
later continuation fails. On exception, the actual executed arrays, contact
stream and failure diagnostics are retained.

Recorded full-opening runs now create both the wide diagnostic camera and a
separate hand close-up, including near-handle starts. These are review cameras;
robot sensor cameras and the actor input contract are unchanged. Existing videos
do not acquire new views retroactively.

Run the CPU checks before the first live job:

```bash
python -m pytest tests/test_isaac_traversal_runner.py \
  tests/test_isaac_post_opening_measurements.py \
  tests/test_continuous_door_teacher.py -q
```

The tests exercise the real CLI prefix without launching Isaac, motor readback
and passive-term timing, independent passage reduction, frozen-prefix failure,
and the absence of new robot pose writers inside the physics loop. Consult the
[controller contract](CONTINUOUS_DOOR_TEACHER.md) before writing another adapter.
