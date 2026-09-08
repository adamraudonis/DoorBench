# Portable post-opening force controller

`PostOpeningTeacher` exposes the physically tested release/stow/rise/passage
chain through numeric measurements suitable for Isaac. It never receives an
active simulator, steps a model, writes a root/foot pose, or emits a door
command. Its separate authored MuJoCo scene supplies geometry and inverse
dynamics. It returns **61 original-capped motor forces** in motor-contract order.
This is privileged development control, not a vision/touch actor.

Two complementary native validations are complete:

| Validation | Result |
|---|---|
| Replay 009's actual measured states and preceding contact intervals | All 32,500 decisions match within 1.65e-12 Nm; 4/4 checks. |
| New physical rollout through the portable interface, trial 010 | 25/25 checks over 65 seconds; zero motor-delivery error and zero QP failures. |
| Trial 010 maximum torso tilt / joint-stop excess | 2.983° / 0.714 mrad. |
| Trial 010 final-second maximum speed / excursion | 0.00326 m/s / 1.68 mm. |

The physical trial still starts from the old attained opening endpoint. It is
an initialized continuation, not a continuous closed-door opening result or an
Isaac result. A different newly attained opening endpoint needs a fresh static
screen and physical qualification. See [continuation evidence and failures](POST_OPENING_CONTINUATION.md).

## Integration contract

```python
from doorbench.dexterous.post_opening_teacher import PostOpeningTeacher

post = PostOpeningTeacher(
    native_robot_xml, motors, stow_posture_config, original_h1_checkpoint,
    door_xml=native_door_xml, passage=True, inward_roll=0.07,
)

# First call only: the current attained state, never a saved robot pose.
first = dict(previous_motor_forces=last_actual_motor_forces,
             release_normal_world=actual_left_panel_outward_normal)

forces, info = post.force(
    t, root_state_w, joint_positions, joint_velocities,
    foot_loads_world_z, hand_body_forces_world,
    door_positions=door_joint_positions,
    door_velocities=door_joint_velocities,
    body_poses=actual_body_poses,
    evidence=actual_evidence,
    pose_time_s=t, contact_interval_s=(max(0.0, t - 0.002), t),
    **first,
)
first = {}  # Every later call uses the same controller and advancing clock.
# Apply only `forces` through the existing audited motor adapter.
```

| Input | Required meaning |
|---|---|
| `t` | Actual episode time, seconds; first call may occur at a nonzero time. |
| `root_state_w` | 13 values: world XYZ, quaternion WXYZ, world linear velocity, world angular velocity. |
| Robot position/velocity dictionaries | Exactly the 69 `motors['joint_names']`, without `robot/`; radians or metres and their rates. |
| Door position/velocity dictionaries | Exactly `leaf_hinge`, `leaf_handle_hinge`, `leaf_latch_bolt_slide`; actual joint positions and rates. |
| `actual_body_poses` | World XYZ/WXYZ **body-origin** poses for `left_ankle_link`, `right_ankle_link`, `lh_palm`, `rh_palm`, `leaf`, `leaf_handle`. Additional bodies are allowed. These are not sole poses or palm touch-site poses. |
| `foot_loads_world_z` | Actual world-up forces on the left and right feet, ordered `[left, right]`, from the preceding interval. |
| `hand_body_forces_world` | Actual normal-plus-friction world force ON every colliding hand body. Full prim paths or unprefixed body names are accepted; duplicate normalized names or missing colliding-body rows are rejected. |
| `previous_motor_forces` | First call only: the preceding actual 61 motor efforts in `motors['actuators']` order, within original caps. |
| `release_normal_world` | First call only: unit vector pointing out of the attained left panel face toward the hand. It is checked against actual leaf orientation and palm side. |

`actual_evidence` has exactly four fields:

```python
actual_evidence = {
    "physics_qualified": True,         # the independent active-plant audit
    "left_hand_contacts": count,       # actual touching contacts, not empty buffers
    "left_hand_load_N": normal_load,   # sum of nonnegative normal contact forces
    "right_environment_contacts": 0,  # RH against the scene; excludes robot self contact
}
```

Do not fabricate these fields. Preserve actual force-buffer epochs before
reading another reusable PhysX buffer. Hand force vectors include both normal
and friction contributions. A right-hand environment contact with nonpositive
gap or positive normal force blocks initialization; a positive-gap, zero-force
buffer entry does not. Valid small robot self contacts are covered by the
independent penetration/joint checks and do not masquerade as a door grasp.

All poses and joint values describe the current endpoint `t`; all force
observations describe the completed interval `[t - 0.002, t]`. Only an episode
starting at `t=0` uses the zero-duration initial contact observation. At a later
opening-to-continuation handoff, preserve the real preceding interval. Never
replace it with `[t, t]` or reset the global clock.

## Initialization and controller history

Create the controller once per attempted continuation. The first call requires
both feet to support at least 10 N, root linear speed at most 0.1 m/s, the right
hand clear of environment contact, and an attained leaf angle of at least
1.2 rad. Actual feet/palms/mechanism poses must agree with authored FK within
3 mm and 0.02 in rotation-matrix norm. The motor contract must match the exact
corrected native XML, original gains/transmissions/control/force caps, 69 joints,
61 motors and eight loopbacks. The original H1 checkpoint is verified on load.
The three-joint Door55 layout and 2 ms timestep are explicit current limits.

The first call generates and screens a new 404-state stow path from the actual
root and joints. It does not use a saved reference root or a previous episode's
path. The `stow_posture_config` is the upper-body neutral target from
`configs/dexterous/door55-readiness-v2/body-reset.json`; its saved initial root,
leg configuration and approach waypoint are not installed in the plant.
A failed route stops initialization. Save `post.plan` and the failure receipt.

The landed-foot QP references the **actual measured ankle positions and
rotations** at handoff. These are planning targets, not physical anchors.
Measured hand loads enter inverse dynamics about body origins; contact moments
remain the same bounded approximation used in 009. Only motor efforts leave
the controller. Maintain all active physics/collision/motor checks independently.

Call `force` once per 2 ms tick. Preserve the instance's stage clock, QP warm
start, last bounded effort, H1 policy history, waypoint state and recorded gait
shape samples. Initialization arguments on a later call are rejected. Do not
recreate the instance each frame, jump over time, or pass duplicate samples.
The local continuation clock starts at the first call; the global episode
clock remains unchanged. The H1 policy initializes its own history only when
rise hands the legs to gait control, approximately 23 seconds into this chain.
No history from an earlier walking phase is required because stow and rise
precede this new gait handoff.

Save `post.initialization`, `post.plan`, returned diagnostics and the active
500 Hz state/contact/motor records. `post.navigator.screens` and `.events`
contain the passage screens and milestones. `info['passage_completed']` marks
the waypoint controller's whole-body crossing/slow-state event; it does not
replace the final audit. Continue the same controller until the evaluator has
a full quiet second with maximum horizontal speed and excursion each below
0.03 in SI units, and the entire actual body clear of the frame. Stop and retain
a failed run on any active safety failure or raised contract/geometry error.

The authored scene currently assumes the same Door55 world placement as the
prepared asset. Changed scene transforms, robots or collision import settings
must be explicitly adapted and validated; mismatch is rejected, not ignored.
In particular, actual PhysX collision checks remain authoritative even when the
unstepped native clearance estimate passes.

## Native reproduction

Add `--portable` to the [documented 65-second continuation command](POST_OPENING_CONTINUATION.md).
The probe supplies only numeric measurements to the controller and records its
actual initialization. It captures real transition forces before refreshing
endpoint kinematics, with no pose writes after reset.

```bash
PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/replay_post_opening_teacher.py \
  --trial out/post-opening/native-009 \
  --output out/post-opening/my-portable-replay --seconds 65

PYTHONPATH=. "$NATIVE_PYTHON" scripts/dexterous/audit_post_opening_record.py \
  --trial out/post-opening/native-010
```

The replay replaces all `mj_step`, `mj_step1` and `mj_step2` entry points with
errors. It is a history/equivalence test on recorded measurements, not a second
physical success. Replay 001's initial rejection is retained: it initially
confused four valid right-hand self contacts with environment contact. Replay
002 checked two seconds; 003 and the source-frozen 004 checked the full sequence.
The corrected contract explicitly distinguishes those contact classes.
