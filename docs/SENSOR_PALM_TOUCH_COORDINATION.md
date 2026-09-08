# Local-touch palm redistribution experiment

This opt-in controller retains the original scripted acquisition and press
route, sensor-only balance, and index/finger preload settings. It adds a tiny
palm translation driven by opposing local tactile loads and the robot's own
joint kinematics. No runtime door state, root pose, object identity, scene
collision query or active-plant handle enters the controller. It is an
instance-specific scripted component, not a learned or vision-based policy.

## Attained geometry and bounded screen

The [index coordination trial](SENSOR_INDEX_TOUCH_COORDINATION.md) reached
29.8 degrees before pausing with weak index projection and strong thumb load.
Its final original opposed distal-pad contacts remain valid. At that actual
contact interval, the average four fingertip palmar normals project positively
onto the index/middle/ring/little outward contact normals (approximately
0.978/0.994/0.984/0.930) and negatively onto the thumb normal (-0.862). A palm
translation in this direction therefore closes the four-finger side while
opening the thumb side in the detached geometry. This is a directional
hypothesis, not a prediction of actual force or grasp stability.

The direction is computed from the four own-robot distal body rotations and
their fixed local palmar `-Y` axes. At this attained state it is approximately
`[-0.104,0.878,0.467]` in the palm frame. Inference recomputes it from encoders
and the declared nominal arm goals; no archived contact normal is supplied.

The proposed displacement is limited to **0.25 mm**. A detached screen covers
51 attained-state increments plus 23,232 nominal combinations of 121 press
poses, all prior five-digit closure/index-offset corners, and three palm
displacements. It passes with maximum nominal nonfoot penetration 2.890 mm,
below the unchanged 3 mm bound. The largest required joint correction is
0.000870 rad. Position/orientation solve residuals pass the explicit 2 micrometer
and 20 microradian geometric tolerances. These kinematic tolerances do not imply
comparable physical tracking precision.

The screen is retained at `out/continuous/sensor-palm-plan-001/screen.json`.
It does not certify continuous configuration volumes, contact loads, or the
actual robot trajectory. Original 500 Hz physical checks remain mandatory.

## Declared feedback and own-robot solver

After 19 seconds, the controller integrates
`thumb_projection/3 - mean(four_finger_projections)/2` using the preceding
accepted decision's filtered local tactile values. Gain is 0.00005 m/s per
normalized load error, rate is limited to ±0.0001 m/s, and displacement to
[0,0.00025] m. These 2 N and 3 N normalizers are the existing finger and thumb
targets; all progression thresholds remain unchanged. Negative error retreats
the additional translation. The extra 2 ms feedback delay is explicit.

`RobotPalmTranslation` owns a separate unstepped robot-only data object. Pelvis
coordinates are a fixed arbitrary gauge. Other joint angles come from encoders;
the eight torso/right-arm/wrist nominal targets come from the declared route.
A bounded Jacobian solve retains the nominal palm orientation while applying
the small translation, and enforces original arm joint limits. It never steps
dynamics, queries object geometry or edits the active model. Its output only
adjusts those eight joint goals. The existing adapter then checks goal slew and
returns the original 61 capped motor forces, owning previous-action history.

All initial 19-second commands must remain identical, and the unchanged grasp,
anatomy, collision, joint/loopback, cap, warning, balance and handle/latch gates
determine success. The current profile is a single new finite 36-second native
comparison; it introduces no additional stiffness change.

Add these flags to the complete preceding index trial command, with a fresh
output directory:

```sh
--palm-protocol configs/dexterous/sensor-palm-touch-v1.json \
--palm-screen out/continuous/sensor-palm-plan-001/screen.json
```

The runner verifies and snapshots the matching screen, source closure and all
profile bytes. Existing stationary, reach, acquisition, tactile, index and
stiffness profiles remain available unchanged when this option is omitted.
