# Scripted handle acquisition with sensor-only balance

On Door55's frozen near-handle reset, the corrected H1/Shadow robot completed a
full scripted acquisition route and held an opposed five-distal-pad grasp while
balancing from encoders, local IMU and local touch. Native trial001 passed **25/25
checks**, including the original12-check grasp audit, over19 seconds /9500
physical2ms steps. This is an instance-specific joint script with sensor feedback,
not a learned or vision-based acquisition policy. The door stayed closed; the
handle was not intentionally actuated. No Isaac result is claimed here.

The qualified45% contact-free configuration and controller remain unchanged.
The new pure `ScriptedAcquisitionSchedule` selects the full static named-joint
path:1 second initial hold,16 seconds with quintic progression through the frozen
precurl route, and2 seconds settle. The30 torso/right-arm/right-hand joint goals
use the same26 motor coordinates,4x right finger feedback and0.05Nm*s/rad
moving-reference damping qualified in [SENSOR_REACH_BALANCE.md](SENSOR_REACH_BALANCE.md).
Maximum sampled goal speed is1.05518rad/s, inside the1.5rad/s bound. Pelvis/leg
and left-side targets remain fixed. All69 actual joint angles and all8 passive
loopbacks retain their original independent physical checks.

## Reproduce

```sh
python scripts/dexterous/probe_sensor_acquisition_balance.py \
  --robot /path/to/h1-shadow-loopback-v2.xml \
  --motors /path/to/h1-import.motors.json \
  --door /path/to/db0055_swing_single \
  --reference configs/dexterous/door55-precurl-v2/reference.json \
  --output out/sensor-acquisition-fresh
python scripts/dexterous/audit_sensor_acquisition_contacts.py \
  --trial out/sensor-acquisition-fresh \
  --output out/sensor-acquisition-fresh/independent-contact-audit.json
python scripts/dexterous/render_sensor_acquisition_balance.py \
  --trial out/sensor-acquisition-fresh --hand
python scripts/dexterous/render_sensor_acquisition_balance.py \
  --trial out/sensor-acquisition-fresh
```

The default frozen protocol is
[`sensor-acquisition-balance-v1.json`](../configs/dexterous/sensor-acquisition-balance-v1.json).
The source reference SHA256 remains
`ff02d42d9000aa6061d317668d3186c3d85ca7b5ee787602707f0d0088775b3e`.
Local native inputs are documented in the source-bound `provenance.json`.
Trial001 is at `/tmp/doorbench-continuous/out/continuous/sensor-acquisition-balance-001`.
It took64.4 seconds CPU wall time; no RunPod/GPU job was launched.

## Admission and evidence boundary

An independent, unstepped full-scene calculator screens1001 nominal poses before
physics. It permits hand contact only on the authored straight lever's original
distal volar pad surfaces: local-Y palmar face,2–40mm distal band,1mm axial margin,
and radial normal alignment>0.8. It rejects all other hand contact and any
contact during the formerly qualified45% prefix. Original joint bounds,
loopbacks and3mm nonfoot penetration tolerance are retained. The static screen
first encounters intended contact at99% of the route, with maximum nominal
penetration0.733mm. It does not predict actual contact loads or certify a grasp.

The runtime controller receives only the numeric sensor packet, local clock and
static named joint goals. Contact object IDs, real root/door poses, geometric
screen results and grasp labels never enter force inference. Scene data is
available only to reset, offline planning and evaluation. There are no active
root/joint/foot pose writes after reset, external support forces, direct door
commands, or physics changes. Contact-based safety termination is evaluator-owned.

Every physical interval retains the actual `mj_step` contact forces and their
matching pre-integration body frames, alongside separately timestamped integrated
joint/root states. All loaded RH lever patches must satisfy the original distal
surface contract throughout the episode. Other RH/LH contacts are rejected.
The final grasp window requires all five qualified pads>=0.2N, four-finger
pairwise radial alignment>0.5, thumb/finger dot<-0.5, and<=5% misplaced force per
digit, sustained for at least0.5 seconds. The source's conservative endpoint-tail
check includes251 samples. No thresholds from the earlier failures were relaxed.

`audit_sensor_acquisition_contacts.py` independently reconstructs the lever
transform and digit-local contact geometry from the archived actual body frames
and forces; it does not trust saved pad labels or rerun contact dynamics. All9500
classifications and pad-force totals matched exactly, with zero invalid loaded
patches. Tests include positive opposed pads and negative controls for dorsal
contact, endcap contact, missing force and an incorrect thumb frame.

## Native trial001

| Measurement | Result |
|---|---:|
| Main checks / original grasp audit |25/25 /12/12|
| First actual contact |15.428s|
| First qualifying interval |16.282s|
| Longest uninterrupted opposed grasp |2.628s, through19.000s|
| Final-window minimum FF/MF/RF/LF/thumb load |0.365 /0.491 /0.561 /0.606 /1.371N|
| Final-window minimum finger alignment |0.9820|
| Final-window maximum thumb/finger dot |−0.9460|
| Maximum torso tilt |0.33247 degrees|
| Pelvis height |0.870684–0.872467m|
| Maximum motor-coordinate error |0.01645rad, below0.04rad|
| Maximum individual nominal-angle error |0.09227rad, retained passive-split diagnostic|
| Maximum joint-stop / loopback violation |2.040 /0.188mrad|
| Maximum nonfoot penetration |0.05262mm|
| Palm vs offline nominal endpoint |3.267mm|
| Minimum final-second per-foot floor support |258.19N|
| Invalid loaded distal patches / unintended hand contacts |0 /0|
| QP failures / native warnings / external assistance |0 /0 /0|
| Maximum incidental leaf / handle angle |2.12e−7 /0.000235rad|

There were no failed physical candidates in this separate contact-enabled
experiment. The earlier contact-free001–004 failures remain preserved under
their original protocols. A test assertion initially expected bitwise decimal
arithmetic at a toy interpolated endpoint; it was corrected to numerical
comparison without changing the controller or physical trial.

Trial001 provenance SHA256:
`5fd1eb64fbae9ace7bd07f6a20e2f3d5de739bced8ca5c3ffff707357ebb5f44`.
Report SHA256:
`df86244124dd6615f770153c980fed87a5ff3f0bc7275cca0cd97ffd38a16536`.
The independent contact receipt is `independent-contact-audit.json`; actual
body/hand replay videos are `scripted-acquisition-body-az150.mp4` and
`scripted-acquisition-hand-az150.mp4`. Their final frames were personally inspected:
the four fingers and thumb occupy opposite sides of the lever, with the body
upright. Visual replay is presentation evidence; the actual contact archive is
the physical evidence.

The next separate experiment can attempt lever depression from this uninterrupted
acquisition. This result alone says nothing about resisting larger manipulation
loads, opening the leaf, walking through, reset variation or other doors.
