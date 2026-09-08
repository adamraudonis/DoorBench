# Right-hand release from the actual walked grasp

**Development only; no release fix is qualified.** The original
`AxialRightRelease` remains the default. These tests keep the v2 robot, original
motor caps, exact Door55 scene, and every anatomy/mechanical acceptance gate.

The continuous native `walking-opening-native-002` baseline acquires the handle
and loads the left palm correctly, but its right hand follows the springing
lever while sliding sideways with curled fingers. The resulting middle and
proximal finger contacts fail the declared pad contract. A geometric screen
that freezes a pressed lever does not certify release of a springing mechanism.
Close-ups from four exact archived pre-states, viewed from two angles, confirm
the contact path. They use the actual source door XML `5b12a0a1…`; an initial
render attempt using the older deep-stance door was rejected by the body-frame
consistency check.

## Isolated continuous comparisons

`probe_walking_release.py` unpacks the baseline's hash-verified source archive
and exact configuration, including the measured landed stance and left-arm
plan. It changes the release primitive in an isolated process. Every completed
raw chunk before the intervention must match the baseline byte-for-byte, or the
run stops. Equal chunks are SHA-verified and hardlinked to conserve disk space.
Both complete comparisons match **all raw evidence through 53.500 s**; the
release intervention starts at 53.504 s. There are no runtime pose resets.

| Variant | Observed result | Decision |
|---|---|---|
| Pressed handle frame bound to measured moving leaf; original 0.4 s preload decay | 18 invalid contacts across 10 intervals, including a 61.5 N thumb-middle impact at preload removal; original joint/motor/collision gates pass | Failed; later palm support is lost and aperture is reached by coasting |
| Same frame; retain original grip preload until measured 20 mm clearance, then decay over 0.4 s | 262 thumb-distal end/side contacts, 157 joint-stop violations, maximum 45.66 mrad | Failed; do not escalate preload |

The first variant reduces the baseline's roughly 1,200 invalid contacts but
does not solve the release. Neither trial passes the sustained palm, right
release completion, or full-opening requirements. The second is also physically
invalid under the unchanged joint-stop tolerance. Final aperture alone is not
success. [Frozen results and archive receipts](../results/dexterous/2026-09-08/walked-right-release-development.json)

## Development API and reproduction

`PressedLeafFrameRightRelease` is an explicit opt-in subclass. Before each
`begin` and `update`, call `observe_leaf(t, leaf_pose)` with the current measured
xyz/wxyz leaf pose on the same local teacher clock. It binds the attained
pressed handle frame once, then follows actual leaf motion rather than copying
the springing operator angle. It changes only analytic teacher targets; the
original force controller still emits the original capped motor commands.

Its optional `retain_grip_until_clear=True` keeps the original digit preload
until the caller's measured clearance freezes the palm goal. The test above
rejects that variant; it is retained solely for reproduction. A stale leaf
measurement is an error. The wrapper records its precise intervention and
source hashes separately from the baseline manifest.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/probe_walking_release.py \
  --source-run /path/to/walking-opening-native-002 \
  --output out/walked-pressed-release-new --seconds 66
# Reproduce the rejected grip-retention comparison by adding:
# --retain-grip-until-clear
```

## Controlled return was rejected by the measured workspace

An explicit `ControlledLeverReturn` development component preserves the actual
opposed grip and moves the operator target smoothly to rest over four seconds,
using the measured operator/leaf joint geometry. The left palm retains the
attained aperture. Reproduce it with `--release-mode controlled-return` in the
same isolated wrapper; the original release remains unchanged.

The 60 s continuous trial again reproduces the baseline raw prefix through
53.500 s. It maintains 0.08566 rad aperture and 3.727 N final left-palm support;
all original mechanical gates pass. However, the lever stalls at 0.2174 rad,
right-palm tracking error grows to 27.83 mm, and 2,106 intervals have invalid
middle-link contact patches. This is **not a qualified controlled return**.

Unstepped inverse-kinematics screens use the exact archived pre-release state
and original joint limits. With a frozen waist, the resting-lever target has
14.64 mm position and 4.16° rotation error and reaches the right elbow and wrist
limits. Allowing the waist and both arms reduces error to 0.43 mm / 1.64°, but
right wrist limits still bind. Allowing the left palm to rotate in its contact
plane or lowering its target does not remove that right-hand restriction.
These are geometric diagnostics, not physical trials. More grip force will not
resolve the required wrist pose.

The first attempt ended before release because the disk filled; its incomplete
prefix is retained. The recorder now compares compressed chunks in memory
before writing, so an equal walking prefix needs only its verified hardlink.
The successful recording and the interrupted attempt are archived separately
from the earlier release failures. [Trial, workspace and archive receipt](../results/dexterous/2026-09-08/controlled-lever-return-development.json)

The next bounded hypothesis is to open the fingers while withdrawing the palm
along a screened curve that respects the returning lever. The earlier measured
release came from an **unpressed canonical grasp**; its timing and 160 mm retreat
are proposals, not evidence that the actual walked, pressed grasp can release.

## A small physical body shift allows the lever to return

The next experiment succeeds at the **controlled lever-return component**.
It is still not a completed right-hand release or full opening.

A whole-body geometric path keeps the two attained foot frames and both palm
poses while returning the lever. Its final pelvis displacement is
`[+8.47, -14.55, +15.11]` mm, with a small orientation change. The 401-state
interpolation audit has at most 13.8 µm palm error, 0.35 µm foot error, 2.336°
torso tilt, and no forbidden collision. The robot COM remains inside the hull
of the measured initial foot contacts. This geometric evidence does not impose
physical foot constraints.

`WholeBodyLeverReturn` supplies the existing stance QP with those pelvis and
leg targets and the existing arm teacher with the screened torso target. The
physical trial retains all original motors, limits, contact geometry and
attained foot references. Its raw walking/acquisition prefix again matches the
original run through 53.500 s, before intervention at 53.504 s.

The 60 s native trial physically returns the lever to rest while retaining the
opposed grasp and left-palm support. An independent audit reads all 30,000 raw
transitions, checks motor caps and state continuity, and reconstructs the actual
body/contact frames for all 3,250 intervals after the intervention. Frame error
is exactly zero and no invalid loaded distal patch occurs. During the final
half second the operator stays within 0.630 mrad of rest and all five pads stay
loaded. Final left-palm load is 13.02 N and aperture is 0.09692 rad. Every
original mechanical, anatomy and stance check passes; the original full task
report correctly fails **usable aperture** and **right release completion**.
[Measured result and canonical archive receipts](../results/dexterous/2026-09-08/whole-body-lever-return-development.json)

```sh
# Regenerate the unstepped geometry path from the exact recorded state.
python scripts/dexterous/screen_whole_body_return.py \
  --source-run /path/to/walking-opening-native-002 \
  --state-run /path/to/walked-controlled-return-002 \
  --source-package /path/to/walked-controlled-return-002-source \
  --at 53.504 --output out/whole-body-return-screen

# Apply only motor targets in a fresh continuous physical comparison.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/dexterous/probe_walking_release.py \
  --source-run /path/to/walking-opening-native-002 \
  --whole-body-path /path/to/controlled-return-whole-body-path-001/report.json \
  --release-mode whole-body-return --seconds 60 \
  --output out/whole-body-return-repeat
```

These targets are tied to the actual walked state and its robot/scene inputs.
The adapter rejects a different attained root/joint configuration. Re-screen a
different robot, scene, or initial stance; this is a privileged teacher component,
not a vision/tactile actor.

## Reject disconnected release configurations

A shorter diagonal ungrip initially appeared feasible when each hand frame was
allowed to choose its own collision-free lever angle. Dense checking revealed
that these feasible sets were disconnected: near 53% progress only a pressed
lever fit, while at 54% only a resting lever fit. No continuous motion connected
them. `release_connectivity.monotone_release_path` now rejects this specific
failure mode; a returned sampled path still requires swept-edge refinement and
physical execution. Seventeen reachable fixed-timing curves and four searches
with independent hand, finger, thumb and operator progress produced no qualified
release. Those failures and the diagnostic plot are retained in the archive.
The unrun curved controller prototype was not installed as a release option.

The next trial must withdraw from the **actually attained resting-lever state**.
The measured canonical ungrip still exceeds the current fixed-waist arm reach,
so another bounded body movement or a newly screened hand path is required.
