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

The next hypothesis is a **controlled return of the lever to rest while the
opposed grasp and left support are retained**, followed by an actual-state
withdrawal plan. It must qualify the return and moving-finger geometry first;
these rejected axial routes cannot be reused as evidence for it.
