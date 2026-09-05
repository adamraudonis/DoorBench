# Constrained motion references

DoorBench's new motion pipeline plans an original articulated adult around the real door collision geometry. It authors footsteps, reach and release phases, solves whole-body inverse kinematics, and adjusts timing to satisfy sampled joint-speed and acceleration limits. A separate validator recomputes the motion from saved joint coordinates.

This is an experimental research component. **Kinematic acceptance does not establish grasp forces, dynamic balance, causal opening, or success within the original benchmark time limit.** Native door movement comes from the frozen scripted-hand recording. Read the [scope and mechanism review](research/planned-reference-scope.md) before treating these trajectories as demonstrations.

The [first complete 1,000-door run](research/planned-reference-baseline.md) accepted 61 traversals and 51 locked-door checks, rejected 404 candidates, and left 484 cases unresolved. Its [experimental release format](research/planned-reference-release.md) preserves every door's status and serves only accepted trajectories for playback.

## Use it locally

The reference extra pins the tested MuJoCo, Mink and DAQP versions. Download the original `assets` and `reference-motions` components following the [release guide](DATASET_RELEASE.md), or use the existing local corpus.

```sh
.venv/bin/python -m pip install -e '.[reference]'
.venv/bin/python -m doorbench.reference.solve \
  --assets assets --recordings out/reference-motions \
  --doors db0079_sliding_single --out out/reference-planned
.venv/bin/python scripts/validate_planned_reference.py \
  --assets assets \
  --clip out/reference-planned/db0079_sliding_single/clip.json \
  --trajectory out/reference-planned/db0079_sliding_single/trajectory.npz \
  --out out/reference-planned/db0079_sliding_single/validation.json
```

Generate a corpus with independent validation and explicit per-door results:

```sh
.venv/bin/python scripts/build_planned_reference_corpus.py \
  --assets assets --recordings out/reference-motions \
  --doors all --out out/reference-planned-corpus --workers 4
```

The output must be separate from inputs. Exact matching attempts can resume; `--force` archives previous attempts before replacement. The index binds the native recording index, input resources, generator files, runtime versions and output checksums. Failed native recordings, unsupported routes, solver rejection and execution errors remain distinct reasons. None proves that a human cannot use the door.

Each door runs in a fresh Python process, with a bounded timeout and its own
stage log. Native solver crashes cannot break the remaining queue. The default
`smooth` gait filters only soft body guidance before IK, preserves authored foot
contacts, and prefers an upright reachable stance. All derivative and geometry
checks run on the final motion after retiming. Some clips remain deliberately
slow; passing those checks does not establish natural timing.

## What is checked

The next generator revision adds three conservative improvements. Eligible powered sources wait and traverse with both hands inactive; their source poses remain prescribed and sensor causality is unverified. Eligible already-unlocked sliding hasps use exact joint/body/site/geometry roles, with a held native pose during withdrawal, repositioning and re-reach. The solver also checks joint-space interpolation during intervals with an unchanged native pose and no active grip at either endpoint. Moving-door and grip intervals remain subject to the independent validator.

These additions do not alter the published first baseline. Every new candidate must pass a fresh independent report; neither successful pilot status nor a stronger generation check is copied into acceptance.

| Check | Evidence |
|---|---|
| Fixed original rig | 38 position coordinates, 37 degrees of freedom; unchanged link geometry and joint ranges |
| Planted feet | Fixed ankle position and orientation; all sole corners supported by the authored floor |
| Hand target | Geometric surface proximity and position tracking, with narrowly named contact exceptions |
| Clearance | Actor self-collision and native collision geometry; saved frames and subdivided joint interpolation |
| Smoothness | Sampled actor speeds and accelerations; root angular acceleration expressed in world axes |
| Actor route | Fresh forward kinematics crosses the declared aperture and reaches the goal |
| Provenance | Original source, trajectory and validation checksums; incomplete clips cannot pass |

The validator's report records all numeric thresholds and the interpolation sampling resolution. Its result is independent of solver success flags. `accepted_kinematic` means these declared checks passed; personal visual review is separate. The [all-door audit of the earlier overlay](research/reference-feasibility-baseline.md) explains why that distinction matters.

The standard rig has a neutral head top of 1.68 m, a rigid 20 cm head, finite feet and spherical hand proxies. It has no fingers. Some apertures are intrinsically too small for that head; other cases need crawling, climbing, access equipment or a better planner. Full dataset coverage includes honest unresolved cases rather than silently changing the actor or the requested task.

Compare complete revisions with matching native inputs, original rig and independent gates:

```sh
.venv/bin/python scripts/compare_planned_reference_corpora.py \
  --before out/reference-planned-corpus-v1 --after out/reference-planned-corpus-v2 \
  --out out/reference-comparison/v1-v2.json
```

The comparison verifies complete artifact inventories and bound acceptance reports, lists both new acceptances and regressions, and separates source-task counts from durations. It rejects pending runs, changed native inputs, and changed validation thresholds. It does not infer visual or dynamic success.

## Read the arrays

`trajectory.npz` is loadable with `numpy.load(..., allow_pickle=False)`. `actor_time` is the authoritative clock. `proposal_time` preserves timing before the local derivative adjustment; `native_time` maps each frame to the immutable source recording. The nominal `fps` field controls rendering, not necessarily uniform sample spacing.

`actor_qpos` and native `qpos` are authoritative coordinates. Body poses, the 16 landmarks, feet, hand targets and contact masks support replay and inspection. `clip.json` records the original actor MJCF, geometry transforms, joint-coordinate mappings, source hashes, phases and timing metrics. No original generalized force or benchmark outcome is transferred to the new clock.

## Review in Motion Lab

The local Motion Lab provides playback, scrubbing, per-door failure reasons and
a brown-door/gold-hardware inspection view. Export a completed corpus, then run
the viewer as described in [human review](HUMAN_REVIEW.md):

```sh
.venv/bin/python scripts/export_planned_reference_web.py \
  --corpus out/reference-planned-corpus --assets assets \
  --out out/planned-reference-web
```

To review an existing prepared release directly, run this command from the
repository root. It reads the release's `web/` directory and the matching local
`assets/` without copying, exporting or uploading anything:

```sh
DOORBENCH_PLANNED_WEB_ROOT=out/planned-release/planned-v2026.09.05.2/web \
VITE_PLANNED_REFERENCE_INDEX=./planned-references/index.json \
  bun run --cwd viewer dev --host 127.0.0.1 --port 5175 --strictPort
```

Open <http://127.0.0.1:5175/#/motions>. This is an unpublished local preview;
the public Motion Lab remains on its pinned release. Relative
`DOORBENCH_PLANNED_WEB_ROOT` paths resolve from the repository root; absolute
paths also work. Omitting the variable keeps `out/planned-reference-web`.

Open `#/motions` in the local viewer. Visual notes use the exact served clip
checksum, so changing a motion does not transfer an old review to it. Pass,
needs-work, issue tags and notes can be exported/imported locally. This review
does not change the independent kinematic result. The left/right arrow keys
navigate doors when a text or playback control is not focused.

## Inspect in Blender

Prepare an appearance job as described in [Reference motions](REFERENCE_MOTIONS.md#animated-blender-scenes), then export the checked rig rather than the old overlay:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/blender_planned_motion.py -- \
  --job out/blender-reference/job.json \
  --clip out/reference-planned/db0079_sliding_single/clip.json \
  --trajectory out/reference-planned/db0079_sliding_single/trajectory.npz \
  --validation out/reference-planned/db0079_sliding_single/validation.json \
  --out out/blender-reference/planned.blend \
  --render-time 20 --image out/blender-reference/planned.png
```

The exporter verifies the report against its exact inputs and retains the original clip bytes. All poses are keyed directly on `actor_time`; packed scenes contain the exact rig surfaces and source door body motion. A report-backed caption states only that sampled kinematic checks passed. Between-keyframe Blender interpolation is for visualization; the stored coordinates and independent audit define the checked motion.

## Research and next steps

The architecture is informed by [human-object animation and retargeting research](research/motion-generation.md) and [constrained whole-body planning](research/kinematic-planning.md). It uses original motion primitives and geometry; no restricted motion-capture datasets, human body models or learned weights are redistributed.

Stronger manipulation references need explicit palm-to-hardware orientation, grasp/regrasp sequences, bimanual coordination, keypad or credential actions, and support for closers and powered mechanisms. Causal demonstrations additionally need contact-driven simulation and balance control. Those claims are deliberately separate from this kinematic route component.
