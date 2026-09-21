# Isaac Sim on the local Windows GPU

`scripts/isaac/run_local_windows.py` runs the existing H1/dual-Shadow import,
one-second readiness check and 14-second walking/stopping fixture directly on
the local GPU. It uses explicit Windows Python paths and does not invoke SSH,
RunPod, allocation scripts or Linux environment wrappers.

Use the `codex/isaac-ready-integration` development code as the migration
baseline. Its latest qualified Isaac result is run042: standing acquisition,
lever operation and about 4.9 degrees of held opening. Transfer046/047 failed
the sustained-grasp and receiving-palm gates. No qualified complete upright
Isaac opening/traversal or learned sensor-only door policy exists yet.

## Prepare the local assets

Use separate Python environments for native asset generation and Isaac Sim.
Isaac Sim 5.1 and Isaac Lab 2.3.2 are the historical simulator versions; record
the installed versions and validate this destination independently. Keep the
runtime and output paths short enough for Windows package and USD paths. This
machine maps `I:` to `C:/Users/adamr/Documents/Codex` using Windows `subst` so
Isaac packages install and load through short paths without an administrator
change to the system long-path policy. The alias is not persistent across
reboots. Restore it with `subst I: C:\Users\adamr\Documents\Codex` after
checking that `I:` is still unused. Do not replace an existing unrelated drive.
The launcher preserves the supplied interpreter alias instead of resolving it
back to the longer physical path; output paths remain physical paths.

The following PowerShell commands prepare the corrected robot and one door.
Adjust only the two interpreter paths and repository path for another machine.
These commands do not initialize the GPU simulator.

```powershell
Set-Location C:/Users/adamr/Documents/Codex/DoorBench
$assetPython = 'C:/Users/adamr/Documents/Codex/doorbench-env/Scripts/python.exe'
$isaacPython = 'I:/isaac-env/Scripts/python.exe'
$env:PYTHONPATH = (Get-Location).Path
& $assetPython scripts/dexterous/setup_robot.py --mechanics-profile shadow-loopback-v2 --output out/local-ready/h1-shadow-loopback-v2.xml
& $assetPython scripts/generate_dataset.py --out assets --ids db0055_swing_single --workers 1 --no-thumbs
& $assetPython scripts/dexterous/prepare_isaac_import.py --robot out/local-ready/h1-shadow-loopback-v2.xml --output out/local-ready/h1-import.xml
& $assetPython scripts/isaac/make_smoke_reference.py --robot out/local-ready/h1-shadow-loopback-v2.xml --output out/local-ready/reference.json
```

Check each exit status before proceeding. `setup_robot.py` downloads pinned
HumanoidBench assets at `cb1189039151c8aadaaa987b442da54383c87fab`.
The explicit `shadow-loopback-v2` flag is required: the historical script
default creates the older hand without the corrected passive loopback model.

## Print and execute local commands

The launcher prints an argument array by default; `--dry-run` is equivalent.
It creates no files and does not initialize Isaac in that mode. Add `--execute`
to perform the stages. Every execution requires a fresh output directory.

```powershell
& $assetPython scripts/isaac/run_local_windows.py --asset-python $assetPython --isaac-python $isaacPython --stage all --output out/local001
& $assetPython scripts/isaac/run_local_windows.py --asset-python $assetPython --isaac-python $isaacPython --stage all --output out/local001 --execute
```

For initial diagnosis, execute import and smoke separately. Use the printed
paths for the imported USD and subsequent readiness receipt:

```powershell
& $assetPython scripts/isaac/run_local_windows.py --asset-python $assetPython --isaac-python $isaacPython --stage import --output out/import001 --execute
& $assetPython scripts/isaac/run_local_windows.py --asset-python $assetPython --isaac-python $isaacPython --stage smoke --robot-usd out/import001/import/robot.usda --output out/smoke001 --execute
& $assetPython scripts/isaac/run_local_windows.py --asset-python $assetPython --isaac-python $isaacPython --stage walking --ready out/smoke001/ready.json --output out/walking001 --execute
```

Walking downloads the pinned official H1 checkpoint when `--checkpoint` is
omitted and verifies its hash in the existing policy adapter. `--no-walking-video`
skips only walking camera output. The readiness stage keeps camera recording
and uses `backend-dry-v2`; the historical standalone walking fixture still
uses its original explicit damping/tanh friction adapter. Its result is a
separate locomotion component check, not parity with the modern manipulation
profile. The native hand's passive loopback mechanics remain present in both.

`plan.json` records exact commands. `manifest.json` records model/input hashes,
source revision and entrypoint hashes, local GPU/driver and both Python package
versions. `result.json` records each process ID, exit status and report hash;
the adjacent stage logs retain simulator output. A failed stage stops the
sequence and keeps its evidence. The default wall-time limit is two hours per
stage and can be changed explicitly with `--timeout-seconds`. Runtime versions
are collected without importing Isaac or printing credentials.

## Continue from readiness to door manipulation

The readiness reference moves a wrist while standing; it is not an acquisition
reference. The historical run042 recipe used an archived
`tangent038-reference.json`, which is not tracked. A fresh local alternative
is reconstructible from `configs/dexterous/door55-standing-acquisition-v1.json`:
run `rescreen_acquisition_reference.py` against the prepared robot and door,
then use the new reference and its geometry audit. This does not reproduce
run042 or qualify its missing measured-state transfer route.

The first fresh Windows native trial completed 36 seconds and opened the door
about 5.1 degrees, but failed the original distal-only sustained grasp check.
Its independently verified raw contacts included inner middle-finger surfaces.
The existing `volar-phalange-v1` anatomical contract can be explicitly selected
for a **new** trial with `--grasp-profile` and a source-bound
`--grasp-profile-definition`; all force, opposition and physical requirements
remain unchanged, and the original distal score is retained. The native probe
requires full transition recording for this selection. Its independent auditor
reads the frozen selection from that trial; it cannot override a historical
trial's declared contract. Do not relabel the first failed trial as passed.

For the portable controller arguments see
[the standing-transfer recipe](ISAAC_STANDING_TRANSFER_RECIPE.md). Its Linux
coordinator hardcodes `asset-venv/bin/python` and `venv/bin/python` and uses
POSIX process-group signals. Do not execute that coordinator unchanged on
Windows. The local launcher deliberately exposes only the currently portable
import/readiness/walking sequence. `run_local_operation.py` provides a separate
Windows-compatible operation launcher from verified local readiness and a
qualified native operation. It verifies the physical, contact and whole-handle
receipts and their source/input hashes, preserves failures, and invokes the
independent Isaac contact auditor. Keep physical success gates and original
motor/door limits when performing new experiments.

Declare the existing anatomy contract on a newly prepared local robot with:

```powershell
& $assetPython scripts/dexterous/declare_local_grasp_profile.py --robot out/local-ready/h1-shadow-loopback-v2.xml --motors out/local-ready/h1-import.motors.json --output out/local-ready/local-volar-profile.json
```

Use a fresh output; the declaration does not certify any recording. Once a
new native trial and its independent audits pass, preview and run Isaac:

```powershell
& $assetPython scripts/isaac/run_local_operation.py --asset-python $assetPython --isaac-python $isaacPython --ready out/local-isaac-smoke-004/ready.json --native-trial out/local-native-volar-001 --output out/local-isaac-operation-002 --seconds 24 --open-on-latch-clear --review-render-profile native-materials-v1
# Add --execute to the same command to run it, using a fresh output directory.
```

The local standing reference reaches the release angle before the original
five-second press timer finishes. The first Isaac operation missed this window;
the explicit `--open-on-latch-clear` controller option begins opening when the
original measured angle and latch thresholds are satisfied. It does not lower
those thresholds. The render option restores intended body materials in a
session layer and uses subdued lighting. An actual 14-second physical prefix
was byte-for-byte identical with and without that rendering change.

An explicitly new target-engine experiment can additionally set
`--experimental-grasp-offset-in-handle-m X Y Z`. The vector must remain within
the original 10 mm reference bound. Its receipt records both the qualified
native baseline and the changed PhysX parameter; native qualification is not
inherited by the changed controller. Every actual Isaac physical/contact check
must still pass. This is useful when contact geometry differs between engines.

`plan_local_standing_transfer.py` can construct a fresh left-hand path from a
qualified native recording and the tracked numeric left-palm preferences.
It verifies the exact recorded state, model and independent audits, then runs
the original dense geometry screen. Failed-source development is explicit
and cannot emit a runtime route. Passing geometry permits a physical trial;
it does not prove palm support or a completed handoff.

For a qualified **Isaac** endpoint use `plan_local_isaac_transfer.py`, which
extracts the recorded root, joints, velocities and body poses and retains the
original destination kinematic admission. A native route supplies numerical
left-hand preferences only. To execute the resulting route, add
`--standing-transfer-source out/QUALIFIED-ISAAC-SOURCE` and
`--standing-transfer-route out/FRESH-ISAAC-PLAN/transfer.json` to the same local
operation command. Use at least the source duration plus 8.5 seconds.
The launcher requires the original controller arguments and source files,
derives the handoff time from the actual qualified source epoch, and reruns the
physical prefix from its original initial state. It does not reset the robot at
the handoff. Qualification additionally requires an exact archived state,
velocity and commanded-motor prefix and an independent palm-support audit.
This stage still establishes only receiving support, not release or traversal.

### Bounded late thumb reference experiment

The local launcher also accepts `--experimental-thumb-reference-joint rh_THJ5`
and `--experimental-thumb-reference-offset-rad .02`. This is a declared change
to one nominal motor reference, within the original authored joint range. It
waits for measured partial opening (0.075–0.10 rad), operator rest (±0.05 rad)
and latch rest (±0.001 m), then uses a one-second quintic ramp. It does not set
the physical joint angle, increase the motor cap, or change contact criteria.

The first 24 s trial, `out/local-isaac-thumb-001`, used palm offset
`--experimental-grasp-offset-in-handle-m .004 -.0025 .0025` and the above thumb
change. It failed the final sustained-grasp check, while passing the other 18
operation checks and independent contact accounting with zero invalid loaded
surfaces. Its final 155/251 grasp samples were valid; the last contiguous window
was 23.692–24.000 s, only 0.308 s elapsed. The failed result is retained.
`independent-thumb-response.json` and its accompanying analysis separate the
nominal target, actual joint motion, motor command and individual contact pairs.
The fresh `out/local-isaac-thumb-002` repeated the same recipe to 28 s and
passes all 19 operation checks plus independent contact/task auditing of all
14,000 intervals with zero invalid loaded patches. Brief earlier grasp
interruptions remain in the record. This qualifies partial opening, not
release or traversal. Its actual endpoint produced the new
`out/local-planning/isaac-qualified-transfer-001/transfer.json` after all 1,001
geometry samples and exact measured-body kinematic admission passed.
The new `out/local-isaac-transfer-001` tests a 42 s episode: the same first 28 s
followed by 14 s for left-palm receiving support. Read its physical result
independently; passing route geometry does not establish palm load.

Both 42 s receiving-palm trials completed normally but failed sustained right
grip and final palm support. All loaded handle patches remained on the declared
surfaces. Transfer002 adds only `--standing-transfer-hybrid-support`; its
physical states and commands match transfer001 through 38.440 s, immediately
after actual palm contact admitted the feedback blend.

| Final half-second, 251 samples | Transfer001 | Hybrid transfer002 |
| --- | ---: | ---: |
| Palm load at least 2 N | 173 | 241 |
| Mean palm load, N | 2.536 | 3.957 |
| Minimum palm load, N | 0 | 0.865 |
| Valid opposed right grip | 247 | 248 |

After the hybrid blend finishes, no simultaneous 0.5 s palm/grip hold exists;
the longest is 0.200 s of interval coverage. The final half-second support dips
remain short and are not steadily disappearing. The next prospective trial
uses `--standing-transfer-support-load-target 6` with the same 42 s duration,
hybrid feedback, source and route. This changes the target only; original motor
caps, anatomy, support threshold and hold length remain unchanged. It is a
force-margin hypothesis, with no claimed physical success before the new audit.

`--standing-transfer-live-prefix-witness` permits a new controller implementation
while preserving original source qualification and captured historical code.
It compares every newly recorded 500 Hz state, velocity, body pose and commanded
motor vector to the qualified source, including exact dtypes and bytes. At the
source's terminal epoch it must authorize stage entry before the first transfer
command. A mismatch stops the episode; the archive is never used to set physical
state or issue saved commands. The result is saved in `live-prefix-witness.json`
and required by both the physical report and launcher. Current source files are
captured separately and must remain unchanged during execution. The default
launcher retains its stricter unchanged-source prerequisite unless opted in.

## Verified destination and Windows fixes

The local destination is Windows 11 with an RTX 5070 (12 GB), driver 591.86,
Python 3.11.9, Isaac Sim 5.1.0, Isaac Lab v2.3.2 and PyTorch 2.7.0+cu128.
The CUDA tensor check, native import and actual PhysX readiness trial passed.
Use separate native and Isaac environments: `usd-core` belongs only in the
native asset environment, because Isaac supplies its own USD runtime.

On this Windows setup MuJoCo must be imported **before importing**
`isaaclab.app.AppLauncher`. The reverse order raises WinError 1114 even before
SimulationApp is constructed. `isaac_opening.py` preloads MuJoCo on Windows.
Kit can return process status 0 after a Python exception, so the launcher also
requires actual stage artifacts and rejects `error.txt`; process exit status
alone is not readiness evidence.

The first actual local walking fixture passed all 16 checks: 2.652 m displacement,
2.646 degrees maximum torso tilt, 0.00852 m/s final-second speed and 3.83 mm
final-second excursion. It is a plane walking/stopping component, not a door
traversal. No runtime root pose writes or external wrenches were used.

## Astra and Jev experiment

Astra authors a phase plan in `configs/isaac/astra-jev-press-plan-v1.json`.
Jev 1.13.0 receives structured measured contact/load and balance summaries,
and proposes `continue_press`, `pause_press` or `request_astra`. It is text-only;
these trials do not use camera perception. The 500 Hz motor feedback continues
while an asynchronous request is pending. Every physics step rechecks local
physical admissibility, phase, reply confidence and both wall/simulation age.
Expired or inadmissible replies pause reference progression. Resuming uses a
200 ms smooth progression ramp; there is no target-clock catch-up jump.

Set `TYPESAFE_API_KEY` in the process environment, then opt in with
`--jev-progress-plan configs/isaac/astra-jev-press-plan-v1.json` and
`--jev-sample-period .2`. The standalone native probe additionally requires
`--portable-wrapper --record-transitions`. Credentials are not recorded.
The default path makes no API calls.

`configs/isaac/astra-jev-press-plan-v2.json` is a separate prospective Isaac
experiment. Its explicit `progress_policy: local_guard_with_jev_advice` permits
fresh full local admission when no fresh model directive is available. A fresh
Jev pause retains its original lease through a missing reply or provider error;
an accepted request for Astra remains latched until the episode or versioned
plan changes. The arbitration class supports this change at its boundary, but
the standalone runner reads one plan at launch and has no live Astra-plan inbox.
Editing its plan file does not clear a running escalation latch. Current contact,
load, balance, mechanical and clock checks still
apply to every step. Records distinguish model decisions, retained model pauses,
local fallback and Astra escalation. The original v1 plan still requires Jev.
The new policy has CPU tests but no completed physical trial yet. Use the existing
`--jev-progress-plan` option to select it; no new runner option is required.
The native probe rejects this policy explicitly because its adapter currently
supports only the original model-required mode.

The operation launcher accepts the same Jev plan and sample-period arguments
for standalone Isaac pressing. The opt-in gate consumes solved PhysX contact
intervals and records unique model replies separately from reused controller
submissions. Transfer, full-sequence and sensor-policy modes remain separate
until their model-control interfaces have been integrated and tested.

The first native paired trial consumed 109 real replies (108 continue, 1 pause),
with 209 ms mean and 1094 ms maximum latency. Its initial 5301 physics intervals
exactly matched the baseline in positions, velocities, commands and forces.
The gate then changed motion. Pressing took 6.578 s instead of 5.000 s; both runs
released the latch and held about 0.09 rad opening, and both failed the final
distal-grasp gate. Some velocity/acceleration measures improved, while invalid
surface contact lasted longer and the peak index load increased. One paired
episode does not establish model superiority; API delays and the resume ramp
also affect the result. The comparison counts unique replies separately from
reused per-step decisions.

The first local Isaac paired trial produced 1,359 parsed model replies
(1,351 continue, eight pause), all matching the request-time local comparator.
At least 146 explicit provider errors were recorded. Its 89 resumptions and
existing smooth ramps removed 6.61 s of reference progress over the 24 s trial.
The final grasp passed, but the maximum handle angle was 0.79734 rad, below the
unchanged 0.80 rad opening trigger, and the door did not open. The different
amount of completed motion prevents interpreting its lower motion measures as
a better opening policy. Provider failures now use nonblocking exponential
retry backoff from 0.25 to 2 wall seconds; logs retain the original evaluation
reason separately from later permission expiry. These transport fixes have
synthetic tests, but were not part of that already-recorded experiment.

API references: [TypeSafe API](https://docs.typesafe.ai/api),
[Jev 1.13 model limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
