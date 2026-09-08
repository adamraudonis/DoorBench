# One-second traversal adapter smoke

The prepared fixture uses source commit
`7016c2a80dc085470eba73c8d3e370449864aa71`. It has **not been uploaded or run**.
Its purpose is to check initialization, original motor limits, contact epochs and
actor-origin velocity in the new continuous Isaac adapter. A one-second trial
cannot qualify opening or traversal; its whole-task report must remain false.

The local package is
`/tmp/doorbench-continuous/out/continuous/traversal-smoke-spec-001/`:

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `spec.json` | Exact argv, 13 remote input hashes, all source hashes and acceptance criteria | `c4f30426de55255aef7d5f041b218446c769cc932d31fb7613b69acc66255e92` |
| `source.tar` | 481 committed source/config files; no generated door assets | `d400538fd06fa779eb2df4748045626e502366e032d54daad61f75af58896baa` |
| `verify_spec.py` | Hash verification only; cannot upload or launch | `0d7f103211a1ba85da368c54bf43b2d91cc10640a9e3f49c83b4c093b683ae7a` |

The real frozen runner parser passed on CPU up to the `AppLauncher` boundary,
with standard launcher argument registration stubbed. The source verifier
passed all 481 hashes and rejected a deliberately corrupted manifest. These are
CPU interface checks, not robot tests. The 13 runtime input files were hashed
through read-only SSH on the existing owned pod; no running job was changed.

## Frozen protocol

The proposed remote source is
`/workspace/continuous-traversal-smoke-source-001`; the fresh result directory is
`/workspace/continuous-traversal-smoke-runs/smoke-001`. The exact command is the
JSON `argv` array, avoiding reconstructed shell quoting.

It uses the original full-v2-003 walking reset, preparation, H1 checkpoint and
volar grasp profile, plus the original imported v2 robot and matching live door.
`--full-opening --traverse --seconds 1` constructs the full controller chain and
runs 500 original 2 ms steps. The explicit unused later-stage options are
`plain-v1`, 8 N palm load and 4 N transfer load. The old 3 N acquisition index
force override is omitted because full-opening rejects it; acquisition cannot
begin during this one-second fixture.

The left-palm targets and axial release are the Linux-validated source004 files.
Their constructor identity checks are exercised; their near-hand geometry is
**not** a claim that they fit the later walking endpoint. A longer run requires
its own actual-landed path validation. No failed older opening is promoted.

The prepared environment is reused. The owner must confirm an available GPU
slot and an active bounded teardown guard before launch. The source and result
directories must be new. After uploading and extracting the frozen source, run
`verify_spec.py --spec spec.json --source <remote-source> --verify-runtime-inputs`
with the asset Python. The verifier checks every source and explicit runtime
input hash and rejects an existing output directory. It does not allocate,
modify a guard, signal a process or execute the simulation.

## Interpreting the result

Smoke acceptance requires 500 complete integrated steps, finite state,
unchanged plant parameters, original joint/loopback/collision/cap gates, input
readback/transmission consistency, complete raw contact/state archives, and no
controller error or early stop. Record the reset contact epoch as `[0,0]`; it
cannot be used as executed contact evidence. Both foot loads must come from the
authored floor, with other contacts retained in the physical audit.

Compare the corrected actor-origin root state and the archived legacy mixed
state using the imported COM offset. Their poses and world angular velocities
must agree. Their world linear velocities must satisfy the rigid-body origin
conversion in [the API review](ISAAC_MEASUREMENT_API_REVIEW.md). Backend motor
readback means submitted actuation input, not independently measured torque.

Expect `passed: false`, `opening_prefix: null`, no controller failure, and
`declared_timeout_or_failure` in the whole-task result. Do not rename that report
a pass. Record a separate smoke acceptance receipt bound to its exact source,
inputs, raw evidence and limitations. The proposed per-process wall cap is
600 seconds; existing owned-pod guards remain unchanged.
