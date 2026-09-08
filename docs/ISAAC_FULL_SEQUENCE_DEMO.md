# Continuous Door55 demo

On the prepared GPU host, this command runs a 55-second H1/Shadow teacher that
walks from 0.7 m away, stops, lowers, prepares its hand, acquires the handle,
presses the lever and attempts to hold a small opening:

```bash
/workspace/venv/bin/python scripts/isaac/run_full_sequence_demo.py \
  --receipt out/isaac-ready/shadow-loopback-v2/ready.json
```

Run from the source checkout created by the one-command Isaac setup. For a
different workspace prefix, set `DOORBENCH_WORK` or supply `--asset-python`.
The launcher uses the current Python executable for Isaac, so use the prepared
Isaac interpreter. It never allocates a node or changes its teardown deadline.

This is a **privileged development teacher**, with measured world poses and
mechanism state. The target is about 0.08 rad (4.6°) of held leaf opening. It does
not demonstrate traversal or a vision/tactile-only learned policy.

## Inputs and evidence

The launcher rejects legacy v1 readiness and checks all recorded ready input
hashes and eight live-verified passive loopbacks. It verifies the frozen walking
reset, arm preparation, acquisition reference and original Unitree **H1** policy
hash. If the default policy file is absent, it downloads the pinned upstream H1
checkpoint and license. An explicitly supplied wrong checkpoint is rejected.

The frozen protocol is
[`configs/dexterous/door55-readiness-v2/isaac-demo-protocol.json`](../configs/dexterous/door55-readiness-v2/isaac-demo-protocol.json).
Two known generated Door55 XML hashes are permitted. Their only XML difference
is the fixed shaft-support inertial eigenbasis; the body-frame tensors agree to
1.09×10⁻¹⁹ kg·m². Other mechanism versions are rejected. The actual XML, robot,
USD, motor contract, checkpoint and controller/auditor source hashes are recorded
and checked again immediately before and after the physical run.

Each invocation creates a fresh directory under `out/isaac-full-sequence/`, with:

- `invocation.json`, copied readiness and frozen inputs, and controller sources.
- `initial-audit.json`: native geometry check of the actual walking reset.
- `run.log`, `pipeline.json`, and `run.pid` for the existing Run Center collector.
- `trial/`: live video, every-step physics/pad evidence, readiness screen and
  `full-sequence-report.json` from the continuous runner.

The launcher registers that directory in the host's
`out/isaac-launch/runs.json`; use `--registry` for another Run Center registry.
For a Run Center running on your laptop, use its existing SSH registration flow
with this remote evidence directory. The host-local launcher does not create a
second remote-control service or silently edit a registry on another machine.

Successful process exit alone is insufficient: all 27 frozen report checks,
55 seconds of evidence, 2 ms timestep, zero runtime robot pose writes and no
direct door commands are required. A failure returns nonzero and preserves the
run directory. A same-environment lock prevents duplicate demo invocations.

## Checks that do not start Isaac

```bash
# Validate files and print the exact command; no processes, writes or downloads.
# Supply an existing pinned checkpoint if it has not been cached yet.
/workspace/venv/bin/python scripts/isaac/run_full_sequence_demo.py \
  --receipt out/isaac-ready/shadow-loopback-v2/ready.json \
  --checkpoint /path/to/original/h1/motion.pt --dry-run

# Also run the CPU native reset screen and register its separate evidence.
/workspace/venv/bin/python scripts/isaac/run_full_sequence_demo.py \
  --receipt out/isaac-ready/shadow-loopback-v2/ready.json --check-only
```

Use `--view hand` for the close review camera, `--no-record` to omit review video,
or `--output` to select a fresh directory. Recording is enabled by default.

September 8, 2026 launcher verification: 29 local tests passed, including legacy
receipt rejection, changed inputs, incorrect checkpoints, failed preflight,
incomplete/failed runtime reports and subprocess error retention. The actual
frozen walking reset passed an independent native geometry check with zero
nonfoot intersections and zero joint-limit failures. Run Center registration
was exercised against an isolated test registry.

**No GPU trial was launched while qualifying this wrapper.** The separately
running full-sequence experiment remains the source of its physical outcome;
wrapper tests are not a successful robot episode.
