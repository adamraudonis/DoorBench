# Corrected-hand Isaac environment

Use this opt-in profile for new H1/Shadow acquisition experiments:

```bash
python3 scripts/isaac/launch.py --config configs/isaac/runtime-v2.json
```

It uses the same pinned Isaac runtime, Run Center and allocation guards as the
[standard launcher](ISAAC_ONE_CLICK.md). A separate remote source/cache identity,
versioned native XML and ready directory prevent reuse of a v1 receipt. The
default `runtime.json` remains explicitly `upstream-v1` for reproduction.

The corrected profile adds the eight passive `J1 <= J2` Shadow finger loopbacks.
It retains the original 69 joints, 61 motor transmissions, mass, joint limits,
force/control caps and free base. This is a versioned mechanics correction;
earlier v1 manipulation results do not validate v2. See the
[mechanics contract](SHADOW_LOOPBACK_MECHANICS.md).

Preparation writes `out/isaac-ready/shadow-loopback-v2/ready.json` only after
the live standing/import smoke passes. The receipt records the native model,
motor contract and robot/door USD hashes, and verifies all eight constraints
through the live PhysX backend. USD attributes alone do not pass. The smoke
checks zero bilateral spring/damping, the intended unilateral limits and
stiffness, unchanged motor caps, physical standing, rendering and native FK.
Readiness establishes the environment; it is not an acquisition success rate.

## Acquire the lever from a contact-free start

After Run Center shows ready, connect using `connection.json`, enter its remote
checkout, and run:

```bash
source isaaclab/cloud/env.sh
python scripts/isaac/run_acquisition_demo.py --check-only
python scripts/isaac/run_acquisition_demo.py --view hand
```

The first command verifies the receipt, frozen reference and native initial
geometry without starting Isaac. The second runs the frozen 10.6-second Door55
acquisition with recording and strict physical/contact gates. It rejects v1 or
changed assets, and treats a failed report as failure even if Isaac exits with
status zero. Use `--output /path/to/fresh-run` to choose an evidence directory,
`--receipt /path/to/ready.json` for an explicit receipt, and `--no-record` to omit
the video. Keep the run directory and its adjacent invocation/initial-audit files.

The reference is pinned in
[`configs/dexterous/door55-precurl-v2`](../configs/dexterous/door55-precurl-v2).
Its initial acquisition pose is contact-free; an older initialized-grasp field
in that file is not used as the reset. The controller receives privileged robot
and mechanism state. This demo qualifies only lever acquisition; approach,
opening, traversal and vision/tactile-only control require separate evidence.

## A different cluster

```bash
python3 scripts/isaac/launch.py --config configs/isaac/runtime-v2.json \
  --host user@gpu-node --port 22 --key ~/.ssh/cluster --work /shared/doorbench
```

An SSH launch makes no RunPod calls; the scheduler must enforce its wall time.
For a scheduler already running in a DoorBench checkout, the identical remote
entry point is:

```bash
DOORBENCH_WORK=/shared/doorbench \
DOORBENCH_MECHANICS_PROFILE=shadow-loopback-v2 bash scripts/isaac/prepare.sh
```

Native XML hashes can differ between clusters because mesh paths are absolute.
Each receipt binds the current native XML to its current imported motor contract;
the historical acquisition model hash remains recorded as provenance. A different
robot requires its own validated embodiment and controller adapter.

## Verification status

The actual opt-in launch completed all five stages on an L40S at **September 8,
2026, 10:25:52 UTC**. It regenerated and imported the v2 model, rendered the
one-second free-base standing smoke and verified all eight live loopbacks.
Native FK differed by at most 0.912 micrometres; mass differed by 4.95e-6 kg. The acquisition `--check-only` command then passed
with the new receipt and contact-free initial geometry. The stricter final
receipt reader was also checked against that same live receipt.

See the [timestamped readiness result](../results/dexterous/2026-09-08/isaac-v2-readiness.json).
Eighteen focused tests pass, covering cache separation, v1 rejection, missing live
constraints, changed mechanics and stale assets. The acquisition wrapper itself
has only completed preflight in this readiness checkout; the long manipulation
trial is tracked separately by the controller work package.

The current development allocation has an independent **11:21 UTC** teardown
on September 8. The approach-to-acquisition chain and repeated live acquisition
trials require another bounded allocation if they do not fit the remaining time.
No readiness launch extends that existing guard. Archive evidence before its
deadline, then use a new owned allocation with the same profile and exact source
manifest; on a cluster, request the equivalent scheduler wall time.
