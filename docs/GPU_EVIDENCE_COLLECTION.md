# Preserve GPU run evidence

The one-command environment launcher now starts a detached local collector.
Its connection receipt links the archive and collector log. The collector
copies partial output during execution, then compares two remote manifests
around the final transfer and verifies every local byte count and SHA256.
Only `final_bytes_verified: true` is a completed archive. This is independent
of whether the simulation passed.

For a separately launched experiment, start the collector before the job:

```sh
python scripts/isaac/collect_run.py \
  --host root@HOST --port PORT --key ~/.ssh/runpod_doorbench \
  --remote /workspace/experiment \
  --destination /absolute/local/archive/experiment \
  --deadline DEADLINE_UNIX --terminal balance-report.json --detach
```

The remote directory must contain `run.pid`, identifying the process that owns
the output. Change `--pid-file` or `--terminal` for another layout. Use a unique
destination per run. `--resume` continues the same unfinished archive after its
old worker exits; an exclusive lock prevents competing writers. Preserve source
bundles and input assets outside the output directory separately.

The collector continues when an assistant turn ends, but the local computer
must remain online and awake. Keep the independent pod teardown timer. A
collector timeout or interrupted transfer leaves an explicitly incomplete
archive; it must never be cited as a verified final result.

A real SSH fixture on September 9 copied and verified all three files of a
completed **failed** dummy task. Python 3.10 on the new node exposed a hash API
compatibility issue, which was fixed with streamed hashing and successfully
retested. The prior `own-imu-grasp-001` run was lost before this collector was
available; only its [retained partial evidence](evidence/own-imu-grasp001-interrupted.json)
may be cited. Do not infer missing raw measurements from that summary.
