# Why the Isaac bimanual transfer did not start

Independent reductions of full-opening runs 004 and 005 found **no continuous
0.5-second overlap** of qualified right-hand grip and at least 2 N of actual left
panel support before their first invalid right-hand contact patch. Both original
failed results remain unchanged. Increasing the support target from 4 N to 6 N
improved the left support, but did not establish the required simultaneous grip.

| Actual evidence | 004, target 4 N | 005, target 6 N |
| --- | ---: | ---: |
| Recorded duration, graceful diagnostic stop | 43.1 s | 43.7 s |
| First invalid right-hand patch | 35.142 s | 35.346 s |
| Longest overlap after release eligibility at 30 s | 0.358 s | 0.458 s |
| Best half-second window | 32.370–32.870 s | 30.656–31.156 s |
| Qualified overlap endpoints in that window | 249/251 | 250/251 |
| Immediate blocker in that window | Two index-finger unloads | One left-support dip |
| Longest left support before first invalid patch, after 30 s | 0.834 s | 2.532 s |

In 005, every right-hand sample in the best window was qualified. The left
support fell to **1.6883 N at 31.076 s**, breaking the hold. The longest overlap
anywhere before the first invalid patch was 0.494 s at 27.564–28.058 s, before
the declared earliest release time of 30 s. Neither interval qualifies, and no
threshold or release time was changed for this analysis.

At 32.812 s, 005 began 2.532 seconds of uninterrupted left support, but the
right grip no longer stayed qualified for half a second. Its longest overlapping
span after 30 s was 31.552–32.010 s. Higher average support therefore does not
by itself solve the right-hand contact geometry problem.

## Right-hand failure categories

These counts cover left approach at 22 s through, but excluding, the first
invalid patch. Categories overlap and must not be added. Opposition statistics
are evaluated only on samples where all five digits have at least 0.2 N of
qualified surface load.

| Failure category | 004 | 005 |
| --- | ---: | ---: |
| Index finger below 0.2 N | 19 | 13 |
| Middle finger below 0.2 N | 40 | 18 |
| Ring finger below 0.2 N | 2 | 3 |
| Little finger below 0.2 N | 0 | 0 |
| Thumb below 0.2 N | 4 | 2 |
| Finger-pair alignment at or below 0.5 | 1,008 | 974 |
| Thumb/finger dot at or above −0.5 | 998 | 1,271 |
| Invalid anatomical contact patch | 0 | 0 |

For 005, the first grip interruption after left approach was a middle-finger
unload at 23.686 s. Thumb opposition first failed at 28.060 s; finger-pair
alignment first failed at 28.796 s. These geometric opposition failures account
for many more samples than simple unloaded digits.

The first invalid patch at 35.346 s was on `rh_thmiddle`, carrying 14.0578 N.
Its contact point in the thumb-middle frame had positive local Y (+6.630 mm),
and its outward normal had positive Y (+0.446), on the other side of the
calibrated minus-Y volar face. This is not a weak-force rejection. The previous
brief thumb unload began at 35.338 s. Once this patch appeared, the controller's
explicit all-episode patch requirement also prevented a later release from
being presented as a qualified prefix.

The evidence supports testing steadier left force during an already stable
right grip and better right-hand contact retention through the long hold.
It does not justify weakening the support, opposition or anatomical-patch
criteria. These are two individual trajectories, not a statistical estimate of
the effect of changing the target force.

## Timing and reproducibility

The reduction aligns `full-opening-steps` and `acquisition-pad-steps` at the
same actual endpoint time. The teacher metadata within each opening row refers
to the preceding interval, 2 ms earlier. That relationship was checked for
every row in both runs, with zero discrepancies. Teacher metadata was not
mistakenly compared with the next interval's contact loads.

`scripts/dexterous/audit_opening_overlap.py` produces the source-bound JSON
report and a five-panel comparison plot from the saved records. It performs no
simulation, does not change any gate, and does not modify its input files.
Three temporal negative controls verify that 251 endpoints are required and a
single bad sample breaks a hold. The final report and inspected plot are at:

`/tmp/doorbench-continuous/out/continuous/full-opening-overlap-audit-001/final/`

The selectively downloaded 005 inputs and original tar are retained beside
that directory; 004 is read from the canonical local run archive. The JSON
report records every consumed input hash, exact first bad patch, per-digit
counts, top overlap spans, best half-second window and its individual failed
samples. These are diagnostic reductions of the already failed executions.
