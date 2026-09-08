# Thumb-only relief before the flexion stop

The restricted-pressure run remains failed26/28. Its final geometry is
infeasible under **fixed palm plus fixed THJ3/4/5 opposition targets**: restoring
THJ2 alone toward its interior pushes the distal pad further into the lever.
The broader five-joint thumb has a feasible local geometric adjustment.

`screen_thumb_interior_coordination.py` starts from four actual states of
`sensor-thumb-flexion-001`, at23.0,23.5,24.0 and24.4s. Only the five thumb joints
change in a detached full-scene calculator. Every other generalized coordinate,
the palm and all four other distal tactile sites stay exactly fixed. The first
three source states precede the40mrad tracking failure at24.122s;24.4s is an
additional pre-stop diagnostic, already past that tracking gate. None is claimed
to be the end of a successful operation.

Each endpoint solves for less signed distal overlap, preserving the actual
qualified thumb contact's axial band and outward normal. THJ2 must end at least
25mrad inside its original lower stop; every thumb joint must retain at least
10mrad of authored range. Adjustments are bounded to±120mrad, although the
largest solution uses only21.735mrad. Each quintic joint path is screened at
1,001 points for original bounds, connected thumb overlap, original distal
surface/axial qualification and3mm hand-penetration limit.

The fresh `out/continuous/sensor-thumb-interior-plan-002/report.json` passes all
four paths, with zero non-thumb coordinate/site movement and no failed geometric
samples. At24.4s the THJ5/4/3/2/1 adjustments are:

| Joint | Change |
|---|---:|
| THJ5 | −12.377mrad |
| THJ4 | +12.513mrad |
| THJ3 | −15.161mrad |
| THJ2 | +21.735mrad |
| THJ1 | −15.006mrad |

That endpoint restores25mrad THJ2 margin and reduces signed overlap from100.1
to30.0µm while changing the thumb surface normal by only6.08µrad. At23.5s a
smaller adjustment, at most1.873mrad, reduces overlap103.0→30.0µm while retaining
25mrad interior margin. These are configuration/path findings, **not** predicted
contact loads, realized motor motion or an anatomical-force qualification.

The planner uses an actual qualified contact centroid and normal only on the
offline evaluator side. Those scene/contact fields must not become runtime
policy inputs. A physical candidate must either consume an explicitly frozen
instance-specific joint schedule or use only own-robot calibrated pad FK,
encoders and local touch to coordinate opposition posture. THJ3/4/5 may change
smoothly as posture targets; they must remain outside the pressure projection.
The original caps, mechanical gates and retained failed reports stay unchanged.

No wrist/palm expansion is needed by this screen. No physical coordination trial
has been run or qualified. Prefer a source before24.122s for a future trial.

```sh
python scripts/dexterous/screen_thumb_interior_coordination.py \
  --trial out/continuous/sensor-thumb-flexion-001 \
  --output out/continuous/sensor-thumb-interior-plan-003
```

Each candidate file retains its complete1,001-point path and every failed
sample, if any. The report binds actual source trajectory/physics/provenance,
robot/door XML and planner source. The original plan001 receipt is retained;
plan002 adds explicit numeric checks of all fixed coordinates/sites.
