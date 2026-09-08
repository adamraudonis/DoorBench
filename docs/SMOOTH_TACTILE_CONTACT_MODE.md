# Bounded recovery from a fresh tactile unload

The failed `sensor-hierarchical-force-001` run lost index contact at25.076s.
When that zero-touch sample became available at25.078s, the controller restored
the full position-feedback torque immediately: FFJ3 changed by−0.572908Nm in2ms.
The thumb had already exceeded the unchanged40mrad tracking gate at25.010s.
These are separate defects; the original run remains failed22/28.

`SensorSmoothContactForceController` is an opt-in repair for contact switching.
It retains the original whole-thumb effort projection. Its input remains a
validated sensor packet plus local clock; no object identities, scene state,
root measurements or evaluator contact labels enter its control interface.

The frozen [profile](../configs/dexterous/sensor-contact-mode-v1.json) starts
after the same19s acquisition. On a fresh zero-load sample it immediately stops
press progression and freezes force, closure and index preload integration.
It retains the established contact-mode weight for20ms, then changes that weight
by at most5/s toward ordinary posture control. Recontact also restores weight
at at most5/s. Continued zero touch for250ms is a terminal controller failure.
Stale or invalid packets are immediately rejected; retention applies only to
the controller's pressure mode, never to sensor evidence or the physical score.

All original caps, joint/loopback limits, distal anatomy, opposed-grasp window,
40mrad motor tracking, load thresholds and maximum closing offsets remain.
The first19s is unchanged. The legacy touch/index/hierarchy defaults retain
their prior behavior through explicit default hooks; no old result is rescored.

The bounded weight and virtual-force rates do **not** impose a global torque
slew bound: changing encoder velocity, pad Jacobian and ordinary damping still
affect submitted motor forces. No returned force is modified after the original
cap and previous-action owner.

## Verification before a physical comparison

`audit_smooth_contact_mode.py` uses retained local loads and poses solely for
counterfactual effort algebra. It cannot predict the altered contact trajectory.
The first screen passed6/6: pre-loss effort matched within4.0e−14Nm; at the first
zero sample the same-pose FFJ3 change became−0.033277Nm. The persistent recorded
loss reached the declared terminal timeout at25.328s. It never became a grasp
success. Evidence is in
`out/continuous/sensor-smooth-contact-plan-001/audit-002.json`, bound to the
profile, controller sources, actual robot and archived packet/trajectory bytes.

Run the existing touch-operation probe with its original force and hierarchy
flags, plus `--contact-mode-protocol configs/dexterous/sensor-contact-mode-v1.json`
and `--contact-mode-screen` pointing to that fresh source-bound audit. A fresh
inner hierarchy screen is also required because the default-preserving weight
hook changes the source hash. Use a fresh output directory and retain every
failed gate. This profile is an instance-specific scripted motor component,
not a learned vision policy or a completed door-opening system.

## First physical comparison: repair isolated, task still failed

`out/continuous/sensor-smooth-contact-001` ran from the same contact-free reset
without later pose writes. Source395573aca, first19s qpos/qvel/forces/clocks/goals
bitwise identical. The original static/grasp admission was retained.

The same index loss occurred in the actual25.076–25.078s interval. With fresh
zero-touch feedback, pressure mode was retained and the press clock froze.
Index contact returned in the next physical interval25.078–25.080s, unlike the
original hierarchy's permanent retraction. Recorded mode replay matches every
numeric tactile projection/weight exactly, with no integration or progression
during zero touch. The largest pressure-weighted virtual-force step is0.020N;
the largest added virtual-work bias step is0.001450Nm. These do not bound the
complete motor command including ordinary velocity feedback.

The distinct whole-thumb defect remained. It exceeded40mrad tracking before
the first index loss and eventually reached150.539mrad maximum motor error.
At27.204s the first invalid loaded thumb patch had0.992525mm axial clearance,
below the unchanged1mm gate. The thumb then unloaded. After250ms without fresh
thumb load, the controller terminated at27.504s/13,752 physical steps. The run
is **failed18/28**, reaching only0.459409rad lever rotation and6.64108mm latch
retraction. The incomplete final-second quiet/foot checks also remain false;
their prescribed35–36s window was never reached.

Original caps, joint/loopback bounds, unintended-contact checks, balance and
warning checks pass over the recorded prefix. Independent raw classification
and qualified-pad loads match exactly. Every recorded projected bias reconstructs
within3.1e−15Nm; every finger motor command within the original adapter tolerance.
The contact auditor correctly rejects the physical task and incomplete manifest.
The new mode/algebra audit's pass is not a task pass.

Evidence includes `comparison.json`, `independent-contact-audit.json`,
`force-mapping-audit.json`, `regulation-decomposition.json`, and
`mode-execution-audit.json`. The personally inspected final hand close-up is
`touch-operation-hand-az150-0002.png`; it shows the thumb displaced toward the
lever end. No successful-contact claim comes from that image.
