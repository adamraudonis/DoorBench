# First actual sensor actor: failed before acquisition

The first sensor-only actor trial fell after 0.522 seconds; it never acquired a
valid grasp. It crossed the 12-degree upright criterion at 0.268 seconds and
reached 47.41 degrees at the final recorded step. Its original motor limits,
passive hand mechanics and torque delivery passed. The original failed report,
all physics samples and the actual pre-step actor decision remain archived.

Its actual initial root and joint state exactly match the qualified teacher002
reset. All seven initial sensor streams were invalid, their timestamps were
-1, and previous action was zero. The initial knee forces had the wrong sign:

| Motor | Actor command Nm | Teacher command Nm | Error Nm |
|---|---:|---:|---:|
| Right knee | +13.774 | -53.533 | +67.307 |
| Left knee | +13.471 | -51.424 | +64.895 |

That packet was absent from the first training data. Fixed 32-step burn-in also
masked the first 64 milliseconds of a true start window, and random interior
windows rarely reached the beginning. Merely making initial sensors valid does
not solve the recurrent initialization problem: with a valid recorded packet
and zero hidden state, the original actor still predicts left/right knee forces
of +1.87/-15.46 Nm instead of -52.52/-54.63 Nm. Even under unchanged teacher
states, body force RMSE remains about 2.17 Nm at 80–100 ms. These are gaps in
start supervision and physical feedback robustness, not a simulator force-cap
failure.

Same-time teacher commands **after** the first physical tick are not corrective
labels for the diverged actor state. They are used only to characterize drift.
At 0.1 s the actor already tilts 2.68 degrees versus the teacher's 0.103 degrees,
with a 0.299-radian norm of joint-state differences. The [independent receipt](evidence/sensor-actor-initial-failure-001.json)
preserves exact initial errors, progression, hashes and the original failure.

## Bounded second training experiment, frozen before execution

Keep the existing actor architecture, motor/observation interface and original
physical gates. Add the actual all-invalid reset observation from actor001,
paired only with teacher002's actual first action. Admission requires identical
robot/door reset, physical joint order, mechanics and sensor calibration. The
failed actor's action is explicitly discarded. The rest of the qualified
acquisition prefix and its causal cutoff remain unchanged.

Train a fresh seed-zero model for 200 CPU AdamW steps. Half the batches start at
the true episode boundary with zero hidden state and no masked prefix; half
retain random interior windows and 32-step burn-in. Keep batch two, supervised
length 32 and learning rate 0.0001. The augmented prefix has 6,457 labels,
including the previously missing t=0 decision. Record early force error and
full-prefix persistence comparison before another physical trial. This
experiment has not run at this protocol commit, and no success is assumed.

For future DAgger corrections, the teacher must be queried on the student's
actual states, with its own continuously advanced controller history. The
current runner preserves 500 Hz root and joint position/door state, but only
50 Hz body-resolved hand loads. A faithful future query stream should separately
record pre-decision clean root13, named joint positions and velocities, handle
pose7 and body-resolved world hand forces at t=0 and every 2 ms. Those privileged
values and returned counterfactual teacher forces belong in evaluator/training
label files, never the actor's numeric sensor packet. Missing impulses must not
be filled by interpolating the sparse contact trace.
