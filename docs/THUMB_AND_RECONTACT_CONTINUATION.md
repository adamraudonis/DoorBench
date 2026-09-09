# Thumb relief and moving-door recontact

September 9, 2026 UTC. Neither experiment below qualifies a complete opening.

## Thumb relief

Two uninterrupted native trials kept the original robot, force caps, distal-pad
contact checks and 15 mrad actual thumb stop margin. Both failed. Raw outputs,
frozen sources and inputs were copied into the local September 9 run archive;
all 117 files per trial were verified against their source hashes.

| Trial | Requested / completed simulation | Result |
|---|---|---|
| `sensor-thumb-admittance-v2-001` | 36 / 24.036 s | Thumb stop-margin guard; maximum motor-coordinate error 44.129 mrad |
| `sensor-thumb-admittance-v3-001` | 36 / 24.254 s | Same guard; maximum motor-coordinate error 47.078 mrad |

Version 2 resolves incompatible soft interior-velocity requests within the
unchanged hard motion bounds. Version 3 additionally responds to the magnitude
of the measured tactile force resultant, including shear. The previous normal
projection saw about 3.14 N while the resultant was about 6.92 N. Responding to
that omitted load delayed the stop but did not solve the posture problem.
Opposed contact remained valid during the force phase; that does not excuse
failed tracking or incomplete lever operation. Further work must address the
thumb's available posture during handle travel, rather than weaken the guard.

[Machine-readable results and archive locations](evidence/thumb-admittance-v2-v3.json).

## Whole-body recontact

The earlier physical recontact trial stopped after 214 ms because the moving
leaf exhausted the waist correction around a nearly fixed body plan.
`WholeBodyContactTargets` instead computes bounded root and ordinary joint
targets for the moving palm goal while preserving the attained feet and
right-hand position. It uses privileged measured geometry and an unstepped
robot model; it is a teacher-planning component, not a sensor policy.

The independent replay of all 107 available source intervals passes its
geometry checks: at least 40 mm right-hand clearance against all scene shapes,
maximum foot position error 55.4 micrometres, and exact agreement between
robot-only and combined-model FK. Six tests cover target integration, rate
bounds, stationary behavior, bad observations and clock gaps. This short
counterfactual replay does not establish new physical contact or completion.

[Replay evidence](evidence/recontact-whole-body-replay-001.json).
Two subsequent six-second constant-leaf-rate extrapolations exposed excessive
root rotation. The revised planner weights the nominal base more strongly and
inscribes its position bounds inside the original root-norm limits, so the
solver encounters those bounds before emitting a command. The final diagnostic
passes all 3,107 actual-plus-synthetic samples, reaches at most 29.45 mrad root
rotation and retains 40 mm right-hand clearance. The extra six seconds are
synthetic kinematics, not additional observed physical motion.
[Full prediction receipt](evidence/recontact-whole-body-prediction-003.json).

The next step is to bind the preceding motor targets at the transition,
integrate the planner, and independently audit a new physical continuation.
