# Opt-in passive-joint adapter: measured fixture, robot qualification pending

`--joint-passive-profile backend-dry-v2` assigns the original robot's passive
friction and damping to the PhysX solver. The default remains `legacy-tanh-v1`
for historical replay. The new profile requires its own whole-robot reach and
grasp tests; the isolated fixture does not establish those outcomes.

The new profile preserves each named joint's original armature and assigns
`[frictionloss, frictionloss, damping]` through the installed tensor API's
static-friction, dynamic-friction and viscous-friction properties. For these
rotary joints, the units are Nm, Nm and Nm·s/rad. It zeroes only the duplicate
explicit passive terms in the submitted force calculation. The original 61
motor outputs, transmissions and caps remain unchanged.

## Independent fixture review

The canonical evidence is
`DoorBench-runs/2026-09-08-shadow-loopback/passive-joint-fixtures`.
The independent audit reproduces all saved trace metrics, force schedules and
report checks exactly. The first Isaac run has an incomplete export because Kit
closed before saving; it remains incomplete. Isaac fixture002 passes 6/6.
Native001 and native002 each retain their original 5/6 result.

Each fixture is a gravity-free, isolated revolute hinge with the original
Shadow-link inertia: 0.017 kg mass, 12.5 mm COM offset, 2.7e-6 kg·m² principal
inertia and 2e-4 kg·m² joint armature. Effective inertia is 0.00020535625 kg·m².
The timestep is 2 ms. The external **fixture-only** excitation is 0.005 or
0.02 Nm for 0.5 seconds, then zero for 0.5 seconds. Passive values are 0.01 Nm
friction and 0.05 Nm·s/rad damping. No fixture excitation is permitted assistance
in a robot task.

| Measurement | Native dry friction | PhysX backend dry friction | Legacy smooth law in PhysX |
|---|---:|---:|---:|
| Motion under 0.005 Nm for 0.5 s | 0.0100085 rad | 5.96e-9 rad | 0.0168266 rad |
| Late speed under 0.02 Nm | 0.2000000 rad/s | 0.1999997 rad/s | 0.1999995 rad/s |
| Late peak speed after removing 0.02 Nm | 4.24e-11 rad/s | 7.67e-10 rad/s | 0.0643683 rad/s |

The native dry-friction fixture fails the frozen 0.002-rad ideal-static motion
threshold. Its numerical creep is retained, not treated as exact engine parity
or hidden by changing that threshold.

The loaded terminal speed independently checks SI scaling:
`(0.02 Nm − 0.01 Nm) / (0.05 Nm·s/rad) = 0.2 rad/s`.
PhysX differs by only 1.66 parts per million. Its first loaded velocity is
0.065497339 rad/s, consistent with the original inertia and implicit viscous
prediction of 0.065497268 rad/s. This is evidence against a degree/radian factor
error or missing armature in this fixture. Float32 backend property readback
also matches the declared values exactly.

The old explicit law is
`−0.05 × velocity − 0.01 × tanh(velocity / 0.001)`.
Both the native emulation and PhysX show an alternating coast velocity even
after external excitation is zero. For saturated friction, the symmetric
explicit-step cycle predicts
`0.01 × dt / (2 × effective_inertia − 0.05 × dt) = 0.064368186 rad/s`.
Native reproduces that value to floating-point precision and PhysX to
7.69e-8 rad/s. This explains why a smooth approximation can destabilize the
light passive finger coordinates at this timestep despite bounded motor sums.
It does not prove the complete grasp failure is fixed by the new profile.

## No double passive force or hidden motor contribution

The runner uses the same profile-selected explicit damping/friction arrays in
both generalized torque submission and `actual_motor_delivery`. They contain
the original coefficients in legacy mode and zeros in backend mode. Backend
solver reactions are therefore not added to the recovered motor commands.
Tests recover all 61 motor inputs in both modes and reject incorrectly adding
passive reactions in the eight unactuated differential coordinates.

`get_dof_actuation_forces` is a readback of submitted generalized actuation input;
it is not an independent joint torque measurement. This distinction remains in
the result contracts.

## Persistence and failed evidence

`configure_backend` verifies both friction properties and original armature at
startup. `PassivePropertyInvariant` then checks their exact expected float32
values after every 2-ms interval. It never writes or repairs plant properties.
Its receipt contains attempted, checked and valid interval counts, per-property
maximum differences and the first failing interval with its actual readback.
A mismatch or missing/repeated clock is terminal; a backend getter exception is
also preserved. The runner writes the receipt before propagating failure and
at normal completion, even if ordinary state rows stop before the failed step.
The receipt reports property persistence only, not completion of a robot task.

The new getter checks are opt-in. Legacy mode has no new per-step property reads.
Source capture includes the helper, and the final plant-invariant dictionary
also includes friction and armature. Robot mass, joint/loopback limits, contact
materials and force caps retain their existing independent checks.

Reproduce the read-only evidence review with
`python scripts/dexterous/audit_passive_fixture.py --directory FIXTURE_DIRECTORY --output NEW_RECEIPT`.
Run CPU contracts with
`python -m pytest -q tests/test_isaac_joint_passive.py tests/test_isaac_traversal_runner.py`.
The review receipt is in `docs/evidence/passive-joint-fixture-review.json`.
