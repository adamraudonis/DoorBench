# Detached nominal bridge: numerical first stage

`build_panel_nominal_bridge(seed, duration_s=...)` consumes the reviewed
`prepare_isaac_panel_bridge_seed(...)` result. The caller must prepare that seed
through its actual-source admissions. This numerical helper rechecks current
input hashes and cross-bindings; it does **not** reexecute physical admission.
No actual successful released source or feasible bridge has yet been assumed.

The initial position and stored prior velocity are the exact 75-coordinate
nominal state at **T−0.002**, in the original withdrawal chart. Measured state at
T stays in the seed as a separate observation. The mapped endpoint uses the
panel first pose, proper rotation composition, and the same 69-joint order.

The 31 changing coordinates use quintic Hermite interpolation with the stored
initial velocity and zero final velocity/acceleration. Initial acceleration
zero is a **prospective coefficient choice**, not restored controller state.
Existing repository quintics are rest-to-rest and cannot preserve this boundary;
the existing root-chart mapper is reused.

The other 44 nominal joint positions remain exactly constant. Their polynomial
derivatives are zero, even if their stored prior limiter velocities are nonzero.
These are separate fields. No analytic C1 continuity is claimed for those 44.
They can meet the first discrete acceleration gate only when their prior
velocity magnitude is at most `3*.002 + 1e-12` rad/s. Incompatible seeds reject;
the helper does not move a finger to hide the conflict or zero the old velocity.

`sample_panel_nominal_bridge(candidate, elapsed_s)` returns copies of positions
and polynomial derivatives. At elapsed zero it also returns the separately
stored prior velocity. The first *new* command is at elapsed .002, or epoch T.
Sampling outside the duration rejects instead of clamping or extrapolating.
Duration is explicit, a whole number of .002 s intervals, at most 30,000. The
requested floating duration is retained separately and the sampler uses exactly
`N*.002`, so its final motor-grid sample is in-domain even with float roundoff. Exact
declared endpoint values are used at the boundaries; interior samples evaluate
the stored normalized-time polynomial.

`audit_panel_nominal_bridge(candidate)` independently evaluates the stored
coefficients, without calling the builder or sampler. It validates normalized
Hermite boundary constraints, identities, and the 44 constant coefficients.
It checks **all** 500 Hz finite-difference commands against the existing bounds:
joint speed 1.2 rad/s, successive joint velocity change `3*.002` rad/s, root
translation speed .02 m/s, and old-chart rotvec speed .03 rad/s, with the exact
existing `1e-12` arithmetic tolerance. The first difference compares the new
position against the old position and stored old velocity. It additionally
checks the old velocity boundary and one explicit stationary terminal step.
Every rate failure remains in the receipt; no passing-prefix denominator is
substituted. A rate-failing candidate is still available for diagnosis.

Analytic derivative maxima include the initial polynomial boundary and all new
commands. They are sampled diagnostics, **not** a
continuous polynomial bound. Full FK, collision, source-relative joint bounds,
passive joints, palm/foot tracking, contact loads, all-environment RH clearance,
mechanism domains and motor/controller continuity are not evaluated here. The
44 old nominal joints differ from the original panel candidate's measured-T
remainder, so even a passing old panel geometry audit does not qualify this
new full pose. Geometry must be checked next from a real admitted source.

Candidate and audit carry exact content/input hashes, source/seed identity and
zero authority. No runtime route, controller restoration, plant write, force,
physics step, geometry admission or physical success is produced. The helper
has no CLI or provider/controller wiring. Tests use explicitly synthetic
upstream admission seams and block physics steps; they are mathematical tests.
