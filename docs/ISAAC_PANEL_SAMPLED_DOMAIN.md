# Detached panel mechanism-domain sampling

`isaac_panel_domain_audit.audit_panel_domain(candidate_path, plan_path, static_audit_path, sampling_spec, progress=None)` extends an existing **successful static panel geometry screen** to an explicitly sampled mechanism grid. It is CPU-only, uses an isolated unstepped MuJoCo geometry calculator, and grants **zero runtime/physical stage authority**. No qualified actual Isaac withdrawal has been assumed or manufactured for development; tests use explicit synthetic admission seams.

The auditor freshly admits the actual released Isaac source. It checks the candidate with `IsaacPanelGeometryProbe`, then requires the original passing static plan/audit, exact embedded receipt, unchanged source context and complete current input hashes. It reconstructs the plan coordinates/progress from the candidate, verifies measured source qpos/qvel and original joint/address order, and recomputes the original analytic spline rate envelope. The original static geometry limits, 40 mm RH/environment clearance, 3 mm elbow clearance, original collision checks and rate bounds cannot be overridden by the sampling specification.

The original static screen is supplied as an immutable bound prerequisite. This new auditor validates its structure/bindings and recomputes its analytic rate bound; it does not pretend a finite mechanism grid re-executes the original time-parametrized static trajectory audit.

## Explicit sampling contract

An example shape, **not an admitted operating range**, is:

```json
{
  "schema": "doorbench.isaac-panel-domain-sampling.v1",
  "reference_aperture_rad": [0.1, 0.4],
  "reference_samples": 2001,
  "leaf_lag_rad": [-0.005, 0.005],
  "leaf_lag_samples": 3,
  "operator_rad": [-0.01, 0.01],
  "operator_samples": 3,
  "latch_m": [-0.001, 0.001],
  "latch_samples": 3,
  "maximum_samples": 250000
}
```

Every endpoint and count is required; unknown fields reject. Reference aperture must be a nonempty subsegment of the source-bound plan. Its uniform grid has 2,001–20,001 nodes, augmented by **every exact spline knot progress within that subsegment**. No time warp or clipping is used when evaluating the cubic at those progress coordinates.

Lag means **reference aperture minus measured leaf aperture**. Positive lag places the measured leaf behind the reference; negative lag places it ahead. Both declared endpoints are sampled, and zero is inserted whenever it lies within the requested interval. Actual source operator/latch coordinates must already lie within their declared intervals and are then inserted as exact nodes. The auditor never silently expands a requested domain to include source values. Degenerate one-value axes require exactly one sample; nondegenerate axes require at least two.

The exact actual released-source point is evaluated separately, even when the requested aperture segment starts later. Its measured mechanism coordinates are preserved, including tiny nonzero operator/latch values. It is labelled `kind=exact_source` and adds one to the total; it never replaces or alters any Cartesian domain point. The fixed remainder of the robot's joints stays at the actual measured source values, matching the original static plan. Although the point probe can inspect complete robot-target mappings, this batch auditor does not accept a hidden extra target route.

The declared denominator is:

`1 source point + reference nodes × lag nodes × operator nodes × latch nodes`.

Inserted spline/source/zero nodes count toward the denominator. The complete count must fit the explicit budget, at most 250,000, before sampling begins. Every requested point is attempted. Mechanism inputs outside the probe's original source-relative joint envelope are recorded as rejected samples; they are never clipped, omitted, or removed from the denominator. Ordinary failed point checks, collisions, rejected coordinates and nonfinite point results are retained in the complete failure list. Progress callbacks report attempted, total, failed and rejected counts. The callback receives no plant handle.

All consumed source/candidate/static/probe/auditor hashes are checked before/after sampling. The CLI also binds the explicit sampling file. An input mutation aborts without a passing receipt. Failed source admission or corrupt prerequisite proof rejects before any new geometry samples. Existing failed artifacts are never overwritten.

## Interpretation

The distinct `doorbench.isaac-panel-sampled-mechanism-domain-audit.v1` receipt reports sampled pass/failure and the unchanged analytic **nominal-reference** rate envelope. It explicitly sets `authorized_stages=0`, `physics_steps=0`, `active_state_writes=0`, `geometric_admission=false`, `physical_admission=false`, `runtime_route_exported=false`, and `continuous_domain_proved=false`.

A passing sample set does **not** prove collision clearance at points between samples, any continuous lead/lag volume, velocities of measured mechanism motion, dynamic support or tracking, a corrected/rate-limited runtime target, a new controller handoff, or actual full aperture/passage. The nominal spline rate envelope does not account for additional live correction/projection/filter terms. Those need their own original guards and a distinct runtime admission; this artifact cannot be supplied as the existing native or Isaac runtime route.

Distances are capped exact queries from authored collision geometry, not physical contact loads or direct PhysX distance measurements. The input mechanism coordinates are prospective sample values; only the separately labelled source point contains the measured actual source mechanism coordinates. No mechanism joints are commanded and no source state is installed into a live plant.

## CLI

Only after an actual released source and its static panel candidate/plan/audit qualify:

```powershell
& C:/Users/adamr/Documents/Codex/doorbench-env/Scripts/python.exe scripts/dexterous/audit_local_isaac_panel_domain.py --candidate <candidate.json> --plan <plan.json> --static-audit <static-audit.json> --sampling <prospective-sampling.json> --output <fresh-domain-audit.json>
```

Exit 0 means the finite sampled screen passed; exit 1 with an emitted receipt preserves every sampled failure. Input/admission errors raise before producing a geometry receipt. The source run, candidate and plan are always read-only. No real-source screen or physics run is part of this implementation task.
