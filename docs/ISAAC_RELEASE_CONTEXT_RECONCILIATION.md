# Release context portability

An actual source comparison between the planning Python stack and Isaac Kit
found four differences while all original source admissions passed and the
normalized initial qpos remained exactly equal. Three were derived scalar
roundoff: transformed free-root angular qvel (1.73e-18 rad/s), material-body
rotation error (3.18e-18 rad), and one contact's inward radial alignment
(1.11e-16). The fourth was the corresponding measured-payload SHA256, whose
payload includes that transformed qvel. The original failed constructor run
and original artifacts are retained.

`reconcile_isaac_release_context(candidate, fresh_context)` is a narrow identity
comparison **after fresh actual source admission**. It returns the original
immutable `IsaacReleasePlanningContext` and a separate comparison receipt.
It does not update the archived candidate, substitute raw measurements, or
grant a runtime stage. Fresh physical/contact checks and original 2 µm / 2 µrad
kinematic admission remain prerequisites. The source files are rehashed before
and after inspection.

Only these finite floating-point fields may differ by an absolute **1e-15**:

- The three transformed body-frame angular qvel coordinates of the model's
  single free root. Their indices come from the original model's DOF address;
  scalar joints and free-root translation rates remain exact.
- `material_frame_admission.maximum_body_rotation_error_rad`.
- Each endpoint contact's `inward_radial_normal_alignment`.

The dependent `measured_sha256` may differ only when **each** checksum verifies
its own complete measured payload. The archived canonical context checksum must
also verify. Every other field, type, list order, raw value, state identity,
classification, asset hash, threshold, and normalized qpos must match exactly.
There is no relative tolerance and no generic approximate dictionary comparison.
An unlisted difference rejects the binding.

The separate receipt records both context hashes, exact changed paths and
values, the numerical environment performing the fresh check, model-derived
angular indices, and unchanged input hashes. It keeps `authorized_stages=0`,
`physics_steps=0`, and `active_state_writes=0`. The loader retains the archived
canonical identity for downstream candidate/audit/map comparisons and exposes
the fresh reconciliation evidence separately. A diagnostic is printed only
when differences exist.

Tests reproduce the exact four changed numerical values inside an explicitly
synthetic source fixture, then reject out-of-scope coordinates, raw contact
changes, corrupted payload checksums, changed source files, types and labels.
Moving the free root in a tiny private model checks that no hard-coded qvel
slice can be admitted. A local read-only integration test also uses the frozen
actual Kit comparison when those untracked artifacts are present; portable
checkouts skip that case. No test advances physics or fabricates a completed
withdrawal.
