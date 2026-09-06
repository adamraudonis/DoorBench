# Flexible PVC curtain reconstruction

The eight strip curtains use independently suspended, articulated PVC strips.
The moving material keeps its actual width, thickness, density and height. Each
strip is divided into pieces no longer than 0.30 m, with native bending joints,
positive material inertia and no fictitious joint armature. Contacts remain
enabled between neighboring strips in all three export tiers.
The rail and clamp plates are fixed hardware. Their former 1.1 kg catalogue
allowance is excluded from moving-strip mass; every moving segment carries
exactly its PVC material mass.

Each cut strip includes a further **28 mm fixed PVC tab** through the clamping
jaws. Its width, thickness, cut-stock length and static material mass are
explicit. The fixed tab meets the first moving material segment at the same
plane and bears on both clamp jaws; the jaws connect to the hanging rail.
The tab retains its own fixed native body. An exact body-pair exclusion covers
only that bonded material interface, matching the treatment of consecutive
segments of one continuous strip. Tests prove that neighboring tabs and world
obstacles still contact the moving strip. A displaced, floating tab fails the
attachment gate.

The previous geometry placed some curtains in three or four overlapping layers
while claiming a two-layer layout. The revised count and pitch admit two layers
without same-layer overlap. The recorded overlap is the actual neighboring-strip
width fraction. Each side has a real, ray-tested sheet-face contact near 1 m
height. A finite load at that contact bends the complete material chain.

## Construction and sources

The original geometry follows the alternating front/back installation described
by [PVC-Strip's hook-system instructions](https://www.pvc-strip.co.uk/news/installing-pvc-strips-hook-type-pvc-curtain-kits/).
Their combined-overlap percentages have a different convention from the
single-neighbor fraction stored here. [Thermokor's strip-curtain systems](https://thermokor.com/pvc-strip-curtains/)
also distinguish individual strips and their hanging support. No manufacturer
CAD or third-party geometry is incorporated.

The existing material table's density of 1,250 kg/m³ and Young's modulus of
10 MPa define an authored, uniform flexible-PVC approximation. Bending stiffness
is `E * width * thickness³ / (12 * segment_length)`; the first joint represents
the half-segment clamp boundary. Segment damping is an explicit linear
approximation. The model is planar: torsion, lateral bending, nonlinear material
response, temperature dependence, aerodynamics, tears and fatigue are outside
its scope. The hanging boundary is an ideal native clamp constraint. Native
force-response validation does not establish measured-product compliance or
safe human operating force.

## Native verification

`doorbench.strip_mechanics_qa.run_strip_mechanics_qa(model, metadata)` creates
fresh simulation data and applies 20 N loads at the actual sheet-face sites.
It runs forward load, passive release, reverse load and passive release twice.
There are no prescribed intermediate joint poses and no disabled neighboring
contacts. The acceptance checks require:

- At least 100 mm of load-directed face displacement in each loading phase.
- Maximum penetration below both 1 mm and half the sheet thickness.
- Finite states, no MuJoCo warnings and no unexplained energy increase above 1 J.
- Net dissipation during each passive release, with the complete applied-work
  budget reported separately.

The thin sheets use a **0.1 ms maximum native timestep**, constant contact
impedance `(0.95, 0.95, 0.0001)` and contact reference `(0.0002, 1)`. The usual
16 MiB native arena is retained. A finer timestep is permitted. These assets
are more expensive to simulate than the rigid-door models.

The rejected prototypes are retained under
`out/mechanical-foundations/strips/`: an activation margin preloaded coplanar
edges; strongly varying impedance became unstable; less resolved contacts
allowed millimetre-scale crossing during release. None of those trials is an
accepted mechanism result. The final native export serializes the actual geom
impedance into both model JSON and MJCF.

The worst thin/tall case, DB0535, was also tested at 0.05 ms with unchanged
contact parameters and the final fixed clamping tabs. Repeated-cycle maximum
penetration changed from 0.988 mm to 0.531 mm. The first forward displacement
maximum differed by 2.3 mm. The largest load-phase displacement difference was
33.5 mm during the first reverse load. Later
contact-rich oscillatory trajectories differ more, so this comparison is
evidence of resolved contact and similar load response, not exact trajectory
convergence. Both runs dissipated energy during every release and stayed within
the applied-work budget. The detailed comparison is in
`out/mechanical-foundations/strips/final-clamped/timestep-convergence.json`
(SHA256 `bbbb50eea829d7dbe7c9d2264543b38cb09a972f8c33db3f28943a5d4b9b57ab`).

All eight final models passed both complete bidirectional cycles:

| Door | Strips / moving segments | Maximum penetration (mm) |
| --- | ---: | ---: |
| DB0037 | 5 / 40 | 0.171 |
| DB0163 | 8 / 56 | 0.121 |
| DB0350 | 11 / 110 | 0.065 |
| DB0406 | 5 / 35 | 0.166 |
| DB0535 | 9 / 90 | 0.988 |
| DB0628 | 11 / 77 | 0.126 |
| DB0641 | 5 / 40 | 0.192 |
| DB0687 | 8 / 56 | 0.065 |

The final receipt SHA256 is
`f30d2f295db1b7919f4b6d58c126e63150603b45d0f0e8d00fab1fa1b149dbc5`.
A fresh export of every final model reproduced all three tested spec/model/XML
hashes byte-for-byte; its receipt is
`out/mechanical-foundations/strips/rebuild-clamped-current/receipt.json`
(SHA256 `9cb80c527644181e6868a3386eda04c1836761a8c090695d0541147ae724e92a`).
The combined strip, mass-scope and mechanical-inventory regressions passed
61 tests.

The native tests use MuJoCo 3.12.0. USD/URDF structure checks do not establish
equivalent thin-sheet dynamics in another physics engine.

The source-bound final receipt and per-door reports are under
`out/mechanical-foundations/strips/final-clamped/`. The earlier `final/`
directory retains the superseded result with the fixed-hanger mass still
allocated to moving strips; `final-material-mass/` retains the subsequent
motion-only test before the fixed clamping tab was added. The permanent regressions are in
`tests/test_strip_mechanics.py`; they include all-eight/all-tier material and
contact checks, compiled solver-parameter checks, genuine native loading, and
an obstruction behind an initially accessible contact. Negative fixtures also
reject contact removal, off-surface sites, false stiffness, contact preload and
an excessive timestep.

Generic independent sweeps of material bends are unsuitable: they hold
neighboring strips fixed, and the generic released pose folds every mechanism
joint to its limit. The scoped native checks replace those fabricated material
configurations. Initial geometry, structural mounting, mass and export checks
remain separate requirements. This work does not establish a successful
benchmark traversal or certify an appearance render.

At the fixed boundary, neighboring coplanar strips may meet edge-to-edge;
overlapping layers and clamp bottoms have a 1 mm rest separation. These are
contact-capable material/support interfaces, not 3 mm free-running machine
clearances. An exact finite-box check confirms zero gap means a seam without
volumetric overlap. This distinction does not permit interstrip penetration,
remove neighboring contacts, or relax floor, wall, header and rail clearances.
