# Positive keeper for the six manual rolling-door chains

The physical hand chain transmitted opening/closing force, but the original mechanism did not retain the curtain after hand release. The DB0419 recorded final native state fell from a bottom reference site of 1.91235 m to 0.05012 m in five seconds with external hand forces removed. All original state, counterbalance, friction and mass were preserved. This was a real service defect, independent of the earlier approach-side and passage-classification defects.

The original proof used an explicitly installed prototype in `doorbench/geometry/hoist_keeper.py`. The keeper is now installed by the six production chain-hoist builders; original prototype receipts below retain their historical source hashes and scope. Its native controller is `doorbench/hoist_keeper.py`. Neither routine writes native coordinates or substitutes a shaft torque for a hand force.

## Physical and documentary basis

Janus instructs installation of a chain keeper on the wall/jamb (Model 2000, installation step 12E), and Wayne Dalton describes a wall-mounted keeper around four feet above the floor. Cornell distinguishes its proprietary integral load brake from ordinary chain-hoist systems that can descend on hand release. These sources establish the need for a physical retaining device; they do not provide the CAD, dimensions, rating or validation for this original keeper.

- [Janus Model 2000 installation](https://www.janusintl.com/hubfs/janus_2019/pdf/model2000-install-Rev.-12.9.16.pdf)
- [Wayne Dalton rolling-sheet installation](https://waynedalton-prod-assets-new.azureedge.net/assets/docs/default-source/install/comm/sheet/291341.pdf?sfvrsn=ead9fd5e_20)
- [Cornell ControlGard description](https://www.cornelliron.com/product/controlgard-chain-hoist)

A 10 mm steel pin enters the central space between real 40 mm-pitch chain rollers along world+Y. Its far end enters a bored receiver. Axial guide plates connect that receiver to a bored, wall-supported housing. The captured pin blocks the chain through native contact; guide friction is the same lubricated-steel coefficient as the existing chain. A guided compression spring supplies 2 N at engagement and 18 N at the 80 mm withdrawn stroke. This is a manually operated positive keeper, **not an automatic load brake**.

The shaft, shoulder and pull knob have geometry-derived mass added to the assembly. Every pre-existing body mass remains unchanged. For a raised curtain, the controller first applies an actual chain-site force to unload the pin, observes native pin/roller normal forces, then withdraws the actual pull knob. Closed-state floor support is handled separately below. During operation a second hand keeps the pin withdrawn. Engagement finds a roller gap while holding the chain, transfers load to the pin, and releases both hands. The 120 N cap applies to each physical input and is not an effort target.

## Evidence and exact scope

Evidence is under ignored `out/mechanical-foundations/hoist-keeper/`:

- `geometry-all/report.json`: all six source IDs 0258,0313,0419,0636,0754,0888; full/simple/minimal tiers; engaged/withdrawn poses; no initial penetration above 1 mm; every existing body mass unchanged.
- `prototype-hold.json`: exact DB0419 source recording state, controlled engagement and five seconds without either hand. Curtain reference ends at 1.90080 m after 12.1 mm of load take-up; maximum penetration 0.492 mm; chain hand peak 55.99 N; keeper hand 18 N; no native warnings. The XML and source recording hashes are recorded.
- `removed_pin.json`: same actual held state; only the pin contact is disabled as a component-removal negative. Curtain falls 1.19348 m in three seconds. Guide friction, spring, masses, other contacts and native state are retained.
- `release_close.json`: actual held-state unloading, pin withdrawal and full controlled close. Unloaded by 0.6005 s, withdrawn by 1.6525 s, final bottom reference 0.02997 m after 12.283 s. Peak chain force 78.39 N, keeper 18.13 N, penetration 0.151 mm, no warnings.
- `repeat-engagement.json`: subsequent actual-state repeat pin engagement and five seconds without hands. That early diagnostic deliberately jogged the curtain upward to find a roller gap, leaving a 22 mm opening; it is **not evidence of a fully closed final state**.
- `helper-attempt-01/`: preserved failed reusable-controller roundtrip. It completed close and a two-second hands-free floor-supported interval, then waited too long to withdraw an unloaded pin and loaded the next roller. This is a controller failure, not a keeper holding pass or force-cap justification.
- `helper-cycle.json`: the frozen reusable controller passes the complete actual-state roundtrip in 39.4205 native seconds: full close, two seconds with both hands absent at bottom reference 0.02254 m, re-open, repeated engagement, then five total seconds without either hand. Final raised reference is 1.91891 m; maximum penetration 0.3003 mm; largest per-phase hand force 69.40 N; no warnings. Native measured unloading replaces the failed attempt’s arbitrary height/timing condition.
- `freeze.json`: original prototype geometry/controller/test hashes, nine passing focused regressions, and a then-current rebuild matching the proof’s three XML tiers and model JSON byte-for-byte. It predates production installation and the closed-state controller revision below.

`tests/test_hoist_keeper.py` checks all six source mechanisms, exact spring force, original-body mass preservation, retained native collision, actual-site/no-coordinate-write behavior, and a real 100 N material-chain pull with/without the positive pin. These are component checks. They do not certify all six full-height cycles.

No evidence here certifies material strength, fatigue, human reach/hand trajectories, automatic braking, or whole-task traversal. Previous incomplete all-height service attempts for 0258 and 0754 remain unresolved and are not converted into passes. No source counterbalance fraction, lock scenario, friction or manual force cap is increased to force a result.


## Production closed-state release

The original always-preload sequence failed at production `qpos0`: the closed curtain already had measured floor support, so the opening-chain pull loaded the opposite side of the seated pin. Release now begins with a short zero-input settling interval, requiring actual floor-to-bottom-bar contact, measured upward floor reaction and a pin load below 5 N. A 10 micrometre numerical contact band is checked against the actual floor and seal geometry; reported closed height alone cannot supply support. The withdrawal target clock pauses whenever support is absent. Removing the floor or loading the pin cannot bypass the load check.

Four source variants complete release with no chain hand input and a keeper peak of 18.10 N. On the two weak-source variants, DB0313 and DB0636, chain slack loads the partly withdrawn pin. The controller stops its withdrawal, holds the actual knob, takes the load with a material-chain grip, and continues only after the original measured unloading gate passes. Their peaks are 81.48 N at the chain and 19.67 N at the keeper; both finish in 2.1385 native seconds. No spring, friction, mass or 120 N input cap is changed.

All six production closed releases and their guard regressions pass: 18 tests in 236.24 wall seconds, with the process-global native warning callback captured in every test scope. The largest contact penetration is 0.268 mm and no native warnings occur. Exact native XML and controller hashes, per-source forces and state transitions are recorded in ignored `hoist-keeper/closed-release-v3/closed-release.json`. Earlier failed attempts remain separate diagnostics. This is closed-state mechanical release evidence, not an all-six full-height or human-task certificate.

Native feedback must refresh contact forces with a full `mj_forward` using the previously applied real input before reading `mj_contactForce`. Rebuilding contact positions alone can change constraint indices while retaining stale force entries. The full-cycle and initialized-open continuation receipts use refreshed forces and remain separate from these closed-state checks.


## Refreshed-force continuation and warning-sensitive initialization

The refreshed-force production DB0419 roundtrip completes actual held-state release, full close, two seconds on the floor, re-opening, repeat engagement and five total seconds without either hand in 39.401 native seconds. Peak hand force is 80.03 N, maximum penetration 0.218 mm, and final raised bottom reference 1.91890 m. The exact input state, source snapshots and full log are retained in `hoist-keeper/closed-release-v2/held-final/`. This remains a bounded service cycle; the earlier wrong-side root task recording is not a traversal certificate.

The shared initialized-open helper now captures MuJoCo's process-global warning callback, including messages which leave every `MjData.warning` counter at zero. A source DB0419 `implicitfast` initialization produces `Linesearch objective is not convex` at 6.6115 native seconds and correctly returns no accepted qpos/qvel. Final refreshed constraints, contacts, finite arrays and warnings are checked before any successful state enters the cache.

Private full-implicit comparisons change only the native integrator, with original geometry, mass, springs, contacts and 120 N per-input caps. DB0419 reaches two seconds of hands-free retention at 0.5 ms and 0.25 ms; final bottom references differ by 1.78 mm and completion times by 0.801 s. Peak chain forces differ (73.06 versus 64.52 N), so this is bounded numerical consistency rather than identical trajectory or force convergence. Both report zero global warning messages and zero warning counters. At this full-height pose the fixed upper stops carry measured load (18.72/11.09 N peak); the keeper is seated but its measured pin load is zero. These results must not be described as proof that the keeper carries the full-height load. The separate intermediate-height removed-pin negative establishes that different load path.

MuJoCo documents that full implicit integration retains coupled Coriolis/centripetal derivatives omitted by `implicitfast`, and can benefit fast coupled pendulums. That is a reason for this isolated comparison, not proof of the warning's unique cause or permission to silently alter source models. [MuJoCo numerical integration](https://mujoco.readthedocs.io/en/latest/computation.html#numerical-integration).

Exact private receipts are `hoist-keeper/closed-release-v3/initialize-{source,implicit}.json` and `pytest-halfstep/hoist0/db0419_rollup/keeper-open-initialization.json`. The frozen algorithm-3 all-six comparison is complete in `out/mechanical-foundations/hoist-keeper/initializer-six/comparison.json`, with original helper sources retained under `source-frozen/` and all input/result hashes verified. It used MuJoCo 3.12.0, the legacy controller, 0.5 ms steps and a 60 native-second ceiling.

| Door | Source integrator | Private full implicit |
|---|---|---|
| DB0258 | Global solver warning | Opening target not reached |
| DB0313 | Opening target not reached | Native process crash |
| DB0419 | Global solver warning | Sampled hands-free state passes; upper stops carry load |
| DB0636 | Opening target not reached | Native process crash |
| DB0754 | Global solver warning | Global solver warning |
| DB0888 | Global solver warning | Contact tolerance failure |

Thus none of the six source trials returns an accepted full-open state; one of six private full-implicit trials does. Combined counts are one sampled pass, nine native failures and two execution failures. The two crashes are bound to their actual worker PIDs (47420/48887) by macOS reports, with fault stacks beginning in `mj_Jdotv`. They are not physical infeasibility findings. Full implicit is not a universal warning or crash remedy and is not promoted by this comparison.

Three near-open failures expose a separate legacy-controller defect. At rest its force is approximately −750 times the remaining height error: DB0258's 58.499 mm error predicts −43.874 N (recorded −43.832 N), while DB0313/0636's 21.102 mm error predicts and records −15.826 N. A finite supporting force therefore requires a persistent position error; more waiting cannot reach the unchanged 10 mm full-open gate. Load compensation is separate follow-up work. No target tolerance, contact limit or hand-force cap was widened.

### Finite-load opening controller follow-up

Algorithm 4 adds bounded integral speed feedback to the actual material-chain force, retaining the source integrator, 0.5 ms step and 120 N cap. DB0313 now reaches a full-open state and two seconds without hand input in 27.966 native seconds, with 111.52 N peak chain force, 0.268 mm maximum penetration and no native warnings. DB0419 reaches full height without warnings but fails the subsequent keeper-seating transition; its pin remains 31 mm withdrawn. These exact source-bound results are retained in `hoist-pi/db0313_rollup.json` and `hoist-pi/db0419_rollup.json`. Fourteen controller/component tests passed. This partial result does not close the six-door audit. A separate bounded seating-feedback correction is under native test; it is not yet accepted evidence.
