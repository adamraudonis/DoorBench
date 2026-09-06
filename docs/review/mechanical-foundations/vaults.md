# Vault and blast boltwork

Production `build_vault` now builds supported crank-and-rod boltwork and frame-mounted crane journals. All 14 sources pass integrated QA, and all 42 native tier cases pass the continuous service and removal-control gates. The corrected local assets and recordings have been regenerated for mechanical verification; the public deployment still contains the earlier versions.

The 14 sources are DB0124, DB0179, DB0288, DB0352, DB0426, DB0458, DB0530, DB0623, DB0672, DB0748, DB0772, DB0913, DB0921 and DB0960. Seven use a handwheel and seven use two independently operated levers. All retain their original condition, hinge friction, task and credential declarations.

## Resulting construction

Each wheel drives a supported 20:100 spur pair. Its output crank moves a bored, pin-connected steel rod, which drives a rigid carrier and four or eight solid bolts. The two-lever construction gives each lever its own crank, rod and bolt. No direct operator-to-bolt follower equality remains. The single gear equality explicitly represents ideal keyed gearing; gear teeth are visible but do not provide a second, redundant contact constraint.

The original thick-leaf running gaps remain. Actual bolt travel is the catalogue throw plus the gap beyond a 4 mm running allowance. Metadata reports these values separately. This matters on DB0124: a literal 50 mm projection failed to reach the frame across its 58 mm gap. The corrected mechanism has 104 mm travel and enters the original strike pocket in the closed state.

Both operator shafts and bolt guides have actual open bores, retained shafts, mounting brackets and prepared leaf stock. Connecting rods have two bored eyes and retained pins. Carrier end stops and a crank return stop carry the travel-limit reactions through actual native contact pairs; wider joint ranges are safety bounds. Separate surface grips on both leaf faces apply leaf-opening force after the boltwork is released.

Crane hinges now contain frame-mounted journals, bored moving sleeves, thrust washers, arms and leaf mounting plates. A frame-backed closing rebate overlaps the leaf by 20 mm. Its handgrip window clears the non-swing-side pull during initial opening. The small negative primary safety range lets the rebate, rather than the q=0 limit, carry closing load.

Prepared stock is subtracted once using each original slab's effective material density. The old operator and half-hinge catalogue allowances are removed; actual added geometry retains its own mass and inertia in every tier. A vault composite remains a homogeneous material/inertia representation, not a resolved laminate, reinforcing cage or armor design.

## Evidence and scope

`tests/test_vault_hardware.py` passes 100 structural, contact-selection and inspection cases; the combined run with the marine mount regressions passes 123 cases. This covers all 14 sources, three native tiers, 49 input positions, 361 released leaf positions, exact rod closure, surface-normal acquisition rays, frame/leaf mounting continuity, bore gaps, material accounting, and controls with detached supports, floating rebates or disabled bolt colliders. Clearance stops the locked sweep at the first actual bolt/strike contact and sweeps only real independent operators over their nominal travel. The separately completed 42 native cases are deselected from this short test run.

The continuous gate uses at most 66.7 N on an operator surface and 120 N on a fixed leaf pull. It requires two complete release/open/close/rethrow/load cycles, hands-off retention, actual bolt/strike and end-stop reactions, no native warnings, penetration and loop residual below 1 mm, and no primary joint-limit reaction. Disabling the rod's contacts and pin connection must sever bolt actuation; disabling the actual locking-bolt contacts must remove the opening arrest. Inertial bodies remain in these negative controls. Gate-controller changes affect test-applied forces only; they do not alter source friction, geometry or native state during stepping.

The final 42-case matrix contains 84 complete service cycles and 84 negative-control cases. Maximum penetration is **0.126264 mm**, native warnings are **zero**, and the primary joint-limit reaction is **zero**. Maximum absolute equality coordinate is 0.000244269, across gear-angle and Cartesian point constraints; multiplying by √3 gives a conservative point-residual bound below **0.424 mm**. Every final XML differs from its proven input only in the relative compiler texture directory; all compiled numeric model arrays and checked simulation options match exactly. Fresh integrated QA, including the native gate, passes **14/14**.

The source/function hashes, all input and hardware hashes, per-door results, exact parity check, and preserved original proof inputs are bound by `out/mechanical-foundations/vault/final-receipt.json`. Evidence is under `out/mechanical-foundations/vault/final-source/`. Earlier development receipts remain under the same class directory as failed or superseded evidence, including the incomplete throw, far-face pull/rebate collision, and controller braking failures. The post-RNG catalogue comparison changes only seven vault operator/latch/lock descriptions and tags; the other 993 generated specifications are identical.

These checks establish the authored native load paths and service operation. They do not certify material stress, fatigue, gear tooth compliance, blast resistance, burglary resistance, sealing, human reach, credential manipulation, or cross-engine dynamics. Native ideal bearing joints carry the journal reactions. The authored faceted bore and thrust gaps are modeled values, not manufacturer measurements. All MJCF, URDF and USD files serialize, but full USD explicitly omits the rod's native point-loop constraint, reduced USD additionally merges native degrees of freedom, and URDF does not supply closed-loop dynamics. Those exports are not mechanically equivalent to the validated native mechanism; no Isaac parity is asserted.

## Primary design references

[Fort Knox's manufacturer catalogue](https://www.ftknox.com/wp-content/uploads/2017/05/Ft-Knox-2017-Catalog_Light_V6_Web_SM.pdf), printed pages 4–5, describes a 5:1 geared vault handle and ball-bearing hinges. It supports the general geared topology, not the dimensions or force law authored here. The PDF text was read; a successful diagram screenshot was not obtained.

[US879229A](https://patents.google.com/patent/US879229A/en), the original crane-hinge patent description, identifies a journal, surrounding sleeve/bearing box and thrust support. [Kiesler Machine's vault-hinge documentation](https://www.kieslermachine.com/ca/vault-hinges/) provides contemporary manufacturer context for heavy vault hinges. These references inform a generic supported construction; this is not a reproduction of a rated product.

## Frame fit follow-up

Personal inspection of all 14 corrected closed frames confirmed continuous head and sill backing across the former gaps. This adds 14 reset-state inspections to the earlier 63 opening, closing and input close-ups; the image hashes and observations are in `out/mechanical-foundations/vault/native-view/frame-inspection/receipt.json`. These are diagnostic viewport captures, not new appearance renders.

Fresh whole-door QA passes all 14 frame-fit sources. Native task recordings pass all 14 opening tasks, all 14 closing-only tasks and all 14 opening-then-closing tasks: **42/42**. Opening uses fixed pulls after actual bolt withdrawal; closing seats the leaf before throwing every bolt. The three validation indices are `0d636cb58f0a09ffde77cd9194fef72f7aba36cbc15cc0afa553966ea96733e9`, `59289d31dd2d78fb3e0aabbd17234bdd12cde80098da244572c98ce9cb322f00` and `378169c14b94ecbfb2b5680b67d1134c00ae99faff178bbf6000a1f65a651a70`. They establish file/state correspondence separately from the continuous mechanism gates. These are bounded scripted mechanism inputs; no embodied human ground truth is claimed.
