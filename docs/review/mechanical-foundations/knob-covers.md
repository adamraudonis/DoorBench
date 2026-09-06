# Knobs with freely rotating covers

The original eight covers were rigid parts of the knob. They could therefore
turn the latch by applying force to the supposedly protective shell. The
replacement is an independently rotating, retained shell with two opposing
finger openings and a separate inner knob/spindle.

This is original generic geometry informed by the access-hole operating class
described by [Safety 1st](https://safety1st.com/products/home-safeguarding-set-80-piece-hs265).
It does not claim product replication or certified child resistance.

The shell has a real cavity in both visible and collision geometry. Individual
convex sectors preserve its openings; an enclosing convex hull would fill them.
The fixed mounting rose has a spindle bore. Its free bearing uses the shell's
geometric inertia, without the old added operator inertia.

`tests/test_knob_covers.py` checks all eight doors in all three physics tiers.
A finite 12 mm finger probe traverses each opening and the required knob turn.
An external shell load rotates the cover without turning the knob; two opposed
inner-knob forces turn the spindle and operate the door. Filled-aperture and
rigidly-coupled-shell counterexamples must fail. Every primary native baseline
also succeeds with two recorded surface forces, each bounded at 20 N.

All eight generated full QA reports passed. The native recording index at
`out/mechanical-foundations/knob-covers/reference-motions/index.json` has SHA-256
`ed36984a97a58c078b388bccdf9d7a481bea1f8ee0a321aa1e2b6c883bd145a9`.
Source hashes and every saved body/contact frame passed the native validator.
The review included 16 browser operation/open-door captures, followed by closer
approach-side views where the initial camera obscured the openings. These are
diagnostic inspection captures; appearance assets were not regenerated.

IDs: DB0120, DB0278, DB0364, DB0578, DB0703, DB0819, DB0938, DB0954.
This verifies the stated mechanism and finite finger-clearance scope. It does
not validate human reach, balance, grasp synthesis or child-resistant behavior.
