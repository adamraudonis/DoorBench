# Paddle mechanics: future native correction

This change affects subsequently generated native doors only. Existing assets,
hardware mesh files, native recordings and the running v2 reference corpus were
not regenerated. It is not an input-compatible repair for a released motion.

## Evidence and interpretation

The former paddle shared a door-normal spindle with both visible plate/base
assemblies. Its broad-face normal was parallel to that joint axis, so an applied
normal face force had zero hinge moment. In the eight paddle cases from the
268-case missing-contact audit, the authored site was also 10 mm outside its
collision box. A tangential twist could turn that spindle, but that does not
implement the declared push/pull operation.

The [manufacturer's HL6-9000 installation instructions, page 2](https://commercial.schlage.com/content/dam/allegion-us-2/web-documents-2/InstallInstructions/Glynn-Johnson_HL6-9000_Push_Pull_Mortise_Latch_Installation_Instructions_107563.pdf)
show separate horizontal paddle pivots, fixed mounting plates, and push/pull
cams operating actuator pins. The
[HL catalogue, page 4](https://commercial.schlage.com/content/dam/allegion-us-2/web-files/schlage/information-documents/Schlage_HL_Series%20Catalog_101400.pdf)
identifies short-throw push/pull operation from either side. These support a
vertical rocker interpretation; they do not establish the exact cam profile,
backdrivability, spring calibration, or dimensions of this generic model.

## Implemented mechanism

Each requested face has a horizontal, face-mounted rocker pivot. The mounting
plate, bearing ears and pin remain on the leaf. The paddle and neck move about
that pin. The visible grip plate is the collision primitive itself, removing the
old visual/proxy discrepancy. The rest lean equals the existing 0.4 rad travel:
the push paddle ends upright, rather than sweeping its lower edge through the
leaf. Nominal grip moment arms remain 90 mm and 120 mm for the two catalogue
types. Widths and heights remain based on the existing proxy dimensions; this
is a generic hospital rocker, not a dimensionally exact branded reproduction.

The primary operator joint name and its existing latch coupling are preserved.
A second face has a noninteractive follower joint and an explicit **ideal 1:1
cam equality**. This retains the earlier model's ganged-face assumption while
giving each face a physically plausible pivot. Internal cam contact, lost
motion, separate clutch functions and real backdrivability are not modeled or
validated. The new topology must be treated as a new native dataset revision.

The contact site lies on the outer plate face for pushing and the inner plate
face for pulling. Its local +Z points out of that contact surface. For unit
force into that face, `(r × F) · axis` is +0.09 or +0.12 N·m throughout travel;
positive rotation operates the latch and moves the grip in the door-opening
direction. The pull point presumes access by fingers behind the plate; no
humanoid grasp, wrist orientation, or contact-wrench feasibility is certified.

The previous native 1.5 N·m return-preload floor is retained once per paired
assembly, without duplicating it on the follower. The catalogue motion label
now says `rotate_horizontal`. The spring-only force estimate recognizes that
motion and the actual preload floor: 23.33 N for the ordinary paddle and 16.50 N
for the hospital arm. These are estimates, excluding latch/cam friction,
gravity and other loads; they are not accessibility certifications.

## Export boundaries

- MJCF carries both rocker joints and the cam equality. The exporter now
  preserves authored `Site.quat` for both fixed and moving sites; previously it
  silently emitted identity orientation.
- URDF carries the follower's bilateral `mimic` relation. This exporter does
  not emit IR sites/contact frames; consumers need the accompanying model JSON.
- Full `door.usda` carries the follower's `PhysxMimicJointAPI` relation, with
  gearing −1, and oriented site Xforms. These authored properties were reopened
  and inspected with OpenUSD; no Isaac/PhysX simulation parity is claimed.
  Rechecked against the publishing merge of `origin/master` through
  `ed4e8281f`: both paddle axes are rotational, so this coupling remains in
  `mimic` mode. The newer exporter's emulated bilateral path for unsupported
  prismatic couplings does not replace the paddle cam. A focused test checks
  the authored coupling mode and both native/USD coefficient pairs.
- **Canonical `door_rl.usda` is not equivalent for these corrected paddles.**
  Its fixed slot reduction welds each follower into `leaf` or `leaf2`. All 13
  paired assemblies in the isolated export exhibit that limitation. It retains
  the primary rocker and source site Xforms, but its JSON grip list contains
  positions only. That reduced export requires separate future work before
  advertising paired-paddle mechanism equivalence in Isaac training.

## Validation receipt

`tests/test_paddle_mechanics.py` checks both catalogue types, both leaf
handednesses and opening directions, single/both faces, locked backlash,
65-step contact and leaf-clearance sweeps, actual MuJoCo applied-force moments,
and operation/return when either face is loaded. The overload dynamics fixture
bounds the soft cam constraint's induced grip mismatch to 0.2 mm. Negative
fixtures reproduce the old zero-moment axis and 10 mm site error. Export tests
cover unchanged primary/latch binding, MJCF/URDF cam relations, full USD cam/site
properties and fixed/moving native site orientation.
The focused tests plus the six existing core tests pass: **27 passed**.

Separately, all **11 authored paddle doors / 13 paired assemblies** were
exported with every output and hardware mesh directed to
`/tmp/doorbench-paddle-mechanics/assets`. At 65 operator configurations per door,
the query used actual compiled MuJoCo primitive distance checks between moving
paddle/neck colliders and nearby other colliders. A bounding-sphere check only
culled pairs provably farther than the 20 mm query horizon. There were **zero
penetrations**, with a minimum returned clearance of **5.99988 mm**. This sweep
held the main door leaves at their authored initial pose; it is not a combined
door-sweep, full latch dynamics, contact-friction or simulator-parity certificate.
Every file in each corresponding shared source-door directory was hashed before
and after and remained unchanged. The detailed receipt is
`out/reference-contact-audit/paddle-mechanics.json`.

Fixtures: db0039, db0074, db0116, db0158, db0347, db0536, db0615, db0648,
db0660, db0884, db0973. The two double doors contribute two paired assemblies
each. The eight scheduling candidates from the earlier audit remain a distinct
set and are not retroactively repaired by this source change.

Eight source-bound Blender close-ups of db0074 and db0347 were personally
inspected: both faces at rest and full actuation. The pin supports/backplates
remain attached and stationary, the neck remains attached to the paddle, and
the visible motion agrees with push versus pull operation. No visible plate
penetration or detachment was found. These are mechanism-review renders, not a
human-grasp demonstration. Images and hashes are in
`out/reference-contact-audit/paddle-closeups` and the receipt above. The far-side
room is dimmer in the existing lighting configuration but the geometry remains
visible; no appearance renderer or environment source was changed.
