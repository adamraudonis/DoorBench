# Internal lock stock and contact surfaces

The initial repaired construction removes the swept bolt volume from the actual panel and case geometry. Its source-bound historical direct audit covers **467 doors / 542 bolt instances**, in all three native tiers. All pass a **0.75 mm internal guide clearance** at 17 travel samples; the smallest measured gap is 0.7499986 mm. Rotating thumbturns and spindles additionally pass a **0.5 mm** parent-stock clearance across their native travel. This is an internal lock construction check, not a whole-door mechanical, security, accessibility or strength certificate.

The earlier conventional inventory found 83 deadbolt stock intersections hidden by MuJoCo's parent-child contact filtering. Seven were multipoint assemblies, now independently rebuilt by the parent task; 76 were conventional deadbolts. Extending the direct check to all parent colliders also exposed **35 solid exit-device cases around moving bolts**. Those cases now contain real open passages. Restored-stock and restored-guide negative fixtures fail the direct gate even when the native contact filter hides the overlap.

`geometry/lock_stock.py` splits the visible and collision boxes, preserves material density and prorates any explicit per-geom mass override over the remaining volume. Original metal cartridge walls guide the bolt with the stated gap. The geometric cut includes the full scalar travel and respects `modeled_at`; it cannot simply be a hole around the initial pose. Exact case/stock cuts and guide geometry names are in `model.meta.lock_stock`. Native assembly mass continues to use the separately documented material/catalogue reconciliation; this audit does not independently certify a cartridge mass or commercial lock rating.

There are **19 locally thickened cartridges on 16 doors**, where the real bolt guide exceeds the available sheet thickness: DB0011, DB0016, DB0039 (2), DB0161 (2), DB0232, DB0295, DB0308, DB0418, DB0425, DB0479, DB0615, DB0817 (2), DB0902, DB0912, DB0919 and DB0922. These use a prepared edge notch and metal face straps contacting intact sheet through protective pads. They do not turn a 10–19 mm glass panel into a fictitious 35 mm slab. Glass fabrication stresses, tempered-glass processing, fastener preload and structural clamp capacity remain outside this simplified model.

The thumbturn has a real 6 mm spindle through an 8 mm prepared stock bore, a separate hub and paddle, and a fixed open rosette. Its paddle sites are named `<bolt>_thumbturn_grip_-1` and `_1`; these are opposite radial surfaces, **not** opposite door faces. `thumbturn_face` and `thumbturn_accessible_from_robot` state which physical face carries it. The former bolt/thumbturn contact exclusion is removed. A bilateral native joint relation still represents the concealed cam transmission; the internal cam/rack is not individually simulated. A visible keyed cylinder explicitly records `key_input.supported=false`: no key insertion or key-contact mechanism is claimed.

Ten keypad housings now have real rear spindle sockets. Their original box-union meshes are decomposed into identical boxes before subtraction; a convex collision hull cannot silently fill the new opening. DB0063, DB0201 and DB0912 also have prepared bores through the overlapping push plates. Thin cartridge mounting straps receive the matching local stop/seal notch. Fresh whole geometric and running-clearance probes pass DB0063, DB0086, DB0182, DB0201, DB0295, DB0304, DB0767 and DB0912; that eight-door check does not certify the remaining unrelated hardware.

The four rim night latches place the bolt inside their actual surface case. Their case/spindle passages and strike offset follow that placement. The other spring latches retain the real closing bevel and one-sided handle coupling. The final **490 latch-to-operator tendon endpoints** are checked against their actual joint ranges. DB0767's rotary handle formerly demanded 13.5 mm beyond the bolt stop. Eight handleset thumbpieces (DB0114, DB0149, DB0330, DB0408, DB0599, DB0767, DB0836 and DB0929) additionally demanded 1.19–1.78 mm beyond it because their ratios subtracted dead travel without modeling a dead-travel offset. Ratios now respect the real endpoints; no unmodeled lost-motion interval is claimed. Realized spring-latch ratios are recorded in `lock_stock[].handle_coupling`, including the final multipoint lever travel.

Existing six-case bevel/strike return tests still pass, as do four capped-torque, two-cycle thumbturn fixtures and the long-travel DB0767 operator fixture. Restoring solid stock in the keypad or push-plate bores fails the direct rotating-spindle gate. The combined focused run is **46 tests passed**.

Manufacturer guidance confirms that installing conventional hardware requires real cross/edge bores and stock preparation: [Schlage door-preparation checklist](https://www.schlage.com/en/blog/product_updates/install-door-hardware-with-ease-with-this-door-prep-checklist.html) and [B60/B62 preparation template](https://www.schlage.com/content/dam/sch-us/documents/pdf/installation-manuals/P515-771.pdf). Our compact cartridges and spindle bore are original simplified geometry, **not** copies of a Schlage product or its commercial crossbore dimensions. No OEM CAD is incorporated.

Reproduce the current full-catalogue gate without writing generated assets:

```sh
PYTHONPATH=. .venv/bin/python scripts/audit_lock_stock.py \
  --out out/mechanical-foundations/lock-stock/final-inventory.json
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_lock_stock.py tests/test_latch_strikes.py \
  --basetemp=out/mechanical-foundations/lock-stock/pytest-review
```

The local inventory binds every spec, IR model and compiled XML plus all source hashes and MuJoCo version. Receipt SHA256: `07daed285c24ef015e7113d70c1843a5c2b175f1ede72f73f1c7e95c0a6686d9`; `sources_unchanged=true`, MuJoCo 3.12.0, 38.07 seconds. Subsequent source changes require fresh generation and a new receipt. The separate source-only conventional inventory is not a substitute for these direct checks or the remaining native mechanism and whole-door QA.

## Installed-door follow-up

Fresh whole-door testing exposed additional interfaces beyond the original
internal-stock proof. All 16 thin-panel cartridge doors now pass whole-door QA.
Their clamping straps lie 15 mm inboard of the panel edge. DB0232 and DB0425 use
8.25 mm supported knob offsets, bored steel spacers, a 0.75 mm neck-to-spacer gap
and an actual 13.5 mm prepared panel bore. Eight focused tests cover all three
tiers and reject restored knob interference and refilled bores.

The two panic-device/exterior-trim combinations, DB0286 and DB0548, previously
summed both inputs in one tendon and could demand too much latch travel. They
now use two independent one-sided ideal cam relations. Each input alone fully
withdraws the bolt; operating both does not double its stroke. Both doors pass
fresh whole-door QA and four native regression cases cover A, B, both, return
and the restored additive defect. The internal cams remain idealized.

Physical inside thumbturns and auxiliary bolts now remain present with full
travel in every tier even when the outside robot has no release. Robot input
permissions remain separate. A bounded native service fixture can use the real
inside control without changing joint limits. Eight inside-lock cases and four
inside-panic cases pass fresh whole-door QA. The conventional independent rotary
lockset rebuild is still in progress; this section does not certify it.
