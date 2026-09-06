# Marine open-door holdbacks

The ten ship doors specified a hook holdback but previously had no retaining hardware. The new original assembly has a floor-anchored tubular station, bored and retained pivot, spring-return steel hook, physical load shoulder, rubber opening stops and a crossbar striker welded to the leaf through two ears. Six/eight-dog layouts put the holder between the lowest actual dog rows; placing every holder at quarter height collided with returned dog handles on DB0384 and DB0674.

The topology is informed by A. L. Hansen's steel hook/catch door holder and optional spring catch. The station, hook profile, spring and mounting dimensions are original engineering. They are not recovered manufacturer CAD or a rated marine restraint. [A. L. Hansen 29 Hook & Catch Door Holder](https://alhansen.com/products/29-hook-catch-door-holder).

The shaft passes through an actual 11.2 mm cheek aperture and faceted bearing bores. Separate thrust washers and retaining heads constrain axial movement. Four modeled M8 anchors join the plate to the structural floor. The leaf striker and moving hook keep their geometry-derived mass; the station stays fixed to the floor. The spring has real supported endpoints and native extension force. Gravity also returns the hook, so removing the spring is tested as a loss of its closing-torque contribution, not falsely described as proof that gravity alone cannot return it.

`doorbench.ship_holdback_qa.run_ship_holdback_qa(model, meta)` first operates the real dogs or wheel grip. It then runs two continuous cycles: open through actual leaf-hand force, capture, remove both hands for two seconds under an external 80 N·m closing test load, remove that artificial load, unload the hook with the leaf hand, lift the real hook grip and close completely. It measures contact loads through both jaw and fixed shoulder, so a numerical hinge limit or friction alone cannot satisfy the retention check. The source opening bumper must also carry measured load during the release phase. Native state is never prescribed after initialization.

Manual surface forces remain capped at 120 N. The penetration limit remains 1 mm, and every contact is checked. A 2 ms prototype exceeded that limit on DB0168's minimal tier (1.00494 mm); its identical geometry at 1 ms measured 0.53004 mm. The assembly therefore requests a local 1 ms maximum timestep. The original failed receipts are preserved. A further half-step test uses the same gates. Removing only jaw collision after an actual loaded capture must lose retention; no geometry dimensions are mutated without recompilation.

The physical opening bumper lies before the leaf's farther numerical safety limit. `geometry.ship_holdback.first_ship_holdback_stop_angle(model, meta)` finds its first rigid contact using private inspection data with the dogs released and hook lifted. It never initializes a native task. Inspection limits the primary to this measured geometric stop and the hook to its operating lift range; the independently forced native gate verifies the load path. No contact exclusions or blanket penetration allowances are added.

Development and final receipts are under ignored `out/mechanical-foundations/ship-holdback/`. Until final source promotion and bound native/clearance checks are complete, the production `hook_holdback` incomplete flag remains. These checks do not establish an embodied human task, complete traversal, weld/anchor strength, marine pressure resistance, fatigue, corrosion performance or equilibrium under arbitrary loads.

```sh
PYTHONPATH=. python -m pytest tests/test_ship_holdback.py --basetemp=out/ship-holdback-test
```
