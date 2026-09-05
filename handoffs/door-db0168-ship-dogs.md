# Ship watertight doors: all dogs must actuate (db0168 and family) — task G9c

Read `handoffs/README.md` first. Start from `master` on a new branch (partial edits may exist on
`worktree-agent-a68f30762fa4dd816`; optional).

## Why (owner's words)

"for this door it only does 1 of 6 hinges https://adamraudonis.github.io/DoorBench/#/door/db0168_ship_watertight"

## Current state

`db0168_ship_watertight`: operator `dog_lever`, latch `dogs_6`, hinge `ship_hinge` x2. `model.json` has bodies
`dog_0` .. `dog_5`, each with its own hinge joint, but only the operator's dog is driven: the other five never
move (viewer "Open door" and the QA drive only touch `meta["operator_joint"]`). Family sampler:
`doorbench/spec.py` ~line 1087 (`n_dogs` in {4, 6, 8}; operator `dog_lever` or `wheel_ship_hatch`).
Geometry: `doorbench/geometry/other.py` (search `dog`).

## Goal

1. `dog_lever` doors (individually dogged): every dog is an independent operator that must be turned; each dog is
   a latch with its own keeper/wedge on the frame; the leaf cannot open while any dog is engaged (QA check);
   `DoorEnv` and the benchmark's unlatch event handle multiple operators (all dogs undogged -> unlatched).
2. `wheel_ship_hatch` doors (quick-acting): the central handwheel drives all dogs simultaneously through a real
   coupling (equality constraints / tendons) with visible linkage rods.
3. Viewer (`viewer/src/doorLogic.ts`): "Open door" undogs all dogs (sequentially for levers, together for the
   wheel) before swinging; the joints panel lists every dog.
4. Check other multi-latch doors for the same defect (cremone bolts, vertical rods, multi-point locks, dutch door
   bolts) and fix consistently.

## Done when

Regenerate -> 1000 signed off; clearance 1000/1000; a new QA check `all_latches_release` passes on every
ship_watertight door; tests; viewer build; `docs/media/db0168_dogs_{closed,undogged,open}.jpg` rendered with
MuJoCo showing all six dogs moving.
