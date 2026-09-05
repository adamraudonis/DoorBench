# Twist handles snap back with a spring (task G9a)

Read `handoffs/README.md` first. Branch with partial work: `worktree-agent-a68f30762fa4dd816` (one snapshot
commit with uncommitted edits to `doorbench/{build,hardware,ir,physics,qa}.py` and
`doorbench/geometry/{common,hinged,other}.py`; the agent was about to add `operator_dynamics()` to
`physics.py`). Run `git diff master...HEAD` first and keep only what is coherent; the same branch also holds
partial keypad/dog/track edits (`keypad-codes.md`, `door-db0168-ship-dogs.md`, `door-db0079-sliding-track.md`).

## Why (owner's words)

"Do you ensure that for some of the twist to open door handle doors, that if you stop twisting it snaps back
with a spring like motion? That seems important to the physical correctness."

## Current state

`doorbench/hardware.py::OPERATORS` entries carry `spring_torque_preload` / `spring_rate`;
`doorbench/physics.py` copies them to `phys["operator_spring_preload"] / ["operator_spring_rate"]`;
`doorbench/geometry/common.py::add_rotary_operator` builds the operator joint; `doorbench/qa.py` has a
`return` check (the spring latch re-extends after release) but nothing asserts that the HANDLE returns.

## Goal

1. Every operator that has a return spring in reality (levers, knobs, paddles, push/touch bars, thumb latches,
   panic bars, keypad levers ...) carries preload + rate (realistic: ANSI/BHMA grade 1 lever return ~0.5-1.5 N m
   preload) and damping so that when the drive is released the handle returns to rest with a damped spring-like
   motion (return time ~0.2-0.5 s, no chatter), also while the leaf is open (latch held retracted by nothing).
   Thumbturns, deadbolt turns, handwheels and dog levers stay where put: list them and why, from the catalogue.
2. `spec.json["physics"]["operator"]` documents the parameters with units (preload, rate, damping, armature,
   expected return time, return kind).
3. QA `operator_returns` for all 1000 doors: drive to full travel, release, assert return to rest within
   tolerance/time and no residual offset. Add tests.
4. Viewer (`viewer/src/doorLogic.ts`, `viewer/src/DoorView.tsx`): when the operator slider is released, or after
   "Open / close door" actuates the handle, animate the snap-back with the same damped-spring profile; label the
   slider "spring return".

## Done when

Regenerate -> 1000 signed off, clearance 1000/1000, `operator_returns` 1000/1000 (or documented exceptions),
tests green, viewer typecheck/build/test clean, a short clip or frame sequence of one lever returning under
`docs/media/`.
