# Five rising-hinge cold-storage doors whose closer loop cannot close — part of task G5

Read `handoffs/README.md` first. Start from `master` on a new branch (or fold into `closer-mechanisms.md`).

## What is wrong

Doors `db0188`, `db0432`, `db0549`, `db0937` (13.3 mm) and `db0585` (12.0 mm) — all `cold_storage` doors with a
rising hinge: the leaf has a `leaf_rise` slide joint coupled to `leaf_hinge` (`0.00764 * leaf_hinge`, equality)
so the whole leaf, and the closer body on it, lifts as the door opens, while the closer's forearm shoe stays on
the frame. A planar two-bar arm cannot follow a lifting pinion, so in MuJoCo the `connect` equality and the rise
coupling fight (the viewer's loop solver reports the residual). Found by `viewer/src/kinematics.test.ts` (the
dataset-wide sweep prints these as geometry notes).

## Goal

Model what real cold-room doors do: rising hinges are normally paired with a **gravity self-close** (the rise
itself closes the door) and no overhead arm closer, or with a closer whose arm has a ball-joint / vertical slack
at the shoe. Pick per door from `spec["closer"]["model"]`: if the closer is an arm type on a rising-hinge door,
either (a) switch the spec post-processing (`doorbench/spec.py`) to `gravity_rise` self-closing for rising-hinge
cold-storage doors (documented; regenerate; the physics `closer` block then describes the rise-based closing
torque), or (b) give the forearm tip a vertical slide degree of freedom at the shoe (real "slotted shoe") so the
loop closes at every angle. Either way: `connect` residual < 1 mm over the sweep, closing behaviour still passes
QA, and the viewer's dataset-wide linkage test prints no notes.

## Done when

`cd viewer && npm test` shows 0 geometry notes; regenerate -> 1000 signed off; clearance 1000/1000; tests green;
`docs/media/cold_storage_rise_db0188_{0,45,90}.png` rendered.
