# Browser physics playground: MuJoCo (WebAssembly) with live-tunable constants (task V7)

Read `handoffs/README.md` first. Branch to resume: `worktree-agent-a0650dd8b52f7f671` (one snapshot commit:
WIP `viewer/src/Playground.tsx`, `viewer/src/mujoco/` loader, `viewer/package.json` + `bun.lock` changes, nav
line in `App.tsx`, styles). Check which WASM package the previous agent chose (`git diff master...HEAD --
viewer/package.json`) and whether it loads.

## Why (owner's words)

"Perhaps also somehow have some physics playground (maybe need to be linked to actual Isaac lab live) which allows
you to tune the physical constants and view their effect."

## Goal

1. A "Playground" route for any door that runs the door's real MJCF (`assets/doors/<id>/scene.xml` + meshes
   under `assets/hardware/`) in the browser with MuJoCo compiled to WebAssembly (MuJoCo >= 3.x; needs equality
   constraints, tendons, OBJ/STL meshes; bundle the wasm with vite; the site is plain GitHub Pages). Step in real
   time, render with three.js (reuse `viewer/src/scene.ts` where possible, or draw from MuJoCo geom state).
2. Sliders with units/ranges/defaults from `spec.json["physics"]`: hinge friction / damping / stiction, leaf mass,
   closer spring preload / rate / sweep-latch-backcheck damping / hold-open, latch spring + throw, operator spring
   return, seal friction, gravity/axis tilt (sagging doors), maglock holding force, track friction. Apply them
   1:1 to the MJCF (rewrite XML + rebuild, or set mjModel fields live: `dof_damping`, `dof_frictionloss`,
   `jnt_stiffness`, `body_mass`, equality/tendon params, actuator gains). "Copy as spec override" emits the
   `spec.json["physics"]` override JSON, and the UI states that the USD / Isaac Lab export consumes the same
   parameters (`docs/RUNPOD.md`, `isaaclab/cloud/README.md` for the command).
3. Interaction: torque/force on any joint (buttons or mouse drag on the leaf / handle), presets ("fling open" ->
   backcheck, "release from 60 deg" -> closing time, "push at latch" -> does it hold), pause/step/reset, real-time
   factor; live plots (angle, joint torque, contact force) with the measured closing time / peak speed / final
   angle next to the dataset QA metrics from `qa.json`.
4. Robustness: clear message when a door cannot load in WASM; lazy-load the wasm only on the playground route;
   `npm run build` clean and bundle size reasonable; test db0012_swing_single (closer), a sliding door, a garage
   door, a keypad door.
5. `docs/PLAYGROUND.md`: how it works, parameter map, how to reproduce a tuned door in MuJoCo Python and in Isaac
   Lab, limitations. A bun/vitest test for the parameter -> MJCF mapping (and the WASM load of one door if the
   runtime allows).

## Done when

Playground works on the four test doors from the dev server; screenshots `docs/media/playground_{door,plots}.png`;
viewer typecheck/build/test clean; docs written.
