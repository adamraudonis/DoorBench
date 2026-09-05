// bun test — the MuJoCo WebAssembly build loads real doors (meshes, connect equalities, tendons) and the playground
// wrapper steps them, applies live parameters, recompiles with the state kept and runs the QA-style scripts.
// Skipped when ../../assets is absent.
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "../types";
import { computeTargets, defaults, paramDefs, rebuildTargets, rewriteMjcf, setParam } from "./params";
import { loadMujocoModule, type AssetReader } from "./loader";
import { DoorSim } from "./sim";

const ASSETS = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "..", "assets");
const have = existsSync(path.join(ASSETS, "doors"));
const reader: AssetReader = async (rel) => new Uint8Array(readFileSync(path.join(ASSETS, rel)));

function load(id: string) {
  const dir = path.join(ASSETS, "doors", id);
  return { model: JSON.parse(readFileSync(path.join(dir, "model.json"), "utf8")) as ModelJ, spec: JSON.parse(readFileSync(path.join(dir, "spec.json"), "utf8")), qa: JSON.parse(readFileSync(path.join(dir, "qa.json"), "utf8")) };
}

describe.skipIf(!have)("MuJoCo WASM", () => {
  test("module loads and reports MuJoCo 3.x", async () => {
    const mj = await loadMujocoModule();
    expect(mj.mj_versionString()).toMatch(/^3\./);
  });

  test.each(["db0012_swing_single", "db0007_sliding_single", "db0148_garage_sectional", "db0056_swing_single"])("%s compiles and settles", async (id) => {
    const mj = await loadMujocoModule();
    const { model, spec } = load(id);
    const sim = await DoorSim.create(mj, id, model, spec, reader);
    try {
      expect(sim.model.njnt).toBe(model.bodies.filter((b) => b.joint).length);
      expect(sim.primary?.name).toBe(model.meta.primary_joint);
      sim.step(500);
      expect(sim.warnings).toEqual([]);
      for (const j of sim.joints) expect(Number.isFinite(sim.q(j))).toBe(true);
      expect(Math.abs(sim.q(sim.primary))).toBeLessThan(0.05);     // a closed door stays closed
      expect(sim.recorder.n).toBeGreaterThan(100);
    } finally { sim.dispose(); }
  });

  test("closer release test reproduces qa.json (door returns) and reacts to the tuned constants", async () => {
    const mj = await loadMujocoModule();
    const id = "db0012_swing_single";
    const { model, spec, qa } = load(id);
    const sim = await DoorSim.create(mj, id, model, spec, reader);
    try {
      const defs = paramDefs(spec, model);
      sim.applyLive(computeTargets(spec, model, defaults(defs), defs));
      sim.releaseTest();
      while (sim.script) sim.step(100);
      expect(sim.measures.finalAngle!).toBeLessThan(6 * Math.PI / 180);   // QA closer_returns threshold
      expect(Math.abs(sim.measures.finalAngle! - qa.metrics.closer_final_angle)).toBeLessThan(0.02);
      expect(sim.measures.closingTime!).toBeGreaterThan(0.3);
      expect(sim.measures.closingTime!).toBeLessThan(12);
      const t1 = sim.measures.closingTime!;
      // a much weaker closer spring: the rusty hinges win and the door no longer closes
      let v = setParam(defaults(defs), defs, "closer.spring_preload_Nm", 5);
      v = setParam(v, defs, "closer.spring_stiffness_Nm_per_rad", 2);
      sim.applyLive(computeTargets(spec, model, v, defs));
      expect((sim.model.jnt_stiffness as Float64Array)[sim.primary!.id]).toBeCloseTo(2);
      sim.releaseTest();
      while (sim.script) sim.step(100);
      expect(sim.measures.finalAngle!).toBeGreaterThan(20 * Math.PI / 180);
      // heavier closing damping: slower (or no longer closed within the 12 s horizon)
      v = setParam(defaults(defs), defs, "closer.damping_closing", 3 * spec.physics.closer.damping_closing);
      sim.applyLive(computeTargets(spec, model, v, defs));
      sim.releaseTest();
      while (sim.script) sim.step(100);
      expect(sim.measures.closingTime ?? Infinity).toBeGreaterThan(t1 * 1.5);
      expect(sim.measures.finalAngle!).toBeGreaterThan(-0.01);
    } finally { sim.dispose(); }
  });

  test("push test matches qa.json hold / free_opens metrics; fling test reports a peak", async () => {
    const mj = await loadMujocoModule();
    const id = "db0056_swing_single";
    const { model, spec, qa } = load(id);
    const sim = await DoorSim.create(mj, id, model, spec, reader);
    try {
      const defs = paramDefs(spec, model);
      sim.applyLive(computeTargets(spec, model, defaults(defs), defs));
      sim.pushTest(qa.metrics.qa_push);
      while (sim.script) sim.step(100);
      expect(sim.measures.pushDisplacement!).toBeLessThan(2 * Math.PI / 180);      // latched: holds
      expect(Math.abs(sim.measures.pushDisplacement! - qa.metrics.hold_displacement)).toBeLessThan(0.01);
      sim.flingTest();
      while (sim.script) sim.step(100);
      expect(sim.measures.flingPeak!).toBeGreaterThan(0.3);
      sim.actuateTest(qa.metrics.qa_push);
      while (sim.script) sim.step(200);
      expect(sim.measures.actuateOpened!).toBeGreaterThan(20 * Math.PI / 180);    // lever retracts the bolt, door opens
    } finally { sim.dispose(); }
  });

  test("rebuild from a rewritten XML keeps the state and applies the new range", async () => {
    const mj = await loadMujocoModule();
    const id = "db0056_swing_single";
    const { model, spec } = load(id);
    const sim = await DoorSim.create(mj, id, model, spec, reader);
    try {
      const defs = paramDefs(spec, model);
      sim.setJoint(sim.primary!, 0.4);
      const t = computeTargets(spec, model, setParam(defaults(defs), defs, "latch.throw_m", 2 * spec.physics.latch.throw_m), defs);
      await sim.rebuild(rewriteMjcf(sim.xml0, rebuildTargets(t)));
      sim.applyLive(t);
      expect(sim.q(sim.primary)).toBeCloseTo(0.4, 3);
      const bolt = sim.byName.get("leaf_latch_bolt_slide")!;
      expect((sim.model.jnt_range as Float64Array)[2 * bolt.id + 1]).toBeCloseTo(2 * spec.physics.latch.throw_m);
      sim.step(200);
      expect(sim.warnings).toEqual([]);
    } finally { sim.dispose(); }
  });

  test("a broken MJCF surfaces as a readable error", async () => {
    const mj = await loadMujocoModule();
    const { model, spec } = load("db0007_sliding_single");
    await expect(DoorSim.create(mj, "db0007_sliding_single", model, spec, reader, '<mujoco><worldbody><body><geom type="nope"/></body></worldbody></mujoco>')).rejects.toThrow(/nope|XML Error/);
  });
});
