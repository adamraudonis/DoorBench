// bun test — parameter map -> MJCF rewriting against real dataset files in ../../assets (skipped when absent).
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "../types";
import { computeTargets, defaults, fmtNum, needsRebuild, paramDefs, rebuildTargets, rewriteMjcf, setParam, specOverride, tiltAxis } from "./params";

const ASSETS = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "..", "assets", "doors");
const have = existsSync(ASSETS);

function load(id: string) {
  const dir = path.join(ASSETS, id);
  const model: ModelJ = JSON.parse(readFileSync(path.join(dir, "model.json"), "utf8"));
  const spec = JSON.parse(readFileSync(path.join(dir, "spec.json"), "utf8"));
  const xml = readFileSync(path.join(dir, "door.xml"), "utf8");
  return { model, spec, xml };
}

const attr = (xml: string, joint: string, name: string): number | null => {
  const m = new RegExp(`<joint\\b[^>]*\\bname="${joint}"[^>]*\\b${name}="([^"]+)"`).exec(xml);
  return m ? parseFloat(m[1]) : null;
};

const DOORS = ["db0012_swing_single", "db0056_swing_single", "db0007_sliding_single", "db0148_garage_sectional", "db0002_swing_single", "db0019_swing_double", "db0066_revolving", "db0123_saloon"];

describe.skipIf(!have)("playground parameter map", () => {
  test("defaults come from spec.json physics and the groups match the door", () => {
    const { model, spec } = load("db0012_swing_single");   // closer, rusty butt hinges, no operator
    const defs = paramDefs(spec, model);
    const keys = defs.map((d) => d.key);
    expect(keys).toContain("closer.spring_preload_Nm");
    expect(keys).toContain("hinge.coulomb_torque_Nm");
    expect(keys).not.toContain("latch.operator_spring_rate");      // pull handle: no return spring
    expect(keys).not.toContain("roller.coulomb_force_N");
    const d = defs.find((x) => x.key === "closer.spring_preload_Nm")!;
    expect(d.default).toBeCloseTo(spec.physics.closer.spring_preload_Nm);
    expect(d.unit).toBe("N·m");
    for (const x of defs) { expect(x.min).toBeLessThanOrEqual(x.default); expect(x.max).toBeGreaterThanOrEqual(x.default); expect(x.what.length).toBeGreaterThan(10); expect(x.mjcf.length).toBeGreaterThan(3); }
  });

  test("sliding and vertical doors expose track friction / counterbalance instead of hinge and closer", () => {
    const s = paramDefs(load("db0007_sliding_single").spec, load("db0007_sliding_single").model).map((d) => d.key);
    expect(s).toContain("roller.coulomb_force_N");
    expect(s).not.toContain("closer.spring_preload_Nm");
    expect(s).not.toContain("hinge.coulomb_torque_Nm");
    const g = paramDefs(load("db0148_garage_sectional").spec, load("db0148_garage_sectional").model).map((d) => d.key);
    expect(g).toContain("roller.counterbalance_fraction");
  });

  test("keypad lever door exposes latch, operator and closer groups", () => {
    const keys = paramDefs(load("db0056_swing_single").spec, load("db0056_swing_single").model).map((d) => d.key);
    for (const k of ["latch.bolt_spring_preload_N", "latch.bolt_spring_rate_N_per_m", "latch.throw_m", "latch.operator_spring_preload", "latch.operator_spring_rate", "closer.damping_closing", "closer.backcheck_angle_rad"]) expect(keys).toContain(k);
  });

  test.each(DOORS)("identity: default values leave door.xml unchanged (%s)", (id) => {
    const { model, spec, xml } = load(id);
    const defs = paramDefs(spec, model);
    const t = computeTargets(spec, model, defaults(defs), defs);
    expect(Object.keys(t.joints)).toEqual([]);
    expect(Object.keys(t.bodyMassScale).every((b) => Math.abs(t.bodyMassScale[b] - 1) < 1e-12)).toBe(true);
    expect(rewriteMjcf(xml, t)).toBe(xml);
    expect(needsRebuild(t)).toBe(false);
  });

  test("closer preload / rate -> stiffness and springref of the leaf hinge (live)", () => {
    const { model, spec, xml } = load("db0012_swing_single");
    const defs = paramDefs(spec, model);
    let v = defaults(defs);
    v = setParam(v, defs, "closer.spring_preload_Nm", 60);
    v = setParam(v, defs, "closer.spring_stiffness_Nm_per_rad", 20);
    const t = computeTargets(spec, model, v, defs);
    expect(t.joints.leaf_hinge.stiffness).toBeCloseTo(20);
    expect(t.joints.leaf_hinge.springref).toBeCloseTo(-60 / 20);
    expect(needsRebuild(t)).toBe(false);
    const out = rewriteMjcf(xml, t);
    expect(attr(out, "leaf_hinge", "stiffness")).toBeCloseTo(20);
    expect(attr(out, "leaf_hinge", "springref")).toBeCloseTo(-3);
    // only that joint changed
    expect(out.replace(/<joint\b[^>]*\bname="leaf_hinge"[^>]*\/>/, "")).toBe(xml.replace(/<joint\b[^>]*\bname="leaf_hinge"[^>]*\/>/, ""));
    // the law carries the closing-side damping
    expect(t.law.closers[0].joint).toBe("leaf_hinge");
    expect(t.law.closers[0].dampingClosing).toBeCloseTo(spec.physics.closer.damping_closing);
  });

  test("hinge friction is a delta on frictionloss (coulomb + stiction / 2), seal moves the total", () => {
    const { model, spec } = load("db0012_swing_single");
    const defs = paramDefs(spec, model);
    const j = model.bodies.find((b) => b.joint?.name === "leaf_hinge")!.joint!;
    let v = setParam(defaults(defs), defs, "hinge.coulomb_torque_Nm", spec.physics.hinge.coulomb_torque_Nm + 10);
    expect(computeTargets(spec, model, v, defs).joints.leaf_hinge.frictionloss).toBeCloseTo(j.frictionloss + 10);
    v = setParam(v, defs, "hinge.stick_torque_Nm", spec.physics.hinge.stick_torque_Nm + 4);
    expect(computeTargets(spec, model, v, defs).joints.leaf_hinge.frictionloss).toBeCloseTo(j.frictionloss + 12);
    const before = v["hinge.coulomb_torque_Nm"];
    v = setParam(v, defs, "hinge.seal_contribution_Nm", v["hinge.seal_contribution_Nm"] + 1);
    expect(v["hinge.coulomb_torque_Nm"]).toBeCloseTo(before + 1);
  });

  test("leaf mass scales the leaf inertial (mass + diaginertia) and nothing else", () => {
    const { model, spec, xml } = load("db0012_swing_single");
    const defs = paramDefs(spec, model);
    const v = setParam(defaults(defs), defs, "mass.total_kg", 2 * spec.physics.mass.total_kg);
    const t = computeTargets(spec, model, v, defs);
    expect(t.bodyMassScale.leaf).toBeCloseTo(2);
    const out = rewriteMjcf(xml, t);
    const m0 = /<body name="leaf"[^>]*>\s*<inertial[^>]*mass="([^"]+)"[^>]*diaginertia="([^"]+)"/.exec(xml)!;
    const m1 = /<body name="leaf"[^>]*>\s*<inertial[^>]*mass="([^"]+)"[^>]*diaginertia="([^"]+)"/.exec(out)!;
    expect(parseFloat(m1[1])).toBeCloseTo(2 * parseFloat(m0[1]), 4);
    const d0 = m0[2].split(" ").map(parseFloat), d1 = m1[2].split(" ").map(parseFloat);
    for (let i = 0; i < 3; i++) expect(d1[i]).toBeCloseTo(2 * d0[i], 6);
    // other inertials untouched
    expect((out.match(/<inertial/g) ?? []).length).toBe((xml.match(/<inertial/g) ?? []).length);
    expect(out.split("\n").filter((l) => l.includes("<inertial") && !l.includes(m1[1])).join("\n")).toBe(xml.split("\n").filter((l) => l.includes("<inertial") && !l.includes(m0[1])).join("\n"));
  });

  test("gravity lands in <option gravity>", () => {
    const { model, spec, xml } = load("db0007_sliding_single");
    const defs = paramDefs(spec, model);
    const t = computeTargets(spec, model, setParam(defaults(defs), defs, "gravity", 1.62), defs);
    const out = rewriteMjcf(xml, t);
    expect(/<option\b[^>]*gravity="0 0 -1.62"/.test(out)).toBe(true);
  });

  test("latch throw is a rebuild: bolt range and tendon coefficient scale together", () => {
    const { model, spec, xml } = load("db0056_swing_single");
    const defs = paramDefs(spec, model);
    const thr0 = spec.physics.latch.throw_m;
    const t = computeTargets(spec, model, setParam(defaults(defs), defs, "latch.throw_m", 2 * thr0), defs);
    expect(needsRebuild(t)).toBe(true);
    expect(t.joints.leaf_latch_bolt_slide.range![1]).toBeCloseTo(2 * thr0);
    expect(t.tendonCoefScale.leaf_latch_bolt_coupling).toBeCloseTo(2);
    const out = rewriteMjcf(xml, rebuildTargets(t));
    const rng = /<joint name="leaf_latch_bolt_slide"[^>]*range="([^"]+)"/.exec(out)![1].split(" ").map(parseFloat);
    expect(rng[1]).toBeCloseTo(2 * thr0);
    const coef0 = /<fixed name="leaf_latch_bolt_coupling"[\s\S]*?<joint joint="leaf_handle_hinge" coef="([^"]+)"/.exec(xml)![1];
    const coef1 = /<fixed name="leaf_latch_bolt_coupling"[\s\S]*?<joint joint="leaf_handle_hinge" coef="([^"]+)"/.exec(out)![1];
    expect(parseFloat(coef1)).toBeCloseTo(2 * parseFloat(coef0), 5);
    // the bolt's own term is untouched
    expect(/<joint joint="leaf_latch_bolt_slide" coef="1"\/>/.test(out)).toBe(true);
    // rebuild-only targets carry no live edits
    expect(rebuildTargets(t).joints.leaf_latch_bolt_slide.stiffness).toBeUndefined();
  });

  test("operator return spring maps onto the handle joint", () => {
    const { model, spec } = load("db0056_swing_single");
    const defs = paramDefs(spec, model);
    let v = setParam(defaults(defs), defs, "latch.operator_spring_preload", 0.7);
    let t = computeTargets(spec, model, v, defs);
    expect(t.joints.leaf_handle_hinge.stiffness).toBeUndefined();          // rate unchanged
    expect(t.joints.leaf_handle_hinge.springref).toBeCloseTo(-0.7);        // -preload / rate with rate = 1
    v = setParam(v, defs, "latch.operator_spring_rate", 2.0);
    t = computeTargets(spec, model, v, defs);
    expect(t.joints.leaf_handle_hinge.stiffness).toBeCloseTo(2.0);
    expect(t.joints.leaf_handle_hinge.springref).toBeUndefined();          // -0.7 / 2 = the shipped -0.35: unchanged
  });

  test("counterbalance fraction scales the garage spring; roller friction is a delta", () => {
    const { model, spec } = load("db0148_garage_sectional");
    const defs = paramDefs(spec, model);
    const j = model.bodies.find((b) => b.joint?.name === "door_slide")!.joint!;
    let v = setParam(defaults(defs), defs, "roller.counterbalance_fraction", 0.5);
    let t = computeTargets(spec, model, v, defs);
    expect(t.joints.door_slide.stiffness).toBeCloseTo(j.stiffness * 0.5 / 0.95);
    expect(t.joints.door_slide.springref).toBeUndefined();     // travel / 0.3 does not depend on cb
    v = setParam(v, defs, "roller.coulomb_force_N", spec.physics.roller.coulomb_force_N + 20);
    t = computeTargets(spec, model, v, defs);
    expect(t.joints.door_slide.frictionloss).toBeCloseTo(j.frictionloss + 20);
  });

  test("axis tilt rotates the hinge axis about x and needs a rebuild", () => {
    const { model, spec, xml } = load("db0012_swing_single");
    const defs = paramDefs(spec, model);
    const t = computeTargets(spec, model, setParam(defaults(defs), defs, "hinge.axis_tilt_deg", 5), defs);
    expect(needsRebuild(t)).toBe(true);
    const ax = t.joints.leaf_hinge.axis!;
    expect(Math.hypot(...ax)).toBeCloseTo(1);
    expect(ax[0]).toBeCloseTo(0);
    expect(Math.abs(ax[1])).toBeCloseTo(Math.sin(5 * Math.PI / 180), 5);
    const out = rewriteMjcf(xml, rebuildTargets(t));
    expect(/<joint name="leaf_hinge"[^>]*axis="0 [-0-9.]+ [-0-9.]+"/.test(out)).toBe(true);
    expect(tiltAxis([0, 0, 1], Math.PI / 2)[1]).toBeCloseTo(-1);
  });

  test("spec override lists only changed keys at their spec paths", () => {
    const { model, spec } = load("db0056_swing_single");
    const defs = paramDefs(spec, model);
    let v = setParam(defaults(defs), defs, "closer.spring_preload_Nm", 50);
    v = setParam(v, defs, "latch.throw_m", 0.02);
    v = setParam(v, defs, "hinge.axis_tilt_deg", 2);
    const o = specOverride("db0056_swing_single", defs, v);
    expect(o).toEqual({ id: "db0056_swing_single", hinge: { axis_tilt_deg: 2 }, physics: { closer: { spring_preload_Nm: 50 }, latch: { throw_m: 0.02 } } });
    expect(specOverride("x", defs, defaults(defs))).toEqual({ id: "x" });
  });

  test("number formatting matches the exporter", () => {
    expect(fmtNum(10.168374)).toBe("10.168374");
    expect(fmtNum(-3.2079740000001)).toBe("-3.207974");
    expect(fmtNum(0)).toBe("0");
    expect(fmtNum(1e-14)).toBe("0");
    expect(fmtNum(2)).toBe("2");
  });
});
