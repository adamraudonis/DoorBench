// bun test — closed-loop linkage solver against real model.json files from ../assets (run after generating the dataset).
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { Vector3 } from "three";
import type { ModelJ, TelescopingLinkageJ, TwoBarLinkageJ } from "./types";
import { LoopSolver, hasLoops } from "./kinematics";

const ASSETS = path.resolve(process.env.DOORBENCH_TEST_ASSETS || path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "assets"), "doors");
const have = existsSync(ASSETS);

function load(id: string): ModelJ {
  return JSON.parse(readFileSync(path.join(ASSETS, id, "model.json"), "utf8"));
}

/** Joint -> [c0, c1] for joints driven by a polynomial coupling off the leaf joints (rising hinges, ...), as scene.ts applies them. */
function leafCouplings(model: ModelJ): Map<string, [number, number]> {
  const leaves = new Set([model.meta.primary_joint, model.meta.secondary_joint].filter(Boolean));
  const out = new Map<string, [number, number]>();
  for (const e of model.equalities ?? []) if (e.kind === "joint" && e.b && leaves.has(e.b)) out.set(e.a, [e.polycoeff[0], e.polycoeff[1]]);
  return out;
}

/** Sweep the primary joint over its range (plus the secondary joint of pairs and the joints coupled to them) and return the
 *  worst separation per loop. */
function sweep(model: ModelJ, opts: { forceGeneric?: boolean } = {}, steps = 60) {
  const s = new LoopSolver(model, opts);
  const primary = s.art.joints.get(model.meta.primary_joint);
  if (!primary) throw new Error("no primary joint");
  const [lo, hi] = primary.range ?? [0, 1.5];
  const worst = new Map<string, number>();
  const trace = new Map<string, number[]>();
  const couplings = leafCouplings(model);
  for (let i = 0; i <= steps; i++) {
    const q = lo + (hi - lo) * i / steps;
    s.setQ(primary.name, q);
    if (model.meta.secondary_joint) s.setQ(model.meta.secondary_joint, q);
    for (const [j, [c0, c1]] of couplings) s.setQ(j, c0 + c1 * q);
    for (const r of s.solve()) {
      worst.set(r.name, Math.max(worst.get(r.name) ?? 0, r.separation));
      for (const j of r.joints) { const t = trace.get(j) ?? []; t.push(s.getQ(j)); trace.set(j, t); }
    }
  }
  return { solver: s, worst, trace };
}

describe.skipIf(!have)("closed-loop linkage solver", () => {
  test("db0012 closer: two-bar arm stays on the shoe from 0 to full open", () => {
    const model = load("db0012_swing_single");
    expect(hasLoops(model)).toBe(true);
    const { solver, worst, trace } = sweep(model);
    expect(solver.loops.length).toBe(1);
    expect(solver.loops[0].type).toBe("two_bar");
    expect(solver.loops[0].source).toBe("derived");
    expect(solver.owned.has("closer_pinion")).toBe(true);
    expect(solver.owned.has("closer_elbow")).toBe(true);
    expect(solver.owned.has("leaf_hinge")).toBe(false);
    for (const [, sep] of worst) expect(sep).toBeLessThan(1e-3);
    // at rest the solver reproduces the authored pose
    solver.setQ("leaf_hinge", 0);
    solver.solve();
    expect(Math.abs(solver.getQ("closer_pinion"))).toBeLessThan(1e-6);
    expect(Math.abs(solver.getQ("closer_elbow"))).toBeLessThan(1e-6);
    // the arms actually fold: both joints move, continuously
    const pin = trace.get("closer_pinion")!, elb = trace.get("closer_elbow")!;
    expect(Math.abs(pin[pin.length - 1] - pin[0])).toBeGreaterThan(0.3);
    expect(Math.abs(elb[elb.length - 1] - elb[0])).toBeGreaterThan(0.3);
    for (let i = 1; i < pin.length; i++) { expect(Math.abs(pin[i] - pin[i - 1])).toBeLessThan(0.2); expect(Math.abs(elb[i] - elb[i - 1])).toBeLessThan(0.2); }
    expect(solver.warnings).toEqual([]);
  });

  test("numeric fallback agrees with the analytic two-bar solution", () => {
    const model = load("db0012_swing_single");
    const a = new LoopSolver(model), g = new LoopSolver(model, { forceGeneric: true });
    expect(g.loops[0].type).toBe("generic");
    for (const q of [0, 0.3, 0.8, 1.2, 1.7]) {
      a.setQ("leaf_hinge", q); g.setQ("leaf_hinge", q);
      const ra = a.solve()[0], rg = g.solve()[0];
      expect(ra.separation).toBeLessThan(1e-4);
      expect(rg.separation).toBeLessThan(1e-4);
      expect(Math.abs(a.getQ("closer_pinion") - g.getQ("closer_pinion"))).toBeLessThan(1e-3);
      expect(Math.abs(a.getQ("closer_elbow") - g.getQ("closer_elbow"))).toBeLessThan(1e-3);
    }
  });

  test("model.json linkages block (schema path) is honoured", () => {
    const model = load("db0012_swing_single");
    const link: TwoBarLinkageJ = {
      name: "closer", type: "two_bar",
      pinion: { body: "closer_arm_main", joint: "closer_pinion", parent: "leaf" },
      elbow: { body: "closer_arm_fore", joint: "closer_elbow" },
      anchor: { body: "world", pos: [-0.3095, -0.093, 2.177] },
      equality: "closer_arm_connect", axis: [0, 0, 1], L1: 0.28, L2: 0.26, elbow_sign: 1,
    };
    // derive the true elbow sign / anchor first, then feed them back through the schema
    const derived = new LoopSolver(model);
    const two = derived.loops[0] as any;
    link.elbow_sign = two.elbowSign;
    const anchor = new Vector3();
    derived.art.worldPoint(two.anchorBody, two.anchorLocal, anchor);
    link.anchor.pos = [anchor.x, anchor.y, anchor.z];
    const m2: ModelJ = { ...model, linkages: [link] };
    const { solver, worst } = sweep(m2);
    expect(solver.loops[0].source).toBe("schema");
    expect(solver.loops[0].name).toBe("closer");
    expect(solver.warnings).toEqual([]);
    for (const [, sep] of worst) expect(sep).toBeLessThan(1e-3);
    // a wrong elbow_sign is reported, the loop still closes (mirror solution)
    const m3: ModelJ = { ...model, linkages: [{ ...link, elbow_sign: (-link.elbow_sign) as 1 | -1 }] };
    const s3 = sweep(m3);
    expect(s3.solver.warnings.some((w) => w.includes("elbow_sign"))).toBe(true);
    for (const [, sep] of s3.worst) expect(sep).toBeLessThan(1e-3);
  });

  test("rising-hinge cold storage: sliding shoe follows lift with a closed arm loop", () => {
    const model = load("db0188_cold_storage");
    const { worst, solver } = sweep(model);
    expect(solver.loops.length).toBe(1);
    expect(solver.loops[0].type).toBe("generic");
    expect(solver.coupled.has("leaf_rise")).toBe(true);
    expect(solver.owned.has("leaf_rise")).toBe(false);
    expect(solver.owned.has("closer_pinion")).toBe(true);
    expect(solver.owned.has("closer_shoe_slide")).toBe(true);
    for (const [, sep] of worst) expect(sep).toBeLessThan(1e-3);
    expect(Math.abs(solver.getQ("closer_shoe_slide") - solver.getQ("leaf_rise"))).toBeLessThan(1e-4);
  });

  test("every door with a connect loop closes it (< 1 mm) over the whole leaf travel", () => {
    // manifest -> only doors that can carry a mechanism loop (closer / operator), then whatever model.json declares
    const manifest = JSON.parse(readFileSync(path.join(ASSETS, "..", "manifest.json"), "utf8"));
    const failures: string[] = [];
    let nLoops = 0, nDoors = 0;
    for (const d of manifest.doors as { id: string; closer: string; operator: string }[]) {
      let model: ModelJ;
      try { model = load(d.id); } catch { continue; }
      if (!hasLoops(model)) continue;
      nDoors++;
      const { solver, worst } = sweep(model, {}, 36);
      for (const w of solver.warnings) failures.push(`${d.id}: ${w}`);
      for (const [name, sep] of worst) {
        nLoops++;
        expect(solver.owned.has(name)).toBe(false);
        for (const j of solver.coupled) expect(solver.owned.has(j)).toBe(false);
        if (!(sep < 1e-3)) failures.push(`${d.id} ${name}: worst separation ${(sep * 1000).toFixed(2)} mm`);
      }
    }
    expect(nDoors).toBeGreaterThan(0);
    expect(nLoops).toBeGreaterThanOrEqual(nDoors);
    expect(failures).toEqual([]);
  });

  test("doors without loops build no solver work", () => {
    for (const id of ["db0002_swing_single", "db0007_sliding_single", "db0013_swing_single"]) {
      const model = load(id);
      expect(hasLoops(model)).toBe(false);
      const s = new LoopSolver(model);
      expect(s.loops.length).toBe(0);
      expect(s.owned.size).toBe(0);
    }
  });
});

describe("telescoping strut (synthetic model)", () => {
  // a hatch leaf hinged about x at the origin; a gas strut: cylinder hinged at a world bracket, rod sliding along the
  // cylinder's x axis, rod tip pinned to a bracket on the leaf
  function strutModel(): ModelJ {
    const body = (name: string, parent: string | null, pos: number[], quat: number[], joint: any, semantic = "closer"): any => ({
      name, parent, pos, quat, joint, geoms: [], sites: [], tiers: ["full"], semantic, label: name, static: false, mass: 1, com: [0, 0, 0], inertia: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    });
    const joint = (name: string, type: "hinge" | "slide", axis: number[], range: [number, number] | null, role: string, interactive: boolean): any => ({
      name, type, axis, pos: [0, 0, 0], range, damping: 0, frictionloss: 0, stiffness: 0, springref: 0, armature: 0, role, label: name, robot_interactive: interactive, initial: 0, modeled_at: 0, notes: "",
    });
    // leaf bracket at (0, 0.6, 0.02) on the leaf; cylinder bracket in the world at (0, 0.2, -0.5)
    const bx = 0, by = 0.2, bz = -0.5;
    const tx = 0, ty = 0.6, tz = 0.02;
    const d = Math.hypot(ty - by, tz - bz);
    const ang = Math.atan2(tz - bz, ty - by);           // strut aim in the y-z plane, hinge axis = x
    // cylinder frame: local x = strut line (rotate x->aim about the x axis? use quaternion about x for a y-z direction)
    // cylinder local x axis must point along (0, cos ang, sin ang): rotate +x onto that: use quaternion about z by 90deg then about x by ang
    const qz = [Math.cos(Math.PI / 4), 0, 0, Math.sin(Math.PI / 4)];     // x -> y
    const qx = [Math.cos(ang / 2), Math.sin(ang / 2), 0, 0];             // y -> (0, cos, sin)
    const mul = (a: number[], b: number[]) => [
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
    const qc = mul(qx, qz);
    const L = 0.3;                                       // rod frame sits L along the cylinder; tip at rest = d - L further
    return {
      name: "strut_test", tier: "full", materials: {}, tendons: [], meta: { primary_joint: "leaf_hinge" },
      bodies: [
        body("world_env", null, [0, 0, 0], [1, 0, 0, 0], null, "structure"),
        body("leaf", null, [0, 0, 0], [1, 0, 0, 0], joint("leaf_hinge", "hinge", [1, 0, 0], [0, 1.4], "primary", true), "leaf"),
        // joint axes are in the body frame: the world x axis is the cylinder's local -y (qc maps local y -> world -x)
        body("strut_cyl", null, [bx, by, bz], qc, joint("strut_hinge", "hinge", [0, -1, 0], null, "mechanism", false)),
        body("strut_rod", "strut_cyl", [L, 0, 0], [1, 0, 0, 0], joint("strut_slide", "slide", [1, 0, 0], [-0.2, 0.5], "mechanism", false)),
      ],
      equalities: [{ kind: "connect", name: "strut_connect", a: "strut_rod", b: "leaf", polycoeff: [0, 0, 0, 0, 0], anchor: [d - L, 0, 0], label: "", active: true }],
    } as ModelJ;
  }

  test("aim + extension follow the leaf; slide = distance - offset", () => {
    const model = strutModel();
    const s = new LoopSolver(model);
    expect(s.loops.length).toBe(1);
    expect(s.loops[0].type).toBe("telescoping");
    expect(s.warnings).toEqual([]);
    const rest = new LoopSolver(model);
    const P = new Vector3(0, 0.2, -0.5);
    for (let i = 0; i <= 40; i++) {
      const q = 1.2 * i / 40;
      s.setQ("leaf_hinge", q);
      const r = s.solve()[0];
      expect(r.separation).toBeLessThan(1e-6);
      expect(r.stretched).toBe(false);
      // leaf bracket after rotating about x: (0, 0.6 cos q - 0.02 sin q, 0.6 sin q + 0.02 cos q)
      const T = new Vector3(0, 0.6 * Math.cos(q) - 0.02 * Math.sin(q), 0.6 * Math.sin(q) + 0.02 * Math.cos(q));
      const dist = T.distanceTo(P);
      const dRest = Math.hypot(0.4, 0.52);
      expect(Math.abs(s.getQ("strut_slide") - (dist - dRest))).toBeLessThan(1e-6);
      void rest;
    }
    // the numeric solver reaches the same configuration
    const g = new LoopSolver(model, { forceGeneric: true });
    g.setQ("leaf_hinge", 0.9); s.setQ("leaf_hinge", 0.9);
    expect(g.solve()[0].separation).toBeLessThan(1e-5);
    s.solve();
    expect(Math.abs(g.getQ("strut_hinge") - s.getQ("strut_hinge"))).toBeLessThan(1e-3);
    expect(Math.abs(g.getQ("strut_slide") - s.getQ("strut_slide"))).toBeLessThan(1e-3);
  });

  test("model.json linkages block (telescoping schema) is honoured", () => {
    const model = strutModel();
    const d = Math.hypot(0.4, 0.52);
    const link: TelescopingLinkageJ = {
      name: "gas_strut", type: "telescoping",
      base: { body: "strut_cyl", joint: "strut_hinge", parent: "world", pos: [0, 0.2, -0.5] },
      slide: { body: "strut_rod", joint: "strut_slide", axis_local: [1, 0, 0], offset: d },
      anchor: { body: "leaf", pos: [0, 0.6, 0.02] },
      equality: "strut_connect",
    };
    const s = new LoopSolver({ ...model, linkages: [link] });
    expect(s.loops[0].type).toBe("telescoping");
    expect(s.loops[0].source).toBe("schema");
    expect(s.loops[0].name).toBe("gas_strut");
    expect(s.warnings).toEqual([]);
    for (const q of [0, 0.4, 0.9, 1.2]) { s.setQ("leaf_hinge", q); expect(s.solve()[0].separation).toBeLessThan(1e-6); }
    // a schema offset that disagrees with the geometry is reported (the geometry wins)
    const s2 = new LoopSolver({ ...model, linkages: [{ ...link, slide: { ...link.slide, offset: d + 0.05 } }] });
    expect(s2.warnings.some((w) => w.includes("offset"))).toBe(true);
    s2.setQ("leaf_hinge", 0.9);
    expect(s2.solve()[0].separation).toBeLessThan(1e-6);
  });

  test("reach limit is reported, not hidden", () => {
    const model = strutModel();
    (model.bodies[3].joint as any).range = [-0.01, 0.01];   // the rod can barely move
    const s = new LoopSolver(model);
    s.setQ("leaf_hinge", 1.2);
    const r = s.solve()[0];
    expect(r.stretched).toBe(true);
    expect(r.ok).toBe(false);
    expect(r.separation).toBeGreaterThan(1e-3);
  });
});
