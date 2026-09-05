// bun test — honest open / close sequencing against real model.json files from ../assets (run after generating the dataset).
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "./types";
import { activeLeaf, boltJointsFor, easeFor, gravityEase, openClosePhases, operatorReturnPhase, returnLabel, sliderReaction, springEase, type JointLike } from "./doorLogic";

const ASSETS = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..", "assets", "doors");

function load(id: string): { model: ModelJ; joints: Map<string, JointLike> } | null {
  const p = path.join(ASSETS, id, "model.json");
  if (!existsSync(p)) return null;
  const model: ModelJ = JSON.parse(readFileSync(p, "utf8"));
  const joints = new Map<string, JointLike>();
  for (const b of model.bodies) if (b.joint) joints.set(b.joint.name, { name: b.joint.name, q: b.joint.modeled_at ?? 0, range: b.joint.range, modeledAt: b.joint.modeled_at ?? 0 });
  return { model, joints };
}

const have = existsSync(ASSETS);

describe.skipIf(!have)("open / close sequencing", () => {
  test("knob + spring latch: operator, leaf, release", () => {
    const d = load("db0002_swing_single")!;
    expect(boltJointsFor(d.model, "leaf_handle_hinge")).toEqual(["leaf_latch_bolt_slide"]);
    const { phases, note } = openClosePhases(d.model, d.joints);
    expect(phases.map((p) => p.joint)).toEqual(["leaf_handle_hinge", "leaf_hinge", "leaf_handle_hinge"]);
    expect(phases[0].to).toBeCloseTo(0.87);            // full travel retracts the bolt through the tendon
    expect(phases[1].to).toBeCloseTo(d.joints.get("leaf_hinge")!.range![1]);
    expect(phases[2].to).toBe(0);
    expect(note).toContain("bolt retracted");
    // closing from open: same honest sequence
    d.joints.get("leaf_hinge")!.q = 1.0;
    const back = openClosePhases(d.model, d.joints);
    expect(back.phases[1].to).toBe(0);
    expect(back.phases.length).toBe(3);
  });

  test("dutch door with the joining bolt engaged moves both leaves; disengaged moves only the lower leaf", () => {
    const joined = load("db0118_dutch")!;
    expect(joined.joints.get("join_bolt_slide")!.q).toBeLessThan(0.01);
    const a = openClosePhases(joined.model, joined.joints);
    const leafPhase = a.phases.find((p) => p.joint === "leaf_lower_hinge")!;
    expect(leafPhase.followers?.map((f) => f.joint)).toEqual(["leaf_upper_hinge"]);
    expect(leafPhase.followers![0].to).toBe(leafPhase.to);
    const free = load("db0095_dutch")!;
    expect(free.joints.get("join_bolt_slide")!.q).toBeGreaterThan(0.01);
    const b = openClosePhases(free.model, free.joints);
    expect(b.phases.find((p) => p.joint === "leaf_lower_hinge")!.followers).toEqual([]);
    // slider mirror only when joined
    expect(sliderReaction(joined.model, joined.joints, "leaf_lower_hinge", 0.5).mirror).toEqual({ joint: "leaf_upper_hinge", q: 0.5 });
    expect(sliderReaction(free.model, free.joints, "leaf_lower_hinge", 0.5).mirror).toBeNull();
  });

  test("pair: only the active leaf (the operator's leaf) moves", () => {
    const d = load("db0097_swing_double")!;
    expect(activeLeaf(d.model)).toBe("leaf_a_hinge");
    const { phases } = openClosePhases(d.model, d.joints);
    const leafPhases = phases.filter((p) => p.joint === "leaf_a_hinge" || p.joint === "leaf_b_hinge");
    expect(leafPhases.map((p) => p.joint)).toEqual(["leaf_a_hinge"]);
    expect(leafPhases[0].followers).toEqual([]);
  });

  test("locked operator (range < 0.006) is never driven and the user is told", () => {
    const d = load("db0546_stall")!;                 // occupied stall: slide latch range 1 mm
    const op = d.joints.get(d.model.meta.operator_joint)!;
    expect(op.range![1] - op.range![0]).toBeLessThan(0.006);
    const { phases, note } = openClosePhases(d.model, d.joints);
    expect(phases.some((p) => p.joint === op.name)).toBe(false);
    expect(note).toContain("locked");
    expect(sliderReaction(d.model, d.joints, d.model.meta.primary_joint, 0.4).operatorTo).toBeNull();
    // a deadbolt-locked door whose handle still turns (backlash) is workable in the viewer: the handle is driven
    const e = load("db0058_swing_single")!;
    expect(openClosePhases(e.model, e.joints).phases.map((p) => p.joint)[0]).toBe("leaf_handle_hinge");
  });

  test("dragging a latched leaf drives the operator to full travel (bolt clears the strike)", () => {
    const d = load("db0002_swing_single")!;
    const r = sliderReaction(d.model, d.joints, "leaf_hinge", 0.3);
    expect(r.operatorTo).toBeCloseTo(0.87);
    expect(r.note).toContain("latch retracted");
    // bolt already retracted: nothing to do
    d.joints.get("leaf_latch_bolt_slide")!.q = 0.0127;
    expect(sliderReaction(d.model, d.joints, "leaf_hinge", 0.3).operatorTo).toBeNull();
    // back to closed: nothing to do
    d.joints.get("leaf_latch_bolt_slide")!.q = 0;
    expect(sliderReaction(d.model, d.joints, "leaf_hinge", 0.0).operatorTo).toBeNull();
    // a non-leaf joint never triggers it
    expect(sliderReaction(d.model, d.joints, "leaf_handle_hinge", 0.5).operatorTo).toBeNull();
  });

  test("slide-bolt gate: the operator is the bolt itself and is driven first", () => {
    const d = load("db0033_gate_sliding")!;
    const op = d.model.meta.operator_joint as string;
    expect(d.model.bodies.find((b) => b.joint?.name === op)!.joint!.type).toBe("slide");
    expect(boltJointsFor(d.model, op)).toEqual([op]);
    const { phases } = openClosePhases(d.model, d.joints);
    expect(phases.map((p) => p.joint)).toEqual([op, "leaf_slide", op]);
    expect(phases[0].to).toBeCloseTo(0.08);
  });
});

// ---------------------------------------------------------------------------
// Operator snap-back (docs/PHYSICS.md "Operator return")
// ---------------------------------------------------------------------------
describe("operator release profile", () => {
  test("springEase starts slowly, arrives at rest, and is monotone", () => {
    for (const shape of [0.2, 0.5, 0.9, 1.0]) {
      expect(springEase(0, shape)).toBeCloseTo(0, 6);
      expect(springEase(1, shape)).toBeCloseTo(1, 3);
      let prev = -1;
      for (let s = 0; s <= 1.0001; s += 0.05) {
        const e = springEase(s, shape);
        expect(e).toBeGreaterThanOrEqual(prev - 1e-9);
        prev = e;
      }
      expect(springEase(0.1, shape)).toBeLessThan(0.2);   // a released handle does not jump: it accelerates
    }
  });

  test("a hard spring accelerates the whole way; a soft one is already coasting at half time", () => {
    // shape = how much of the spring's pull is used up before the rest stop.  Small = the equilibrium sits far past
    // the stop, so the handle is still accelerating when it slams home; 1 = it decays onto the stop asymptotically.
    expect(springEase(0.5, 0.2)).toBeLessThan(0.5);
    expect(springEase(0.5, 1.0)).toBeGreaterThan(0.5);
    expect(springEase(0.5, 0.2)).toBeLessThan(springEase(0.5, 1.0));
  });

  test("gravityEase is a constant-acceleration fall", () => {
    expect(gravityEase(0)).toBe(0);
    expect(gravityEase(0.5)).toBeCloseTo(0.25, 6);
    expect(gravityEase(1)).toBe(1);
  });

  test("easeFor falls back to the ease-in-out used by every other phase", () => {
    const ph = { joint: "j", from: 0, to: 1, dur: 100 };
    expect(easeFor(ph, 0.5)).toBeCloseTo(0.5, 6);
    expect(easeFor({ ...ph, ease: "spring" as const, shape: 0.5 }, 1)).toBeCloseTo(1, 3);
  });
});

describe.skipIf(!have)("operator snap-back against real doors", () => {
  test("a sprung knob returns to its rest stop with the derived spring profile", () => {
    const d = load("db0002_swing_single")!;
    const op = "leaf_handle_hinge";
    d.joints.get(op)!.q = 0.8;
    const ph = operatorReturnPhase(d.model, d.joints, op)!;
    expect(ph).not.toBeNull();
    expect(ph.joint).toBe(op);
    expect(ph.to).toBe(0);
    expect(ph.ease).toBe("spring");
    expect(ph.dur).toBeGreaterThan(150);
    expect(ph.dur).toBeLessThan(700);
    expect(returnLabel(d.model, op)).toContain("spring return");
  });

  test("a released operator already at rest has nothing to animate", () => {
    const d = load("db0002_swing_single")!;
    expect(operatorReturnPhase(d.model, d.joints, "leaf_handle_hinge")).toBeNull();
  });

  test("open / close releases the handle with the spring profile, not a linear slide", () => {
    const d = load("db0002_swing_single")!;
    const { phases } = openClosePhases(d.model, d.joints);
    const release = phases[phases.length - 1];
    expect(release.joint).toBe("leaf_handle_hinge");
    expect(release.ease).toBe("spring");
    expect(release.to).toBe(0);
  });

  test("a handwheel is a detent operator: it stays where it is put and says so", () => {
    const d = load("db0179_vault");
    if (!d) return;
    const wheel = "wheel_hinge";
    d.joints.get(wheel)!.q = 3.0;
    expect(operatorReturnPhase(d.model, d.joints, wheel)).toBeNull();
    expect(returnLabel(d.model, wheel)).toContain("stays where put");
  });

  test("a ring pull returns under gravity, to wherever gravity puts it", () => {
    const d = load("db0380_hatch_floor");
    if (!d) return;
    d.joints.get("ring_hinge")!.q = 1.4;
    const ph = operatorReturnPhase(d.model, d.joints, "ring_hinge")!;
    expect(ph.ease).toBe("gravity");
    expect(returnLabel(d.model, "ring_hinge")).toContain("gravity return");
  });
});
