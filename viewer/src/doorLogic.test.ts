// bun test — honest open / close sequencing against real model.json files from ../assets (run after generating the dataset).
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "./types";
import { activeLeaf, boltJointsFor, openClosePhases, operatorJoints, operatorsAreIndividual, sliderReaction, type JointLike } from "./doorLogic";

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

  test("individually dogged watertight door: every dog is undogged, then the leaf swings, then all are re-dogged", () => {
    const d = load("db0168_ship_watertight")!;
    expect(operatorsAreIndividual(d.model)).toBe(true);
    const dogs = operatorJoints(d.model);
    expect(dogs).toEqual(["dog_0_hinge", "dog_1_hinge", "dog_2_hinge", "dog_3_hinge", "dog_4_hinge", "dog_5_hinge"]);
    const { phases, note } = openClosePhases(d.model, d.joints);
    // 6 dogs released, the leaf, 6 dogs re-engaged
    expect(phases.map((p) => p.joint)).toEqual([...dogs, "leaf_hinge", ...[...dogs].reverse()]);
    for (const p of phases.slice(0, 6)) expect(p.to).toBeCloseTo(d.joints.get(p.joint)!.range![1]);
    for (const p of phases.slice(7)) expect(p.to).toBe(0);
    expect(phases[6].to).toBeCloseTo(d.joints.get("leaf_hinge")!.range![1]);
    expect(note).toContain("6 latches released");
    // dragging the leaf undogs ALL of them, not just the first
    const r = sliderReaction(d.model, d.joints, "leaf_hinge", 0.4);
    expect(r.operatorsTo.map((o) => o.joint)).toEqual(dogs);
    expect(r.note).toContain("all 6 latches");
  });

  test("quick-acting watertight door: one handwheel, and the coupling moves every dog", () => {
    const d = load("db0744_ship_watertight")!;
    expect(operatorsAreIndividual(d.model)).toBe(false);
    expect(operatorJoints(d.model)).toEqual(["wheel_hinge"]);
    const driven = d.model.equalities.filter((e) => e.kind === "joint" && e.b === "wheel_hinge").map((e) => e.a);
    expect(driven.filter((n) => n.startsWith("dog_")).length).toBeGreaterThanOrEqual(4);
    expect(driven).toContain("linkage_rod_r_slide");
    expect(boltJointsFor(d.model, "wheel_hinge")).toEqual(driven.filter((n) => n.startsWith("dog_")));
    const { phases } = openClosePhases(d.model, d.joints);
    expect(phases.map((p) => p.joint)).toEqual(["wheel_hinge", "leaf_hinge", "wheel_hinge"]);
  });

  test("blast door: both lever bolts are operators", () => {
    const d = load("db0288_blast")!;
    expect(operatorsAreIndividual(d.model)).toBe(true);
    expect(operatorJoints(d.model)).toEqual(["dog_0_hinge", "dog_1_hinge"]);
    expect(openClosePhases(d.model, d.joints).phases.map((p) => p.joint)).toEqual(["dog_0_hinge", "dog_1_hinge", "leaf_hinge", "dog_1_hinge", "dog_0_hinge"]);
  });

  test("single-operator doors list exactly one operator", () => {
    const d = load("db0002_swing_single")!;
    expect(operatorJoints(d.model)).toEqual(["leaf_handle_hinge"]);
    expect(operatorsAreIndividual(d.model)).toBe(false);
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
