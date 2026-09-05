// bun test — honest open / close sequencing against real model.json files from ../assets (run after generating the dataset).
import { describe, expect, test } from "bun:test";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import path from "node:path";
import type { ModelJ } from "./types";
import { activeLeaf, boltJointsFor, openClosePhases, previewOperatorForLeaf, sliderReaction, type JointLike, type Phase } from "./doorLogic";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..");
const ASSETS = path.resolve(ROOT, process.env.DOORBENCH_SITE_ROOT || ".", "assets", "doors");
const moves = (phases: Phase[]) => phases.flatMap(p => [p, ...(p.followers ?? [])]);

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

  test("pair: both independently operable leaves move", () => {
    const d = load("db0097_swing_double")!;
    expect(activeLeaf(d.model)).toBe("leaf_a_hinge");
    const { phases } = openClosePhases(d.model, d.joints);
    const leafPhases = phases.filter((p) => p.joint === "leaf_a_hinge" || p.joint === "leaf_b_hinge");
    expect(leafPhases.map((p) => p.joint)).toEqual(["leaf_a_hinge"]);
    expect(leafPhases[0].followers?.map(f => f.joint)).toEqual(["leaf_b_hinge"]);
  });

  test("DB0019 previews both inside panic bars without changing approach-side accessibility", () => {
    const d = load("db0019_swing_double")!;
    expect(d.model.meta.operator_joint).toBeNull();
    const { phases } = openClosePhases(d.model, d.joints);
    expect(phases).toHaveLength(3);
    for (const [leaf, operator] of [["leaf_a_hinge", "leaf_a_exit_device_slide"], ["leaf_b_hinge", "leaf_b_exit_device_slide"]]) {
      expect(previewOperatorForLeaf(d.model, leaf)).toBe(operator);
      expect(moves([phases[0]]).find(m => m.joint === operator)?.to).toBeCloseTo(0.016);
      expect(moves([phases[1]]).find(m => m.joint === leaf)?.to).toBeCloseTo(Math.PI / 2);
      expect(moves([phases[2]]).find(m => m.joint === operator)?.to).toBe(0);
      const slider = sliderReaction(d.model, d.joints, leaf, 0.4);
      expect(slider.operator).toBe(operator);
      expect(slider.operatorTo).toBeCloseTo(0.016);
      expect(slider.mirror).toBeNull(); // A manual slider stays independent.
    }
    expect(d.model.meta.operator_joint).toBeNull();
    // Opposing physical axes each use their native positive range; do not negate B.
    const axes = d.model.bodies.filter(b => ["leaf_a_hinge", "leaf_b_hinge"].includes(b.joint?.name ?? "")).map(b => b.joint!.axis[2]);
    expect(axes).toEqual([-1, 1]);
  });

  test("a partially open pair opens fully, then closes both from their own coordinates", () => {
    const d = load("db0019_swing_double")!;
    d.joints.get("leaf_a_hinge")!.q = Math.PI / 2;
    const opening = moves(openClosePhases(d.model, d.joints).phases);
    expect(opening.find(m => m.joint === "leaf_b_hinge")?.to).toBeCloseTo(Math.PI / 2);
    expect(opening.some(m => m.joint === "leaf_a_hinge")).toBe(false);
    d.joints.get("leaf_b_hinge")!.q = Math.PI / 2;
    const closing = moves(openClosePhases(d.model, d.joints).phases);
    for (const leaf of ["leaf_a_hinge", "leaf_b_hinge"]) expect(closing.find(m => m.joint === leaf)?.to).toBe(0);
    // Generalized native ranges can have an offset or a negative opening coordinate.
    const b = d.joints.get("leaf_b_hinge")!;
    b.modeledAt = 0.2; b.range = [-1.1, 0.2]; b.q = 0.2;
    expect(moves(openClosePhases(d.model, d.joints).phases).find(m => m.joint === b.name)?.to).toBe(-1.1);
  });

  test("inactive flush-bolted leaves and active maglocks stay secured in buttons and sliders", () => {
    const inactive = load("db0149_swing_double")!;
    const preview = openClosePhases(inactive.model, inactive.joints);
    expect(moves(preview.phases).some(m => m.joint === "leaf_b_hinge")).toBe(false);
    expect(moves(preview.phases).some(m => m.joint === "leaf_a_hinge")).toBe(true);
    expect(preview.note).toContain("secured");
    expect(sliderReaction(inactive.model, inactive.joints, "leaf_b_hinge", 1).blocked).toBe(true);
    const locked = load("db0396_swing_double")!;
    expect(openClosePhases(locked.model, locked.joints).phases).toEqual([]);
    expect(openClosePhases(locked.model, locked.joints).note).toContain("locked");
    expect(sliderReaction(locked.model, locked.joints, "leaf_b_hinge", 1).blocked).toBe(true);
  });

  test("separate deadbolts must clear the edge before the handle can open a paired leaf", () => {
    const fixed = load("db0534_swing_double")!;
    expect(openClosePhases(fixed.model, fixed.joints).phases).toEqual([]);
    expect(openClosePhases(fixed.model, fixed.joints).note).toContain("separate deadbolt");
    expect(sliderReaction(fixed.model, fixed.joints, "leaf_a_hinge", 0.5).blocked).toBe(true);
    const locked = load("db0792_swing_double")!;
    expect(openClosePhases(locked.model, locked.joints).phases).toEqual([]);
    const bolt = locked.joints.get("leaf_a_deadbolt_slide")!;
    bolt.q = bolt.range![1];
    expect(moves(openClosePhases(locked.model, locked.joints).phases).some(m => m.joint === "leaf_a_hinge")).toBe(true);
    const unlocked = load("db0714_swing_double")!;
    expect(moves(openClosePhases(unlocked.model, unlocked.joints).phases).some(m => m.joint === "leaf_a_hinge")).toBe(true);
    // A fixed bolt modeled inside the leaf is also unlocked; fixed alone does not imply thrown.
    const body = fixed.model.bodies.find(b => b.name === "leaf_a_deadbolt")!;
    const box = body.geoms.find(g => g.name === "leaf_a_deadbolt_box")!;
    box.pos[0] -= Math.sign(body.pos[0]) * 0.0254;
    expect(moves(openClosePhases(fixed.model, fixed.joints).phases).some(m => m.joint === "leaf_a_hinge")).toBe(true);
  });

  test("a manually thrown auxiliary barrel bolt blocks the leaf until withdrawn", () => {
    const d = load("db0127_swing_double")!;
    const bolt = d.joints.get("leaf_a_aux_bolt_slide")!;
    expect(moves(openClosePhases(d.model, d.joints).phases).some(m => m.joint === "leaf_a_hinge")).toBe(true);
    bolt.q = 0;
    expect(moves(openClosePhases(d.model, d.joints).phases).some(m => m.joint === "leaf_a_hinge")).toBe(false);
    expect(sliderReaction(d.model, d.joints, "leaf_a_hinge", 0.4).blocked).toBe(true);
    bolt.q = bolt.range![1];
    expect(sliderReaction(d.model, d.joints, "leaf_a_hinge", 0.4).blocked).toBe(false);
  });

  test("overlapping astragal: A opens first, B closes first, and a locked A also blocks B", () => {
    const d = load("db0871_swing_double")!;
    expect(sliderReaction(d.model, d.joints, "leaf_b_hinge", 0.5).blocked).toBe(true);
    expect(sliderReaction(d.model, d.joints, "leaf_a_hinge", 0.5).blocked).toBe(false);
    const leaves = new Set(["leaf_a_hinge", "leaf_b_hinge"]);
    const opening = openClosePhases(d.model, d.joints).phases.filter(p => leaves.has(p.joint));
    expect(opening.map(p => p.joint)).toEqual(["leaf_a_hinge", "leaf_b_hinge"]);
    expect(opening.every(p => p.followers?.length === 0)).toBe(true);
    for (const leaf of leaves) d.joints.get(leaf)!.q = d.joints.get(leaf)!.range![1];
    expect(sliderReaction(d.model, d.joints, "leaf_a_hinge", 0).blocked).toBe(true);
    expect(sliderReaction(d.model, d.joints, "leaf_b_hinge", 0).blocked).toBe(false);
    expect(openClosePhases(d.model, d.joints).phases.filter(p => leaves.has(p.joint)).map(p => p.joint)).toEqual(["leaf_b_hinge", "leaf_a_hinge"]);
    d.joints.get("leaf_b_hinge")!.q = 0.01;
    expect(sliderReaction(d.model, d.joints, "leaf_a_hinge", 0).blocked).toBe(true);
    const locked = load("db0183_swing_double")!;
    expect(openClosePhases(locked.model, locked.joints).phases).toEqual([]);
    expect(openClosePhases(locked.model, locked.joints).note).toContain("astragal");
    expect(sliderReaction(locked.model, locked.joints, "leaf_b_hinge", 0.5).blocked).toBe(true);
    // A French door's T astragal does not prevent A opening while inactive B remains bolted shut.
    const french = load("db0832_swing_double")!;
    const free = moves(openClosePhases(french.model, french.joints).phases);
    expect(free.some(m => m.joint === "leaf_a_hinge")).toBe(true);
    expect(free.some(m => m.joint === "leaf_b_hinge")).toBe(false);
  });

  test("every swing pair keeps preview targets in native limits and preserves secured leaves", () => {
    const ids = readdirSync(ASSETS).filter(id => id.endsWith("_swing_double"));
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) {
      const d = load(id)!;
      const all = moves(openClosePhases(d.model, d.joints).phases);
      for (const move of all) {
        const h = d.joints.get(move.joint)!;
        if (h.range) { expect(move.to).toBeGreaterThanOrEqual(h.range[0]); expect(move.to).toBeLessThanOrEqual(h.range[1]); }
      }
      for (const name of [d.model.meta.primary_joint, d.model.meta.secondary_joint]) {
        const leaf = d.joints.get(name)!;
        const body = d.model.bodies.find(b => b.joint?.name === name)!.name;
        const secured = leaf.range && leaf.range[1] - leaf.range[0] < 0.006 || d.model.equalities.some(e => e.kind === "weld" && e.active !== false && e.a === body && e.b === "world");
        if (secured) expect(all.some(m => m.joint === name)).toBe(false);
      }
      // With a full travel range, every independent latch driver must precede its leaf's motion.
      const phases = openClosePhases(d.model, d.joints).phases;
      for (const name of [d.model.meta.primary_joint, d.model.meta.secondary_joint]) {
        const operator = previewOperatorForLeaf(d.model, name);
        const leafIndex = phases.findIndex(p => moves([p]).some(m => m.joint === name));
        if (leafIndex < 0 || !operator || !boltJointsFor(d.model, operator).length) continue;
        const op = d.joints.get(operator)!;
        if (op.range && op.range[1] - op.range[0] >= 0.006) expect(phases.findIndex(p => moves([p]).some(m => m.joint === operator))).toBeLessThan(leafIndex);
      }
    }
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
