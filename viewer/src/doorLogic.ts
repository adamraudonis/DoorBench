// Pure helpers about the door mechanism as described by model.json (used by the viewer's honest open / close).
import type { ModelJ } from "./types";
import type { JointHandle } from "./scene";

/** A joint whose range is (almost) zero is locked: the mechanism cannot be worked (e.g. a keyed lever, a jammed bolt). */
export function isLocked(h: JointHandle | undefined): boolean {
  return !!h && !!h.range && h.range[1] - h.range[0] < 0.006;
}

/** The leaf joint (primary or secondary) whose body is an ancestor of the given joint's body. */
export function leafOfJoint(model: ModelJ, joint: string | undefined): string | undefined {
  const meta = model.meta ?? {};
  const leafJoints = new Set([meta.primary_joint, meta.secondary_joint].filter(Boolean) as string[]);
  const byName = new Map(model.bodies.map((b) => [b.name, b]));
  let body = model.bodies.find((b) => b.joint?.name === joint);
  while (body) {
    if (body.joint && leafJoints.has(body.joint.name)) return body.joint.name;
    body = body.parent ? byName.get(body.parent) : undefined;
  }
  return undefined;
}

/** The leaf the robot works: the operator's leaf for pairs, else the primary joint. */
export function activeLeaf(model: ModelJ): string | undefined {
  const meta = model.meta ?? {};
  return leafOfJoint(model, meta.operator_joint) ?? meta.primary_joint;
}

/** Bolt / lock joints that the operator drives (one-sided tendons and polynomial couplings). */
export function boltJointsFor(model: ModelJ, operator: string | undefined): string[] {
  if (!operator) return [];
  const out = new Set<string>();
  const opRole = model.bodies.find((b) => b.joint?.name === operator)?.joint?.role;
  if (opRole === "latch" || opRole === "lock") out.add(operator);        // slide bolts, pins, hooks: the operator IS the bolt
  for (const t of model.tendons ?? []) {
    const [[bolt], [driver]] = t.sites as [string, number][];
    if (driver === operator) out.add(bolt);
  }
  for (const e of model.equalities) {
    if (e.kind === "joint" && e.b === operator) {
      const role = model.bodies.find((b) => b.joint?.name === e.a)?.joint?.role;
      if (role === "latch" || role === "lock") out.add(e.a);
    }
  }
  return Array.from(out);
}

// ---------------------------------------------------------------------------
// Honest open / close sequencing (pure; used by DoorView and unit-tested in doorLogic.test.ts)
// ---------------------------------------------------------------------------
export interface JointLike { name: string; q: number; range: [number, number] | null; modeledAt: number }
export interface Phase { joint: string; from: number; to: number; dur: number; followers?: { joint: string; from: number; to: number }[]; t0?: number }

function thrown(joints: Map<string, JointLike>, bolts: string[]): boolean {
  for (const name of bolts) {
    const h = joints.get(name);
    if (!h || !h.range) continue;
    if (h.range[1] > 1e-6 && h.q < 0.8 * h.range[1]) return true;
  }
  return false;
}

/**
 * Phases for the "Open / close door" button: work the operator (which retracts the bolt through its coupling), move
 * the active leaf (both leaves of a dutch door when the joining bolt is engaged; only the active leaf of a pair),
 * release the operator.  A locked operator (range < 0.006) is never driven.
 */
export function openClosePhases(model: ModelJ, joints: Map<string, JointLike>): { phases: Phase[]; note: string | null } {
  const meta = model.meta ?? {};
  const operator: string | undefined = meta.operator_joint ?? undefined;
  const leafName = activeLeaf(model);
  const leaf = leafName ? joints.get(leafName) : undefined;
  if (!leaf) return { phases: [], note: null };
  const closed = leaf.modeledAt ?? 0;
  const open = !leaf.range ? closed + 1.2 : leaf.range[1] > closed + 1e-6 ? leaf.range[1] : leaf.range[0];
  const isClosed = Math.abs(leaf.q - closed) < 0.02;
  const target = isClosed ? open : closed;
  const op = operator ? joints.get(operator) : undefined;
  const bolts = boltJointsFor(model, operator);
  const locked = !!op && !!op.range && op.range[1] - op.range[0] < 0.006;
  const needsOp = !!op && !!op.range && !locked && bolts.length > 0;
  const phases: Phase[] = [];
  let note: string | null = null;
  if (locked && isClosed && bolts.length && thrown(joints, bolts)) note = "locked: the operator cannot be worked from this side (range < 0.006)";
  if (needsOp && op) phases.push({ joint: op.name, from: op.q, to: op.range![1], dur: 450 });
  const followers: NonNullable<Phase["followers"]> = [];
  const jb = joints.get("join_bolt_slide");
  const secondary: string | undefined = meta.secondary_joint ?? undefined;
  if (secondary && jb && jb.q < 0.01) { const s = joints.get(secondary); if (s) followers.push({ joint: secondary, from: s.q, to: target }); }
  phases.push({ joint: leaf.name, from: leaf.q, to: target, dur: 1400, followers });
  if (needsOp && op) phases.push({ joint: op.name, from: op.range![1], to: op.modeledAt ?? op.range![0], dur: 400 });
  if (!note) note = needsOp ? (isClosed ? "operator actuated → bolt retracted → leaf opens → operator released" : "operator actuated → bolt retracted → leaf closes → bolt re-extends") : followers.length ? "joining bolt engaged: both leaves move together" : null;
  return { phases, note };
}

/**
 * What else must move when a leaf slider is dragged to q: the operator is driven to full travel when the latch is still
 * thrown (so the bolt clears the strike), and the other leaf of a joined dutch door follows.
 */
export function sliderReaction(model: ModelJ, joints: Map<string, JointLike>, leafJoint: string, q: number): { operatorTo: number | null; mirror: { joint: string; q: number } | null; note: string | null } {
  const meta = model.meta ?? {};
  const primary: string | undefined = meta.primary_joint ?? undefined;
  const secondary: string | undefined = meta.secondary_joint ?? undefined;
  const operator: string | undefined = meta.operator_joint ?? undefined;
  const h = joints.get(leafJoint);
  const out = { operatorTo: null as number | null, mirror: null as { joint: string; q: number } | null, note: null as string | null };
  if (!h || (leafJoint !== primary && leafJoint !== secondary)) return out;
  const away = Math.abs(q - (h.modeledAt ?? 0)) > 0.02;
  const opLeaf = leafOfJoint(model, operator);
  const ownsOperator = opLeaf === leafJoint || (!opLeaf && leafJoint === primary);
  const op = operator ? joints.get(operator) : undefined;
  const locked = !!op && !!op.range && op.range[1] - op.range[0] < 0.006;
  if (away && ownsOperator && op && op.range && !locked && thrown(joints, boltJointsFor(model, operator))) {
    out.operatorTo = op.range[1];
    out.note = "latch retracted: operator driven to full travel so the bolt clears the strike";
  }
  const jb = joints.get("join_bolt_slide");
  if (primary && secondary && jb && jb.q < 0.01) out.mirror = { joint: leafJoint === primary ? secondary : primary, q };
  return out;
}
