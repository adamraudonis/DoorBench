// Pure helpers about the door mechanism as described by model.json (used by the kinematic preview).
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

/** Paired swing leaves are independent; Dutch, bifold and sliding couplings keep their own behavior. */
export function isSwingPair(model: ModelJ): boolean {
  return model.meta?.family === "swing_double" && !!model.meta?.secondary_joint;
}

/** A preview may work the inside panic bar even when no operator is accessible from the robot's approach side.
 * Keep meta.operator_joint unchanged: it describes benchmark accessibility, not every modeled operator. */
export function previewOperatorForLeaf(model: ModelJ, leaf: string): string | undefined {
  const operator = model.meta?.operator_joint as string | undefined;
  if (operator && (leafOfJoint(model, operator) === leaf || (!leafOfJoint(model, operator) && leaf === model.meta?.primary_joint))) return operator;
  if (!isSwingPair(model)) return undefined;
  const followers = new Set(model.equalities.filter(e => e.kind === "joint").map(e => e.a));
  return model.bodies.find(b => b.joint?.role === "operator" && !followers.has(b.joint.name) && leafOfJoint(model, b.joint.name) === leaf)?.joint?.name;
}

/**
 * EVERY operator the robot has to work, in the order it works them.  A watertight door has one lever per dog and a
 * blast door one per lever bolt (`operator_coupling === "individual"`): the leaf is held while any single one of them
 * is still engaged, so all of them must be released before it can swing.  A door whose lock points are driven together
 * from one operator (a ship handwheel, a vault handwheel, a cremone knob, a multipoint lever) lists just that one.
 */
export function operatorJoints(model: ModelJ): string[] {
  const meta = model.meta ?? {};
  const named: string[] = Array.isArray(meta.operator_joints) ? meta.operator_joints : [];
  const have = new Set(model.bodies.map((b) => b.joint?.name).filter(Boolean) as string[]);
  const out = named.filter((n) => have.has(n));
  if (!out.length && meta.operator_joint && have.has(meta.operator_joint)) out.push(meta.operator_joint);
  return out;
}

/** True when each operator is an independent latch that must be released on its own (dog levers, lever bolts). */
export function operatorsAreIndividual(model: ModelJ): boolean {
  return (model.meta ?? {}).operator_coupling === "individual" && operatorJoints(model).length > 1;
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

function openPosition(leaf: JointLike): number {
  return !leaf.range ? leaf.modeledAt + 1.2 : leaf.range[1] > leaf.modeledAt + 1e-6 ? leaf.range[1] : leaf.range[0];
}

/** Do not use the inspection button to bypass authored inactive-leaf bolts or active lock welds. */
function pairLeafBlock(model: ModelJ, joints: Map<string, JointLike>, leaf: JointLike): string | null {
  if (leaf.range && leaf.range[1] - leaf.range[0] < 0.006) return `${leaf.name}: secured by its authored joint limit`;
  const body = model.bodies.find(b => b.joint?.name === leaf.name)?.name;
  const weld = model.equalities.find(e => e.kind === "weld" && e.active !== false &&
    ((e.a === body && (e.b === "world" || !e.b)) || (e.b === body && e.a === "world")));
  if (weld) return `${leaf.name}: locked (${weld.label || weld.name})`;
  const operator = previewOperatorForLeaf(model, leaf.name);
  const driven = new Set(boltJointsFor(model, operator));
  // common.add_deadbolt places each bolt body's origin on the leaf edge, its box along local X.
  // Measure projection past that edge, including the modeled-at offset; fixed retracted bolts stay free.
  for (const bolt of model.bodies.filter(b => b.parent === body && b.semantic === "lock")) {
    const box = bolt.geoms.find(g => g.name === `${bolt.name}_box` && g.type === "box" && g.part_label === "Deadbolt");
    if (!box || (bolt.joint && driven.has(bolt.joint.name))) continue;
    const h = bolt.joint ? joints.get(bolt.joint.name) : undefined;
    const projection = Math.sign(bolt.pos[0]) * box.pos[0] + box.size[0] - (h ? h.q - h.modeledAt : 0);
    if (projection > 0.001) return `${leaf.name}: locked by a separate deadbolt${h ? "; retract its thumbturn first" : " (fixed; no release)"}`;
  }
  // Independent native lock slides (e.g. auxiliary barrel bolts) use full positive travel for withdrawal.
  for (const b of model.bodies) {
    const joint = b.joint;
    if (joint?.role !== "lock" || joint.type !== "slide" || driven.has(joint.name) || leafOfJoint(model, joint.name) !== leaf.name) continue;
    const h = joints.get(joint.name);
    if (h?.range && h.q < h.range[1] - 1e-6) return `${leaf.name}: locked; withdraw ${joint.label || joint.name} first`;
  }
  const op = operator ? joints.get(operator) : undefined;
  if (op?.range && op.range[1] - op.range[0] < 0.006 && thrown(joints, boltJointsFor(model, operator))) return `${leaf.name}: locked operator`;
  return null;
}

function simultaneous(moves: { joint: string; from: number; to: number }[], dur: number): Phase[] {
  const [first, ...followers] = moves;
  return first ? [{ ...first, dur, followers }] : [];
}

function hasAstragal(model: ModelJ): boolean {
  const secondaryBody = model.bodies.find(b => b.joint?.name === model.meta.secondary_joint);
  return !!secondaryBody?.geoms.some(g => g.name === "astragal");
}

/** B's astragal overlaps A. A can open with B bolted shut; B cannot open until A is clear. */
function astragalBlock(model: ModelJ, joints: Map<string, JointLike>, leaf: JointLike, target: number): string | null {
  if (!hasAstragal(model)) return null;
  const opening = Math.abs(target - leaf.modeledAt) > Math.abs(leaf.q - leaf.modeledAt) + 1e-6;
  const closing = Math.abs(target - leaf.modeledAt) < Math.abs(leaf.q - leaf.modeledAt) - 1e-6;
  const a = joints.get(model.meta.primary_joint), b = joints.get(model.meta.secondary_joint);
  if (leaf === b && opening && a && Math.abs(a.q - openPosition(a)) > 1e-6) return "overlapping astragal: open leaf A fully before opening leaf B";
  if (leaf === a && closing && b && Math.abs(b.q - b.modeledAt) > 1e-6) return "overlapping astragal: close leaf B before closing leaf A";
  return null;
}

function pairedOpenClosePhases(model: ModelJ, joints: Map<string, JointLike>): { phases: Phase[]; note: string | null } {
  const leaves = [model.meta.primary_joint, model.meta.secondary_joint].map(name => joints.get(name)).filter((h): h is JointLike => !!h);
  const notes: string[] = [];
  let movable = leaves.filter(leaf => {
    const reason = pairLeafBlock(model, joints, leaf);
    if (reason) notes.push(reason);
    return !reason;
  });
  // The astragal attached to B overlaps A: clear A first when opening, seat B first when closing.
  const overlaps = hasAstragal(model);
  // A partly open pair opens fully on the next click; only a fully open pair toggles closed.
  const opening = movable.some(leaf => Math.abs(leaf.q - openPosition(leaf)) > 0.02);
  movable = movable.filter(leaf => {
    const other = leaves.find(h => h !== leaf);
    // Both free leaves will be sequenced below. Otherwise check the stationary leaf's position.
    const reason = other && !movable.includes(other) ? astragalBlock(model, joints, leaf, opening ? openPosition(leaf) : leaf.modeledAt) : null;
    if (reason) notes.push(reason);
    return !reason;
  });
  const moves = movable.map(leaf => ({ joint: leaf.name, from: leaf.q, to: opening ? openPosition(leaf) : leaf.modeledAt }))
    .filter(move => Math.abs(move.to - move.from) > 1e-6);
  const operators = moves.map(move => previewOperatorForLeaf(model, move.joint))
    .filter((name): name is string => !!name && boltJointsFor(model, name).length > 0)
    .map(name => joints.get(name)).filter((op): op is JointLike => !!op && !!op.range && op.range[1] - op.range[0] >= 0.006);
  const phases = simultaneous(operators.map(op => ({ joint: op.name, from: op.q, to: op.range![1] })), 450);
  phases.push(...(overlaps
    ? (opening ? moves : [...moves].reverse()).map(move => ({ ...move, dur: 1400, followers: [] }))
    : simultaneous(moves, 1400)));
  phases.push(...simultaneous(operators.map(op => ({ joint: op.name, from: op.range![1], to: op.modeledAt })), 400));
  if (moves.length) notes.unshift(`mechanism preview: ${operators.length ? "operators actuated → bolts retracted → " : ""}${moves.length === 2 ? "both leaves" : "free leaf"} ${opening ? "open" : "close"}${overlaps ? " in astragal order" : ""}`);
  return { phases, note: notes.join("; ") || null };
}

/**
 * Phases for the "Open / close door" button: work the operator (which retracts the bolt through its coupling), move
 * the active leaf (both leaves of a dutch door when the joining bolt is engaged; both free leaves of a swing pair),
 * release the operator.  A locked operator (range < 0.006) is never driven.
 */
export function openClosePhases(model: ModelJ, joints: Map<string, JointLike>): { phases: Phase[]; note: string | null } {
  if (isSwingPair(model)) return pairedOpenClosePhases(model, joints);
  const meta = model.meta ?? {};
  const operator: string | undefined = meta.operator_joint ?? undefined;
  const leafName = activeLeaf(model);
  const leaf = leafName ? joints.get(leafName) : undefined;
  if (!leaf) return { phases: [], note: null };
  const closed = leaf.modeledAt ?? 0;
  const open = !leaf.range ? closed + 1.2 : leaf.range[1] > closed + 1e-6 ? leaf.range[1] : leaf.range[0];
  const isClosed = Math.abs(leaf.q - closed) < 0.02;
  const target = isClosed ? open : closed;
  const ops = operatorJoints(model).map((n) => joints.get(n)).filter(Boolean) as JointLike[];
  const individual = operatorsAreIndividual(model);
  const op = operator ? joints.get(operator) : undefined;
  const bolts = boltJointsFor(model, operator);
  const locked = !!op && !!op.range && op.range[1] - op.range[0] < 0.006;
  // each dog / lever bolt IS its own latch, so an individually latched door always has to be worked, whether or not
  // the first operator happens to drive a separate bolt joint
  const workable = ops.filter((h) => !!h.range && h.range[1] - h.range[0] >= 0.006 && (individual || boltJointsFor(model, h.name).length > 0));
  const needsOp = !locked && workable.length > 0;
  const phases: Phase[] = [];
  let note: string | null = null;
  if (locked && isClosed && bolts.length && thrown(joints, bolts)) note = "locked: the operator cannot be worked from this side (range < 0.006)";
  // release: one phase per operator (a robot turns dog levers one after another), 450 ms for a single operator and
  // a shorter beat each when there are several so the whole sequence stays watchable
  const beat = workable.length > 1 ? Math.max(220, Math.round(1200 / workable.length)) : 450;
  if (needsOp) for (const h of workable) phases.push({ joint: h.name, from: h.q, to: h.range![1], dur: beat });
  const followers: NonNullable<Phase["followers"]> = [];
  const jb = joints.get("join_bolt_slide");
  const secondary: string | undefined = meta.secondary_joint ?? undefined;
  if (secondary && jb && jb.q < 0.01) { const s = joints.get(secondary); if (s) followers.push({ joint: secondary, from: s.q, to: target }); }
  phases.push({ joint: leaf.name, from: leaf.q, to: target, dur: 1400, followers });
  // re-engage in the reverse order: the last one released is the first one thrown again
  if (needsOp) for (const h of [...workable].reverse()) phases.push({ joint: h.name, from: h.range![1], to: h.modeledAt ?? h.range![0], dur: Math.round(beat * 0.9) });
  if (!note) {
    if (needsOp && individual) note = `${workable.length} latches released one by one → leaf ${isClosed ? "opens" : "closes"} → all ${workable.length} re-engaged`;
    else if (needsOp) note = isClosed ? "operator actuated → bolt retracted → leaf opens → operator released" : "operator actuated → bolt retracted → leaf closes → bolt re-extends";
    else if (followers.length) note = "joining bolt engaged: both leaves move together";
  }
  return { phases, note };
}

/**
 * What else must move when a leaf slider is dragged to q: the operator is driven to full travel when the latch is still
 * thrown (so the bolt clears the strike), and the other leaf of a joined dutch door follows.
 */
export function sliderReaction(model: ModelJ, joints: Map<string, JointLike>, leafJoint: string, q: number): { operator: string | null; operatorTo: number | null; operatorsTo: { joint: string; q: number }[]; blocked: boolean; mirror: { joint: string; q: number } | null; note: string | null } {
  const meta = model.meta ?? {};
  const primary: string | undefined = meta.primary_joint ?? undefined;
  const secondary: string | undefined = meta.secondary_joint ?? undefined;
  const operator = previewOperatorForLeaf(model, leafJoint);
  const h = joints.get(leafJoint);
  const out = { operator: null as string | null, operatorTo: null as number | null, operatorsTo: [] as { joint: string; q: number }[], blocked: false, mirror: null as { joint: string; q: number } | null, note: null as string | null };
  if (!h || (leafJoint !== primary && leafJoint !== secondary)) return out;
  if (isSwingPair(model)) {
    const reason = pairLeafBlock(model, joints, h) || astragalBlock(model, joints, h, q);
    if (reason) return { ...out, blocked: true, note: reason };
  }
  const away = Math.abs(q - (h.modeledAt ?? 0)) > 0.02;
  const opLeaf = leafOfJoint(model, operator);
  const ownsOperator = opLeaf === leafJoint || (!opLeaf && leafJoint === primary);
  const op = operator ? joints.get(operator) : undefined;
  const locked = !!op && !!op.range && op.range[1] - op.range[0] < 0.006;
  const individual = operatorsAreIndividual(model);
  if (away && ownsOperator && op && op.range && !locked && (individual || thrown(joints, boltJointsFor(model, operator)))) {
    // every independent latch has to be clear before the leaf can be dragged anywhere, not just the first one
    const all = operatorJoints(model).map((n) => joints.get(n)).filter((x): x is JointLike => !!x && !!x.range && x.range[1] - x.range[0] >= 0.006);
    if (individual && all.some((x) => x.q < 0.8 * x.range![1])) {
      out.operator = op.name;
      out.operatorsTo = all.map((x) => ({ joint: x.name, q: x.range![1] }));
      out.operatorTo = op.range[1];
      out.note = `all ${all.length} latches released: the leaf is held while any one of them is engaged`;
    } else if (!individual && thrown(joints, boltJointsFor(model, operator))) {
      out.operator = op.name;
      out.operatorTo = op.range[1];
      out.note = "latch retracted: operator driven to full travel so the bolt clears the strike";
    }
  }
  const jb = joints.get("join_bolt_slide");
  if (primary && secondary && jb && jb.q < 0.01) out.mirror = { joint: leafJoint === primary ? secondary : primary, q };
  return out;
}

/** Deep link `q=<joint>:<value>[,<joint>:<value>...]` — value in rad / m, or with a `deg` / `mm` suffix (e.g. `q=leaf_hinge:45deg`). */
export function parsePoseQuery(query: string): [string, number][] {
  const raw = new URLSearchParams(query).get("q");
  if (!raw) return [];
  const out: [string, number][] = [];
  for (const part of raw.split(",")) {
    const m = /^\s*([\w.-]+)\s*[:=]\s*(-?[\d.]+(?:e-?\d+)?)\s*(deg|mm|rad|m)?\s*$/i.exec(part);
    if (!m) continue;
    let v = parseFloat(m[2]);
    if (!Number.isFinite(v)) continue;
    const unit = (m[3] ?? "").toLowerCase();
    if (unit === "deg") v *= Math.PI / 180; else if (unit === "mm") v /= 1000;
    out.push([m[1], v]);
  }
  return out;
}
