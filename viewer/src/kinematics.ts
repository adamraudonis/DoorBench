// Closed kinematic loops for the 3D viewer: door closers (two-bar arm linkages), gas struts / pneumatic tubes
// (telescoping hinge + slide), automatic-operator arms and any other `connect` equality in model.json.
//
// MuJoCo closes these loops with a `connect` equality (a point on body A pinned to the same world point on body B).
// The viewer drives the door joints kinematically, so this module recomputes the *mechanism* joints of every loop each
// time a driver joint moves:
//   * two_bar      analytic planar two-link IK (pinion + elbow hinges) with a fixed elbow side, clamped when |PT| > L1+L2
//   * telescoping  analytic aim (hinge) + extension (slide)
//   * generic      damped Gauss-Newton over the loop's mechanism joints, warm-started from the previous frame
// Analytic solutions are refined by the numeric solver when the geometry is slightly off (e.g. an anchor a hair out of
// the arm plane), so the tip stays on its anchor whenever that is kinematically possible.
//
// Pure (no DOM, no scene graph): runs in the browser and under `bun test`.  Uses three.js math only.
import { Matrix3, Matrix4, Quaternion, Vector3 } from "three";
import type { BodyJ, EqualityJ, LinkageJ, ModelJ, Vec3 } from "./types";

export type LoopType = "two_bar" | "telescoping" | "generic";

export interface KJoint {
  name: string;
  body: number;
  type: "hinge" | "slide";
  axis: Vector3;            // unit, body frame
  pos: Vector3;             // body frame
  range: [number, number] | null;
  modeledAt: number;
  role: string;
  interactive: boolean;
  q: number;
}

export interface KBody {
  name: string;
  index: number;
  parent: number;           // -1 = world
  local: Matrix4;           // pos/quat relative to the parent frame
  joint: KJoint | null;
  static: boolean;
}

const ONE = new Vector3(1, 1, 1);
const IDQ = new Quaternion();
const IDENTITY = new Matrix4();
const _q = new Quaternion();
const _v = new Vector3();
const _v2 = new Vector3();
const _m3 = new Matrix3();

function v3(a: Vec3 | number[] | undefined, fallback = 0): Vector3 {
  return new Vector3(a?.[0] ?? fallback, a?.[1] ?? fallback, a?.[2] ?? fallback);
}

/** Joint transform in the body frame: hinge = rotate about `axis` through `pos`; slide = translate along `axis`. */
function jointMatrix(j: KJoint, out: Matrix4): Matrix4 {
  const dq = j.q - j.modeledAt;
  if (j.type === "hinge") {
    _q.setFromAxisAngle(j.axis, dq);
    _v.copy(j.pos).applyQuaternion(_q);
    _v2.copy(j.pos).sub(_v);
    return out.compose(_v2, _q, ONE);
  }
  _v.copy(j.axis).multiplyScalar(dq);
  return out.compose(_v, IDQ, ONE);
}

/** Rotate a (non-unit) vector by the rotation part of a rigid transform. */
function rotate(v: Vector3, m: Matrix4): Vector3 {
  return v.applyMatrix3(_m3.setFromMatrix4(m));
}

/** Signed angle from `a` to `b` about unit axis `n` (both assumed roughly perpendicular to n). */
function angleAbout(a: Vector3, b: Vector3, n: Vector3): number {
  const cx = a.y * b.z - a.z * b.y, cy = a.z * b.x - a.x * b.z, cz = a.x * b.y - a.y * b.x;
  return Math.atan2(cx * n.x + cy * n.y + cz * n.z, a.dot(b));
}

/** Rotate `v` about unit axis `n` by `ang` (Rodrigues). */
function rotateAbout(v: Vector3, n: Vector3, ang: number, out: Vector3): Vector3 {
  const c = Math.cos(ang), s = Math.sin(ang);
  const dot = n.dot(v);
  const cx = n.y * v.z - n.z * v.y, cy = n.z * v.x - n.x * v.z, cz = n.x * v.y - n.y * v.x;
  return out.set(v.x * c + cx * s + n.x * dot * (1 - c), v.y * c + cy * s + n.y * dot * (1 - c), v.z * c + cz * s + n.z * dot * (1 - c));
}

/** Keep a hinge angle continuous with its previous value (hinges are 2π-periodic). */
function unwrap(q: number, prev: number): number {
  return q + 2 * Math.PI * Math.round((prev - q) / (2 * Math.PI));
}

function clampRange(j: KJoint, q: number): number {
  return j.range ? Math.min(Math.max(q, j.range[0]), j.range[1]) : q;
}

// ---------------------------------------------------------------------------
// Articulated tree with forward kinematics (mirrors the three.js scene graph built in scene.ts)
// ---------------------------------------------------------------------------
export class Articulation {
  readonly bodies: KBody[] = [];
  readonly index = new Map<string, number>();
  readonly joints = new Map<string, KJoint>();
  readonly pre: Matrix4[] = [];      // world transform of the body frame before its joint (joint pos/axis live here)
  readonly post: Matrix4[] = [];     // world transform after the joint (geoms, sites and children live here)
  private readonly jm = new Matrix4();

  constructor(model: ModelJ) {
    const byName = new Map<string, BodyJ>(model.bodies.map((b) => [b.name, b]));
    // parents are written before children (ir.Model.validate); tolerate any order anyway
    const pending = model.bodies.slice();
    let guard = 0;
    while (pending.length && guard++ < 100000) {
      const b = pending.shift()!;
      const hasParent = !!b.parent && byName.has(b.parent);
      if (hasParent && !this.index.has(b.parent!)) { pending.push(b); continue; }
      const index = this.bodies.length;
      const parent = hasParent ? this.index.get(b.parent!)! : -1;
      const local = new Matrix4().compose(v3(b.pos), new Quaternion(b.quat[1], b.quat[2], b.quat[3], b.quat[0]), ONE);
      let joint: KJoint | null = null;
      if (b.joint) {
        const j = b.joint;
        joint = {
          name: j.name, body: index, type: j.type, axis: v3(j.axis).normalize(), pos: v3(j.pos), range: j.range, modeledAt: j.modeled_at ?? 0,
          role: j.role, interactive: !!j.robot_interactive, q: j.modeled_at ?? 0,
        };
        this.joints.set(j.name, joint);
      }
      this.bodies.push({ name: b.name, index, parent, local, joint, static: !!b.static });
      this.index.set(b.name, index);
      this.pre.push(new Matrix4());
      this.post.push(new Matrix4());
    }
    this.update();
  }

  setQ(name: string, q: number): void {
    const j = this.joints.get(name);
    if (j) j.q = q;
  }

  getQ(name: string): number {
    return this.joints.get(name)?.q ?? 0;
  }

  /** Full forward kinematics (bodies are in parent-first order; ~100 small matrix products). */
  update(): void {
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i];
      const parentPost = b.parent >= 0 ? this.post[b.parent] : IDENTITY;
      this.pre[i].multiplyMatrices(parentPost, b.local);
      if (b.joint) this.post[i].multiplyMatrices(this.pre[i], jointMatrix(b.joint, this.jm));
      else this.post[i].copy(this.pre[i]);
    }
  }

  /** World position of a point given in a body's (post-joint) frame; body -1 = world. */
  worldPoint(body: number, local: Vector3, out: Vector3): Vector3 {
    out.copy(local);
    return body >= 0 ? out.applyMatrix4(this.post[body]) : out;
  }

  /** World position and unit axis of a joint (independent of the joint's own value). */
  jointWorld(j: KJoint, p: Vector3, n: Vector3): void {
    p.copy(j.pos).applyMatrix4(this.pre[j.body]);
    n.copy(j.axis).transformDirection(this.pre[j.body]);
  }

  /** Bodies from `body` up to the root (inclusive). */
  chain(body: number): number[] {
    const out: number[] = [];
    let b = body;
    let guard = 0;
    while (b >= 0 && guard++ < 1000) { out.push(b); b = this.bodies[b].parent; }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Loop descriptions
// ---------------------------------------------------------------------------
interface LoopJoint { joint: KJoint; side: 1 | -1 }     // +1: the joint moves the tip (body A side), -1: it moves the anchor (body B side)

interface LoopBase {
  name: string;
  equality: string;
  source: "schema" | "derived";
  tipBody: number;          // the body carrying the constrained point that the mechanism moves
  tipLocal: Vector3;        // in tipBody frame
  anchorBody: number;       // -1 = world
  anchorLocal: Vector3;     // in anchorBody frame
  joints: LoopJoint[];      // mechanism joints this loop owns
  warnings: string[];
  stretched: boolean;       // last solve hit the reach limit
  lastSeparation: number;
}
interface TwoBarLoop extends LoopBase {
  type: "two_bar";
  pinion: KJoint;           // hinge on the main arm body
  elbow: KJoint;            // hinge on the forearm body (child of the main arm body)
  restDir1: Vector3;        // pinion -> elbow direction in the main-arm frame (at q = modeledAt)
  restDir2: Vector3;        // elbow -> tip direction in the forearm frame
  L1: number;
  L2: number;
  elbowSign: 1 | -1;
}
interface TelescopingLoop extends LoopBase {
  type: "telescoping";
  hinge: KJoint;            // on the cylinder body
  slide: KJoint;            // on the rod body (child of the cylinder body)
  restDir: Vector3;         // hinge -> tip direction in the cylinder frame at rest
  rRest: number;            // hinge -> tip distance at rest
  slideSign: number;        // +1 when +slide extends the strut along restDir
}
interface GenericLoop extends LoopBase { type: "generic" }
type Loop = TwoBarLoop | TelescopingLoop | GenericLoop;

export interface LoopResult {
  name: string;
  type: LoopType;
  source: "schema" | "derived";
  equality: string;
  joints: string[];
  separation: number;       // m, anchor-point separation after the solve
  stretched: boolean;       // reach limit hit (|PT| outside [|L1-L2|, L1+L2]) -> the tip cannot reach the anchor
  ok: boolean;              // separation < 1 mm
}

export interface LoopSolverOptions {
  forceGeneric?: boolean;   // testing: ignore the analytic solvers
  tolerance?: number;       // m, default 1e-4 (the numeric solver stops below this)
}

const DRIVER_ROLES = new Set(["primary", "secondary", "operator"]);

/** True when model.json describes at least one closed loop the viewer must solve. */
export function hasLoops(model: ModelJ): boolean {
  return (model.equalities ?? []).some((e) => e.kind === "connect" && e.active !== false) || !!model.linkages?.length;
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------
export class LoopSolver {
  readonly art: Articulation;
  readonly loops: Loop[] = [];
  readonly owned = new Set<string>();
  readonly coupled = new Set<string>();     // joints driven by a joint equality / tendon: inputs to the loops, never unknowns
  readonly warnings: string[] = [];
  private readonly tol: number;
  private readonly forceGeneric: boolean;
  // scratch
  private readonly P = new Vector3(); private readonly N = new Vector3(); private readonly T = new Vector3();
  private readonly D = new Vector3(); private readonly U = new Vector3(); private readonly E = new Vector3();
  private readonly A = new Vector3(); private readonly B = new Vector3(); private readonly R = new Vector3();

  constructor(model: ModelJ, opts: LoopSolverOptions = {}) {
    this.tol = opts.tolerance ?? 1e-4;
    this.forceGeneric = !!opts.forceGeneric;
    this.art = new Articulation(model);
    // polynomial joint couplings (rising hinges, bolt <- handle) and one-sided tendons are set by the scene each frame
    for (const e of model.equalities ?? []) if (e.kind === "joint" && e.b) this.coupled.add(e.a);
    for (const t of model.tendons ?? []) if (t.sites?.[0]?.[0]) this.coupled.add(t.sites[0][0]);
    const schema = new Map<string, LinkageJ>();
    for (const l of model.linkages ?? []) if (l && typeof l === "object" && l.equality) schema.set(l.equality, l);
    for (const e of model.equalities ?? []) {
      if (e.kind !== "connect" || e.active === false) continue;
      const loop = this.build(e, schema.get(e.name));
      if (loop) { this.loops.push(loop); for (const lj of loop.joints) this.owned.add(lj.joint.name); }
    }
    // schema linkages whose equality is not in the equalities list (should not happen; the loop needs the anchor)
    for (const [eq, l] of schema) if (!this.loops.some((x) => x.equality === eq)) this.warnings.push(`linkage ${l.name}: equality ${eq} not found in model.json equalities; ignored`);
    this.solve();
  }

  setQ(name: string, q: number): void { this.art.setQ(name, q); }
  getQ(name: string): number { return this.art.getQ(name); }
  get linkageNames(): string[] { return this.loops.map((l) => l.name); }

  // ---- construction -------------------------------------------------------
  private build(e: EqualityJ, link: LinkageJ | undefined): Loop | null {
    const art = this.art;
    const a = art.index.get(e.a);
    if (a === undefined) { this.warnings.push(`${e.name}: body ${e.a} not found`); return null; }
    const b = e.b && e.b !== "world" ? art.index.get(e.b) : -1;
    if (b === undefined) { this.warnings.push(`${e.name}: body ${e.b} not found`); return null; }
    const anchorA = v3(e.anchor);
    // rest pose: every joint at modeled_at (MuJoCo qpos0 = ref) -> the world point both bodies share
    const saved = new Map<string, number>();
    for (const j of art.joints.values()) { saved.set(j.name, j.q); j.q = j.modeledAt; }
    art.update();
    const restWorld = art.worldPoint(a, anchorA, new Vector3());
    const anchorB = restWorld.clone();
    if (b >= 0) anchorB.applyMatrix4(new Matrix4().copy(art.post[b]).invert());
    // the sub-chain between the two bodies
    const chainA = art.chain(a), chainB = b >= 0 ? art.chain(b) : [];
    const setB = new Set(chainB);
    const common = chainA.find((x) => setB.has(x));
    const segA = common === undefined ? chainA : chainA.slice(0, chainA.indexOf(common));
    const segB = common === undefined ? chainB : chainB.slice(0, chainB.indexOf(common));
    const jointsOf = (seg: number[], side: 1 | -1): LoopJoint[] => seg.map((i) => art.bodies[i].joint).filter((j): j is KJoint => !!j).map((joint) => ({ joint, side }));
    const allA = jointsOf(segA, 1), allB = jointsOf(segB, -1);
    const free = (x: LoopJoint) => !this.coupled.has(x.joint.name);
    let ownedA = allA.filter((x) => free(x) && x.joint.role === "mechanism"), ownedB = allB.filter((x) => free(x) && x.joint.role === "mechanism");
    if (!ownedA.length && !ownedB.length) {
      ownedA = allA.filter((x) => free(x) && !x.joint.interactive && !DRIVER_ROLES.has(x.joint.role));
      ownedB = allB.filter((x) => free(x) && !x.joint.interactive && !DRIVER_ROLES.has(x.joint.role));
    }
    const warnings: string[] = [];
    const base = { name: link?.name ?? e.name.replace(/_connect$/, ""), equality: e.name, source: link ? "schema" as const : "derived" as const, warnings, stretched: false, lastSeparation: 0 };
    const restore = () => { for (const [n, q] of saved) art.setQ(n, q); art.update(); };
    if (!ownedA.length && !ownedB.length) {
      warnings.push(`${e.name}: no mechanism joint between ${e.a} and ${e.b ?? "world"}; the loop cannot be solved`);
      this.warnings.push(...warnings);
      restore();
      return { ...base, type: "generic", tipBody: a, tipLocal: anchorA, anchorBody: b, anchorLocal: anchorB, joints: [] };
    }
    // orient the loop: the "tip" is the constrained point moved by the mechanism joints; the anchor is the other point
    const mobileIsA = ownedA.length > 0 && ownedB.length === 0;
    const mobileIsB = ownedB.length > 0 && ownedA.length === 0;
    const tipBody = mobileIsB ? b : a, tipLocal = mobileIsB ? anchorB : anchorA;
    const anchorBody = mobileIsB ? a : b, anchorLocal = mobileIsB ? anchorA : anchorB;
    const joints = [...ownedA, ...ownedB];
    const generic = (): GenericLoop => ({ ...base, type: "generic", tipBody, tipLocal, anchorBody, anchorLocal, joints });
    let loop: Loop | null = null;
    if (!this.forceGeneric && (mobileIsA || mobileIsB)) {
      const M = (mobileIsB ? ownedB : ownedA).map((x) => x.joint);   // nearest the tip first
      const tipB = art.bodies[tipBody];
      const wantType = link?.type;
      if (M.length === 2 && M[0].body === tipBody && M[1].body === tipB.parent && M[0].type === "hinge" && M[1].type === "hinge" && wantType !== "telescoping") {
        loop = this.buildTwoBar(base, tipBody, tipLocal, anchorBody, anchorLocal, joints, M[1], M[0], link);
      } else if (M.length === 2 && M[0].body === tipBody && M[1].body === tipB.parent && M[0].type === "slide" && M[1].type === "hinge" && wantType !== "two_bar") {
        loop = this.buildTelescoping(base, tipBody, tipLocal, anchorBody, anchorLocal, joints, M[1], M[0], link);
      }
      if (link && !loop) warnings.push(`${link.name}: declared ${link.type} but the body tree does not match (${M.map((j) => `${j.name}:${j.type}`).join(" > ")}); using the numeric solver`);
    }
    if (!loop) loop = generic();
    if (link) {
      // the schema anchor must be where the rest pose puts the tip
      const p = v3(link.anchor?.pos);
      const ab = link.anchor?.body === "world" || !link.anchor?.body ? -1 : art.index.get(link.anchor.body);
      if (ab === undefined) warnings.push(`${link.name}: anchor body ${link.anchor.body} not found; using the rest-pose anchor`);
      else {
        const w = art.worldPoint(ab, p, new Vector3());
        const d = w.distanceTo(restWorld);
        if (d > 1e-3) warnings.push(`${link.name}: schema anchor is ${(d * 1000).toFixed(1)} mm from the rest-pose tip of ${e.a}; using the schema anchor`);
        loop.anchorBody = ab; loop.anchorLocal = p;
      }
    }
    this.warnings.push(...warnings);
    restore();
    return loop;
  }

  private buildTwoBar(base: Omit<LoopBase, "tipBody" | "tipLocal" | "anchorBody" | "anchorLocal" | "joints">, tipBody: number, tipLocal: Vector3, anchorBody: number, anchorLocal: Vector3, joints: LoopJoint[], pinion: KJoint, elbow: KJoint, link: LinkageJ | undefined): TwoBarLoop | null {
    const art = this.art;
    const arm1 = art.bodies[pinion.body], arm2 = art.bodies[elbow.body];
    // elbow point in the main-arm frame: forearm frame origin + its joint pos
    const elbowLocal = elbow.pos.clone().applyMatrix4(arm2.local);
    const d1 = elbowLocal.clone().sub(pinion.pos), d2 = tipLocal.clone().sub(elbow.pos);
    const L1 = d1.length(), L2 = d2.length();
    if (L1 < 1e-6 || L2 < 1e-6) return null;
    // both hinge axes must be parallel (planar linkage)
    const n1 = pinion.axis.clone(), n2 = rotate(elbow.axis.clone(), arm2.local);
    if (Math.abs(Math.abs(n1.dot(n2)) - 1) > 1e-3) { base.warnings.push(`${base.name}: pinion and elbow axes are not parallel (${Math.acos(Math.min(1, Math.abs(n1.dot(n2)))).toFixed(3)} rad); numeric solver`); return null; }
    // in-plane arm directions (the rest offsets may include a small out-of-plane component; keep the planar part)
    const restDir1 = d1.clone().addScaledVector(n1, -d1.dot(n1)).normalize();
    const restDir2 = d2.clone().addScaledVector(elbow.axis, -d2.dot(elbow.axis)).normalize();
    // elbow side at the rest pose (art is at the rest pose here)
    art.jointWorld(pinion, this.P, this.N);
    art.worldPoint(anchorBody, anchorLocal, this.T);
    art.worldPoint(arm1.index, elbowLocal, this.E);
    this.D.copy(this.T).sub(this.P); this.D.addScaledVector(this.N, -this.D.dot(this.N));
    this.E.sub(this.P);
    const cross = this.D.clone().cross(this.E).dot(this.N);
    let elbowSign: 1 | -1 = cross >= 0 ? 1 : -1;
    if (link && link.type === "two_bar") {
      if (link.elbow_sign === 1 || link.elbow_sign === -1) {
        if (link.elbow_sign !== elbowSign && Math.abs(cross) > 1e-6) base.warnings.push(`${base.name}: schema elbow_sign ${link.elbow_sign} disagrees with the rest pose (${elbowSign}); using the schema`);
        elbowSign = link.elbow_sign;
      }
      if (Number.isFinite(link.L1) && Math.abs(link.L1 - L1) > 5e-4) base.warnings.push(`${base.name}: schema L1 ${link.L1} differs from the geometry (${L1.toFixed(4)})`);
      if (Number.isFinite(link.L2) && Math.abs(link.L2 - L2) > 5e-4) base.warnings.push(`${base.name}: schema L2 ${link.L2} differs from the geometry (${L2.toFixed(4)})`);
    }
    return { ...base, type: "two_bar", tipBody, tipLocal, anchorBody, anchorLocal, joints, pinion, elbow, restDir1, restDir2, L1, L2, elbowSign };
  }

  private buildTelescoping(base: Omit<LoopBase, "tipBody" | "tipLocal" | "anchorBody" | "anchorLocal" | "joints">, tipBody: number, tipLocal: Vector3, anchorBody: number, anchorLocal: Vector3, joints: LoopJoint[], hinge: KJoint, slide: KJoint, link: LinkageJ | undefined): TelescopingLoop | null {
    const art = this.art;
    const rod = art.bodies[slide.body];
    // tip in the cylinder frame at rest; the strut line is hinge -> tip
    const tipInCyl = tipLocal.clone().applyMatrix4(rod.local);
    const d = tipInCyl.sub(hinge.pos);
    const restDir = d.clone().addScaledVector(hinge.axis, -d.dot(hinge.axis));
    const rRest = restDir.length();
    if (rRest < 1e-6) return null;
    restDir.divideScalar(rRest);
    const slideAxisCyl = rotate(slide.axis.clone(), rod.local);
    const along = slideAxisCyl.dot(restDir);
    if (Math.abs(along) < 0.99) { base.warnings.push(`${base.name}: slide axis is ${Math.acos(Math.min(1, Math.abs(along))).toFixed(2)} rad off the strut line; numeric solver`); return null; }
    if (link && link.type === "telescoping" && Number.isFinite(link.slide?.offset)) {
      const off = rRest - slide.modeledAt * Math.sign(along);
      if (Math.abs(off - link.slide.offset) > 5e-4) base.warnings.push(`${base.name}: schema slide offset ${link.slide.offset} differs from the geometry (${off.toFixed(4)})`);
    }
    return { ...base, type: "telescoping", tipBody, tipLocal, anchorBody, anchorLocal, joints, hinge, slide, restDir, rRest, slideSign: Math.sign(along) };
  }

  // ---- solving ------------------------------------------------------------
  /** Solve every loop for the current driver joint values.  Call after changing driver joints; reads/writes `art`. */
  solve(): LoopResult[] {
    const art = this.art;
    const out: LoopResult[] = [];
    art.update();
    for (const loop of this.loops) {
      if (loop.joints.length) {
        if (loop.type === "two_bar") this.solveTwoBar(loop);
        else if (loop.type === "telescoping") this.solveTelescoping(loop);
        let sep = this.separation(loop);
        // numeric refinement: generic loops always (warm-started from the previous frame; converges in a handful of
        // iterations for small door increments, the cap only matters for slider jumps); analytic ones only when the
        // closed form left a gap it could fix (e.g. an anchor slightly out of the arm plane)
        if (loop.type === "generic" || (sep > this.tol && !loop.stretched)) sep = this.refine(loop, loop.type === "generic" ? 40 : 4);
        // a generic loop that cannot be closed is reported the same way as an over-reached analytic one
        if (loop.type === "generic") loop.stretched = sep >= 1e-3;
        loop.lastSeparation = sep;
      } else { loop.lastSeparation = this.separation(loop); loop.stretched = loop.lastSeparation >= 1e-3; }
      out.push({ name: loop.name, type: loop.type, source: loop.source, equality: loop.equality, joints: loop.joints.map((j) => j.joint.name), separation: loop.lastSeparation, stretched: loop.stretched, ok: loop.lastSeparation < 1e-3 });
    }
    return out;
  }

  private separation(loop: Loop): number {
    this.art.worldPoint(loop.tipBody, loop.tipLocal, this.A);
    this.art.worldPoint(loop.anchorBody, loop.anchorLocal, this.B);
    return this.A.distanceTo(this.B);
  }

  private solveTwoBar(loop: TwoBarLoop): void {
    const art = this.art;
    const { P, N, T, D, U, E } = this;
    art.jointWorld(loop.pinion, P, N);
    art.worldPoint(loop.anchorBody, loop.anchorLocal, T);
    D.copy(T).sub(P);
    D.addScaledVector(N, -D.dot(N));                    // project the anchor into the arm plane
    const r = D.length();
    if (r < 1e-9) { U.copy(loop.restDir1).transformDirection(art.pre[loop.pinion.body]); } else U.copy(D).divideScalar(r);
    const lo = Math.abs(loop.L1 - loop.L2) + 1e-9, hi = loop.L1 + loop.L2 - 1e-9;
    const rr = Math.min(Math.max(r, lo), hi);
    loop.stretched = r > hi || r < lo;
    const cosA = (loop.L1 * loop.L1 + rr * rr - loop.L2 * loop.L2) / (2 * loop.L1 * rr);
    const alpha = Math.acos(Math.min(1, Math.max(-1, cosA)));
    // main arm: rotate the in-plane direction to the elbow by the elbow side
    rotateAbout(U, N, loop.elbowSign * alpha, this.R);   // R = dir1
    E.copy(P).addScaledVector(this.R, loop.L1);
    this.A.copy(loop.restDir1).transformDirection(art.pre[loop.pinion.body]);
    const q1 = unwrap(loop.pinion.modeledAt + angleAbout(this.A, this.R, N), loop.pinion.q);
    loop.pinion.q = clampRange(loop.pinion, q1);
    art.update();
    // forearm: from the elbow to the (clamped) target on the line
    this.B.copy(P).addScaledVector(U, rr).sub(E).normalize();
    this.A.copy(loop.restDir2).transformDirection(art.pre[loop.elbow.body]);
    this.D.copy(loop.elbow.axis).transformDirection(art.pre[loop.elbow.body]);
    const q2 = unwrap(loop.elbow.modeledAt + angleAbout(this.A, this.B, this.D), loop.elbow.q);
    loop.elbow.q = clampRange(loop.elbow, q2);
    art.update();
  }

  private solveTelescoping(loop: TelescopingLoop): void {
    const art = this.art;
    const { P, N, T, D, U } = this;
    art.jointWorld(loop.hinge, P, N);
    art.worldPoint(loop.anchorBody, loop.anchorLocal, T);
    D.copy(T).sub(P);
    D.addScaledVector(N, -D.dot(N));
    const r = D.length();
    this.A.copy(loop.restDir).transformDirection(art.pre[loop.hinge.body]);
    if (r < 1e-9) U.copy(this.A); else U.copy(D).divideScalar(r);
    const qh = unwrap(loop.hinge.modeledAt + angleAbout(this.A, U, N), loop.hinge.q);
    loop.hinge.q = clampRange(loop.hinge, qh);
    const qs = loop.slide.modeledAt + loop.slideSign * (r - loop.rRest);
    const qc = clampRange(loop.slide, qs);
    loop.stretched = Math.abs(qc - qs) > 1e-9;
    loop.slide.q = qc;
    art.update();
  }

  /**
   * Damped Gauss-Newton on the loop's mechanism joints: minimise |tip - anchor|.  Tiny system (m <= ~6 joints), a few
   * iterations per frame; warm-started from the current joint values.  Returns the final separation.
   */
  private refine(loop: Loop, maxIter: number): number {
    const art = this.art;
    const js = loop.joints;
    const m = js.length;
    if (!m) return this.separation(loop);
    const J = new Array<number>(3 * m);
    const H = new Array<number>(m * m);
    const g = new Array<number>(m);
    const delta = new Array<number>(m);
    const { A, B, R, P, N, E } = this;
    let lambda = 1e-6;
    let sep = 0;
    for (let it = 0; it < maxIter; it++) {
      art.worldPoint(loop.tipBody, loop.tipLocal, A);
      art.worldPoint(loop.anchorBody, loop.anchorLocal, B);
      R.copy(A).sub(B);
      sep = R.length();
      if (sep < this.tol * 0.01) break;
      for (let k = 0; k < m; k++) {
        const lj = js[k];
        art.jointWorld(lj.joint, P, N);
        // d(point)/dq: hinge -> n x (point - p); slide -> n   (the point on the joint's own side of the loop)
        if (lj.joint.type === "hinge") { E.copy(lj.side > 0 ? A : B).sub(P); E.crossVectors(N, E); } else E.copy(N);
        E.multiplyScalar(lj.side);
        J[3 * k] = E.x; J[3 * k + 1] = E.y; J[3 * k + 2] = E.z;
      }
      // normal equations (J^T J + lambda I) delta = -J^T r
      for (let i = 0; i < m; i++) {
        g[i] = -(J[3 * i] * R.x + J[3 * i + 1] * R.y + J[3 * i + 2] * R.z);
        for (let k = 0; k < m; k++) H[i * m + k] = J[3 * i] * J[3 * k] + J[3 * i + 1] * J[3 * k + 1] + J[3 * i + 2] * J[3 * k + 2];
      }
      let trace = 0;
      for (let i = 0; i < m; i++) trace += H[i * m + i];
      const damp = lambda * Math.max(trace / m, 1e-12) + 1e-14;
      for (let i = 0; i < m; i++) H[i * m + i] += damp;
      if (!solveSPD(H, g, delta, m)) break;
      // take the step (clamped to joint ranges); back off if it makes things worse
      const before = js.map((lj) => lj.joint.q);
      let step = 1;
      let improved = false;
      for (let tries = 0; tries < 4 && !improved; tries++) {
        for (let k = 0; k < m; k++) js[k].joint.q = clampRange(js[k].joint, before[k] + step * delta[k]);
        art.update();
        const s2 = this.separation(loop);
        if (s2 <= sep) { improved = true; sep = s2; lambda = Math.max(lambda * 0.3, 1e-9); }
        else { step *= 0.5; lambda *= 10; }
      }
      if (!improved) { for (let k = 0; k < m; k++) js[k].joint.q = before[k]; art.update(); break; }
      let dn = 0;
      for (let k = 0; k < m; k++) dn += (js[k].joint.q - before[k]) ** 2;
      if (dn < 1e-20) break;
    }
    return this.separation(loop);
  }
}

/** Solve the small SPD system H x = g (row-major m x m) by Gaussian elimination with partial pivoting. */
function solveSPD(H: number[], g: number[], x: number[], m: number): boolean {
  const a = H.slice(), b = g.slice();
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(a[r * m + c]) > Math.abs(a[piv * m + c])) piv = r;
    if (Math.abs(a[piv * m + c]) < 1e-18) return false;
    if (piv !== c) {
      for (let k = 0; k < m; k++) { const t = a[c * m + k]; a[c * m + k] = a[piv * m + k]; a[piv * m + k] = t; }
      const t = b[c]; b[c] = b[piv]; b[piv] = t;
    }
    for (let r = c + 1; r < m; r++) {
      const f = a[r * m + c] / a[c * m + c];
      if (!f) continue;
      for (let k = c; k < m; k++) a[r * m + k] -= f * a[c * m + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = m - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < m; k++) s -= a[r * m + k] * x[k];
    x[r] = s / a[r * m + r];
  }
  return x.every((v) => Number.isFinite(v));
}
