// A door running in MuJoCo (WebAssembly): stepping, held torques, mouse drag forces, the direction-dependent laws
// the MJCF cannot carry (closer sweep/latch/backcheck damping, hold-open, ratchets, pet-flap magnet, maglock
// breakaway), live parameter edits, recompiles with the state carried over, a recorder for the plots and the
// measurements shown next to the dataset's QA metrics.
//
// Everything here is DOM-free so it also runs under `bun test` (see wasm.test.ts).
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";
import type { ModelJ } from "../types";
import type { LawSpec, Targets } from "./params";
import { stageDoor, type AssetReader } from "./loader";

export interface SimJoint {
  name: string;
  id: number;
  qposadr: number;
  dofadr: number;
  type: "hinge" | "slide";
  range: [number, number] | null;
  role: string;
  label: string;
  interactive: boolean;
  body: number;
}

export interface Sample { t: number; q: number; v: number; tauApplied: number; tauPassive: number; tauConstraint: number; contactF: number }

/** Fixed-size ring buffer of samples for the plots. */
export class Recorder {
  readonly cap: number;
  t: Float64Array; q: Float32Array; v: Float32Array; a: Float32Array; p: Float32Array; c: Float32Array; f: Float32Array;
  n = 0; head = 0;
  constructor(cap = 6000) {
    this.cap = cap;
    this.t = new Float64Array(cap); this.q = new Float32Array(cap); this.v = new Float32Array(cap); this.a = new Float32Array(cap);
    this.p = new Float32Array(cap); this.c = new Float32Array(cap); this.f = new Float32Array(cap);
  }
  push(s: Sample) {
    const i = this.head;
    this.t[i] = s.t; this.q[i] = s.q; this.v[i] = s.v; this.a[i] = s.tauApplied; this.p[i] = s.tauPassive; this.c[i] = s.tauConstraint; this.f[i] = s.contactF;
    this.head = (i + 1) % this.cap;
    if (this.n < this.cap) this.n++;
  }
  clear() { this.n = 0; this.head = 0; }
  /** Oldest-first index list. */
  indices(): number[] {
    const out: number[] = [];
    const start = (this.head - this.n + this.cap) % this.cap;
    for (let k = 0; k < this.n; k++) out.push((start + k) % this.cap);
    return out;
  }
}

export interface Drag { body: number; local: [number, number, number]; target: [number, number, number]; kp: number; maxF: number; point: [number, number, number]; force: [number, number, number] }

/** A scripted experiment: called before each step; returns true when finished. */
export type Script = { name: string; onStep: (sim: DoorSim, k: number) => boolean; k: number };

export interface Measurements {
  peakSpeed: number;                 // rad/s or m/s since reset
  peakContact: number;               // N
  closingTime: number | null;        // release test: s from release to |q| < 2 deg
  finalAngle: number | null;         // release test: q after 12 s
  relatched: boolean | null;         // release test: latch bolts extended at the end
  pushDisplacement: number | null;   // push test: q after 1 s of QA push
  flingPeak: number | null;          // fling test: peak q
  flingHitStop: boolean | null;
  actuateOpened: number | null;      // actuate test: q at the end
  lastTest: string | null;
  testRunning: boolean;
}

const WARN_NAMES = ["INERTIA", "CONTACTFULL", "CNSTRFULL", "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL"];

export class DoorSim {
  readonly mj: MainModule;
  readonly id: string;
  readonly modelJ: ModelJ;
  readonly spec: any;
  readonly reader: AssetReader;
  readonly xml0: string;             // the shipped door.xml
  xml: string;                       // the compiled XML (xml0 with the recompile-only edits)
  model!: MjModel;
  data!: MjData;
  joints: SimJoint[] = [];
  byName = new Map<string, SimJoint>();
  primary: SimJoint | null = null;
  operator: SimJoint | null = null;
  bolts: SimJoint[] = [];
  leafBodies: number[] = [];
  law: LawSpec = { closers: [], ratchets: [], magnet: null, maglock: null };
  held = new Map<number, number>();  // dof -> generalized force held by a button
  drag: Drag | null = null;
  script: Script | null = null;
  recorder = new Recorder();
  recordEvery = 4;                   // 500 Hz physics -> 125 Hz samples
  lastContactF = 0;
  measures: Measurements = { peakSpeed: 0, peakContact: 0, closingTime: null, finalAngle: null, relatched: null, pushDisplacement: null, flingPeak: null, flingHitStop: null, actuateOpened: null, lastTest: null, testRunning: false };
  warnings: string[] = [];
  events: string[] = [];
  maglockBroken = false;
  stepsDone = 0;
  private base: { mass: Float64Array; inertia: Float64Array } | null = null;   // compile-time inertials (live mass edits are relative to these)
  private eqIds = new Map<string, number>();
  private stateBuf: any = null;

  private constructor(mj: MainModule, id: string, modelJ: ModelJ, spec: any, xml: string, reader: AssetReader) {
    this.mj = mj; this.id = id; this.modelJ = modelJ; this.spec = spec; this.xml0 = xml; this.xml = xml; this.reader = reader;
  }

  static async create(mj: MainModule, id: string, modelJ: ModelJ, spec: any, reader: AssetReader, xml?: string): Promise<DoorSim> {
    const text = xml ?? new TextDecoder().decode(await reader(`doors/${id}/door.xml`));
    const sim = new DoorSim(mj, id, modelJ, spec, text, reader);
    await sim.compile(text);
    return sim;
  }

  /** Compile `xml` (stage + mj_loadXML).  Throws a plain Error with MuJoCo's message. */
  private async compile(xml: string) {
    const path = await stageDoor(this.mj, this.id, xml, this.reader);
    let model: MjModel;
    try { model = this.mj.MjModel.from_xml_path(path); } catch (e) { throw new Error(typeof e === "string" ? e : (e as Error)?.message ?? String(e)); }
    const data = new this.mj.MjData(model);
    this.model = model; this.data = data; this.xml = xml;
    this.index();
    this.base = { mass: Float64Array.from(model.body_mass as ArrayLike<number>), inertia: Float64Array.from(model.body_inertia as ArrayLike<number>) };
    this.mj.mj_forward(model, data);
  }

  private index() {
    const { mj, model } = this;
    const J = mj.mjtObj.mjOBJ_JOINT.value, B = mj.mjtObj.mjOBJ_BODY.value, E = mj.mjtObj.mjOBJ_EQUALITY.value;
    this.joints = []; this.byName.clear(); this.bolts = []; this.leafBodies = []; this.eqIds.clear();
    const jinfo = new Map<string, { role: string; label: string; interactive: boolean; range: [number, number] | null }>();
    for (const b of this.modelJ.bodies) if (b.joint) jinfo.set(b.joint.name, { role: b.joint.role, label: b.joint.label, interactive: b.joint.robot_interactive, range: b.joint.range });
    const types = model.jnt_type as ArrayLike<number>, qadr = model.jnt_qposadr as ArrayLike<number>, dadr = model.jnt_dofadr as ArrayLike<number>, rng = model.jnt_range as ArrayLike<number>, jb = model.jnt_bodyid as ArrayLike<number>;
    for (let i = 0; i < model.njnt; i++) {
      const name = mj.mj_id2name(model, J, i);
      const info = jinfo.get(name);
      const type = types[i] === 3 ? "hinge" : "slide";     // mjJNT_HINGE = 3, mjJNT_SLIDE = 2
      const r: [number, number] | null = info?.range ?? ((rng[2 * i] !== 0 || rng[2 * i + 1] !== 0) ? [rng[2 * i], rng[2 * i + 1]] : null);
      const sj: SimJoint = { name, id: i, qposadr: qadr[i], dofadr: dadr[i], type, range: r, role: info?.role ?? "", label: info?.label ?? name, interactive: info?.interactive ?? true, body: jb[i] };
      this.joints.push(sj); this.byName.set(name, sj);
      if (sj.role === "latch") this.bolts.push(sj);
    }
    this.primary = this.byName.get(this.modelJ.meta?.primary_joint) ?? this.joints.find((j) => j.role === "primary") ?? null;
    this.operator = this.modelJ.meta?.operator_joint ? this.byName.get(this.modelJ.meta.operator_joint) ?? null : null;
    for (const b of this.modelJ.bodies) if (b.semantic === "leaf" && !b.static) { const id = mj.mj_name2id(model, B, b.name); if (id >= 0) this.leafBodies.push(id); }
    for (let i = 0; i < model.neq; i++) this.eqIds.set(mj.mj_id2name(model, E, i), i);
  }

  get timestep(): number { return this.model.opt.timestep; }
  set timestep(dt: number) { this.model.opt.timestep = dt; }
  get time(): number { return this.data.time; }
  q(j: SimJoint | null): number { return j ? (this.data.qpos as ArrayLike<number>)[j.qposadr] : 0; }
  v(j: SimJoint | null): number { return j ? (this.data.qvel as ArrayLike<number>)[j.dofadr] : 0; }

  /** Recompile from a rewritten XML, carrying the state over when the structure is unchanged. */
  async rebuild(xml: string) {
    const old = { model: this.model, data: this.data };
    const qpos = Float64Array.from(old.data.qpos as ArrayLike<number>), qvel = Float64Array.from(old.data.qvel as ArrayLike<number>), t = old.data.time;
    const held = new Map(this.held);
    await this.compile(xml);
    if (this.model.nq === qpos.length) { (this.data.qpos as Float64Array).set(qpos); (this.data.qvel as Float64Array).set(qvel); this.data.time = t; this.mj.mj_forward(this.model, this.data); }
    this.held = held;
    this.maglockBroken = false;
    old.data.delete(); old.model.delete();
  }

  /** Apply the live part of the targets to the compiled model (absolute values relative to the shipped baseline). */
  applyLive(t: Targets) {
    const { model } = this;
    const dd = model.dof_damping as Float64Array, fl = model.dof_frictionloss as Float64Array, ks = model.jnt_stiffness as Float64Array, qs = model.qpos_spring as Float64Array;
    for (const j of this.joints) {
      const src = this.modelJ.bodies.find((b) => b.joint?.name === j.name)?.joint;
      if (!src) continue;
      const tj = t.joints[j.name] ?? {};
      dd[j.dofadr] = tj.damping ?? src.damping;
      fl[j.dofadr] = tj.frictionloss ?? src.frictionloss;
      ks[j.id] = tj.stiffness ?? src.stiffness;
      qs[j.qposadr] = tj.springref ?? src.springref;
    }
    if (this.base) {
      const bm = model.body_mass as Float64Array, bi = model.body_inertia as Float64Array;
      const B = this.mj.mjtObj.mjOBJ_BODY.value;
      for (let b = 0; b < model.nbody; b++) {
        const name = this.mj.mj_id2name(model, B, b);
        const s = t.bodyMassScale[name] ?? 1;
        bm[b] = this.base.mass[b] * s;
        for (let k = 0; k < 3; k++) bi[3 * b + k] = this.base.inertia[3 * b + k] * s;
      }
    }
    (model.opt.gravity as Float64Array)[2] = -t.gravity;
    // mj_setConst recomputes the qpos0-dependent constants and uses mjData as scratch: keep the state across it
    const qpos = Float64Array.from(this.data.qpos as ArrayLike<number>), qvel = Float64Array.from(this.data.qvel as ArrayLike<number>), time = this.data.time;
    this.mj.mj_setConst(model, this.data);
    (this.data.qpos as Float64Array).set(qpos); (this.data.qvel as Float64Array).set(qvel); this.data.time = time;
    this.mj.mj_forward(model, this.data);
    this.law = t.law;
  }

  reset() {
    this.mj.mj_resetData(this.model, this.data);
    this.held.clear(); this.drag = null; this.script = null;
    this.recorder.clear();
    this.measures = { ...this.measures, peakSpeed: 0, peakContact: 0, testRunning: false };
    this.warnings = []; this.events = [];
    this.maglockBroken = false;
    if (this.law.maglock) this.setWeldsActive(true);
    this.mj.mj_forward(this.model, this.data);
    this.stepsDone = 0;
  }

  setJoint(j: SimJoint, q: number, v = 0) {
    (this.data.qpos as Float64Array)[j.qposadr] = q;
    (this.data.qvel as Float64Array)[j.dofadr] = v;
    this.mj.mj_forward(this.model, this.data);
  }

  hold(j: SimJoint, tau: number) { if (tau === 0) this.held.delete(j.dofadr); else this.held.set(j.dofadr, tau); }
  release(j: SimJoint) { this.held.delete(j.dofadr); }
  releaseAll() { this.held.clear(); }

  /** Generalized force on the primary dof this step from a spring pulling the grabbed point to the drag target. */
  private applyDrag(qfrc: Float64Array) {
    const d = this.drag;
    if (!d) return;
    const xpos = this.data.xpos as ArrayLike<number>, xmat = this.data.xmat as ArrayLike<number>;
    const b = d.body, o = 9 * b;
    const p: [number, number, number] = [
      xpos[3 * b] + xmat[o] * d.local[0] + xmat[o + 1] * d.local[1] + xmat[o + 2] * d.local[2],
      xpos[3 * b + 1] + xmat[o + 3] * d.local[0] + xmat[o + 4] * d.local[1] + xmat[o + 5] * d.local[2],
      xpos[3 * b + 2] + xmat[o + 6] * d.local[0] + xmat[o + 7] * d.local[1] + xmat[o + 8] * d.local[2],
    ];
    let F: [number, number, number] = [d.kp * (d.target[0] - p[0]), d.kp * (d.target[1] - p[1]), d.kp * (d.target[2] - p[2])];
    const n = Math.hypot(...F);
    if (n > d.maxF) F = [F[0] * d.maxF / n, F[1] * d.maxF / n, F[2] * d.maxF / n];
    d.point = p; d.force = F;
    this.mj.mj_applyFT(this.model, this.data, F, [0, 0, 0], p, b, qfrc);
  }

  /** The direction-dependent laws (DoorEnv._install_passive_callback / Isaac Lab DoorMechanismAction). */
  private applyLaw(qfrc: Float64Array) {
    const qpos = this.data.qpos as ArrayLike<number>, qvel = this.data.qvel as ArrayLike<number>, dd = this.model.dof_damping as ArrayLike<number>;
    for (const cl of this.law.closers) {
      const j = this.byName.get(cl.joint);
      if (!j) continue;
      const q = qpos[j.qposadr], v = qvel[j.dofadr];
      let bTarget = v < 0 ? cl.dampingClosing : cl.dampingOpening;
      if (cl.backcheckAngle != null && v > 0 && q > cl.backcheckAngle) bTarget += cl.backcheckDamping;
      qfrc[j.dofadr] += -(bTarget - dd[j.dofadr]) * v;
      if (cl.holdOpen != null && q > cl.holdOpen - 0.02) qfrc[j.dofadr] += cl.springStiffness * (q - cl.springref);   // cancel the spring: the arm rests on the hold-open cam
    }
    for (const name of this.law.ratchets) {
      const j = this.byName.get(name);
      if (j && qvel[j.dofadr] < 0) qfrc[j.dofadr] += -200.0 * qvel[j.dofadr];
    }
    if (this.law.magnet) {
      const j = this.byName.get(this.law.magnet.joint);
      if (j) { const q = qpos[j.qposadr], lim = 3 * Math.PI / 180; if (Math.abs(q) < lim) qfrc[j.dofadr] += -Math.sign(q) * this.law.magnet.forceN * this.law.magnet.armM * (1 - Math.abs(q) / lim); }
    }
  }

  /** Maglock: total constraint force on the weld(s) vs the holding force; break = deactivate the equality. */
  private checkMaglock() {
    const ml = this.law.maglock;
    if (!ml || this.maglockBroken) return;
    const efcType = this.data.efc_type as ArrayLike<number>, efcId = this.data.efc_id as ArrayLike<number>, efcF = this.data.efc_force as ArrayLike<number>;
    let sum = 0;
    for (let i = 0; i < this.data.nefc; i++) {
      if (efcType[i] !== 0) continue;   // mjCNSTR_EQUALITY
      for (const w of ml.welds) if (this.eqIds.get(w) === efcId[i]) { sum += efcF[i] * efcF[i]; }
    }
    const f = Math.sqrt(sum);
    if (f > ml.holdingForceN) { this.setWeldsActive(false); this.maglockBroken = true; this.events.push(`t=${this.time.toFixed(2)} s: maglock forced (${f.toFixed(0)} N > ${ml.holdingForceN.toFixed(0)} N)`); }
  }

  private setWeldsActive(active: boolean) {
    const ml = this.law.maglock;
    if (!ml) return;
    const spec = this.mj.mjtState.mjSTATE_EQ_ACTIVE.value;
    const n = this.mj.mj_stateSize(this.model, spec);
    if (n <= 0) return;
    if (!this.stateBuf || this.stateBuf.GetElementCount() !== n) { this.stateBuf?.delete(); this.stateBuf = new this.mj.DoubleBuffer(n); }
    this.mj.mj_getState(this.model, this.data, this.stateBuf, spec);
    const view = this.stateBuf.GetView() as Float64Array;
    for (const w of ml.welds) { const i = this.eqIds.get(w); if (i !== undefined && i < n) view[i] = active ? 1 : 0; }
    this.mj.mj_setState(this.model, this.data, Array.from(view), spec);
  }

  /** Sum of contact normal forces on the leaf bodies (computed by the UI once per frame; expensive: copies contacts). */
  contactForceOnLeaf(): number {
    const { mj, model, data } = this;
    const contacts = data.contact;
    const gb = model.geom_bodyid as ArrayLike<number>;
    const buf = new mj.DoubleBuffer(6);
    let total = 0;
    try {
      const n = contacts.size();
      for (let i = 0; i < n; i++) {
        const c = contacts.get(i)!;
        const b1 = gb[c.geom1], b2 = gb[c.geom2];
        if (this.leafBodies.includes(b1) || this.leafBodies.includes(b2)) { mj.mj_contactForce(model, data, i, buf); total += Math.abs((buf.GetView() as Float64Array)[0]); }
        // elements returned by MjContactVec.get are references owned by the vector: only the vector is deleted
      }
    } finally { buf.delete(); contacts.delete(); }
    this.lastContactF = total;
    if (total > this.measures.peakContact) this.measures.peakContact = total;
    return total;
  }

  /** Advance `n` physics steps. */
  step(n: number) {
    const { mj, model, data } = this;
    for (let s = 0; s < n; s++) {
      if (this.script) { const done = this.script.onStep(this, this.script.k++); if (done) { this.script = null; this.measures.testRunning = false; } }
      const qfrc = data.qfrc_applied as Float64Array;
      qfrc.fill(0);
      for (const [dof, tau] of this.held) qfrc[dof] += tau;
      this.applyLaw(qfrc);
      this.applyDrag(qfrc);
      mj.mj_step(model, data);
      this.stepsDone++;
      if (this.law.maglock) this.checkMaglock();
      if (this.primary) {
        const v = Math.abs((data.qvel as ArrayLike<number>)[this.primary.dofadr]);
        if (v > this.measures.peakSpeed) this.measures.peakSpeed = v;
        if (this.stepsDone % this.recordEvery === 0) {
          const d = this.primary.dofadr;
          this.recorder.push({ t: data.time, q: (data.qpos as ArrayLike<number>)[this.primary.qposadr], v: (data.qvel as ArrayLike<number>)[d], tauApplied: qfrc[d], tauPassive: (data.qfrc_passive as ArrayLike<number>)[d], tauConstraint: (data.qfrc_constraint as ArrayLike<number>)[d], contactF: this.lastContactF });
        }
      }
      if (this.stepsDone % 250 === 0) this.pollWarnings();
    }
  }

  private pollWarnings() {
    // `data.warning` is a reference into mjData (unlike `data.contact`, which is a copy): nothing to delete
    const w = this.data.warning;
    const out: string[] = [];
    for (let i = 0; i < w.size(); i++) { const s = w.get(i)!; if (s.number > 0) out.push(`${WARN_NAMES[i] ?? i} ×${s.number}`); }
    if (out.join() !== this.warnings.join()) this.warnings = out;
  }

  boltsExtended(): boolean | null {
    if (!this.bolts.length) return null;
    return this.bolts.every((b) => this.q(b) < 0.006);
  }

  // ------------------------------------------------------------------------------------------------------------
  // scripted experiments (mirrors doorbench/qa.py so the numbers are comparable with qa.json)
  // ------------------------------------------------------------------------------------------------------------
  /** Closer test: release from 60 deg (or 80 % of the range) with the latch extended; closing time to < 2 deg,
   *  final angle after 12 s, re-latched?. */
  releaseTest() {
    const p = this.primary;
    if (!p) return;
    this.reset();
    const isHinge = p.type === "hinge";
    const maxq = p.range ? p.range[1] : (isHinge ? Math.PI / 2 : 1);
    const q0 = isHinge ? Math.min(60 * Math.PI / 180, 0.8 * maxq) : 0.8 * maxq;
    (this.data.qpos as Float64Array)[p.qposadr] = q0;
    for (const b of this.bolts) (this.data.qpos as Float64Array)[b.qposadr] = 0.0;
    this.mj.mj_forward(this.model, this.data);
    const t0 = this.time, thr = isHinge ? 2 * Math.PI / 180 : 0.02, horizon = 12.0;
    this.measures = { ...this.measures, closingTime: null, finalAngle: null, relatched: null, lastTest: "release", testRunning: true };
    this.script = { name: "release", k: 0, onStep: (sim) => {
      const q = sim.q(p);
      if (sim.measures.closingTime === null && q < thr) sim.measures.closingTime = sim.time - t0;
      if (sim.time - t0 >= horizon) { sim.measures.finalAngle = q; sim.measures.relatched = sim.boltsExtended(); return true; }
      return false;
    } };
  }

  /** Backcheck test: slam the door open from closed at the slam velocity; peak angle, did it hit the stop. */
  flingTest(speed?: number) {
    const p = this.primary;
    if (!p) return;
    this.reset();
    const isHinge = p.type === "hinge";
    const v0 = speed ?? (isHinge ? 4.0 : 1.5);
    for (const b of this.bolts) (this.data.qpos as Float64Array)[b.qposadr] = b.range ? b.range[1] : 0.02;   // latch retracted: the door is free to fly
    (this.data.qvel as Float64Array)[p.dofadr] = v0;
    this.mj.mj_forward(this.model, this.data);
    const t0 = this.time, stop = p.range ? p.range[1] : Infinity;
    this.measures = { ...this.measures, flingPeak: 0, flingHitStop: false, lastTest: "fling", testRunning: true };
    this.script = { name: "fling", k: 0, onStep: (sim) => {
      const q = sim.q(p);
      if (q > (sim.measures.flingPeak ?? 0)) sim.measures.flingPeak = q;
      if (q > stop - (isHinge ? 0.5 * Math.PI / 180 : 0.005)) sim.measures.flingHitStop = true;
      return sim.time - t0 > 4.0 || (sim.time - t0 > 0.3 && sim.v(p) < 0);
    } };
  }

  /** QA hold / free_opens test: the QA push (qa.json metrics.qa_push) on the primary dof for 500 steps from rest. */
  pushTest(push: number) {
    const p = this.primary;
    if (!p) return;
    this.reset();
    this.measures = { ...this.measures, pushDisplacement: null, lastTest: "push", testRunning: true };
    this.script = { name: "push", k: 0, onStep: (sim, k) => {
      if (k < 500) { sim.held.set(p.dofadr, push); return false; }
      sim.held.delete(p.dofadr);
      sim.measures.pushDisplacement = sim.q(p);
      return true;
    } };
  }

  /** QA actuate test: work the operator (and auxiliary bolts) then push; angle reached after 3200 steps. */
  actuateTest(push: number) {
    const p = this.primary, o = this.operator;
    if (!p) return;
    this.reset();
    const eff = o ? (o.type === "hinge" ? (o.name.startsWith("dog_") ? 14 : o.name.includes("wheel") ? 10 : o.name.includes("exit_device") ? 8 : 4) : 120) : 0;
    const aux = this.joints.filter((j) => ["leaf_aux_bolt_slide", "slide_latch_slide", "leaf_slide_bolt_slide", "leaf_pin_slide", "leaf_thumb_hinge", "hatch_bolt_slide", "join_bolt_slide", "garage_slide_lock_slide", "leaf_hook_thumbturn_hinge", "leaf_a_hook_thumbturn_hinge"].includes(j.name));
    const tt = this.byName.get("leaf_deadbolt_thumbturn_hinge") ?? this.byName.get("leaf_a_deadbolt_thumbturn_hinge");
    this.measures = { ...this.measures, actuateOpened: null, lastTest: "actuate", testRunning: true };
    this.script = { name: "actuate", k: 0, onStep: (sim, k) => {
      sim.held.clear();
      if (tt && k < 600) sim.held.set(tt.dofadr, 2.0);
      for (const a of aux) sim.held.set(a.dofadr, a.type === "hinge" ? 3.0 : 60.0);
      if (o && k >= 300) sim.held.set(o.dofadr, eff);
      if (k >= 600 && (p.type !== "hinge" || sim.q(p) < 50 * Math.PI / 180)) sim.held.set(p.dofadr, push);
      if (k >= 3200) { sim.held.clear(); sim.measures.actuateOpened = sim.q(p); return true; }
      return false;
    } };
  }

  dispose() {
    this.stateBuf?.delete(); this.stateBuf = null;
    try { this.data.delete(); } catch { /* already deleted */ }
    try { this.model.delete(); } catch { /* already deleted */ }
  }
}
