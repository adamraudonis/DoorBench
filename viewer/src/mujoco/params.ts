// Physics playground: the tunable constants of a door and their 1:1 mapping onto the MJCF the dataset ships.
//
// Every slider is a value of `spec.json["physics"]` (same key path, same unit), so a tuned door is expressed as a
// spec override that `doorbench/export/playground.py` feeds back through the real exporters (MJCF for MuJoCo, USD
// for Isaac Lab).  In the browser the same override is applied to the shipped `door.xml` two ways:
//
//   * live      – fields MuJoCo lets us edit on the compiled mjModel (dof_damping, dof_frictionloss, jnt_stiffness,
//                 qpos_spring, body_mass / body_inertia, opt.gravity) followed by mj_setConst
//   * rebuild   – attributes that need a recompile (joint axis, joint range, tendon coefficients): the XML is
//                 rewritten (`rewriteMjcf`) and the model recompiled with the state carried over
//
// The mapping mirrors doorbench/geometry/{hinged,other,common}.py.  It is applied as *deltas / scale factors on the
// exported joint values*, not as absolute formulas, so every family keeps its builder-specific offsets and the
// identity holds exactly: at the default values `rewriteMjcf(door.xml)` == `door.xml` (asserted by params.test.ts).
//
// Pure: no DOM, no MuJoCo.  Runs in the browser and under `bun test`.
import type { JointJ, ModelJ } from "../types";

export type ParamGroup = "world" | "leaf" | "hinge" | "closer" | "roller" | "latch" | "operator" | "lock";

export interface ParamDef {
  key: string;                 // spec.json["physics"] path, e.g. "closer.spring_preload_Nm" (also the slider id)
  group: ParamGroup;
  label: string;
  unit: string;                // unit of the stored value (spec units)
  min: number;
  max: number;
  step: number;
  default: number;             // the dataset's value
  live: boolean;               // editable on the compiled model (no recompile)
  what: string;                // plain-language meaning
  mjcf: string;                // where it lands in door.xml
  usd: string;                 // where the USD / Isaac Lab export carries it
  display?: { scale: number; unit: string; digits?: number };  // UI conversion (rad -> deg, m -> mm)
  offLabel?: string;           // value 0 means "off" (hold-open)
}

export type ParamValues = Record<string, number>;

export interface JointTarget {
  damping?: number;
  frictionloss?: number;
  stiffness?: number;
  springref?: number;
  axis?: [number, number, number];
  range?: [number, number];
}

/** Direction-dependent / event laws MuJoCo cannot express in the MJCF; DoorEnv applies them in a passive-force
 *  callback, Isaac Lab in `DoorMechanismAction`, the playground before every mj_step (see sim.ts). */
export interface CloserLaw { joint: string; dampingClosing: number; dampingOpening: number; backcheckAngle: number | null; backcheckDamping: number; holdOpen: number | null; springStiffness: number; springref: number }
export interface LawSpec {
  closers: CloserLaw[];
  ratchets: string[];                                   // one-way rotors (turnstiles)
  magnet: { joint: string; forceN: number; armM: number } | null;   // pet-flap magnet detent near closed
  maglock: { welds: string[]; holdingForceN: number } | null;      // maglock weld breaks above the holding force
}

export interface Targets {
  joints: Record<string, JointTarget>;
  bodyMassScale: Record<string, number>;                // body name -> scale on mass and inertia
  gravity: number;                                      // m/s^2 (positive magnitude)
  tendonCoefScale: Record<string, number>;              // tendon name -> scale on its driver coefficient
  law: LawSpec;
}

/** MuJoCo's default gravity; the exporter writes no <option gravity>, so this is what every door.xml runs with. */
export const MJCF_GRAVITY = 9.81;
const HINGE_KIN = (k: string | undefined) => !!k && (k.startsWith("hinge") || k === "rotor");
const num = (x: any, d = 0) => (typeof x === "number" && Number.isFinite(x) ? x : d);
const rangeAround = (v: number, lo: number, hi: number, minSpan: number) => {
  const a = Math.abs(v);
  return [Math.min(lo, v), Math.max(hi, a * 3, minSpan)] as [number, number];
};

interface Ctx { spec: any; model: ModelJ; phys: any; joints: JointJ[]; primary?: JointJ; leafJoints: JointJ[]; boltJoints: JointJ[]; operator?: JointJ; hinged: boolean; sliding: boolean }

function ctx(spec: any, model: ModelJ): Ctx {
  const phys = spec?.physics ?? {};
  const joints = model.bodies.filter((b) => b.joint).map((b) => b.joint!) as JointJ[];
  const primary = joints.find((j) => j.name === model.meta?.primary_joint) ?? joints.find((j) => j.role === "primary");
  const leafJoints = primary ? joints.filter((j) => (j.role === "primary" || j.role === "secondary") && j.type === primary.type) : [];
  const boltJoints = joints.filter((j) => j.role === "latch" && j.stiffness > 0);
  const operator = joints.find((j) => j.name === model.meta?.operator_joint);
  const kin = spec?.kinematics?.type as string | undefined;
  return { spec, model, phys, joints, primary, leafJoints, boltJoints, operator, hinged: !!primary && primary.type === "hinge" && HINGE_KIN(kin), sliding: !!primary && primary.type === "slide" };
}

const DEG = { scale: 180 / Math.PI, unit: "°", digits: 1 };
const MM = { scale: 1000, unit: "mm", digits: 1 };

/** The parameter list of one door: only the groups that exist on it (a sliding door has no closer, a pull has no
 *  operator spring ...).  Defaults come from spec.json["physics"]. */
export function paramDefs(spec: any, model: ModelJ): ParamDef[] {
  const c = ctx(spec, model);
  const { phys } = c;
  const defs: ParamDef[] = [];
  defs.push({ key: "gravity", group: "world", label: "gravity", unit: "m/s²", min: 0, max: 20, step: 0.01, default: MJCF_GRAVITY, live: true,
    what: "Gravitational acceleration (MuJoCo's default 9.81 m/s², the value every door.xml simulates with; the spec lists g = 9.80665 for the derivations). Turn it down to see how much of a door's behaviour is weight: sagging closers, counterbalanced garage doors, hatch lids.",
    mjcf: "<option gravity=\"0 0 -g\">", usd: "/PhysicsScene physics:gravityMagnitude" });
  const mass0 = num(phys.mass?.total_kg, model.bodies.filter((b) => b.semantic === "leaf" && !b.static).reduce((s, b) => s + b.mass, 0));
  if (mass0 > 0) defs.push({ key: "mass.total_kg", group: "leaf", label: "leaf mass", unit: "kg", min: Math.max(0.5, 0.2 * mass0), max: Math.max(3 * mass0, 20), step: 0.1, default: mass0, live: true,
    what: "Mass of the moving leaf (slab + glazing + the hardware riding on it). The inertia tensor scales with it, so a heavier door is also harder to accelerate and slams harder.",
    mjcf: "<inertial mass diaginertia> of every leaf body, scaled by m / m₀", usd: "physics:mass + physics:diagonalInertia on the leaf link" });
  if (c.hinged && phys.hinge) {
    const cou = num(phys.hinge.coulomb_torque_Nm), seal = num(phys.hinge.seal_contribution_Nm), stick = num(phys.hinge.stick_torque_Nm), air = num(phys.hinge.air_damping_Nms_per_rad);
    defs.push({ key: "hinge.coulomb_torque_Nm", group: "hinge", label: "hinge friction (Coulomb, incl. seal)", unit: "N·m", min: 0, max: rangeAround(cou, 0, 60, 60)[1], step: 0.1, default: cou, live: true,
      what: "Constant friction torque the hinges (and the steady part of the seal drag) oppose to any motion. Bearing-load model μ·(m·g·r_thrust + 2·F_h·r_pin)·k_condition + seal.",
      mjcf: `joint ${c.primary!.name} frictionloss = coulomb + stiction / 2`, usd: "physxJointAxis:staticFrictionEffort (N·m)" });
    defs.push({ key: "hinge.seal_contribution_Nm", group: "hinge", label: "of which seal drag", unit: "N·m", min: 0, max: Math.max(6, 3 * seal), step: 0.05, default: seal, live: true,
      what: "Steady drag of the weather / smoke seal. Part of the Coulomb torque above: moving this slider moves the total by the same amount.",
      mjcf: "included in frictionloss", usd: "included in staticFrictionEffort" });
    defs.push({ key: "hinge.stick_torque_Nm", group: "hinge", label: "stiction (stuck door)", unit: "N·m", min: 0, max: Math.max(20, 3 * stick), step: 0.1, default: stick, live: true,
      what: "Extra break-away torque of a swollen / rusty / sagging door. Half of it is exported as extra Coulomb friction (MuJoCo has no separate static term); the compliance check uses all of it.",
      mjcf: `joint ${c.primary!.name} frictionloss += stiction / 2`, usd: "staticFrictionEffort += stiction / 2" });
    defs.push({ key: "hinge.air_damping_Nms_per_rad", group: "hinge", label: "air damping", unit: "N·m·s/rad", min: 0, max: Math.max(5, 4 * air), step: 0.01, default: air, live: true,
      what: "Aerodynamic drag on the leaf linearised at 1 rad/s (½·ρ·Cd·H·W⁴/4). Small for normal doors, noticeable on wide light leaves.",
      mjcf: `joint ${c.primary!.name} damping (symmetric part)`, usd: "drive damping (symmetric part)" });
    if ((c.spec?.kinematics?.type ?? "") === "hinge_vertical") defs.push({ key: "hinge.axis_tilt_deg", group: "hinge", label: "hinge axis tilt (sagging frame)", unit: "°", min: -8, max: 8, step: 0.1, default: num(c.spec?.hinge?.axis_tilt_deg), live: false,
      what: "Tilt of the hinge line out of vertical (a sagging frame or rising-butt hinge). A tilted axis lets gravity swing the door open or closed on its own.",
      mjcf: `joint ${c.primary!.name} axis rotated about the wall's x axis (recompile)`, usd: "joint frame rotation (physics:localRot0/1)" });
  }
  if (c.hinged && phys.closer && phys.closer.kind && phys.closer.kind !== "none") {
    const k = num(phys.closer.spring_stiffness_Nm_per_rad), pre = num(phys.closer.spring_preload_Nm), dc = num(phys.closer.damping_closing), dop = num(phys.closer.damping_opening);
    const gas = phys.closer.kind === "gas_strut";
    defs.push({ key: "closer.spring_preload_Nm", group: "closer", label: gas ? "strut assist at closed" : "closer spring preload", unit: "N·m", min: gas ? Math.min(-400, 3 * pre) : 0, max: gas ? 0 : Math.max(150, 3 * pre), step: 0.5, default: pre, live: true,
      what: gas ? "Lift assist of the gas strut at the closed position (negative = pushes the lid open)." : "Closing torque at the closed position (EN 1154 closing moment × 1.15). The robot must beat this plus the friction to start opening; below the friction the door will not close on its own.",
      mjcf: `joint ${c.primary!.name} springref = −preload / stiffness`, usd: "drive targetPosition = −preload / stiffness" });
    defs.push({ key: "closer.spring_stiffness_Nm_per_rad", group: "closer", label: gas ? "strut rate" : "closer spring rate", unit: "N·m/rad", min: gas ? Math.min(-200, 3 * k) : 0, max: gas ? 0 : Math.max(120, 3 * k), step: 0.1, default: k, live: true,
      what: "Rise of the closing torque per radian of opening: τ(θ) = preload + k·θ (EN 1154 opening moment at 90° → 85 % of the maximum).",
      mjcf: `joint ${c.primary!.name} stiffness`, usd: "drive stiffness (N·m/deg in USD)" });
    defs.push({ key: "closer.damping_opening", group: "closer", label: "damping while opening", unit: "N·m·s/rad", min: 0, max: Math.max(60, 3 * dop), step: 0.1, default: dop, live: true,
      what: "Hydraulic damping of the closer against opening (light, so people can open the door).",
      mjcf: `joint ${c.primary!.name} damping (the MJCF carries the symmetric opening value)`, usd: "drive damping" });
    defs.push({ key: "closer.damping_closing", group: "closer", label: "damping while closing (sweep + latch valves)", unit: "N·m·s/rad", min: 0, max: Math.max(300, 3 * dc), step: 0.5, default: dc, live: true,
      what: "Hydraulic damping on the closing stroke (sweep and latch valves). Not representable natively: DoorEnv / Isaac Lab / this playground add (b_closing − b_opening)·θ̇ while θ̇ < 0.",
      mjcf: "model.json damping_closing → passive-force callback", usd: "doorbench:closer → DoorMechanismAction feed-forward effort" });
    defs.push({ key: "closer.backcheck_angle_rad", group: "closer", label: "backcheck starts at", unit: "rad", display: DEG, min: 0, max: Math.max(1.8, num(phys.closer.backcheck_angle_rad, 1.31)), step: 0.01, default: num(phys.closer.backcheck_angle_rad, 0), live: true, offLabel: "off",
      what: "Opening angle beyond which the backcheck valve adds damping so a flung door does not hit the wall. 0 = no backcheck.",
      mjcf: "model.json backcheck_angle → callback", usd: "doorbench:closer backcheck_angle → DoorMechanismAction" });
    defs.push({ key: "closer.backcheck_damping", group: "closer", label: "backcheck damping", unit: "N·m·s/rad", min: 0, max: Math.max(150, 3 * num(phys.closer.backcheck_damping)), step: 0.5, default: num(phys.closer.backcheck_damping), live: true,
      what: "Extra damping while opening past the backcheck angle.", mjcf: "model.json backcheck_damping → callback", usd: "doorbench:closer backcheck_damping → DoorMechanismAction" });
    if (!gas) defs.push({ key: "closer.hold_open_rad", group: "closer", label: "hold-open at", unit: "rad", display: DEG, min: 0, max: Math.max(1.8, num(c.primary!.range?.[1], 1.6)), step: 0.01, default: num(phys.closer.hold_open_rad, 0), live: true, offLabel: "off",
      what: "Hold-open detent: past this angle the closer stops pulling (the spring torque is cancelled) until the door is pushed back below it. 0 = no hold-open.",
      mjcf: "model.json / spec closer.hold_open_rad → callback", usd: "doorbench:closer hold_open_rad → DoorMechanismAction" });
  }
  if (c.sliding && phys.roller) {
    const f = num(phys.roller.coulomb_force_N), b = num(phys.roller.viscous_damping_N_s_per_m), cb = num(phys.roller.counterbalance_fraction);
    defs.push({ key: "roller.coulomb_force_N", group: "roller", label: "track / roller friction", unit: "N", min: 0, max: Math.max(120, 3 * f), step: 0.1, default: f, live: true,
      what: "Rolling / sliding resistance of the carriages, guides or curtain: μ_roll · m · g · k_condition (dirty tracks and wood-on-wood are much worse than sealed bearings).",
      mjcf: `joint ${c.primary!.name} frictionloss`, usd: "physxJointAxis:staticFrictionEffort (N)" });
    defs.push({ key: "roller.viscous_damping_N_s_per_m", group: "roller", label: "viscous damping", unit: "N·s/m", min: 0, max: Math.max(30, 4 * b), step: 0.1, default: b, live: true,
      what: "Velocity-proportional resistance of the track (soft-close dampers, grease, air).", mjcf: `joint ${c.primary!.name} damping`, usd: "drive damping" });
    const vertical = (c.spec?.kinematics?.type ?? "") === "slide_vertical" || cb > 0;
    if (vertical) defs.push({ key: "roller.counterbalance_fraction", group: "roller", label: "counterbalance (share of weight)", unit: "×", min: 0, max: 1.2, step: 0.01, default: cb, live: true,
      what: "Torsion / extension spring counterbalance of a vertical door as a fraction of its weight (0.95 = a well-adjusted garage door needs only 5 % of its weight to lift). Modelled as a spring whose force declines 30 % over the travel.",
      mjcf: `joint ${c.primary!.name} stiffness = 0.3·cb·m·g / travel, springref = travel / 0.3`, usd: "drive stiffness + targetPosition" });
  }
  if (c.boltJoints.length && phys.latch && num(phys.latch.throw_m) > 0) {
    const pre = num(phys.latch.bolt_spring_preload_N), rate = num(phys.latch.bolt_spring_rate_N_per_m), thr = num(phys.latch.throw_m);
    defs.push({ key: "latch.bolt_spring_preload_N", group: "latch", label: "latch spring preload", unit: "N", min: 0, max: Math.max(30, 3 * pre), step: 0.1, default: pre, live: true,
      what: "Force pushing the latch bolt out at full extension. It must overcome the bolt's own friction to snap back into the strike; too weak and the door does not re-latch when it closes slowly.",
      mjcf: `joint ${c.boltJoints.map((j) => j.name).join(", ")} springref = −preload / rate`, usd: "latch drive targetPosition" });
    defs.push({ key: "latch.bolt_spring_rate_N_per_m", group: "latch", label: "latch spring rate", unit: "N/m", min: 1, max: Math.max(3000, 3 * rate), step: 1, default: rate, live: true,
      what: "Stiffness of the latch spring: how much harder the bolt pushes when depressed by the strike lip.", mjcf: "bolt joint stiffness", usd: "latch drive stiffness" });
    defs.push({ key: "latch.throw_m", group: "latch", label: "latch throw", unit: "m", display: MM, min: 0.004, max: Math.max(0.03, 2 * thr), step: 0.0005, default: thr, live: false,
      what: "How far the bolt projects into the strike (BHMA: 12.7 mm grade 3, 19 mm grade-1 deadlatch). Rewrites the bolt travel and the operator→bolt coupling; the bolt mesh keeps its shipped length.",
      mjcf: "bolt joint range 0..throw; tendon coefficient −throw / (operator travel − dead travel) (recompile)", usd: "prismatic joint upper limit + PhysxMimicJointAPI gearing" });
  }
  if (c.operator && c.operator.stiffness > 0 && phys.latch) {
    const rot = c.operator.type === "hinge";
    const pre = num(phys.latch.operator_spring_preload), rate = num(phys.latch.operator_spring_rate);
    defs.push({ key: "latch.operator_spring_preload", group: "operator", label: "operator return-spring preload", unit: rot ? "N·m" : "N", min: 0, max: Math.max(rot ? 3 : 80, 3 * pre), step: rot ? 0.01 : 0.5, default: pre, live: true,
      what: "Torque (force) the handle's return spring exerts at rest. The robot has to overcome it before the latch starts to move; it also snaps the handle back when released.",
      mjcf: `joint ${c.operator.name} springref = −preload / rate`, usd: "operator drive targetPosition" });
    defs.push({ key: "latch.operator_spring_rate", group: "operator", label: "operator return-spring rate", unit: rot ? "N·m/rad" : "N/m", min: 0.01, max: Math.max(rot ? 10 : 3000, 3 * rate), step: rot ? 0.01 : 1, default: rate, live: true,
      what: "Rise of the return torque per radian (metre) of handle travel.", mjcf: `joint ${c.operator.name} stiffness`, usd: "operator drive stiffness" });
  }
  const welds = (model.meta?.breakable_welds ?? []) as { name: string; holding_force_N: number }[];
  if (welds.length) defs.push({ key: "lock.maglock_holding_force_N", group: "lock", label: "maglock holding force", unit: "N", min: 0, max: Math.max(8000, 2 * num(welds[0].holding_force_N)), step: 10, default: num(welds[0].holding_force_N), live: true,
    what: "Force the electromagnetic lock holds the armature plate with (600 lbf ≈ 2.7 kN, 1200 lbf ≈ 5.3 kN). The weld constraint breaks when the constraint force exceeds it (DoorEnv labels this maglock_forced damage).",
    mjcf: `equality weld ${welds.map((w) => w.name).join(", ")} deactivated above the force (model.json breakable_welds)`, usd: "doorbench:breakable_welds → DoorMechanismAction" });
  const magnet = num(c.spec?.kinematics?.magnet_force_N);
  if (magnet > 0 && c.primary) defs.push({ key: "kinematics.magnet_force_N", group: "lock", label: "flap magnet force", unit: "N", min: 0, max: Math.max(20, 3 * magnet), step: 0.1, default: magnet, live: true,
    what: "Magnetic strip holding a pet flap shut: a detent torque of F × leaf height within ±3° of closed.", mjcf: "spec kinematics.magnet_force_N → callback", usd: "doorbench:magnet → DoorMechanismAction" });
  return defs;
}

export function defaults(defs: ParamDef[]): ParamValues {
  const v: ParamValues = {};
  for (const d of defs) v[d.key] = d.default;
  return v;
}

/** Slider edit with the coupled bookkeeping (seal drag is part of the Coulomb total). */
export function setParam(values: ParamValues, defs: ParamDef[], key: string, x: number): ParamValues {
  const out = { ...values };
  const def = defs.find((d) => d.key === key);
  if (!def) return out;
  x = Math.min(Math.max(x, def.min), def.max);
  if (key === "hinge.seal_contribution_Nm" && "hinge.coulomb_torque_Nm" in out) out["hinge.coulomb_torque_Nm"] = Math.max(0, out["hinge.coulomb_torque_Nm"] + (x - out[key]));
  if (key === "hinge.coulomb_torque_Nm" && "hinge.seal_contribution_Nm" in out) out["hinge.seal_contribution_Nm"] = Math.min(out["hinge.seal_contribution_Nm"], x);
  out[key] = x;
  return out;
}

/** Joint attribute targets + laws for a set of parameter values (deltas / scales on the exported joint values). */
export function computeTargets(spec: any, model: ModelJ, values: ParamValues, defs?: ParamDef[]): Targets {
  const c = ctx(spec, model);
  const D = defs ?? paramDefs(spec, model);
  const d0 = defaults(D);
  const v = (k: string) => (k in values ? values[k] : d0[k]);
  const has = (k: string) => k in d0;
  const joints: Record<string, JointTarget> = {};
  const jt = (name: string): JointTarget => {
    // entries are created lazily on first write so untouched joints never appear in the targets
    if (joints[name]) return joints[name];
    const store: JointTarget = {};
    return new Proxy(store, { set(o, k, val) { (o as any)[k] = val; joints[name] = o; return true; } });
  };
  const bodyMassScale: Record<string, number> = {};
  const tendonCoefScale: Record<string, number> = {};
  const law: LawSpec = { closers: [], ratchets: [], magnet: null, maglock: null };
  const bothWays = !!model.meta?.both_ways || !!c.spec?.kinematics?.both_ways;

  // leaf mass: every leaf body scaled by the same factor (inertia follows)
  if (has("mass.total_kg") && d0["mass.total_kg"] > 0) {
    const s = v("mass.total_kg") / d0["mass.total_kg"];
    for (const b of model.bodies) if (b.semantic === "leaf" && !b.static && b.mass > 0) bodyMassScale[b.name] = s;
  }
  // hinged leaves: friction / damping deltas, closer spring, axis tilt
  for (const j of c.leafJoints) {
    if (c.hinged) {
      let fl = j.frictionloss, damp = j.damping, k = j.stiffness, ref = j.springref;
      if (has("hinge.coulomb_torque_Nm")) fl += (v("hinge.coulomb_torque_Nm") - d0["hinge.coulomb_torque_Nm"]) + 0.5 * (v("hinge.stick_torque_Nm") - d0["hinge.stick_torque_Nm"]);
      if (has("hinge.air_damping_Nms_per_rad")) damp += v("hinge.air_damping_Nms_per_rad") - d0["hinge.air_damping_Nms_per_rad"];
      if (has("closer.damping_opening")) damp += v("closer.damping_opening") - d0["closer.damping_opening"];
      if (has("closer.spring_stiffness_Nm_per_rad")) {
        const r = springTarget(j, d0["closer.spring_stiffness_Nm_per_rad"], v("closer.spring_stiffness_Nm_per_rad"), d0["closer.spring_preload_Nm"], v("closer.spring_preload_Nm"));
        k = r.k;
        ref = bothWays ? 0 : r.ref;
      }
      const t = jt(j.name);
      if (Math.abs(fl - j.frictionloss) > 1e-12) t.frictionloss = Math.max(0, fl);
      if (Math.abs(damp - j.damping) > 1e-12) t.damping = Math.max(0, damp);
      if (Math.abs(k - j.stiffness) > 1e-12) t.stiffness = k;
      if (Math.abs(ref - j.springref) > 1e-12) t.springref = ref;
      if (has("hinge.axis_tilt_deg") && Math.abs(v("hinge.axis_tilt_deg") - d0["hinge.axis_tilt_deg"]) > 1e-9) t.axis = tiltAxis(j.axis, (v("hinge.axis_tilt_deg") - d0["hinge.axis_tilt_deg"]) * Math.PI / 180);
      if (j.damping_closing != null && j.damping_opening != null && j.damping_closing > 0 || has("closer.damping_closing")) {
        law.closers.push({ joint: j.name, dampingClosing: has("closer.damping_closing") ? v("closer.damping_closing") : num(j.damping_closing), dampingOpening: has("closer.damping_opening") ? v("closer.damping_opening") : num(j.damping_opening),
          backcheckAngle: has("closer.backcheck_angle_rad") ? (v("closer.backcheck_angle_rad") > 0 ? v("closer.backcheck_angle_rad") : null) : (j.backcheck_angle ?? null), backcheckDamping: has("closer.backcheck_damping") ? v("closer.backcheck_damping") : num(j.backcheck_damping),
          holdOpen: has("closer.hold_open_rad") && v("closer.hold_open_rad") > 0 ? v("closer.hold_open_rad") : null, springStiffness: k, springref: ref });
      }
    } else if (c.sliding) {
      let fl = j.frictionloss, damp = j.damping, k = j.stiffness, ref = j.springref;
      if (has("roller.coulomb_force_N")) fl += v("roller.coulomb_force_N") - d0["roller.coulomb_force_N"];
      if (has("roller.viscous_damping_N_s_per_m")) damp += v("roller.viscous_damping_N_s_per_m") - d0["roller.viscous_damping_N_s_per_m"];
      if (has("roller.counterbalance_fraction")) {
        const cb0 = d0["roller.counterbalance_fraction"], cb = v("roller.counterbalance_fraction");
        const travel = j.range ? j.range[1] - j.range[0] : 1;
        if (cb0 > 1e-9 && j.stiffness > 0) { k = j.stiffness * (cb / cb0); ref = cb > 1e-9 ? j.springref : 0; }
        else if (cb > 1e-9) { const m = v("mass.total_kg") || d0["mass.total_kg"] || 1; k = 0.3 * cb * m * 9.81 / Math.max(travel, 0.1); ref = travel / 0.3; }
        else { k = 0; ref = 0; }
      }
      const t = jt(j.name);
      if (Math.abs(fl - j.frictionloss) > 1e-12) t.frictionloss = Math.max(0, fl);
      if (Math.abs(damp - j.damping) > 1e-12) t.damping = Math.max(0, damp);
      if (Math.abs(k - j.stiffness) > 1e-12) t.stiffness = k;
      if (Math.abs(ref - j.springref) > 1e-12) t.springref = ref;
    }
    if (j.ratchet_one_way) law.ratchets.push(j.name);
  }
  // latch bolts
  if (has("latch.bolt_spring_rate_N_per_m")) {
    const r0 = d0["latch.bolt_spring_rate_N_per_m"], r = v("latch.bolt_spring_rate_N_per_m"), pre = v("latch.bolt_spring_preload_N"), pre0 = d0["latch.bolt_spring_preload_N"];
    const thr0 = d0["latch.throw_m"], thr = v("latch.throw_m");
    for (const j of c.boltJoints) {
      const { k, ref } = springTarget(j, r0, r, pre0, pre);
      const t = jt(j.name);
      if (Math.abs(k - j.stiffness) > 1e-12) t.stiffness = k;
      if (Math.abs(ref - j.springref) > 1e-12) t.springref = ref;
      if (has("latch.throw_m") && Math.abs(thr - thr0) > 1e-12 && j.range && thr0 > 1e-9) {
        t.range = [j.range[0], j.range[1] * (thr / thr0)];
        for (const tn of model.tendons ?? []) if (tn.sites?.[0]?.[0] === j.name) tendonCoefScale[tn.name] = thr / thr0;
      }
    }
  }
  // operator return spring
  if (c.operator && has("latch.operator_spring_rate")) {
    const j = c.operator;
    const { k, ref } = springTarget(j, d0["latch.operator_spring_rate"], v("latch.operator_spring_rate"), d0["latch.operator_spring_preload"], v("latch.operator_spring_preload"));
    const t = jt(j.name);
    if (Math.abs(k - j.stiffness) > 1e-12) t.stiffness = k;
    if (Math.abs(ref - j.springref) > 1e-12) t.springref = ref;
  }
  // maglock / magnet laws
  const welds = (model.meta?.breakable_welds ?? []) as { name: string; holding_force_N: number }[];
  if (welds.length) law.maglock = { welds: welds.map((w) => w.name), holdingForceN: has("lock.maglock_holding_force_N") ? v("lock.maglock_holding_force_N") : num(welds[0].holding_force_N) };
  if (has("kinematics.magnet_force_N") && c.primary) law.magnet = { joint: c.primary.name, forceN: v("kinematics.magnet_force_N"), armM: num(c.spec?.leaf?.height, 0.3) };
  return { joints, bodyMassScale, gravity: has("gravity") ? v("gravity") : MJCF_GRAVITY, tendonCoefScale, law };
}

/** New (stiffness, springref) of a spring joint when the spec rate goes k0 -> k and the preload p0 -> p.
 *  Standard builders write stiffness = rate, springref = -preload / rate; the joint then gets exactly -p / k.  Builders
 *  with their own scaling (n spring hinges, paddles with a raised preload) keep their ratio: the torque at q = 0
 *  (-k·springref) scales with the preload and the stiffness with the rate. */
function springTarget(j: JointJ, k0: number, k: number, p0: number, p: number): { k: number; ref: number } {
  const kk = Math.abs(k0) > 1e-9 ? j.stiffness * (k / k0) : k;
  if (Math.abs(kk) < 1e-9) return { k: 0, ref: 0 };
  const standard = Math.abs(k0) > 1e-9 && Math.abs(-p0 / k0 - j.springref) < 1e-6;
  if (standard) return { k: kk, ref: -p / kk };
  const tau0 = -j.stiffness * j.springref;                      // torque / force at q = 0 as built
  const tau = Math.abs(p0) > 1e-12 ? tau0 * (p / p0) : tau0;
  return { k: kk, ref: -tau / kk };
}

const escRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Rotate a hinge axis about the world x axis (along the wall) by `a` radians: a sagging frame. */
export function tiltAxis(axis: [number, number, number], a: number): [number, number, number] {
  const [x, y, z] = axis;
  const c = Math.cos(a), s = Math.sin(a);
  const out: [number, number, number] = [x, c * y - s * z, s * y + c * z];
  const n = Math.hypot(...out) || 1;
  return [out[0] / n, out[1] / n, out[2] / n];
}

// ---------------------------------------------------------------------------------------------------------------
// MJCF rewriting (the exporter writes one element per line with double-quoted attributes; we edit attributes in
// place and never touch anything else, so untouched doors round-trip byte-for-byte)
// ---------------------------------------------------------------------------------------------------------------
export function fmtNum(x: number, nd = 6): string {
  if (Math.abs(x) < 1e-12) return "0";
  let s = x.toFixed(nd);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "" || s === "-0" ? "0" : s;
}
const fmtVec = (v: number[], nd = 6) => v.map((x) => fmtNum(x, nd)).join(" ");

function setAttr(tag: string, name: string, value: string): string {
  const re = new RegExp(`\\s${name}="[^"]*"`);
  if (re.test(tag)) return tag.replace(re, ` ${name}="${value}"`);
  return tag.replace(/\s*\/?>$/, (m) => ` ${name}="${value}"${m.trim()}`);
}
const getAttr = (tag: string, name: string): string | null => { const m = new RegExp(`\\s${name}="([^"]*)"`).exec(tag); return m ? m[1] : null; };

/** Apply the targets to door.xml.  Gravity is written into <option>; joint attributes, leaf inertials and tendon
 *  coefficients are edited in place. */
export function rewriteMjcf(xml: string, t: Targets): string {
  let out = xml;
  // option gravity (the exporter never writes gravity; MuJoCo's default is 9.81)
  if (Math.abs(t.gravity - MJCF_GRAVITY) > 1e-9) out = out.replace(/<option\b[^>]*?(\/?)>/, (tag) => setAttr(tag, "gravity", `0 0 ${fmtNum(-t.gravity)}`));
  // joints
  out = out.replace(/<joint\b[^>]*\bname="([^"]+)"[^>]*\/>/g, (tag, name: string) => {
    const tj = t.joints[name];
    if (!tj) return tag;
    let s = tag;
    if (tj.damping !== undefined) s = setAttr(s, "damping", fmtNum(tj.damping));
    if (tj.frictionloss !== undefined) s = setAttr(s, "frictionloss", fmtNum(tj.frictionloss));
    if (tj.stiffness !== undefined) { s = setAttr(s, "stiffness", fmtNum(tj.stiffness)); if (getAttr(s, "springref") === null) s = setAttr(s, "springref", fmtNum(tj.springref ?? 0)); }
    if (tj.springref !== undefined) s = setAttr(s, "springref", fmtNum(tj.springref));
    if (tj.axis) s = setAttr(s, "axis", fmtVec(tj.axis));
    if (tj.range) s = setAttr(s, "range", fmtVec(tj.range));
    return s;
  });
  // leaf inertials: the <inertial .../> line directly after <body name="X" ...>
  for (const [body, scale] of Object.entries(t.bodyMassScale)) {
    if (Math.abs(scale - 1) < 1e-12) continue;
    const re = new RegExp(`(<body\\b[^>]*\\bname="${escRe(body)}"[^>]*>\\s*)(<inertial\\b[^>]*/>)`);
    out = out.replace(re, (_m, head: string, inertial: string) => {
      let s = inertial;
      const m = getAttr(s, "mass"), di = getAttr(s, "diaginertia");
      if (m !== null) s = setAttr(s, "mass", fmtNum(parseFloat(m) * scale));
      if (di !== null) s = setAttr(s, "diaginertia", fmtVec(di.trim().split(/\s+/).map((x) => parseFloat(x) * scale), 9));
      return head + s;
    });
  }
  // tendon driver coefficients (fixed tendon: first <joint> term is the bolt, the others drive it)
  for (const [tendon, scale] of Object.entries(t.tendonCoefScale)) {
    if (Math.abs(scale - 1) < 1e-12) continue;
    const re = new RegExp(`(<fixed\\b[^>]*\\bname="${escRe(tendon)}"[^>]*>)([\\s\\S]*?)(</fixed>)`);
    out = out.replace(re, (_m, open: string, body: string, close: string) => {
      let first = true;
      const b = body.replace(/<joint\b[^>]*\/>/g, (jt) => { if (first) { first = false; return jt; } const c = getAttr(jt, "coef"); return c === null ? jt : setAttr(jt, "coef", fmtNum(parseFloat(c) * scale)); });
      return open + b + close;
    });
  }
  return out;
}

/** True when the targets need a recompile (anything rewriteMjcf handles that mjModel cannot take live). */
export function needsRebuild(t: Targets): boolean {
  for (const j of Object.values(t.joints)) if (j.axis || j.range) return true;
  return Object.values(t.tendonCoefScale).some((s) => Math.abs(s - 1) > 1e-12);
}

/** The recompile-only part of the targets (axis, range, tendon gearing).  The playground rebuilds the model from the
 *  shipped XML with just these edits and applies everything else live, so live values stay relative to one baseline. */
export function rebuildTargets(t: Targets): Targets {
  const joints: Record<string, JointTarget> = {};
  for (const [n, j] of Object.entries(t.joints)) if (j.axis || j.range) joints[n] = { ...(j.axis ? { axis: j.axis } : {}), ...(j.range ? { range: j.range } : {}) };
  return { joints, bodyMassScale: {}, gravity: MJCF_GRAVITY, tendonCoefScale: t.tendonCoefScale, law: t.law };
}

/** Rebuild signature: rebuild only when this changes. */
export const rebuildKey = (t: Targets) => JSON.stringify(rebuildTargets(t).joints) + JSON.stringify(t.tendonCoefScale);

// ---------------------------------------------------------------------------------------------------------------
// spec.json["physics"] override (what "Copy as spec override" emits; consumed by doorbench/export/playground.py)
// ---------------------------------------------------------------------------------------------------------------
export function specOverride(doorId: string, defs: ParamDef[], values: ParamValues): Record<string, any> {
  const phys: Record<string, any> = {};
  const top: Record<string, any> = {};
  for (const d of defs) {
    const val = values[d.key];
    if (val === undefined || Math.abs(val - d.default) < 1e-12) continue;
    const path = d.key.split(".");
    const root = path[0] === "kinematics" || path[0] === "hinge" && path[1] === "axis_tilt_deg" ? top : phys;
    let node = root;
    for (let i = 0; i < path.length - 1; i++) node = node[path[i]] ??= {};
    node[path[path.length - 1]] = d.key === "closer.hold_open_rad" && val === 0 ? null : round(val);
  }
  const out: Record<string, any> = { id: doorId, ...top };
  if (Object.keys(phys).length) out.physics = phys;
  return out;
}
const round = (x: number) => Math.round(x * 1e6) / 1e6;

export function formatValue(d: ParamDef, x: number): string {
  if (d.offLabel && x === 0) return d.offLabel;
  const disp = d.display;
  const val = disp ? x * disp.scale : x;
  const digits = disp?.digits ?? (Math.abs(val) >= 100 ? 0 : Math.abs(val) >= 10 ? 1 : Math.abs(val) >= 1 ? 2 : 3);
  return `${val.toFixed(digits)} ${disp?.unit ?? d.unit}`;
}
