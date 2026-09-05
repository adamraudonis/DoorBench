export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // w x y z

export interface GeomJ {
  name: string;
  type: "box" | "cylinder" | "capsule" | "sphere" | "mesh";
  size: number[];
  pos: Vec3;
  quat: Quat;
  material: string;
  collision: boolean;
  visual: boolean;
  semantic: string;
  part_label: string;
  mesh_name?: string;
  tiers: string[];
}

export interface JointJ {
  name: string;
  type: "hinge" | "slide";
  axis: Vec3;
  pos: Vec3;
  range: [number, number] | null;
  damping: number;
  frictionloss: number;
  stiffness: number;
  springref: number;
  armature: number;
  role: string;
  label: string;
  robot_interactive: boolean;
  initial: number;
  modeled_at?: number;
  notes: string;
  damping_closing?: number | null;
  damping_opening?: number | null;
  ratchet_one_way?: boolean;
}

export interface SiteJ {
  name: string;
  pos: Vec3;
  quat: Quat;
  size: number;
  role: string;
}

export interface BodyJ {
  name: string;
  parent: string | null;
  pos: Vec3;
  quat: Quat;
  joint: JointJ | null;
  geoms: GeomJ[];
  sites: SiteJ[];
  tiers: string[];
  semantic: string;
  label: string;
  static: boolean;
  mass: number;
  com: Vec3;
  inertia: number[][];
}

export interface MaterialJ {
  name: string;
  rgba: [number, number, number, number];
  roughness: number;
  metallic: number;
  texture: string | null;
  transparent: boolean;
  emissive: Vec3;
}

export interface EqualityJ {
  kind: "joint" | "connect" | "weld";
  name: string;
  a: string;
  b: string | null;
  polycoeff: number[];
  anchor?: Vec3;          // connect: anchor point in body `a`'s frame (MJCF semantics; body b's point = same world point at the rest pose)
  label: string;
  active: boolean;
}

/** Closed kinematic loops declared by the generator (model.json["linkages"]).  Optional: when absent the viewer derives the
 *  same information from the bodies + `connect` equalities (kinematics.ts). */
export interface TwoBarLinkageJ {
  name: string;
  type: "two_bar";
  pinion: { body: string; joint: string; parent: string };
  elbow: { body: string; joint: string };
  anchor: { body: string; pos: Vec3 };
  equality: string;
  axis: Vec3;
  L1: number;
  L2: number;
  elbow_sign: 1 | -1;
}
export interface TelescopingLinkageJ {
  name: string;
  type: "telescoping";
  base: { body: string; joint: string; parent: string; pos: Vec3 };
  slide: { body: string; joint: string; axis_local: Vec3; offset: number };
  anchor: { body: string; pos: Vec3 };
  equality: string;
}
export type LinkageJ = TwoBarLinkageJ | TelescopingLinkageJ;

export interface TendonJ {
  name: string;
  sites: [string, number][];
  range: [number, number];
  label: string;
}

export interface ModelJ {
  name: string;
  tier: string;
  bodies: BodyJ[];
  materials: Record<string, MaterialJ>;
  equalities: EqualityJ[];
  tendons: TendonJ[];
  linkages?: LinkageJ[];
  meta: Record<string, any>;
}

export interface HumanJ {
  radius_m: number;
  height_m: number;
  speed_m_s: number;
  start_t_s: number;
  direction: "same_as_robot" | "opposite_to_robot";
  path: [number, number, number][];   // (t, x, y)
  waits_at_closed_door: boolean;
  note?: string;
}

export interface ScenarioJ {
  name: string;
  suite?: "core" | "human";        // core = default benchmark (no person); human = advanced opt-in suite
  requires_human?: boolean;
  description: string;
  initial_state: { door: "closed" | "open"; lock_engaged: boolean; latched: boolean };
  start: { center: Vec3; radius: number; yaw: number; yaw_range: [number, number]; randomize: { position: string; radius: number; yaw_jitter_rad: number; seed_base: number; formula: string } };
  approach_point: Vec3;
  handle_targets: string[];
  pass_plane: { center: Vec3; normal: Vec3; width: number; height: number; traverse_direction: Vec3 };
  goal: { center: Vec3; radius: number } | null;
  human: HumanJ | null;
  thresholds: { open_rad: number | null; open_m: number | null; clear_rad: number | null; clear_m: number | null };
  rewards: Record<string, number>;
  success: string[];
  time_budget_s: number;
  expected_transit_s: number;
  expected_transit_terms: Record<string, number>;
}

export interface BenchmarkJ {
  schema_version: string;
  robot: Record<string, number | string>;
  human: Record<string, number>;
  primary_scenario: string;
  scenarios: ScenarioJ[];
  reward_values: Record<string, number>;
  event_descriptions: Record<string, string>;
}

export interface BenchmarkSummary {
  scenarios: string[];
  primary: string;
  core?: string[];      // the door's core-suite scenarios (no simulated person; the default benchmark)
  human?: string[];     // the door's human-suite scenarios (advanced, opt-in)
  time_budget_s: number;
  expected_transit_s: number;
  has_human: boolean;
}

/** Isaac parity badge (scripts/merge_isaac_results.py): ok = grade A or B in every tested USD kind, fail = a status
 *  disagreement or not comparable, untested = not yet run in Isaac Sim. */
export type IsaacParityStatus = "ok" | "fail" | "untested";

export interface ManifestDoor {
  benchmark_eligibility?: { eligible: boolean; collection: string; reason_code: string | null; reason: string | null };
  reference_motion_available?: boolean;
  reference_motion_unavailable_reason?: string;
  id: string;
  index: number;
  family: string;
  context: string;
  use_case: string;
  task: string;
  difficulty: number;
  mass_kg: number;
  leaf: { width: number; height: number; thickness: number; slab: string; panel_style: string };
  operator: string;
  latch: string;
  lock: string;
  lock_engaged: boolean;
  robot_side_release: boolean;
  closer: string;
  hinge: string;
  condition: string;
  swing: string;
  hinge_side: string;
  extras: string[];
  tags: string[];
  n_bodies: number;
  n_joints: number;
  signed_off: boolean;
  qa_failed: string[];
  thumbs: string[];
  files: Record<string, any>;
  physics_summary: Record<string, number | boolean | null>;
  benchmark?: BenchmarkSummary | null;
  isaac_parity?: IsaacParityStatus;          // MuJoCo vs Isaac Sim / PhysX parity gate (docs/ISAAC_PARITY.md)
  isaac_parity_grade?: string | null;        // A | B | C | X (worst tested USD kind)
  error?: string;
}

export interface Manifest {
  name: string;
  version: string;
  generated: string;
  n_doors: number;
  n_signed_off: number;
  families: string[];
  doors: ManifestDoor[];
  isaac_parity?: { version: string; date: string; commit: string | null; n_ok: number; n_fail: number; n_untested: number };
}

export const FAMILY_LABELS: Record<string, string> = {
  swing_single: "Swing (single)", swing_double: "Swing (pair)", dutch: "Dutch", saloon: "Saloon", pivot: "Pivot",
  sliding_single: "Sliding", sliding_bypass: "Bypass closet", bifold: "Bifold", accordion: "Accordion", revolving: "Revolving",
  turnstile_tripod: "Tripod turnstile", turnstile_fullheight: "Full-height turnstile", garage_sectional: "Garage (sectional)",
  garage_tiltup: "Garage (tilt-up)", rollup: "Roll-up", pet_door: "Pet door", hatch_floor: "Floor hatch", hatch_ceiling: "Ceiling hatch",
  ship_watertight: "Watertight (marine)", vault: "Vault", blast: "Blast door", gate_swing: "Gate (swing)", gate_sliding: "Gate (sliding)",
  baby_gate: "Baby gate", stall: "Toilet stall", strip_curtain: "Strip curtain", cold_storage: "Cold storage", automatic_sliding: "Automatic sliding",
  automatic_swing: "Automatic swing", elevator: "Elevator",
};
