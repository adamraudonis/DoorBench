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
  time_budget_s: number;
  expected_transit_s: number;
  has_human: boolean;
}

export interface ManifestDoor {
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

// --- taxonomy hierarchy (viewer/public/taxonomy.json, generated by scripts/taxonomy_report.py) -----------------
export interface TaxRep { id: string; thumb: string | null; use_case: string; mass_kg: number | null }
export interface TaxSizes { leaf_width_m: [number, number]; leaf_height_m: [number, number]; mass_kg: [number, number]; mass_median_kg: number }
export type TaxMechanism = "operator" | "latch" | "lock" | "closer" | "hinge";
export interface TaxSummary {
  count: number;
  signed_off: number;
  sizes: TaxSizes;
  hardware: Record<string, Record<string, number>>;   // operator, operator_kind, latch, latch_kind, lock, lock_kind, closer, closer_kind, hinge, hinge_kind
  kinematics: Record<string, number>;                  // kinematics.type -> doors
  leaves: Record<string, number>;                      // leaf count -> doors
  flags: Record<string, number>;                       // pair, both_ways, self_closing, powered, ...
  conditions: Record<string, number>;
  tasks: Record<string, number>;
  scenarios: Record<string, number>;
  locked: number;
  locked_no_release: number;
  difficulty_mean: number;
  reps: TaxRep[];
}
export interface TaxVariant extends TaxSummary {
  id: string;
  label: string;
  description?: string;
  setting?: string;
  filter: Record<string, string>;                      // catalogue query parameters selecting exactly these doors
  ids: string[];
}
export interface TaxFamily extends TaxSummary {
  id: string;
  label: string;
  description: string;
  kinematics_type: string;
  leaves_note: string;
  examples: string[];
  standards: string[];
  robot: string;
  quota: number;
  variant_rule: string;
  variants: TaxVariant[];
}
export interface TaxClass extends TaxSummary {
  id: string;
  label: string;
  description: string;
  families: TaxFamily[];
}
export interface TaxRelation { rows: string[]; cols: string[]; matrix: number[][] }
export interface TaxonomyJ {
  n_doors: number;
  n_signed_off: number;
  generated?: string;
  manifest_generated?: string;
  seed?: number;
  version?: string;
  motion_classes: TaxClass[];
  kinematics_types: Record<string, string>;
  family_labels: Record<string, string>;
  motion_class_of: Record<string, string>;
  context_info: Record<string, { label: string; setting: string; description: string }>;
  settings: Record<string, { label: string; count: number }>;
  relations: Record<TaxMechanism, TaxRelation>;
  kind_examples: Record<TaxMechanism, Record<string, string[]>>;
  shared_mechanisms: { mechanism: TaxMechanism; kind: string; families: string[]; n_doors: number }[];
}
