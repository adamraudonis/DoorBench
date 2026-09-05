#!/usr/bin/env python
"""Isaac Sim / PhysX side of the DoorBench parity gate (runs INSIDE the Isaac Lab python on a GPU box).

  ./isaaclab.sh -p scripts/isaaclab/isaac_parity.py --doors all --which both --batch 20 --headless
  ./isaaclab.sh -p scripts/isaaclab/isaac_parity.py --doors db0002_swing_single,db0021_swing_single --which full --headless
  bash isaaclab/cloud/parity.sh --limit 40                       # wrapper (sources isaaclab/cloud/_env.sh)

Runs the protocol of ``doorbench/parity/protocol.py`` - the same phases, efforts, thresholds and metric code as the
MuJoCo reference (``scripts/parity_reference_mujoco.py``) - on door.usda (kind ``full``) and door_rl.usda (kind ``rl``),
``batch`` doors per stage, each door its own Isaac Lab ``Articulation`` (spawning / reset / stop-handle fix as in
validate_usd_isaacsim.py).  Per-door inputs (adaptive push etc.) are read from results/parity/mujoco.json when present
so both simulators use identical numbers; otherwise they are derived from spec / model / qa.json.

Isaac-side mechanics (see the module docstring of protocol.py and isaaclab/STATUS.md):
  * the USD drives are the MuJoCo springs: gains untouched (ImplicitActuatorCfg stiffness=None / damping=None),
    position targets restored to ``doorbench:target_si`` EVERY step (Isaac Lab zero-initialises them, which erased
    all spring preloads in the first probe), velocity targets 0
  * robot actions only through set_joint_effort_target (PhysX joint actuation force == MuJoCo qfrc_applied);
    N*m on revolute, N on prismatic joints
  * the one-sided MJCF latch tendon (bolt >= scale * operator) has no PhysX counterpart: kinematic clamp of the
    latch joint state each step (write_joint_state_to_sim), identical to the tendon limit; by default the latch
    drive target is raised to the same minimum while the tendon pulls (--latch-mode clamp+target), otherwise the
    300 N/m latch spring re-extends the bolt by ~2.5 mm every 1/120 s step between clamps (--latch-mode clamp)
  * ``release`` pins the primary joint each step (as qa.py writes qpos / qvel)
  * doors of a batch sit on a centred grid with --spacing 20,14 m (gate leaves sweep / slide up to 8.2 m from
    their origin, fences and floor-hatch decks extend up to 9.9 m: the 6 m grid of the first probe let neighbours
    collide) on a ground plane sized to the grid; batches group doors with the same phase schedule so idle phases
    are not stepped
  * automatic doors: the MJCF position servo (kp, kv, forcerange; ctrl = 0) IS the PhysX drive of spring-less servo
    joints (``doorbench:servo_in_drive``, exporter folds kp / kv / forcerange into stiffness / damping / maxForce);
    servo joints that also carry a spring (automatic swing operators on a closer) get the servo as a clipped
    feed-forward effort (--no-servo disables only that emulation)
  * Coulomb joint friction: the exporter's PhysxJointAxisAPI:angular|linear efforts are read back through Isaac Lab
    (``joint_friction_coeff`` = static friction effort on Isaac Sim >= 5); a mismatch > 1 % is a structure error, and
    when ``write_joint_friction_coefficient_to_sim`` exists the runner first writes the authored efforts and re-reads
    (emulation ``joint_friction_written``) so the physics never runs without friction
  * every link's ``physxRigidBody:maxAngularVelocity`` must be >= 1000 deg/s after Isaac Lab applied the cfg
    (round 1 ran at 100 deg/s = 1.75 rad/s and clamped every leaf)
  * rising / helical hinges (cold_storage, stall): door_rl.usda locks the riser, so the runner applies the gravity
    closing torque -m g dz/dq from ``doorbench:rl["rise_coupling"]`` on the door joint (emulation ``rise_gravity_torque``)
  * env-released locks (maglocks, delayed egress, electric bolts, interlocks) ARE exported: a breakable
    ``UsdPhysics.FixedJoint`` base -> leaf with ``physics:excludeFromArticulation`` (PhysX loop joint) whose
    ``physics:jointEnabled`` the environment clears on release.  The runner only checks it is there and enabled
    (emulation ``env_release_joint``); ``--emulate-weld`` stays as the fallback for a file exported before that
    (it pins the primary joint during ``hold``) and is a no-op when the joint exists
  * bilateral couplings PhysX cannot represent (``doorbench:coupling_mode = "emulated"``: hinge -> slide and
    slide -> slide equalities - thumbturn -> deadbolt, dogs -> bolts, cremone, helical riser - because PhysX
    articulation mimic joints support rotational axes only) are applied as a first-class bilateral constraint
    (emulation ``coupling_bilateral``): the driven joint tracks ``q_a = c0 + c1 q_b`` kinematically AND the driver
    carries the reaction ``c1 * tau_a_ext`` (spring, damping, Coulomb friction, gravity bias, all authored on the
    driven joint prim) plus the reflected inertia ``c1^2 I_a`` written into its armature.  A pure kinematic write
    applies no reaction: the driver loses the coupled part's weight and sags (16 doors of SETTLE_DRIFT, all of them
    helical risers whose mimic PhysX had dropped).
Writes results/parity/isaac_full.json / isaac_rl.json ({"meta", "doors": {door_id: record}}), partial results after
every batch, resumable (doors already present are skipped unless --force).  One door failing never kills a batch.

*** NOT EXECUTED ON THIS MACHINE (Apple silicon, no NVIDIA GPU): written against the Isaac Lab 2.3.2 API. ***
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, ensure_extension_importable  # noqa: E402

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="DoorBench Isaac parity gate (PhysX side).")
parser.add_argument("--doors", type=str, default="all", help="all | easy-100 | random-50 | family:a,b | one-per-family | id,id | @ids.txt")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--which", type=str, default="both", choices=["rl", "full", "both"])
parser.add_argument("--batch", type=int, default=20)
parser.add_argument("--out-dir", type=str, default=os.path.join(ROOT, "results", "parity"))
parser.add_argument("--inputs", type=str, default=os.path.join(ROOT, "results", "parity", "mujoco.json"), help="MuJoCo reference file (per-door inputs + pose0)")
parser.add_argument("--hz", type=int, default=120, help="physics rate (120; 240 for the sensitivity rerun)")
parser.add_argument("--iters", type=str, default="16,4", help="solver position,velocity iterations (32,8 for the sensitivity rerun)")
parser.add_argument("--emulate-weld", action="store_true", help="pin env-released welded doors during the hold phase")
parser.add_argument("--no-servo", action="store_true", help="do not emulate the MJCF position servo of automatic doors")
parser.add_argument("--force", action="store_true", help="re-run doors already present in the output file")
parser.add_argument("--retry-errors", action="store_true", help="re-run doors whose previous record is a spawn / inspect / batch error")
parser.add_argument("--tag", type=str, default="", help="suffix for the output files (e.g. _dt240)")
parser.add_argument("--spacing", type=str, default="20,14", help="x,y grid spacing in metres between the doors of a batch (moving parts reach up to 8.2 m in x / 5.6 m in y, static fences and decks up to 9.9 m / 6.6 m)")
parser.add_argument("--latch-mode", type=str, default="clamp+target", choices=["clamp", "clamp+target"],
                    help="one-sided latch tendon emulation: kinematic clamp only, or clamp + latch drive target raised to the tendon minimum (default)")
parser.add_argument("--no-group", action="store_true", help="keep the --doors order instead of grouping doors with the same phase schedule into batches")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import build_simulation_context  # noqa: E402

ensure_extension_importable()
from doorbench_isaaclab import doors as D  # noqa: E402
from doorbench_isaaclab.assets import DOOR_ACTUATORS, DOOR_ARTICULATION_PROPS, DOOR_RIGID_PROPS  # noqa: E402

from doorbench.parity import protocol as P  # noqa: E402

RL_JOINTS = {"door_slide", "door_hinge", "operator_hinge", "operator_slide", "latch_slide", "leaf2_slide", "leaf2_hinge"}
DT = 1.0 / float(args_cli.hz)
SAMPLE_EVERY = max(1, int(round(1.0 / (P.SAMPLE_HZ * DT))))
if abs(SAMPLE_EVERY * DT * P.SAMPLE_HZ - 1.0) > 1e-6:
    print(f"[parity] WARNING: {args_cli.hz} Hz is not a multiple of the {P.SAMPLE_HZ} Hz sample grid; curves are sampled every {SAMPLE_EVERY} steps ({1.0 / (SAMPLE_EVERY * DT):.2f} Hz)")
POS_ITERS, VEL_ITERS = (int(x) for x in args_cli.iters.split(","))
X_SPACING, Y_SPACING = (float(x) for x in args_cli.spacing.split(","))
LATCH_TARGET = args_cli.latch_mode == "clamp+target"
# smoothing velocity of the Coulomb term in an emulated coupling's reaction (rad/s or m/s); the exporter writes the
# same number into doorbench:couplings["friction_vel_eps"]
COUPLING_VEL_EPS = 1e-3
ENGINE = {"isaac_sim": None, "isaac_lab": None, "physx_dt": DT, "solver_iterations": [POS_ITERS, VEL_ITERS]}
try:
    import isaaclab
    ENGINE["isaac_lab"] = getattr(isaaclab, "__version__", None)
except Exception:
    pass
try:
    import isaacsim
    ENGINE["isaac_sim"] = getattr(isaacsim, "__version__", None)
except Exception:
    pass


def _art_props():
    try:
        return DOOR_ARTICULATION_PROPS.replace(solver_position_iteration_count=POS_ITERS, solver_velocity_iteration_count=VEL_ITERS)
    except Exception:
        import copy
        p = copy.deepcopy(DOOR_ARTICULATION_PROPS)
        p.solver_position_iteration_count, p.solver_velocity_iteration_count = POS_ITERS, VEL_ITERS
        return p


def _write_friction_efforts(art: Articulation, static: torch.Tensor):
    """MuJoCo frictionloss -> PhysX static == dynamic friction effort, viscous 0, through whichever Isaac Lab 2.3 API
    the installed version offers (keyword arguments on write_joint_friction_coefficient_to_sim, or the separate
    write_joint_dynamic/viscous_friction_coefficient_to_sim methods)."""
    zeros = torch.zeros_like(static)
    try:
        art.write_joint_friction_coefficient_to_sim(static, joint_dynamic_friction_coeff=static.clone(), joint_viscous_friction_coeff=zeros)
        return
    except TypeError:
        pass
    art.write_joint_friction_coefficient_to_sim(static)
    for name, val in (("write_joint_dynamic_friction_coefficient_to_sim", static.clone()), ("write_joint_viscous_friction_coefficient_to_sim", zeros)):
        fn = getattr(art, name, None)
        if fn is not None:
            fn(val)


def _grid(n: int) -> list[tuple[float, float, float]]:
    """Door origins for a batch of ``n``: a centred grid, square-ish in metres, spaced X_SPACING x Y_SPACING.

    Doors slide along x and swing toward y.  Over the dataset the moving parts reach up to 8.2 m in x (gate leaves:
    3.6-4.8 m of travel, 4 m swing sweeps) and 5.6 m in y, static fences / walls / hatch decks extend up to 9.9 m in
    x and 6.6 m in y (floor slabs excluded: static vs static never collides), so the 6 m grid of
    validate_usd_isaacsim.py let gate leaves run into the neighbouring door's walls; 20 x 14 m keeps every pair apart
    (worst pairing 18.1 m in x, 12.3 m in y).
    """
    cols = max(1, int(math.ceil(math.sqrt(n * Y_SPACING / X_SPACING))))
    rows = int(math.ceil(n / cols))
    return [(X_SPACING * (k % cols - (cols - 1) / 2.0), Y_SPACING * (k // cols - (rows - 1) / 2.0), 0.0) for k in range(n)]


def _door_cfg(door_id: str, kind: str, k: int, origin: tuple[float, float, float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"/World/Doors/door_{k:03d}",
        spawn=sim_utils.UsdFileCfg(usd_path=D.usd_path(door_id, canonical=(kind == "rl")), activate_contact_sensors=False, rigid_props=DOOR_RIGID_PROPS, articulation_props=_art_props()),
        init_state=ArticulationCfg.InitialStateCfg(pos=origin),
        actuators=DOOR_ACTUATORS,
        articulation_root_prim_path="/Articulation",
    )


def _load_inputs(path: str) -> tuple[dict, dict]:
    """{door_id: inputs}, {door_id: pose0} from the MuJoCo reference file (empty when absent)."""
    if not path or not os.path.isfile(path):
        print(f"[parity] no MuJoCo reference at {path}: inputs derived from spec / model / qa.json (push from qa_push)")
        return {}, {}
    with open(path) as f:
        ref = json.load(f)
    doors = ref.get("doors", {})
    return {k: v["inputs"] for k, v in doors.items() if "inputs" in v}, {k: v.get("pose0") for k, v in doors.items()}


def _fallback_inputs(door_id: str) -> dict:
    dd = D.door_dir(door_id)
    with open(os.path.join(dd, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(dd, "model.json")) as f:
        mj = json.load(f)
    qa = None
    if os.path.isfile(os.path.join(dd, "qa.json")):
        with open(os.path.join(dd, "qa.json")) as f:
            qa = json.load(f)
    return P.door_inputs(spec, mj, qa=qa, rl_meta=P.read_rl_meta(dd))


class DoorHandle:
    """One spawned door: joint mapping (MJCF names -> articulation indices), offsets, spring targets, recording."""

    def __init__(self, sim, art: Articulation, kind: str, door_id: str, inputs: dict, pose0_ref: dict | None):
        from pxr import Usd, UsdPhysics

        self.art, self.kind, self.door_id, self.inputs = art, kind, door_id, inputs
        self.pkind = "usd_rl" if kind == "rl" else "usd_full"
        self.sched = inputs["schedule"][self.pkind]
        self.jn = list(art.joint_names)
        self.nj = len(self.jn)
        self.dev = art.device
        # ---- joint prims: zero offsets, spring targets, source joints
        root = sim.stage.GetPrimAtPath(art.cfg.prim_path)
        prim_info = {}
        self.link_max_ang_vel = {}
        self.rl_meta = None
        self.env_release_prims = []
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                g = lambda a, d=None: (prim.GetAttribute(a).Get() if prim.HasAttribute(a) else d)
                prim_info[prim.GetName()] = {"zero_offset": float(g("doorbench:zero_offset", 0.0) or 0.0), "target_si": float(g("doorbench:target_si", 0.0) or 0.0),
                                             "source": g("doorbench:source_joint", None), "friction": g("doorbench:friction_effort", None),
                                             "armature": g("doorbench:armature_si", None), "servo_in_drive": bool(g("doorbench:servo_in_drive", False)),
                                             "servo_kp": float(g("doorbench:servo_kp_si", 0.0) or 0.0), "servo_kv": float(g("doorbench:servo_kv_si", 0.0) or 0.0),
                                             "servo_ctrl": float(g("doorbench:servo_ctrl", 0.0) or 0.0),
                                             "revolute": bool(prim.IsA(UsdPhysics.RevoluteJoint)),
                                             "stiffness": float(g("doorbench:stiffness_si", 0.0) or 0.0), "damping": float(g("doorbench:damping_si", 0.0) or 0.0),
                                             "coupling_mode": g("doorbench:coupling_mode", None), "coupling_driver": g("doorbench:coupling_driver", None),
                                             "coupling_c0": float(g("doorbench:coupling_c0", 0.0) or 0.0), "coupling_c1": float(g("doorbench:coupling_c1", 0.0) or 0.0),
                                             "coupling_gravity_bias": float(g("doorbench:coupling_gravity_bias", 0.0) or 0.0),
                                             "coupling_chain_order": int(g("doorbench:coupling_chain_order", 0) or 0),
                                             "reflected_armature": float(g("doorbench:coupling_reflected_armature", 0.0) or 0.0)}
            elif prim.IsA(UsdPhysics.FixedJoint) and prim.HasAttribute("doorbench:role") and prim.GetAttribute("doorbench:role").Get() == "env_release":
                self.env_release_prims.append({"joint": prim.GetName(), "path": str(prim.GetPath()),
                                               "enabled": bool(prim.GetAttribute("physics:jointEnabled").Get()) if prim.HasAttribute("physics:jointEnabled") else None,
                                               "holding_force_N": float(prim.GetAttribute("doorbench:holding_force_N").Get() or 0.0) if prim.HasAttribute("doorbench:holding_force_N") else None,
                                               "excluded": bool(prim.GetAttribute("physics:excludeFromArticulation").Get()) if prim.HasAttribute("physics:excludeFromArticulation") else False,
                                               "body": prim.GetAttribute("doorbench:weld_body").Get() if prim.HasAttribute("doorbench:weld_body") else None})
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                a = prim.GetAttribute("physxRigidBody:maxAngularVelocity")
                self.link_max_ang_vel[prim.GetName()] = float(a.Get()) if (a and a.IsValid() and a.Get() is not None) else None
            elif prim.HasAttribute("doorbench:rl"):
                try:
                    self.rl_meta = json.loads(prim.GetAttribute("doorbench:rl").Get())
                except Exception:
                    self.rl_meta = None
        self.prim_info = prim_info
        self.offset = np.zeros(self.nj)
        target = np.zeros(self.nj)
        for i, n in enumerate(self.jn):
            info = prim_info.get(n, {})
            self.offset[i] = info.get("zero_offset", 0.0)
            target[i] = info.get("target_si", 0.0)
        # ---- MJCF joint name -> articulation index
        self.map = {}
        if kind == "full":
            for i, n in enumerate(self.jn):
                if n in inputs["joints"]:
                    self.map[n] = i
        else:
            rl = inputs.get("rl") or {}
            slot_of = dict(rl.get("slot_of") or {})
            if not slot_of:   # derive from the prims' source joints
                for n, info in prim_info.items():
                    if info.get("source") and n in self.jn:
                        slot_of[info["source"]] = n
            for src, slot in slot_of.items():
                if slot in self.jn and src in inputs["joints"]:
                    self.map[src] = self.jn.index(slot)
            self.locked_slots = [i for i, n in enumerate(self.jn) if i not in self.map.values()]
        self.inv = {i: n for n, i in self.map.items()}
        self.pj = self.map.get(inputs["primary_joint"])
        if self.pj is None:
            raise RuntimeError(f"primary joint {inputs['primary_joint']} not present in the {kind} articulation ({self.jn})")
        self.spring_target = torch.tensor(target, dtype=torch.float32, device=self.dev).unsqueeze(0)
        self.spring_target_np = np.array(target, dtype=float)
        self.zero = torch.zeros(1, self.nj, device=self.dev)
        self._effort = torch.zeros(1, self.nj, device=self.dev)
        self._target = self.spring_target.clone()
        self._cache = None          # (step id, q_db, v) read after the last sim step, shared by post_step and pre_step
        self.env_origin = np.array(art.cfg.init_state.pos, dtype=float)
        self.q0_usd = np.zeros(self.nj)     # USD joint zero == the authored pose == MJCF qpos0 (initial == modeled_at for every joint)
        # tendon-driven joints (latch bolts): lower range end in DoorBench coordinates, for the drive-target emulation
        self.range_lo = {n: (inputs["joints"][n]["range"][0] if inputs["joints"][n]["range"] else -math.inf) for n in self.map}
        self.latch_target = LATCH_TARGET
        self.emulations = ["spring_targets_restored", "latch_clamp+target" if self.latch_target else "latch_clamp"]
        self.friction_readback = None
        # MJCF joints whose position servo is already the PhysX drive (spring-less servo joints, see usd.py): no
        # feed-forward emulation for them; the remaining servo joints (spring + servo) are emulated unless --no-servo
        self.servo_in_drive = {src for src, i in self.map.items() if prim_info.get(self.jn[i], {}).get("servo_in_drive")}
        emulated_servos = [a["joint"] for a in inputs["coupling"].get("actuators", []) if a.get("joint") in self.map and a["joint"] not in self.servo_in_drive]
        self.servo = bool(inputs["flags"]["automatic"]) and not args_cli.no_servo and bool(emulated_servos)
        if self.servo_in_drive:
            self.emulations.append("servo_in_drive")
        if self.servo:
            self.emulations.append("servo_emulated")
        # rising / helical hinge in the canonical file: riser locked -> constant gravity closing torque on the door joint
        self.rise_torque = 0.0
        rc = (self.rl_meta or {}).get("rise_coupling") if kind == "rl" else None
        if rc and rc.get("gravity_torque_Nm") is not None and math.isfinite(float(rc["gravity_torque_Nm"])):
            self.rise_torque = float(rc["gravity_torque_Nm"])
            self.emulations.append("rise_gravity_torque")
        self.errors = []
        self.coupling_armature = {}
        # environment-released locks: exported as a breakable loop FixedJoint, so PhysX holds the leaf on its own.
        # --emulate-weld stays as the fallback for files exported before the joint existed.
        self.env_release = [e for e in self.env_release_prims if e.get("enabled") is not False]
        if self.env_release:
            self.emulations.append("env_release_joint")
        self.weld = bool(args_cli.emulate_weld and inputs["flags"]["env_release_only"] and inputs["flags"]["has_weld"] and not self.env_release)
        if self.weld:
            self.emulations.append("weld_pinned_hold")
        # bilateral couplings PhysX drops (mimic joints are rotational-only): tracked + reaction on the driver
        self.couplings = self._build_couplings()
        if self.couplings:
            self.emulations.append("coupling_bilateral")
            self._write_coupling_armature()
        self.phases = {}
        self.ctx = {}
        self.q_hold = None
        self.structure = self.check_structure()
        self.pose0 = self.check_pose0(pose0_ref) if (pose0_ref and kind == "full") else None
        self.cur = None

    # ------------------------------------------------------------------
    def check_structure(self) -> dict:
        errors, warnings = [], []
        try:
            lim = self.art.data.joint_pos_limits[0].cpu().numpy() if hasattr(self.art.data, "joint_pos_limits") else self.art.data.soft_joint_pos_limits[0].cpu().numpy()
            st = self.art.data.default_joint_stiffness[0].cpu().numpy()
            dp = self.art.data.default_joint_damping[0].cpu().numpy()
            if self.kind == "rl" and set(self.jn) != RL_JOINTS:
                errors.append(f"joint names {sorted(self.jn)} != canonical")
            if self.kind == "full" and set(self.jn) != set(self.inputs["joints"]):
                errors.append(f"joint names differ from model.json: {sorted(set(self.jn) ^ set(self.inputs['joints']))[:6]}")
            for src, i in self.map.items():
                j = self.inputs["joints"][src]
                pi = self.prim_info.get(self.jn[i], {})
                if j["range"] is not None:
                    lo, hi = j["range"][0] - j["modeled_at"], j["range"][1] - j["modeled_at"]
                    if abs(lim[i][0] - lo) > 2e-3 or abs(lim[i][1] - hi) > 2e-3:
                        errors.append(f"{src}: limits {[float(x) for x in lim[i]]} != IR {[lo, hi]}")
                # a spring-less servo joint carries its MJCF position servo in the drive (k = kp, d = kv + damping)
                want_k = j["stiffness"] + (pi.get("servo_kp", 0.0) if src in self.servo_in_drive else 0.0)
                want_d = j["damping"] + (pi.get("servo_kv", 0.0) if src in self.servo_in_drive else 0.0)
                if abs(st[i] - want_k) > 1e-2 * max(1.0, abs(want_k)):
                    errors.append(f"{src}: stiffness {st[i]:.4g} != IR {want_k:.4g}")
                if abs(dp[i] - want_d) > 1e-2 * max(1.0, abs(want_d)):
                    warnings.append(f"{src}: damping {dp[i]:.4g} != IR {want_d:.4g}")
                tgt = (j["springref"] - j["modeled_at"]) if j["stiffness"] > 0 else ((pi.get("servo_ctrl", 0.0) - j["modeled_at"]) if src in self.servo_in_drive else 0.0)
                if abs(float(self.spring_target[0, i]) - tgt) > 1e-3:
                    errors.append(f"{src}: spring target {float(self.spring_target[0, i]):.4g} != IR {tgt:.4g}")
                if abs(self.offset[i] - j["modeled_at"]) > 1e-6:
                    errors.append(f"{src}: zero_offset {self.offset[i]} != modeled_at {j['modeled_at']}")
            # Coulomb friction read-back.  Isaac Lab on Isaac Sim >= 5 exposes PhysX's static friction effort as
            # joint_friction_coeff (round 1 read 0.0 on every joint: the efforts were authored on the rotX / transX
            # instance that the parser ignores).  A mismatch is corrected through the Isaac Lab write API when it
            # exists (so the physics never runs without friction) and is a structure error when it persists.
            self.friction_readback = self._read_friction()
            if self.friction_readback is not None:
                bad = self._friction_mismatch(self.friction_readback)
                if bad and hasattr(self.art, "write_joint_friction_coefficient_to_sim"):
                    self._write_friction()
                    self.friction_readback = self._read_friction()
                    bad2 = self._friction_mismatch(self.friction_readback or {})
                    warnings.append(f"friction efforts not parsed from the USD ({bad[:3]}); written through Isaac Lab" + ("" if not bad2 else f", still off: {bad2[:3]}"))
                    if bad2:
                        errors.append(f"joint friction {bad2[:4]}")
                elif bad:
                    errors.append(f"joint friction {bad[:4]}")
            arm = getattr(self.art.data, "joint_armature", None)
            if arm is None:
                arm = getattr(self.art.data, "default_joint_armature", None)
            self.armature_readback = {n: float(arm[0, i]) for n, i in self.map.items()} if arm is not None else None
            if self.armature_readback:
                for n, i in self.map.items():
                    # a driver of an emulated coupling carries the reflected inertia c1^2 * I_driven on top of the IR
                    # armature (MuJoCo's equality gives it that inertia through the constraint)
                    want = float(self.inputs["joints"][n]["armature"]) + float(self.coupling_armature.get(self.jn[i], 0.0))
                    if abs(self.armature_readback[n] - want) > 1e-2 * max(1e-3, want):
                        warnings.append(f"{n}: armature {self.armature_readback[n]:.4g} != IR{'+reflected' if self.jn[i] in self.coupling_armature else ''} {want:.4g}")
            # env-released locks must exist as a real (enabled, breakable) joint in the USD
            if self.inputs["flags"].get("env_release_only") and self.inputs["flags"].get("has_weld") and not self.env_release:
                errors.append("env-released lock: no doorbench:role=env_release FixedJoint in the USD (regenerate the dataset; --emulate-weld pins the leaf instead)")
            for e in self.env_release:
                if not e.get("excluded"):
                    errors.append(f"{e['joint']}: env-release joint without physics:excludeFromArticulation (it would become a second parent of the leaf)")
                if not e.get("holding_force_N"):
                    errors.append(f"{e['joint']}: env-release joint without a holding force (breakForce)")
            # every equality the exporter marked "emulated" must have been resolved to two articulation joints
            n_emulated = sum(1 for n, i in self.prim_info.items() if i.get("coupling_mode") == "emulated" and n in self.jn)
            if n_emulated and len(self.couplings) != n_emulated:
                errors.append(f"{len(self.couplings)} of {n_emulated} emulated couplings resolved")
            # PhysX angular velocity cap on the links (deg/s); Isaac Lab writes the cfg value onto the prims at spawn
            low = {n: v for n, v in self.link_max_ang_vel.items() if v is not None and v < 1000.0}
            if low:
                errors.append(f"links capped below 1000 deg/s (17 rad/s): {dict(list(low.items())[:3])} - MuJoCo has no cap; check DOOR_RIGID_PROPS.max_angular_velocity (deg/s)")
        except Exception as e:
            errors.append(f"structure check: {type(e).__name__}: {e}")
        return {"status": "fail" if errors else "pass", "errors": errors, "warnings": warnings, "mapped_joints": sorted(self.map),
                "friction_coeff_readback": self.friction_readback, "friction_effort_authored": {n: self.prim_info.get(self.jn[i], {}).get("friction") for n, i in self.map.items()},
                "armature_readback": getattr(self, "armature_readback", None), "servo_in_drive": sorted(self.servo_in_drive), "rise_gravity_torque_Nm": self.rise_torque,
                "link_max_angular_velocity_deg_s": self.link_max_ang_vel,
                "env_release": self.env_release_prims,
                "couplings_emulated": [{k: c[k] for k in ("driven", "driver", "c0", "c1", "bias", "order")} for c in self.couplings],
                "coupling_reflected_armature": self.coupling_armature}

    def _build_couplings(self):
        """Couplings the exporter marked ``emulated`` (hinge -> slide, slide -> slide), resolved to joint indices.

        ``q_driven = c0 + c1 * q_driver`` in DoorBench coordinates.  The driven joint is tracked kinematically and
        the driver carries the reaction ``c1 * tau_driven_ext`` (``_coupling_effort``) plus the reflected inertia
        (written into its armature by ``_write_coupling_armature``)."""
        out = []
        for jname, info in self.prim_info.items():
            if info.get("coupling_mode") != "emulated" or jname not in self.jn:
                continue
            driver = info.get("coupling_driver")
            dj = next((n for n in self.jn if n == driver or self.prim_info.get(n, {}).get("source") == driver), None)
            if dj is None:
                self.errors.append(f"coupling {jname} <- {driver}: driver joint not in the articulation")
                continue
            ia, ib = self.jn.index(jname), self.jn.index(dj)
            lim = None
            try:
                pl = self.art.data.joint_pos_limits[0].cpu().numpy() if hasattr(self.art.data, "joint_pos_limits") else self.art.data.soft_joint_pos_limits[0].cpu().numpy()
                lim = (float(pl[ia][0]), float(pl[ia][1]))
            except Exception:
                pass
            out.append({"driven": jname, "driver": dj, "ia": ia, "ib": ib, "c0": info["coupling_c0"], "c1": info["coupling_c1"],
                        "off_a": self.offset[ia], "off_b": self.offset[ib], "limits": lim,
                        "k": info["stiffness"], "d": info["damping"], "target": info["target_si"],
                        "friction": float(info.get("friction") or 0.0), "bias": info["coupling_gravity_bias"],
                        "order": info["coupling_chain_order"]})
        out.sort(key=lambda c: c["order"])
        return out

    def _write_coupling_armature(self):
        """Driver joints carry the inertia of everything they drive through an emulated coupling (c1^2 * I_driven).

        MuJoCo's equality gives the driver that inertia for free; PhysX drops the mimic, so it is written into the
        driver's armature - the exact equivalent of the constraint's inertial term ``-c1 * I_a * qdd_a``."""
        extra = {}
        for jname, info in self.prim_info.items():
            if info.get("reflected_armature") and jname in self.jn:
                extra[self.jn.index(jname)] = float(info["reflected_armature"])
        if not extra:
            return
        fn = getattr(self.art, "write_joint_armature_to_sim", None)
        arm = getattr(self.art.data, "joint_armature", None)
        if arm is None:
            arm = getattr(self.art.data, "default_joint_armature", None)
        if fn is None or arm is None:
            self.errors.append("coupling reflected inertia cannot be written (no write_joint_armature_to_sim)")
            return
        want = arm[0].clone().unsqueeze(0)
        for i, v in extra.items():
            want[0, i] = float(want[0, i]) + v
        fn(want)
        self.coupling_armature = {self.jn[i]: v for i, v in extra.items()}

    def _coupling_effort(self, q_db, v):
        """Reaction the emulated couplings put on their drivers: ``c1 * tau_driven_ext`` (N*m / N by joint type).

        ``tau_driven_ext`` is everything PhysX / MuJoCo apply to the driven DOF: its drive spring and damping, the
        Coulomb friction bound (smoothed over ``COUPLING_VEL_EPS``) and the constant gravity bias the exporter
        measured at the authored pose.  Returns {driver index: effort} and the tracked driven positions."""
        eff, track = {}, []
        q_target = {}
        for c in self.couplings:
            qb = float(q_db[c["ib"]]) if c["ib"] not in q_target else q_target[c["ib"]]
            qa = c["c0"] + c["c1"] * qb
            q_target[c["ia"]] = qa
            q_usd = qa - c["off_a"]
            if c["limits"] is not None:
                q_usd = min(max(q_usd, c["limits"][0]), c["limits"][1])
            va = c["c1"] * float(v[c["ib"]])
            tau = c["k"] * (c["target"] - q_usd) - c["d"] * va + c["bias"]
            if c["friction"]:
                tau -= c["friction"] * max(-1.0, min(1.0, va / COUPLING_VEL_EPS))
            eff[c["ib"]] = eff.get(c["ib"], 0.0) + c["c1"] * tau
            track.append((c["ia"], q_usd, va))
        return eff, track

    def _read_friction(self):
        fr = getattr(self.art.data, "joint_friction_coeff", None)
        if fr is None:
            return None
        return {n: float(fr[0, i]) for n, i in self.map.items()}

    def _friction_mismatch(self, readback: dict) -> list:
        bad = []
        for n, i in self.map.items():
            want = float(self.inputs["joints"][n]["frictionloss"])
            got = readback.get(n)
            if got is None or abs(got - want) > 1e-2 * max(1e-3, want):
                bad.append(f"{n}: {got} != {want:.4g}")
        return bad

    def _write_friction(self):
        """Author the MuJoCo Coulomb bound as PhysX static == dynamic friction effort (viscous 0) on every mapped joint."""
        static = torch.zeros(1, self.nj, device=self.dev)
        for n, i in self.map.items():
            static[0, i] = float(self.inputs["joints"][n]["frictionloss"])
        _write_friction_efforts(self.art, static)
        self.emulations.append("joint_friction_written")

    def check_pose0(self, ref: dict) -> dict:
        """Informational frame check: PhysX link origins vs MuJoCo body origins (d.xpos) at the initial state (full kind).

        Isaac Lab 2.3: ``body_pos_w`` is the centre-of-mass frame (== body_com_pos_w); the link (prim) frame that
        matches MuJoCo's body origin is ``body_link_pos_w``."""
        try:
            data = self.art.data
            frame = "link" if hasattr(data, "body_link_pos_w") else "com"
            pos_t = data.body_link_pos_w if frame == "link" else data.body_pos_w
            pos = pos_t[0].cpu().numpy() - self.env_origin
            names = list(self.art.body_names)
            worst = (0.0, None)
            n = 0
            for b, p in ref.get("bodies", {}).items():
                if b in names:
                    e = float(np.linalg.norm(pos[names.index(b)] - np.array(p)))
                    n += 1
                    if e > worst[0]:
                        worst = (e, b)
            return {"n_bodies": n, "max_err_m": worst[0], "worst_body": worst[1], "frame": frame}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    def q_db(self) -> np.ndarray:
        return self.art.data.joint_pos[0].cpu().numpy() + self.offset

    def state(self, step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(q in DoorBench coordinates, joint velocities) from the articulation data; one GPU -> CPU read per step
        (post_step caches it for the following pre_step)."""
        if step is not None and self._cache is not None and self._cache[0] == step:
            return self._cache[1], self._cache[2]
        q = self.q_db()
        v = self.art.data.joint_vel[0].cpu().numpy()
        if step is not None:
            self._cache = (step, q, v)
        return q, v

    def qmap(self, q: np.ndarray) -> dict:
        return {n: float(q[i]) for n, i in self.map.items()}

    def reset(self, overrides: dict):
        q = np.array(self.q0_usd)
        for n, v in overrides.items():
            i = self.map.get(n)
            if i is not None:
                q[i] = v - self.offset[i]
        qt = torch.tensor(q, dtype=torch.float32, device=self.dev).unsqueeze(0)
        self.art.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        self.art.set_joint_position_target(self.spring_target)
        self.art.set_joint_velocity_target(self.zero)
        self.art.set_joint_effort_target(self.zero)
        self.art.write_data_to_sim()
        try:
            self.art.reset()
        except Exception:
            pass
        self.art.update(DT)
        self._cache = None

    def duration(self, phase: str) -> float:
        return P.phase_duration(self.inputs, phase, self.pkind)

    def applies(self, phase: str) -> bool:
        return not self.sched[phase].startswith("na:")

    def begin(self, phase: str):
        q, v = self.state(0)
        self.cur = {"phase": phase, "t": [0.0], "q": {n: [float(q[i])] for n, i in self.map.items()}, "v": {self.inputs["primary_joint"]: [float(v[self.pj])]},
                    "qmin": q.copy(), "qmax": q.copy(), "vmax": np.zeros(self.nj), "finite": True, "warnings": [], "clamps": 0, "target_raised": 0}
        if phase == "release":
            self.q_hold = float(q[self.pj])

    def pre_step(self, phase: str, t: float, active: bool, step: int | None = None):
        art = self.art
        q, v = self.state(step)
        qm = self.qmap(q)
        eff = P.phase_efforts(self.inputs, phase, t, qm, kind=self.pkind) if active else {}
        if self.servo:
            for n, f in P.servo_effort(self.inputs, qm, {n: float(v[i]) for n, i in self.map.items()}).items():
                if n not in self.servo_in_drive:          # spring-less servo joints: the PhysX drive already is the servo
                    eff[n] = eff.get(n, 0.0) + f
        effort = self._effort
        effort.zero_()
        for n, f in eff.items():
            i = self.map.get(n)
            if i is not None:
                effort[0, i] = float(f)
        if self.rise_torque:
            # helical hinge: the canonical file locks the riser; MuJoCo's rise coupling costs m g dz per rad of opening
            effort[0, self.pj] += float(self.rise_torque)
        # bilateral couplings PhysX drops (mimic joints are rotational-only): the driven joint tracks its driver and
        # the driver carries the reaction, so it feels the coupled part's weight, spring and friction
        if self.couplings:
            ceff, ctrack = self._coupling_effort(q, v)
            for i, f in ceff.items():
                effort[0, i] += float(f)
            for i, q_usd, va in ctrack:
                self._write_joint(i, q_usd, vel=va)
        # one-sided tendons (latch bolt >= scale * operator).  MuJoCo enforces them as a stiff constraint; here the
        # joint state is clamped to the tendon minimum every step and (clamp+target) the latch drive target follows
        # that minimum while the tendon pulls (q_min above the joint's lower range end), otherwise the 300 N/m latch
        # spring pulling toward its -8 mm springref re-extends a 0.04 kg bolt by ~2.5 mm within one 1/120 s step and
        # the recorded retraction chatters below the tendon minimum (MuJoCo: none).  Once the operator returns the
        # minimum falls back to the range end and the drive target to the springref, so the bolt re-extends as in
        # MuJoCo (release / relatch).
        target = self.spring_target
        raised = False
        for n, qmin in P.tendon_min_positions(self.inputs, qm).items():
            i = self.map.get(n)
            if i is None:
                continue
            q_usd_min = qmin - self.offset[i]
            if self.latch_target and qmin > self.range_lo[n] + 1e-4 and q_usd_min > self.spring_target_np[i]:
                if not raised:
                    target = self._target
                    target.copy_(self.spring_target)
                    raised = True
                target[0, i] = float(q_usd_min)
            if q[i] < qmin - 1e-6:
                self._write_joint(i, q_usd_min)
                if self.cur is not None:        # idle doors (phase not applicable) are clamped too but not recorded
                    self.cur["clamps"] += 1
        if raised and self.cur is not None:
            self.cur["target_raised"] += 1
        if phase == "release" and active and self.q_hold is not None:
            self._write_joint(self.pj, self.q_hold - self.offset[self.pj])
        if phase == "hold" and self.weld:
            self._write_joint(self.pj, self.q0_usd[self.pj])
        art.set_joint_position_target(target)
        art.set_joint_velocity_target(self.zero)
        art.set_joint_effort_target(effort)
        art.write_data_to_sim()

    def _write_joint(self, i: int, q_usd: float, vel: float = 0.0):
        """Kinematic write of one joint (position + velocity); the other joints keep the values of the last read."""
        ids = torch.tensor([i], device=self.dev)
        pos = torch.tensor([[q_usd]], dtype=torch.float32, device=self.dev)
        vt = torch.tensor([[vel]], dtype=torch.float32, device=self.dev)
        self.art.write_joint_state_to_sim(pos, vt, joint_ids=ids)
        self._cache = None

    def post_step(self, k: int, t: float):
        q, v = self.state(k)
        c = self.cur
        np.minimum(c["qmin"], q, out=c["qmin"]); np.maximum(c["qmax"], q, out=c["qmax"]); np.maximum(c["vmax"], np.abs(v), out=c["vmax"])
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            c["finite"] = False
        if k % SAMPLE_EVERY == 0:
            c["t"].append(round(t, 6))
            for n, i in self.map.items():
                c["q"][n].append(float(q[i]))
            c["v"][self.inputs["primary_joint"]].append(float(v[self.pj]))

    def finish(self, phase: str):
        c = self.cur
        curve = {"t": c["t"], "q": c["q"], "v": c["v"], "minmax": {n: [float(c["qmin"][i]), float(c["qmax"][i])] for n, i in self.map.items()},
                 "vmax": {n: float(c["vmax"][i]) for n, i in self.map.items()}, "finite": c["finite"], "warnings": c["warnings"], "latch_clamps": c["clamps"]}
        if self.kind == "rl":
            q = self.q_db()
            curve["rl_locked_slot_max_abs"] = float(max((abs(q[i] - self.q0_usd[i] - self.offset[i]) for i in self.locked_slots), default=0.0))
        expected = self.sched[phase]
        metrics = P.phase_metrics(self.inputs, phase, curve, self.ctx)
        metrics["latch_clamps"] = c["clamps"]
        metrics["latch_target_raised_steps"] = c["target_raised"]
        if "rl_locked_slot_max_abs" in curve:
            metrics["rl_locked_slot_max_abs"] = curve["rl_locked_slot_max_abs"]
        status = P.phase_status(self.inputs, phase, expected, metrics)
        if phase == "operate":
            self.ctx["opened"] = metrics.get("opened")
        self.phases[phase] = {"expected": expected, "status": status, "metrics": metrics, "informational": expected.endswith("_info"),
                              "curve": {"t": [round(x, 4) for x in c["t"]], "q": {n: [round(x, 5) for x in arr] for n, arr in c["q"].items() if n in (self.inputs["primary_joint"], self.inputs["operator_joint"], self.inputs["latch_bolt_joint"])}, "hz": P.SAMPLE_HZ}}
        self.cur = None

    def record(self) -> dict:
        for phase in P.PHASES:
            if phase not in self.phases:
                self.phases[phase] = {"expected": self.sched[phase], "status": "na", "metrics": {}, "informational": False}
        return {"door_id": self.door_id, "sim": "physx", "kind": self.kind, "engine": ENGINE, "dt": DT, "protocol_version": P.PROTOCOL_VERSION, "inputs_hash": self.inputs.get("inputs_hash"),
                "emulations_used": self.emulations, "structure": self.structure, "pose0": self.pose0, "phases": self.phases, "errors": self.errors,
                "limits": {"violations": [dict(v, phase=p) for p, r in self.phases.items() for v in (r["metrics"].get("limit_violations") or [])]},
                "sanity": {"finite": all(r["metrics"].get("finite", True) for r in self.phases.values() if r["metrics"]), "velocity_cap_hit": any(r["metrics"].get("velocity_cap_hit") for r in self.phases.values() if r["metrics"])},
                "ok": self.structure["status"] == "pass" and all(r["status"] in ("pass", "skip", "na") or r.get("informational") for r in self.phases.values()) and not self.errors}


def run_batch(ids: list[str], kind: str, device: str, inputs_by_id: dict, pose0_by_id: dict) -> dict:
    rows = {}
    origins = _grid(len(ids))
    with build_simulation_context(device=device, dt=DT, gravity_enabled=True, add_ground_plane=False, auto_add_lighting=False) as sim:
        # headless: Isaac Lab's timeline-STOP callback loops render() forever when the context exits (and sim.reset()
        # re-arms its _disable_app_control_on_stop_handle flag), so drop the subscription itself.
        if getattr(sim, "_app_control_on_stop_handle", None) is not None:
            sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
        # ground plane sized to the grid (the default of build_simulation_context is 100 x 100 m; door_rl.usda has no
        # floor slab of its own, so every door must sit on this plane).  Same spawner call as add_ground_plane=True.
        span = max(max(abs(o[0]) for o in origins), max(abs(o[1]) for o in origins))
        ground = sim_utils.GroundPlaneCfg(size=(2.0 * span + 100.0, 2.0 * span + 100.0))
        ground.func("/World/defaultGroundPlane", ground)
        arts = []
        for k, did in enumerate(ids):
            try:
                arts.append(Articulation(_door_cfg(did, kind, k, origins[k])))
            except Exception as e:
                arts.append(None)
                rows[did] = {"door_id": did, "sim": "physx", "kind": kind, "engine": ENGINE, "protocol_version": P.PROTOCOL_VERSION, "load_error": f"spawn: {type(e).__name__}: {e}", "ok": False, "phases": {}}
        sim.reset()
        handles = []
        for did, art in zip(ids, arts):
            if art is None:
                continue
            try:
                art.update(DT)
                inputs = inputs_by_id.get(did) or _fallback_inputs(did)
                handles.append(DoorHandle(sim, art, kind, did, inputs, pose0_by_id.get(did)))
            except Exception as e:
                rows[did] = {"door_id": did, "sim": "physx", "kind": kind, "engine": ENGINE, "protocol_version": P.PROTOCOL_VERSION, "load_error": f"inspect: {type(e).__name__}: {e}", "ok": False, "phases": {},
                             "traceback": traceback.format_exc()[-1500:]}
        live = list(handles)
        for phase in P.PHASES:
            applicable = [h for h in live if h.applies(phase)]
            if phase == "relatch":   # qa.py: relatch only when the door opened past 5 deg
                applicable = [h for h in applicable if (h.ctx.get("opened") or 0.0) > h.inputs["thresholds"]["relatch_min_open"]]
                for h in live:
                    if h.applies(phase) and h not in applicable:
                        m = {"finite": True, "opened_before": h.ctx.get("opened"), "limit_violations": [], "warnings": []}
                        h.phases[phase] = {"expected": h.sched[phase], "status": P.phase_status(h.inputs, phase, h.sched[phase], m), "metrics": m, "informational": False}
            if not applicable:
                continue
            if P.PHASE_RESETS[phase]:
                for h in live:
                    try:
                        h.reset(P.phase_initial_state(h.inputs, phase))
                    except Exception as e:
                        h.errors.append(f"reset {phase}: {type(e).__name__}: {e}")
            started = []
            for h in applicable:
                try:
                    h.begin(phase)
                    started.append(h)
                except Exception as e:
                    h.errors.append(f"begin {phase}: {type(e).__name__}: {e}")
                    h.phases[phase] = {"expected": h.sched[phase], "status": "fail", "metrics": {"finite": False}, "informational": False}
            applicable = started
            if not applicable:
                continue
            n_steps = max(int(round(h.duration(phase) / DT)) for h in applicable)
            durs = {id(h): h.duration(phase) for h in applicable}
            for k in range(n_steps):
                t = k * DT
                for h in live:
                    try:
                        h.pre_step(phase, t, active=(h in applicable and t < durs.get(id(h), 0.0) - 1e-9), step=k)
                    except Exception as e:
                        if len(h.errors) < 5:
                            h.errors.append(f"{phase} step {k}: {type(e).__name__}: {e}")
                sim.step()
                for h in live:
                    try:
                        h.art.update(DT)
                    except Exception as e:
                        if len(h.errors) < 5:
                            h.errors.append(f"{phase} update {k}: {type(e).__name__}: {e}")
                for h in applicable:
                    try:
                        h.post_step(k + 1, (k + 1) * DT)
                    except Exception as e:
                        if len(h.errors) < 5:
                            h.errors.append(f"{phase} record {k}: {type(e).__name__}: {e}")
            for h in applicable:
                try:
                    h.finish(phase)
                except Exception as e:
                    h.errors.append(f"{phase} finish: {type(e).__name__}: {e}")
                    h.phases[phase] = {"expected": h.sched[phase], "status": "fail", "metrics": {"finite": False}, "informational": False}
                    h.cur = None
        for h in live:
            try:
                rows[h.door_id] = h.record()
            except Exception as e:
                rows[h.door_id] = {"door_id": h.door_id, "sim": "physx", "kind": kind, "engine": ENGINE, "protocol_version": P.PROTOCOL_VERSION, "load_error": f"record: {type(e).__name__}: {e}", "ok": False, "phases": {}}
    return rows


def _schedule_key(inputs: dict | None, pkind: str) -> tuple:
    """Batch-grouping key: which phases apply (and whether the hold phase is the 6 s free push)."""
    if not inputs:
        return ("~",)
    sched = inputs["schedule"][pkind]
    return tuple(("" if sched[p].startswith("na:") else ("free" if (p == "hold" and sched[p] != "hold") else "x")) for p in P.PHASES)


def select_ids(spec: str) -> list[str]:
    if spec.startswith("one-per-family"):
        reps = {}
        for d in D.manifest()["doors"]:
            reps.setdefault(d["family"], d["id"])
        return sorted(reps.values())
    return D.select_ids(spec)


def main():
    ids = select_ids(args_cli.doors)
    if args_cli.limit:
        ids = ids[: args_cli.limit]
    kinds = ["rl", "full"] if args_cli.which == "both" else [args_cli.which]
    device = args_cli.device or "cuda:0"
    inputs_by_id, pose0_by_id = _load_inputs(args_cli.inputs)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    t0 = time.time()
    for kind in kinds:
        out = os.path.join(args_cli.out_dir, f"isaac_{kind}{args_cli.tag}.json")
        doors = {}
        if os.path.isfile(out) and not args_cli.force:
            with open(out) as f:
                prev = json.load(f)
            if prev.get("meta", {}).get("protocol_version") == P.PROTOCOL_VERSION:
                doors = prev.get("doors", {})
        todo = [i for i in ids if i not in doors or args_cli.force or (args_cli.retry_errors and doors[i].get("load_error"))]
        if not args_cli.no_group:
            # batches step every phase any of their doors needs (a closer door adds 12 s to the whole batch): group
            # doors with the same applicable phases / hold duration so idle stepping is minimal; order otherwise kept
            pkind = "usd_rl" if kind == "rl" else "usd_full"
            order = {i: k for k, i in enumerate(todo)}
            todo.sort(key=lambda i: (_schedule_key(inputs_by_id.get(i), pkind), order[i]))
        n_batches = math.ceil(len(todo) / args_cli.batch)
        print(f"[parity] {kind}: {len(todo)} doors to run ({len(ids) - len(todo)} already in {out}), {n_batches} batches of {args_cli.batch}, dt 1/{args_cli.hz}, iters {POS_ITERS}/{VEL_ITERS}, "
              f"grid {X_SPACING:g} x {Y_SPACING:g} m, latch {args_cli.latch_mode}")
        for b in range(n_batches):
            batch = todo[b * args_cli.batch: (b + 1) * args_cli.batch]
            print(f"[parity] {kind} batch {b + 1}/{n_batches}: {len(batch)} doors ({batch[0]} .. {batch[-1]})", flush=True)
            tb = time.time()
            try:
                rows = run_batch(batch, kind, device, inputs_by_id, pose0_by_id)
            except Exception as e:  # a crashing batch must not lose the report
                print(f"[parity] {kind} batch {b + 1}: EXCEPTION {type(e).__name__}: {e}")
                traceback.print_exc()
                rows = {i: {"door_id": i, "sim": "physx", "kind": kind, "engine": ENGINE, "protocol_version": P.PROTOCOL_VERSION, "load_error": f"batch exception: {type(e).__name__}: {e}", "ok": False, "phases": {}} for i in batch}
            doors.update(rows)
            meta = {"protocol_version": P.PROTOCOL_VERSION, "sim": "physx", "kind": kind, "engine": ENGINE, "dt": DT, "sample_hz": P.SAMPLE_HZ, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_doors": len(doors), "options": {"emulate_weld": args_cli.emulate_weld, "servo": not args_cli.no_servo, "inputs": args_cli.inputs, "batch": args_cli.batch,
                                                       "spacing_m": [X_SPACING, Y_SPACING], "latch_mode": args_cli.latch_mode, "grouped": not args_cli.no_group}}
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"meta": meta, "doors": doors}, f)
            os.replace(tmp, out)
            n_ok = sum(1 for r in rows.values() if r.get("ok"))
            print(f"[parity] {kind} batch {b + 1}/{n_batches} done in {time.time() - tb:.0f} s: {n_ok}/{len(rows)} ok -> {out}", flush=True)
        n_ok = sum(1 for r in doors.values() if r.get("ok"))
        print(f"[parity] {kind}: {n_ok}/{len(doors)} doors pass every applicable phase ({time.time() - t0:.0f} s total)")
        hist = {}
        for r in doors.values():
            if r.get("load_error"):
                hist.setdefault("load_error", []).append(r["door_id"])
            for p, row in r.get("phases", {}).items():
                if row.get("status") == "fail" and not row.get("informational"):
                    hist.setdefault(f"{p}:{row.get('expected')}", []).append(r["door_id"])
        for k, v in sorted(hist.items(), key=lambda kv: -len(kv[1]))[:20]:
            print(f"  x{len(v)} {k}  e.g. {v[:4]}")


if __name__ == "__main__":
    main()
    simulation_app.close()
