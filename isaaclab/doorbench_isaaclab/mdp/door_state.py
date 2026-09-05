"""Per-environment door metadata + benchmark event tracking for the DoorBench Isaac Lab tasks.

Every environment may hold a *different* door (``MultiUsdFileCfg``).  After the scene is created, ``DoorState`` reads
the ``doorbench:rl`` JSON attribute that ``doorbench/export/usd.py`` wrote on each spawned door prim and builds
per-env tensors (which canonical joint is the door joint, thresholds, grip points, pass plane, damage limits ...).
Each policy step ``update()`` derives the benchmark labels of ``doorbench/benchmark/labels.py`` from PhysX state:

  touched_door / touched_operator   contact (or proximity) between the agent and the leaf / operator link
  operator_actuated                 operator joint >= 70 % of its travel
  latch_released                    latch bolt >= 80 % retracted (or no latch)
  door_opened / door_open_clear     |door joint| >= open / clearance threshold (10 deg / 60 deg, 0.10 m / 0.55 m)
  robot_passed_through              agent base crosses the door plane inside the opening
  door_closed_after                 back within the closed threshold after passing
  door_slammed                      closing speed at the stop above the spec's slam velocity
  door_damaged                      agent contact force on the leaf above the spec's dent / glass threshold

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU): written against the Isaac Lab 2.3 API, syntax-checked only.
"""
from __future__ import annotations

import json
import re

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply

RL_JOINTS = ("door_slide", "door_hinge", "operator_hinge", "operator_slide", "latch_slide", "leaf2_slide", "leaf2_hinge")
EVENTS = ("touched_door", "touched_operator", "operator_actuated", "latch_released", "door_opened", "door_open_clear",
          "robot_passed_through", "door_closed_after", "door_slammed", "door_damaged")
TASK_IDS = {"open_and_traverse": 0, "open_only": 1, "traverse_open": 2, "close": 3, "unlock_open_traverse": 4, "locked_recognize": 5,
            "push_through": 6, "hold_and_pass": 7, "peek": 8}
START_OPEN_TASKS = ("traverse_open", "close")


def get_door_state(env, **kwargs) -> "DoorState":
    """Lazily create (and cache on the env) the DoorState shared by all mdp terms."""
    st = getattr(env, "_doorbench_state", None)
    if st is None:
        st = DoorState(env, **kwargs)
        env._doorbench_state = st
    return st


class DoorState:
    def __init__(self, env, door_name: str = "door", agent_name: str | None = None, tip_body_candidates=None, base_body_candidates=None,
                 leaf_sensor: str = "contact_leaf", operator_sensor: str = "contact_operator", touch_distance: float = 0.10):
        self.env = env
        self.device = env.device
        self.N = env.num_envs
        self.door: Articulation = env.scene[door_name]
        self.door_name = door_name
        # ---- agent (hand articulation or humanoid)
        self.agent_name = agent_name or ("hand" if "hand" in env.scene.keys() else "robot")
        self.agent: Articulation = env.scene[self.agent_name]
        self.tip_ids = self._resolve_bodies(self.agent, tip_body_candidates or ["palm", ".*_palm_link", ".*_hand_link", ".*wrist.*", ".*_elbow_roll_link", ".*_elbow_.*"])
        self.base_ids = self._resolve_bodies(self.agent, base_body_candidates or ["palm", "pelvis", "torso_link", "base"])[:1]
        self.leaf_sensor = env.scene[leaf_sensor] if leaf_sensor in env.scene.keys() else None
        self.operator_sensor = env.scene[operator_sensor] if operator_sensor in env.scene.keys() else None
        self.touch_distance = touch_distance
        # ---- canonical joint / link indices (identical for every door)
        jn = self.door.joint_names
        self.j = {n: jn.index(n) for n in RL_JOINTS}
        bn = self.door.body_names
        self.b = {n: bn.index(n) for n in ("leaf", "operator", "latch", "leaf2", "carriage")}
        self.num_joints = len(jn)
        # ---- per-env metadata from the spawned prims
        metas = self._read_metas()
        self.metas = metas
        self.door_ids = [m["door_id"] for m in metas]
        from ..doors import require_eligible_ids
        require_eligible_ids(self.door_ids)
        N, dev = self.N, self.device
        f = lambda key, default=0.0: torch.tensor([float(key(m) if key(m) is not None else default) for m in metas], device=dev)
        b = lambda key: torch.tensor([bool(key(m)) for m in metas], device=dev)
        i = lambda key: torch.tensor([int(key(m)) for m in metas], device=dev, dtype=torch.long)
        self.is_hinge = b(lambda m: m["slots"]["door"] == "hinge")
        self.door_j = torch.where(self.is_hinge, torch.full((N,), self.j["door_hinge"], device=dev), torch.full((N,), self.j["door_slide"], device=dev))
        self.has_op = b(lambda m: m["slots"]["operator"] != "none")
        self.op_j = i(lambda m: self.j["operator_hinge"] if m["slots"]["operator"] == "hinge" else self.j["operator_slide"])
        self.has_latch = b(lambda m: m["slots"]["latch"] != "none")
        self.latch_j = self.j["latch_slide"]
        self.has_sec = b(lambda m: m["slots"]["secondary"] != "none")
        self.sec_j = i(lambda m: self.j["leaf2_hinge"] if m["slots"]["secondary"] == "hinge" else self.j["leaf2_slide"])
        self.open_thr = f(lambda m: m["open_threshold"])
        self.clear_thr = f(lambda m: m["clear_threshold"])
        self.closed_thr = f(lambda m: m["closed_threshold"])
        self.is_push = b(lambda m: m.get("robot", {}).get("is_push", True))
        self.lock_engaged = b(lambda m: m.get("lock", {}).get("engaged", False))
        self.task_id = i(lambda m: TASK_IDS.get(m.get("task"), 0))
        self.start_open = b(lambda m: m.get("task") in START_OPEN_TASKS)
        self.opening_w = f(lambda m: m.get("opening", {}).get("width"), 0.9)
        self.opening_h = f(lambda m: m.get("opening", {}).get("height"), 2.0)
        self.dent_force = f(lambda m: m.get("damage", {}).get("leaf_dent_force_N"), 1e9)
        self.slam_vel = f(lambda m: m.get("damage", {}).get("slam_velocity_rad_s"), 4.0)
        self.op_yield = f(lambda m: m.get("damage", {}).get("operator_yield_torque_Nm"), 1e9)
        # joint ranges / spring targets per canonical joint (N, J)
        self.q_lo = torch.zeros(N, self.num_joints, device=dev)
        self.q_hi = torch.zeros(N, self.num_joints, device=dev)
        self.spring_target = torch.zeros(N, self.num_joints, device=dev)
        self.b_close = torch.zeros(N, device=dev)
        self.b_open = torch.zeros(N, device=dev)
        self.b_base = torch.zeros(N, device=dev)
        self.backcheck_angle = torch.full((N,), 1e9, device=dev)
        self.backcheck_damping = torch.zeros(N, device=dev)
        for k, m in enumerate(metas):
            for name, info in m["joints"].items():
                jidx = self.j[name]
                self.q_lo[k, jidx], self.q_hi[k, jidx] = float(info["range"][0]), float(info["range"][1])
                self.spring_target[k, jidx] = float(info.get("target", 0.0) or 0.0)
            dj = m["joints"][m["door_joint"]]
            if dj.get("damping_closing") is not None:
                self.b_close[k] = float(dj["damping_closing"])
                self.b_open[k] = float(dj.get("damping_opening") or 0.0)
                self.b_base[k] = float(dj.get("damping") or 0.0)
            if dj.get("backcheck_angle") is not None:
                self.backcheck_angle[k] = float(dj["backcheck_angle"])
                self.backcheck_damping[k] = float(dj.get("backcheck_damping") or 0.0)
        self.q_range = (self.q_hi - self.q_lo).clamp_min(1e-6)
        self.has_closer = self.b_close > 0
        # latch coupling: bolt target = spring_target + scale * operator_q  (one-sided tendon in MuJoCo)
        self.latch_scale = f(lambda m: (m.get("latch_coupling") or {}).get("scale"), 0.0)
        self.latch_op_j = i(lambda m: self.j.get((m.get("latch_coupling") or {}).get("operator_joint", "operator_hinge"), self.j["operator_hinge"]))
        # secondary coupling (bi-parting automatic sliders, dutch joining bolts): q_sec = c0 + c1 * q_door
        self.sec_c0 = f(lambda m: (m.get("secondary_coupling") or {}).get("coeff", [0.0, 0.0])[0])
        self.sec_c1 = f(lambda m: (m.get("secondary_coupling") or {}).get("coeff", [0.0, 0.0])[1])
        self.sec_driven = b(lambda m: (m.get("secondary_coupling") or {}).get("driven") == "secondary")
        # automatic doors (position servo in the spec): kp, kv, open target, sensor range.  ``in_drive``: the exporter
        # folded the servo into the canonical joint's PhysX drive (spring-less servo joints) - the gains are already in
        # the USD and only the position target moves; otherwise DoorMechanismAction writes kp / kv into the sim once.
        self.auto = b(lambda m: bool(m.get("actuators")))
        self.auto_kp = f(lambda m: (m.get("actuators") or [{}])[0].get("kp"), 0.0)
        self.auto_kv = f(lambda m: (m.get("actuators") or [{}])[0].get("kv"), 0.0)
        self.auto_in_drive = b(lambda m: bool(m.get("actuators")) and all(a.get("in_drive") for a in m["actuators"]))
        # rising / helical hinge (cold_storage, stall): the riser is locked in door_rl.usda, MuJoCo's rise coupling
        # costs m g dz per radian of opening -> constant closing torque on the door joint (usd.py rise_coupling_info)
        self.rise_torque = f(lambda m: (m.get("rise_coupling") or {}).get("gravity_torque_Nm"), 0.0)
        # Coulomb joint friction: make sure PhysX holds the authored efforts (Isaac Sim >= 5 exposes the static effort
        # as joint_friction_coeff; a parser that dropped the per-axis API would leave every joint frictionless)
        self._ensure_joint_friction(metas)
        # environment-released locks (mag lock / delayed egress / electric bolt / interlock): a real breakable
        # FixedJoint in the USD that this class enables / disables, instead of nothing at all
        self.env_lock_engaged = torch.zeros(N, dtype=torch.bool, device=dev)
        self._collect_env_release(metas)
        self.env_lock_engaged |= self.has_env_release
        # bilateral couplings PhysX drops (prismatic mimic joints); the driver keeps the reaction
        self._collect_couplings(metas)
        self.auto_open = f(lambda m: ((m.get("actuators") or [{}])[0].get("ctrlrange") or [0, 0])[1], 0.0)
        self.auto_speed = torch.where(self.is_hinge, torch.full((N,), 0.6, device=dev), torch.full((N,), 0.3, device=dev))
        self.auto_range = 1.8
        # sites (env-local frame)
        self.approach = torch.tensor([m["sites"]["approach"] for m in metas], device=dev, dtype=torch.float)
        self.goal = torch.tensor([m["sites"]["goal"] for m in metas], device=dev, dtype=torch.float)
        self.pass_plane = torch.tensor([m["sites"]["pass_plane"] for m in metas], device=dev, dtype=torch.float)
        grip_link, grip_local = [], []
        for m in metas:
            g = m["sites"]["grip"]
            if g:
                grip_link.append(self.b.get(g[0]["link"], self.b["leaf"]))
                grip_local.append(g[0]["pos"])
            else:  # no site: a point on the leaf at handle height
                grip_link.append(self.b["leaf"])
                grip_local.append([float(m.get("leaf_edge_x_local") or 0.6) * 0.9, 0.0, float(m.get("handle_height") or 1.0)])
        self.grip_link = torch.tensor(grip_link, device=dev, dtype=torch.long)
        self.grip_local = torch.tensor(grip_local, device=dev, dtype=torch.float)
        self.grip_w = torch.zeros(N, 3, device=dev)
        self.env_origins = env.scene.env_origins
        self.arange = torch.arange(N, device=dev)
        # ---- episode buffers
        self.flags = {e: torch.zeros(N, dtype=torch.bool, device=dev) for e in EVENTS}
        self.new = {e: torch.zeros(N, dtype=torch.bool, device=dev) for e in EVENTS}
        self.prev_side = torch.zeros(N, dtype=torch.long, device=dev)
        self.prev_door_q = torch.zeros(N, device=dev)
        self.max_door_q = torch.zeros(N, device=dev)
        self.time_to = {k: torch.full((N,), -1.0, device=dev) for k in ("touch", "open", "pass")}
        self.door_q = torch.zeros(N, device=dev)
        self.door_dq = torch.zeros(N, device=dev)
        self.op_frac = torch.zeros(N, device=dev)
        self.latch_frac = torch.zeros(N, device=dev)
        self.tip_w = torch.zeros(N, 3, device=dev)
        self.base_w = torch.zeros(N, 3, device=dev)
        self.leaf_force = torch.zeros(N, device=dev)
        self.operator_force = torch.zeros(N, device=dev)
        self.auto_target = torch.zeros(N, device=dev)
        self._last_update = -1
        self.applied_auto_gains = False

    # ------------------------------------------------------------------ helpers
    def _ensure_joint_friction(self, metas: list[dict]):
        """Write the IR Coulomb efforts (static == dynamic, viscous 0) when PhysX reads back something else.

        The USD carries them as ``physxJointAxis:angular|linear:*FrictionEffort``; this is the belt to that braces
        (Isaac Lab >= 2.3 ``write_joint_friction_coefficient_to_sim``).  Silently skipped on APIs that lack it."""
        fr = getattr(self.door.data, "joint_friction_coeff", None)
        if fr is None or not hasattr(self.door, "write_joint_friction_coefficient_to_sim"):
            return
        want = torch.zeros(self.N, self.num_joints, device=self.device)
        for k, m in enumerate(metas):
            for name, info in m.get("joints", {}).items():
                if info.get("active"):
                    want[k, self.j[name]] = float(info.get("friction") or 0.0)
        if bool(((fr - want).abs() > 1e-2 * want.clamp_min(1e-3)).any()):
            # MuJoCo frictionloss = one Coulomb bound for stick and slip -> static == dynamic effort, no viscous term;
            # Isaac Lab 2.3 offers the dynamic / viscous writes either as keyword arguments or as separate methods
            try:
                self.door.write_joint_friction_coefficient_to_sim(want, joint_dynamic_friction_coeff=want.clone(), joint_viscous_friction_coeff=torch.zeros_like(want))
                return
            except TypeError:
                pass
            self.door.write_joint_friction_coefficient_to_sim(want)
            for name, val in (("write_joint_dynamic_friction_coefficient_to_sim", want.clone()), ("write_joint_viscous_friction_coefficient_to_sim", torch.zeros_like(want))):
                fn = getattr(self.door, name, None)
                if fn is not None:
                    fn(val)

    @staticmethod
    def _resolve_bodies(asset: Articulation, candidates):
        for pat in candidates:
            ids = [i for i, n in enumerate(asset.body_names) if re.fullmatch(pat, n)]
            if ids:
                return ids
        return [0]

    def _read_metas(self) -> list[dict]:
        """doorbench:rl JSON of every env's door prim (spawned by UsdFileCfg / MultiUsdFileCfg)."""
        stage = self.env.scene.stage
        prim_expr = self.door.cfg.prim_path  # e.g. /World/envs/env_.*/Door
        metas = []
        self.door_prim_paths = []
        for k, env_path in enumerate(self.env.scene.env_prim_paths):
            path = prim_expr.replace("{ENV_REGEX_NS}", env_path)
            path = re.sub(r"env_\.\*", env_path.split("/")[-1], path)
            prim = stage.GetPrimAtPath(path)
            attr = prim.GetAttribute("doorbench:rl") if prim.IsValid() else None
            if attr is None or not attr.IsValid() or not attr.Get():
                raise RuntimeError(f"door prim {path} has no doorbench:rl attribute (spawn door_rl.usda from DoorBench, regenerate with scripts/generate_dataset.py)")
            self.door_prim_paths.append(path)
            metas.append(json.loads(attr.Get()))
        return metas

    # ------------------------------------------------------------------ environment-released locks
    def _collect_env_release(self, metas: list[dict]):
        """Breakable FixedJoints of environment-released locks (mag lock, delayed egress, electric bolt, interlock).

        ``doorbench/export/usd.py`` authors them base -> leaf with ``physics:excludeFromArticulation`` so PhysX
        solves them as a loop joint, ``physics:breakForce = breakTorque = holding_force_N`` (the leaf breaks free
        under an excessive push, as ``DoorEnv._lock_logic`` does) and ``physics:jointEnabled``, which is what a badge
        / REX press / delayed-egress timeout clears - the counterpart of MuJoCo's ``d.eq_active[eid] = 0``."""
        self.env_release_joints = []          # per env: [(prim, holding_force_N)]
        stage = self.env.scene.stage
        n = 0
        for k, m in enumerate(metas):
            rows = []
            for e in m.get("env_release", []) or []:
                prim = stage.GetPrimAtPath(f"{self.door_prim_paths[k]}/Articulation/Joints/{e['joint_name']}")
                if prim.IsValid():
                    rows.append((prim, float(e.get("holding_force_N") or 0.0)))
                    n += 1
            self.env_release_joints.append(rows)
        self.has_env_release = torch.tensor([bool(r) for r in self.env_release_joints], device=self.device)
        self.n_env_release = n

    def _warn_if_usd_writes_do_not_reach_physx(self):
        """A runtime write to ``physics:jointEnabled`` only reaches PhysX while the USD stage is the source of truth.

        Isaac Lab's ``SimulationCfg.use_fabric`` defaults to True: physics state then lives in Fabric and the USD
        stage is synchronised for rendering only, so setting an attribute on the prim changes nothing in the solver
        and the mag lock would stay engaged for ever - silently, because the attribute reads back the value we
        wrote.  This cannot be tested without Isaac Sim, so it is checked at the moment of use and says so out loud
        instead of failing quietly.  Fixes, in order of preference: toggle the joint through the PhysX SDK, or set
        ``sim.use_fabric = False`` on a task that uses ``mdp.release_env_lock``.
        """
        if getattr(self, "_env_lock_warned", False):
            return
        self._env_lock_warned = True
        fabric = None
        for owner in (getattr(self.env, "sim", None), getattr(self.env, "cfg", None)):
            cfg = getattr(owner, "cfg", owner)
            if cfg is not None and hasattr(cfg, "use_fabric"):
                fabric = bool(cfg.use_fabric)
                break
            sim_cfg = getattr(cfg, "sim", None)
            if sim_cfg is not None and hasattr(sim_cfg, "use_fabric"):
                fabric = bool(sim_cfg.use_fabric)
                break
        if fabric is not False:
            print(f"[doorbench] WARNING: releasing an environment lock writes physics:jointEnabled on the USD prim, "
                  f"and this simulation runs with use_fabric={fabric} (None = could not tell). With Fabric on, PhysX "
                  f"does not read that write and the {self.n_env_release} env-release joint(s) stay engaged. Verify "
                  f"the leaf actually swings after mdp.release_env_lock, or set sim.use_fabric = False.")

    def set_env_lock(self, env_ids=None, enabled: bool = True):
        """Enable / disable the environment-released locks of those envs (``physics:jointEnabled``).

        Disabling is the release: badge presented, REX button pressed, delayed-egress timer expired, interlock
        satisfied.  ``reset_door`` re-enables them so the next episode starts locked again.

        See ``_warn_if_usd_writes_do_not_reach_physx``: whether this write reaches the solver depends on the sim
        running without Fabric, which is the one thing about the env-release joint that no static check can settle.
        """
        if not getattr(self, "n_env_release", 0):
            return
        if not enabled:
            self._warn_if_usd_writes_do_not_reach_physx()
        ids = range(self.N) if env_ids is None else [int(i) for i in env_ids]
        for k in ids:
            for prim, _ in self.env_release_joints[k]:
                attr = prim.GetAttribute("physics:jointEnabled")
                if attr and attr.IsValid():
                    attr.Set(bool(enabled))
        self.env_lock_engaged[list(ids)] = bool(enabled)

    # ------------------------------------------------------------------ bilateral couplings
    def _collect_couplings(self, metas: list[dict]):
        """Couplings between two canonical joints that PhysX cannot represent (``mode == "emulated"``).

        ``PhysxMimicJointAPI`` supports rotational axes only, so a prismatic coupling is dropped silently; the
        exporter writes the coupling law plus the driven DOF's effective inertia, passive law and gravity bias into
        ``doorbench:rl["couplings"]`` instead.  ``DoorMechanismAction`` tracks the driven joint AND puts the reaction
        ``c1 * tau_driven_ext`` on the driver (a pure kinematic write applies no reaction: the driver would lose the
        coupled part's weight and friction).  ``mode == "servo"`` couplings (the two leaves of an automatic
        bi-parting slider) are left to the drives: both joints carry their own MJCF position servo."""
        rows = []
        for k, m in enumerate(metas):
            for c in m.get("couplings", []) or []:
                if c.get("mode") != "emulated":
                    continue
                da, db = c.get("driven_slot"), c.get("driver_slot")
                if da not in self.j or db not in self.j:
                    continue
                # PhysX joint positions are USD coordinates (q_usd = q_db - zero_offset); the exporter precomputes
                # the same law in those coordinates as ``coeff_usd``
                cu = c.get("coeff_usd") or c["coeff"]
                rows.append({"env": k, "ia": self.j[da], "ib": self.j[db], "c0": float(cu[0]), "c1": float(cu[1]),
                             "k": float(c.get("driven_stiffness") or 0.0), "d": float(c.get("driven_damping") or 0.0),
                             "target": float(c.get("driven_target") or 0.0), "friction": float(c.get("driven_friction") or 0.0),
                             "bias": float(c.get("driven_gravity_bias") or 0.0), "eps": float(c.get("friction_vel_eps") or 1e-3),
                             "lo": (c.get("driven_range") or [-1e9, 1e9])[0], "hi": (c.get("driven_range") or [-1e9, 1e9])[1],
                             "order": int(c.get("chain_order") or 0)})
        rows.sort(key=lambda r: r["order"])
        self.couplings = rows
        # the driver of an emulated coupling carries the reflected inertia c1^2 * I_driven, which MuJoCo's equality
        # gives it for free; write it into the driver's armature once
        extra = {}
        for k, m in enumerate(metas):
            for slot, val in (m.get("coupling_reflected_armature") or {}).items():
                if slot in self.j and float(val) > 0:
                    extra[(k, self.j[slot])] = float(val)
        # a driven joint is no longer a free DoF (its state is written every step) and its spring / damping are
        # reflected onto the driver analytically, so switch its own PhysX drive off (the authored gains stay)
        if rows and hasattr(self.door, "write_joint_stiffness_to_sim"):
            k = self.door.data.joint_stiffness.clone()
            d = self.door.data.joint_damping.clone()
            for r in rows:
                k[r["env"], r["ia"]] = 0.0
                d[r["env"], r["ia"]] = 0.0
            self.door.write_joint_stiffness_to_sim(k)
            self.door.write_joint_damping_to_sim(d)
        if extra and hasattr(self.door, "write_joint_armature_to_sim"):
            arm = getattr(self.door.data, "joint_armature", None)
            if arm is None:
                arm = getattr(self.door.data, "default_joint_armature", None)
            if arm is not None:
                want = arm.clone()
                for (k, i), v in extra.items():
                    want[k, i] = want[k, i] + v
                self.door.write_joint_armature_to_sim(want)
        self.coupling_reflected = extra

    # ------------------------------------------------------------------ per-step update
    def update(self):
        env = self.env
        if self._last_update == env.common_step_counter:
            return
        self._last_update = env.common_step_counter
        N, ar = self.N, self.arange
        jp, jv = self.door.data.joint_pos, self.door.data.joint_vel
        self.door_q = jp[ar, self.door_j]
        self.door_dq = jv[ar, self.door_j]
        op_q = jp[ar, self.op_j]
        self.op_frac = torch.where(self.has_op, (op_q - self.q_lo[ar, self.op_j]) / self.q_range[ar, self.op_j], torch.zeros_like(op_q))
        lq = jp[:, self.latch_j]
        self.latch_frac = torch.where(self.has_latch, (lq - self.q_lo[:, self.latch_j]) / self.q_range[:, self.latch_j], torch.ones_like(lq))
        # grip point in world
        bp = self.door.data.body_link_pos_w[ar, self.grip_link]
        bq = self.door.data.body_link_quat_w[ar, self.grip_link]
        self.grip_w = bp + quat_apply(bq, self.grip_local)
        # agent
        tips = self.agent.data.body_link_pos_w[:, self.tip_ids]            # (N, T, 3)
        d = torch.linalg.norm(tips - self.grip_w[:, None, :], dim=-1)      # (N, T)
        self.tip_dist, imin = d.min(dim=1)
        self.tip_w = tips[ar, imin]
        self.base_w = self.agent.data.body_link_pos_w[:, self.base_ids[0]]
        base_local = self.base_w - self.env_origins
        # contacts (filtered by the agent's bodies when the sensors have filters)
        self.leaf_force = self._sensor_force(self.leaf_sensor)
        self.operator_force = self._sensor_force(self.operator_sensor)
        # ---- labels
        L, new = self.flags, self.new
        for e in EVENTS:
            new[e].zero_()
        t = env.episode_length_buf.float() * env.step_dt
        touched_op = (self.operator_force > 0.5) | (self.tip_dist < self.touch_distance)
        touched = touched_op | (self.leaf_force > 0.5)
        self._set("touched_operator", touched_op & self.has_op)
        self._set("touched_door", touched)
        self.time_to["touch"] = torch.where(new["touched_door"], t, self.time_to["touch"])
        self._set("operator_actuated", self.has_op & (self.op_frac >= 0.7))
        self._set("latch_released", (~self.has_latch) | (self.latch_frac >= 0.8))
        aq = self.door_q.abs()
        self.max_door_q = torch.maximum(self.max_door_q, aq)
        self._set("door_opened", aq >= self.open_thr)
        self.time_to["open"] = torch.where(new["door_opened"], t, self.time_to["open"])
        self._set("door_open_clear", aq >= self.clear_thr)
        # pass-through: base crosses the wall plane (y = pass_plane.y) inside the opening width, door open
        side = torch.where(base_local[:, 1] > self.pass_plane[:, 1], torch.ones_like(self.prev_side), -torch.ones_like(self.prev_side))
        crossed = (self.prev_side != 0) & (side != self.prev_side) & ((base_local[:, 0] - self.pass_plane[:, 0]).abs() < self.opening_w / 2 + 0.3) & (base_local[:, 2] < self.opening_h + 0.5)
        self._set("robot_passed_through", crossed & (L["door_opened"] | (self.task_id == TASK_IDS["traverse_open"])))
        self.time_to["pass"] = torch.where(new["robot_passed_through"], t, self.time_to["pass"])
        self.prev_side = side
        self._set("door_closed_after", L["robot_passed_through"] & (aq < self.closed_thr))
        # slam: closing through the closed threshold faster than the spec allows
        closing = (self.prev_door_q.abs() >= self.closed_thr) & (aq < self.closed_thr)
        self._set("door_slammed", closing & (self.door_dq.abs() > self.slam_vel))
        self.prev_door_q = self.door_q.clone()
        # damage: agent contact force on the leaf above the dent / glass threshold
        self._set("door_damaged", self.leaf_force > self.dent_force)

    def _set(self, name, cond):
        cond = cond & ~self.flags[name]
        self.new[name] = cond
        self.flags[name] |= cond

    def _sensor_force(self, sensor):
        if sensor is None:
            return torch.zeros(self.N, device=self.device)
        data = sensor.data
        fm = getattr(data, "force_matrix_w", None)
        if fm is not None and fm.numel() > 0:
            return torch.linalg.norm(fm, dim=-1).flatten(1).max(dim=1).values
        return torch.linalg.norm(data.net_forces_w, dim=-1).flatten(1).max(dim=1).values

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        """Snapshot the finished episode's labels (``last_*``, read by eval_all_doors.py) and clear the buffers.
        Called by the ``reset_door`` event term, which runs before the managers' own resets."""
        if not hasattr(self, "last_flags"):
            self.last_flags = {e: torch.zeros(self.N, dtype=torch.bool, device=self.device) for e in EVENTS}
            self.last_success = torch.zeros(self.N, dtype=torch.bool, device=self.device)
            self.last_time_to = {k: torch.full((self.N,), -1.0, device=self.device) for k in self.time_to}
            self.last_max_door_q = torch.zeros(self.N, device=self.device)
            self.last_length = torch.zeros(self.N, device=self.device)
        succ = self.success()
        self.last_success[env_ids] = succ[env_ids]
        for e in EVENTS:
            self.last_flags[e][env_ids] = self.flags[e][env_ids]
        for k in self.time_to:
            self.last_time_to[k][env_ids] = self.time_to[k][env_ids]
        self.last_max_door_q[env_ids] = self.max_door_q[env_ids]
        self.last_length[env_ids] = self.env.episode_length_buf[env_ids].float() * self.env.step_dt
        for e in EVENTS:
            self.flags[e][env_ids] = False
            self.new[e][env_ids] = False
        self.prev_side[env_ids] = 0
        self.prev_door_q[env_ids] = 0.0
        self.max_door_q[env_ids] = 0.0
        for k in self.time_to:
            self.time_to[k][env_ids] = -1.0
        self.auto_target[env_ids] = 0.0
        self._last_update = -1

    def door_reset_joint_pos(self, env_ids):
        """Initial joint positions: q = 0 (spec initial state) or 80 % open for traverse_open / close tasks."""
        q = torch.zeros(len(env_ids), self.num_joints, device=self.device)
        so = self.start_open[env_ids]
        dj = self.door_j[env_ids]
        rows = torch.arange(len(env_ids), device=self.device)
        open_q = self.q_lo[env_ids, dj] + 0.8 * self.q_range[env_ids, dj]
        q[rows, dj] = torch.where(so, open_q, torch.zeros_like(open_q))
        # a latch that starts open is retracted by the strike (bolt held in) -> leave at 0 (extended) is fine when open
        return q

    def success(self) -> torch.Tensor:
        """Task-specific success predicate (doorbench.benchmark.labels.LabelTracker.finalize)."""
        L, T = self.flags, self.task_id
        nd = ~L["door_damaged"]
        s = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        s |= (T == 0) & L["door_opened"] & L["robot_passed_through"] & nd
        s |= (T == 1) & L["door_open_clear"] & nd
        s |= (T == 2) & L["robot_passed_through"] & ~L["touched_door"] & nd
        s |= (T == 3) & L["door_closed_after"] & nd
        s |= (T == 4) & L["door_opened"] & L["robot_passed_through"] & nd
        s |= (T == 5) & ~L["door_opened"] & nd
        s |= (T == 6) & L["robot_passed_through"] & nd
        s |= (T == 7) & L["door_opened"] & L["robot_passed_through"] & nd & ~L["door_slammed"]
        s |= (T == 8) & L["door_opened"] & (self.max_door_q < self.clear_thr) & ~L["robot_passed_through"] & nd
        return s

    def episode_record(self, k: int, last: bool = False) -> dict:
        """Labels of env k as a JSON-able dict (used by eval_all_doors.py).  ``last=True`` reads the snapshot of the
        episode that just ended (taken by ``reset``), ``False`` the running episode."""
        F, T, S, M = (self.last_flags, self.last_time_to, self.last_success, self.last_max_door_q) if last else (self.flags, self.time_to, self.success(), self.max_door_q)
        return {"door_id": self.door_ids[k], "task": self.metas[k].get("task"), "success": bool(S[k]),
                "events": [e for e in EVENTS if bool(F[e][k])],
                "time_to_touch": float(T["touch"][k]), "time_to_open": float(T["open"][k]), "time_to_pass": float(T["pass"][k]),
                "max_door_q": float(M[k]), "episode_length_s": float(self.last_length[k]) if last else float(self.env.episode_length_buf[k]) * self.env.step_dt}
