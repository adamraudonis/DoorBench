"""Door mechanism "action" term (zero-dimensional): keeps the passive door physics honest every physics step.

Isaac Lab overwrites the PhysX drive targets of every joint with the articulation's position targets each step, so a
closer spring authored as ``drive:targetPosition = springref`` in the USD would lose its preload.  This term restores
the spring targets, couples the latch bolt to the operator (the one-sided tendon of the MJCF), adds the asymmetric
closer damping / backcheck as feed-forward effort, and drives automatic doors when the agent is in sensor range.

NOT EXECUTED ON THIS MACHINE (no NVIDIA GPU).
"""
from __future__ import annotations

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .door_state import get_door_state


class DoorMechanismAction(ActionTerm):
    """action_dim == 0: nothing for the policy to output, ``apply_actions`` runs every physics step."""

    cfg: "DoorMechanismActionCfg"

    def __init__(self, cfg: "DoorMechanismActionCfg", env):
        super().__init__(cfg, env)
        self.state = get_door_state(env, door_name=cfg.asset_name)
        self._empty = torch.zeros(env.num_envs, 0, device=env.device)
        self._effort = torch.zeros(env.num_envs, self.state.num_joints, device=env.device)
        self._dt = env.physics_dt

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def reset(self, env_ids=None):
        # the door labels are reset by the mandatory `reset_door` event term (which runs before the manager resets
        # and snapshots the finished episode); nothing to do here
        return

    def apply_actions(self):
        st = self.state
        door = st.door
        ar = st.arange
        jp, jv = door.data.joint_pos, door.data.joint_vel
        targets = st.spring_target.clone()
        # latch bolt follows the operator (bolt_q >= scale * operator_q): raise the drive target so the spring retracts it
        op_q = jp[ar, st.latch_op_j]
        retract = (st.latch_scale * op_q).clamp_min(0.0) * st.has_latch.float()
        targets[:, st.latch_j] = targets[:, st.latch_j] + retract
        # automatic doors: servo to open while the agent is within sensor range of the door plane (either side).
        # Spring-less servo joints already carry kp / kv / forcerange in their PhysX drive (doorbench:servo_in_drive,
        # rl actuators[*].in_drive) - only the target moves; the others (servo + closer spring) get the servo gains
        # written once (the spring's own drive gains are replaced, as before).
        if bool(st.auto.any()):
            if not st.applied_auto_gains:
                ids = torch.nonzero(st.auto & ~st.auto_in_drive).flatten()
                if len(ids):
                    kp = torch.zeros(len(ids), st.num_joints, device=st.device)
                    kv = torch.zeros(len(ids), st.num_joints, device=st.device)
                    rows = torch.arange(len(ids), device=st.device)
                    kp[rows, st.door_j[ids]] = st.auto_kp[ids]
                    kv[rows, st.door_j[ids]] = st.auto_kv[ids]
                    kp[rows, st.sec_j[ids]] = torch.where(st.has_sec[ids], st.auto_kp[ids], torch.zeros_like(st.auto_kp[ids]))
                    kv[rows, st.sec_j[ids]] = torch.where(st.has_sec[ids], st.auto_kv[ids], torch.zeros_like(st.auto_kv[ids]))
                    # keep the USD gains where we do not override them
                    kp = torch.where(kp > 0, kp, door.data.joint_stiffness[ids])
                    kv = torch.where(kv > 0, kv, door.data.joint_damping[ids])
                    door.write_joint_stiffness_to_sim(kp, env_ids=ids)
                    door.write_joint_damping_to_sim(kv, env_ids=ids)
                st.applied_auto_gains = True
            near = (st.base_w - st.env_origins - st.pass_plane)[:, :2].norm(dim=-1) < st.auto_range
            desired = torch.where(near & st.auto, st.auto_open, torch.zeros_like(st.auto_open))
            step = st.auto_speed * self._dt
            st.auto_target = st.auto_target + (desired - st.auto_target).clamp(-step, step)
            targets[ar, st.door_j] = torch.where(st.auto, st.auto_target, targets[ar, st.door_j])
            sec_t = st.sec_c0 + st.sec_c1 * st.auto_target
            targets[ar, st.sec_j] = torch.where(st.auto & st.has_sec, sec_t, targets[ar, st.sec_j])
        door.set_joint_position_target(targets)
        # asymmetric closer damping (+ backcheck): feed-forward effort on the door joint
        v = jv[ar, st.door_j]
        q = jp[ar, st.door_j]
        b_dir = torch.where(v < 0, st.b_close, st.b_open)
        b_dir = b_dir + torch.where((v > 0) & (q > st.backcheck_angle), st.backcheck_damping, torch.zeros_like(b_dir))
        extra = torch.where(st.has_closer, -(b_dir - st.b_base) * v, torch.zeros_like(v))
        # rising / helical hinge: the riser is locked in door_rl.usda; MuJoCo's rise coupling costs m g dz per radian
        # of opening, i.e. a constant closing torque on the door joint (0 for every other door)
        extra = extra + st.rise_torque
        self._effort.zero_()
        self._effort[ar, st.door_j] = extra
        door.set_joint_effort_target(self._effort)


@configclass
class DoorMechanismActionCfg(ActionTermCfg):
    class_type: type = DoorMechanismAction
    asset_name: str = "door"
