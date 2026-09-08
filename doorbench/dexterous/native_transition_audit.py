"""Archive actual MuJoCo dynamics, separately from the integrated endpoint.

Call before_step immediately before the one real plant step, and after_step
immediately after it. No forward-dynamics recomputation is performed. The
recorder copies the actual contact solution and its matching pre-integration
body transforms, then refreshes only kinematics for current controller poses.
This is evaluator/teacher infrastructure, never an actor observation adapter.
"""
import mujoco
import numpy as np

from .grasp_verification import native_grasp_sample
from .hand_surface_audit import native_hand_surface_loads
from .native_warning_audit import warning_counts, warning_interval


def _joint_state(sim,qpos):
    m=sim.m;js=np.asarray(sim.joints);q=qpos[m.jnt_qposadr[js]]
    limited=m.jnt_limited[js].astype(bool)
    violations=np.maximum(m.jnt_range[js,0]-q,q-m.jnt_range[js,1])
    loops={}
    for side in ('rh','lh'):
        for digit in ('FF','MF','RF','LF'):
            a=m.joint(f'robot/{side}_{digit}J1').id
            b=m.joint(f'robot/{side}_{digit}J2').id
            loops[f'{side}_{digit}']=float(qpos[m.jnt_qposadr[a]]-qpos[m.jnt_qposadr[b]])
    return dict(max_joint_limit_violation_rad=max(0.,float(violations[limited].max(initial=0.))),
                shadow_loopback_differences_rad=loops,
                max_shadow_loopback_violation_rad=max([0.,*loops.values()]))


def _contact_solution(sim):
    m,d=sim.m,sim.d
    loads={m.body(b).name.removeprefix('robot/'):np.zeros(3) for b in range(m.nbody)
           if m.body(b).name.startswith(('robot/rh_','robot/lh_'))}
    contacts=[];bodies=set()
    for index,c in enumerate(d.contact[:d.ncon]):
        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,index,wrench)
        frame=c.frame.reshape(3,3).copy();world_force=frame.T@wrench[:3]
        pair=[int(m.geom_bodyid[g]) for g in c.geom];bodies.update(pair)
        for sign,body in zip((-1,1),pair):
            name=m.body(body).name.removeprefix('robot/')
            if name in loads:loads[name]+=sign*world_force
        contacts.append(dict(geom=[int(g) for g in c.geom],body=pair,
            position_world_m=c.pos.copy().tolist(),frame_world=frame.tolist(),
            wrench_contact_frame=wrench.tolist(),distance_m=float(c.dist)))
    ids=sorted(bodies)
    return loads,dict(contacts=contacts,body_ids=ids,
        body_positions_world_m=d.xpos[ids].copy().tolist(),
        body_rotations_world=d.xmat[ids].copy().reshape((-1,3,3)).tolist())


class NativeTransitionRecorder:
    def __init__(self,sim,lever_geom,*,handle_joint,profile='distal-pad-v1'):
        if sim.m.opt.integrator==mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError('RK4 intermediate force epochs require a different recorder')
        self.sim=sim;self.lever_geom=lever_geom;self.handle_joint=handle_joint
        self.profile=profile;self.pending=None
        self.hand_forces,_=_contact_solution(sim)
        self.left_surface=native_hand_surface_loads(sim.m,sim.d)
        self.contact_time_s=float(sim.d.time)
        self.contact_interval_end_s=float(sim.d.time)

    def before_step(self):
        if self.pending is not None:raise ValueError('Unconsumed physical step snapshot')
        d=self.sim.d
        self.pending=dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),
            controls=d.ctrl.copy(),diagnostics=self.sim.diagnostics().copy(),
            body_positions=d.xpos.copy(),body_rotations=d.xmat.copy(),
            body_wrench_max=float(np.max(np.abs(d.xfrc_applied))),
            warning_counts=warning_counts(d))

    def after_step(self):
        if self.pending is None:raise ValueError('Missing pre-step state snapshot')
        sim=self.sim;m,d=sim.m,sim.d;before=self.pending
        end=float(d.time);dt=end-before['time']
        if abs(dt-m.opt.timestep)>1e-8:raise ValueError('Expected exactly one unchanged plant step')
        if not np.allclose(d.xpos,before['body_positions'],atol=1e-9,rtol=0) or not np.allclose(d.xmat,before['body_rotations'],atol=1e-9,rtol=0):
            raise ValueError('Actual contact geometry does not match the saved pre-integration pose')
        # No FK or dynamics call may precede these copies. These are the forces,
        # contact frames and body transforms actually used by this mj_step.
        self.hand_forces,raw=_contact_solution(sim)
        self.left_surface=native_hand_surface_loads(m,d)
        sample=native_grasp_sample(sim,self.lever_geom,handle_joint=self.handle_joint,
            pre_step_external_wrench_max=before['body_wrench_max'],profile=self.profile)
        # The legacy helper reads integrated q. Replace its state-derived fields
        # with the saved pre-integration state matching the actual contact frame.
        for key in ('sim_time_s','door_q','root_height_m','torso_tilt_deg','finite'):
            sample[key]=before['diagnostics'][key]
        sample.update(_joint_state(sim,before['qpos']))
        sample['handle_angle_rad']=float(before['qpos'][m.jnt_qposadr[m.joint(self.handle_joint).id]])
        sample['left_surface_audit']=self.left_surface
        pre_fields=('sim_time_s','door_q','root_height_m','torso_tilt_deg','finite',
            'max_joint_limit_violation_rad','max_shadow_loopback_violation_rad',
            'shadow_loopback_differences_rad','handle_angle_rad')
        pre_state={key:sample[key] for key in pre_fields}
        raw.update(interval_start_s=before['time'],interval_end_s=end,
            geometry_time_s=before['time'],qpos_before=before['qpos'].tolist(),
            qvel_before=before['qvel'].tolist(),qpos_after=d.qpos.copy().tolist(),
            qvel_after=d.qvel.copy().tolist(),controls=before['controls'].tolist(),
            actuator_force=d.actuator_force.copy().tolist(),
            mujoco_warning_interval=warning_interval(before['warning_counts'],warning_counts(d)))
        q=d.qpos.copy();v=d.qvel.copy();force=d.actuator_force.copy()
        mujoco.mj_kinematics(m,d)
        if d.time!=end or not np.array_equal(q,d.qpos) or not np.array_equal(v,d.qvel) or not np.array_equal(force,d.actuator_force):
            raise AssertionError('Kinematics refresh changed integrated state or physical motor force')
        # Endpoint state checks are independent of the earlier contact solution.
        row=sample.copy();row.update(sim.diagnostics());row.update(_joint_state(sim,d.qpos))
        row['handle_angle_rad']=float(d.qpos[m.jnt_qposadr[m.joint(self.handle_joint).id]])
        row['external_wrench_max']=sample['external_wrench_max']
        row['applied_generalized_force_max']=sample['applied_generalized_force_max']
        row.update(pre_integration_state=pre_state,joint_state_time_s=end,
            measurement_pose_time_s=end,contact_geometry_time_s=before['time'],
            contact_interval_start_s=before['time'],contact_interval_end_s=end,
            contact_force_source='actual_mj_step_dynamics',
            body_pose_refresh='mj_kinematics_only',pre_integration_body_poses_match=True)
        row['mujoco_warning_interval']=raw['mujoco_warning_interval']
        self.contact_time_s=before['time'];self.contact_interval_end_s=end
        self.pending=None
        return row,raw
