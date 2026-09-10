"""Source-bound, unstepped upright panel targets after an actual hand release."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .screened_panel_path import ScreenedPanelPath, MeasuredAperturePhase
from .screened_panel_teacher import validate_attained_panel_state


class StandingPanelReference:
    def __init__(self,scene,plan_path,left,*,fixed_foot_targets=False):
        if type(fixed_foot_targets) is not bool:raise ValueError('Explicit boolean fixed foot targets required')
        self.fixed_foot_targets=fixed_foot_targets
        self.m=scene.m;self.d=mujoco.MjData(self.m);self.left=left
        with open(plan_path) as f:self.plan=json.load(f)
        p=self.plan;r=p['screen_receipt']
        if p.get('schema')!='doorbench.whole-body-panel-plan.v1' or not r.get('passed') or r['samples']<2001 or not r['exact_initial_state']:
            raise ValueError('Require a dense qualified attained panel plan')
        if r['limits']['torso_tilt_deg']>4:raise ValueError('Upright continuation requires the four-degree planning bound')
        self.initial=np.asarray(p['initial_qpos']);self.rq=p['root_qpos_address'];self.initial_root=self.initial[self.rq:self.rq+7]
        self.rotation=Rotation.from_quat(self.initial_root[[4,5,6,3]])
        self.path=ScreenedPanelPath(p['progress'],p['coordinates'],p['duration_s'])
        e=r['reference_phase_envelope']
        if not e['passed'] or not 0<e['aperture_speed_limit_rad_s']<=.149 or not 0<e['aperture_acceleration_limit_rad_s2']<=.08:raise ValueError('Require audited bounded phase rates')
        self.phase=MeasuredAperturePhase(p['initial_leaf_angle_rad'],p['final_leaf_angle_rad'],p['initial_leaf_velocity_rad_s'],maximum_speed=e['aperture_speed_limit_rad_s'],maximum_acceleration=e['aperture_acceleration_limit_rad_s2'])
        self.start_time=float(p['initial_episode_time_s']);self.started=None
        self.lh=self.m.site('robot/lh_palm_touch').id;self.rh=self.m.site('robot/rh_palm_touch').id;self.leaf=self.m.body('leaf').id;self.lq=self.m.joint('leaf_hinge').qposadr[0]

    def update(self,t,root,joints,angle,stance):
        p=self.plan
        if self.started is None:
            validate_attained_panel_state(p,self.initial_root,root,joints,angle)
            if self.fixed_foot_targets:stance.freeze_foot_targets()
            self.started=t
            self.root_bias=stance.target_root.copy()-self.initial_root[:3]
            self.rotation_bias=stance.target_rotation@self.rotation.as_matrix().T
            self.stance_names=[self.left.teacher.m.joint(int(j)).name for j in stance.joints]
            self.joint_bias=stance.joint_target.copy()-np.array([joints[n] for n in self.stance_names])
            self.offset=self.left.offset
        reference,velocity,acceleration=self.phase.update(t,angle)
        span=p['final_leaf_angle_rad']-p['initial_leaf_angle_rad'];s=np.clip((reference-p['initial_leaf_angle_rad'])/span,0,1)
        q=self.path.spline(s);dq=self.path.spline(s,1)*velocity/span;ddq=self.path.spline(s,2)*(velocity/span)**2+self.path.spline(s,1)*acceleration/span
        if max(abs(dq[6:]))>1.2 or max(abs(ddq[6:]))>3 or np.linalg.norm(dq[:3])>.02 or np.linalg.norm(dq[3:6])>.03:raise ValueError('Panel reference exceeded audited motion limits')
        targets={**p['initial_robot_joints'],**dict(zip(p['joint_names'],q[6:]))}
        position=self.initial_root[:3]+q[:3];rotation=(Rotation.from_rotvec(q[3:6])*self.rotation).as_matrix()
        stance.target_root[:]=position+self.root_bias;stance.target_rotation=self.rotation_bias@rotation
        stance.joint_target[:]=np.array([targets[n] for n in self.stance_names])+self.joint_bias
        d=self.d;m=self.m;d.qpos[:]=self.initial;d.qpos[self.rq:self.rq+3]=position
        quat=Rotation.from_matrix(rotation).as_quat();d.qpos[self.rq+3:self.rq+7]=quat[[3,0,1,2]]
        for name,value in targets.items():d.qpos[m.joint('robot/'+name).qposadr[0]]=value
        d.qpos[self.lq]=reference;mujoco.mj_kinematics(m,d)
        lr=d.xmat[self.leaf].reshape(3,3);local=lr.T@(d.site_xpos[self.lh]-d.xpos[self.leaf]);local[1]-=self.offset
        self.left.path[-1]['position']=local;self.left.path[-1]['nominal']=np.array([targets[n] for n in self.left.names])
        self.left.panel_palm_rotation=lr.T@d.site_xmat[self.lh].reshape(3,3)
        return targets,d.site_xpos[self.rh].copy(),d.site_xmat[self.rh].reshape(3,3).copy(),dict(panel_fixed_foot_targets=bool(stance.fixed_foot_rotations is not None),panel_started_s=self.started,panel_progress=float(s),panel_reference_aperture_rad=float(reference),panel_measured_aperture_rad=float(angle),panel_maximum_joint_speed_rad_s=float(max(abs(dq[6:]))),panel_maximum_joint_acceleration_rad_s2=float(max(abs(ddq[6:]))))
