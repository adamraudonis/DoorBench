"""Opt-in native panel target controller for an independently screened state.

No active plant handle is accepted. Outputs are existing arm and landed-foot
stance targets, consumed by the unchanged original motor force interfaces.
This experimental path retains historical failed prefix evidence; it does not
qualify a release, support, or opening merely by reaching its reference end.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .screened_panel_path import MeasuredAperturePhase, ScreenedPanelPath


def validate_attained_panel_state(plan, initial_root, root, joints, angle):
    """Reject a plan from another attained body/hand state, including fingers."""
    root=np.asarray(root,float);expected=plan['initial_robot_joints']
    if root.shape!=(13,) or set(expected)!=set(joints):
        raise ValueError('Require complete current root and scalar joint observations')
    if not np.isfinite(np.r_[root,list(joints.values()),angle]).all():
        raise ValueError('Require finite current actual panel observations')
    if not np.allclose(root[:7],initial_root,atol=1e-5,rtol=0):
        raise ValueError('Require this panel screen’s exact attained root')
    if not np.allclose([joints[n] for n in expected],list(expected.values()),atol=1e-5,rtol=0):
        raise ValueError('Require this panel screen’s exact complete attained joint state')
    if abs(angle-plan['initial_leaf_angle_rad'])>1e-5:
        raise ValueError('Require this panel screen’s measured starting aperture')


def validate_tracking_lead_receipt(plan, receipt, lead, start):
    """Require the independent shifted-geometry screen for an increased lead."""
    if lead<=.005:return
    if (not isinstance(receipt,dict) or receipt.get('passed') is not True
            or receipt.get('samples',0)<2001
            or receipt.get('screen_sha256')!=plan['screen_receipt']['screen_sha256']
            or not np.isfinite(receipt.get('actual_leaf_lag_rad',np.nan))
            or receipt['actual_leaf_lag_rad']<lead):
        raise ValueError('Require a passed shifted-geometry audit of this exact screened path')
    audit_start=receipt.get('lag_start_angle_rad')
    if audit_start is not None and (not np.isfinite(audit_start) or start is None or audit_start>start):
        raise ValueError('The shifted-geometry audit starts later than the declared lead change')


class ScreenedWholeBodyPanel:
    def __init__(self,left,path,*,normal_feedforward_N=3.5,tracking_lead_rad=.005,lead_start_angle=None,lead_ramp_rad=.1,lead_receipt=None,**legacy_options):
        if type(normal_feedforward_N) not in (int,float) or not np.isfinite(normal_feedforward_N) or not 0 < normal_feedforward_N <= 8.:
            raise ValueError('Require a finite declared normal feedforward in (0,8] N; original motor caps remain unchanged')
        self.normal_feedforward_N=float(normal_feedforward_N)
        self.left=left;self.teacher=left.teacher
        self.plan=json.loads(Path(path).read_text());plan=self.plan
        if plan.get('schema')!='doorbench.whole-body-panel-plan.v1' or plan['screen_receipt'].get('passed') is not True:
            raise ValueError('Require the independently passed dense actual-state panel screen')
        if plan['screen_receipt']['samples']<2001 or not plan['screen_receipt']['exact_initial_state']:
            raise ValueError('Require dense screening including the exact actual state')
        self.path=ScreenedPanelPath(plan['progress'],plan['coordinates'],plan['duration_s'])
        self.delta=plan['final_leaf_angle_rad']-plan['initial_leaf_angle_rad']
        self.phase=MeasuredAperturePhase(plan['initial_leaf_angle_rad'],plan['final_leaf_angle_rad'],plan['initial_leaf_velocity_rad_s'],tracking_lead=tracking_lead_rad,lead_start_angle=lead_start_angle,lead_ramp_rad=lead_ramp_rad)
        receipt=None if lead_receipt is None else json.loads(Path(lead_receipt).read_text())
        validate_tracking_lead_receipt(plan,receipt,tracking_lead_rad,lead_start_angle)
        self.names=plan['joint_names'];self.indices=[self.teacher.names.index(n) for n in self.names]
        self.qa=np.array([self.teacher.m.jnt_qposadr[self.teacher.m.joint(n).id] for n in self.names])
        self.initial=np.array(plan['initial_qpos']);rq=plan['root_qpos_address']
        self.initial_root=self.initial[rq:rq+7]
        self.root_rotation=Rotation.from_quat([*self.initial_root[4:],self.initial_root[3]])
        self.reference_data=mujoco.MjData(self.teacher.m)
        self.started=None;self.last_update=None;self.clear=False;self.previous=None
        self.loaded_since=None;self.latest=None
        self.legacy_options=dict(legacy_options)

    def begin(self,t,root,joints,leaf_pose,angle):
        if self.started is not None:raise ValueError('Panel already started')
        validate_attained_panel_state(self.plan,self.initial_root,root,joints,angle)
        if self.left.started is None or self.left.progress<.999:
            raise ValueError('Require the completed actual left contact approach')
        self.started=float(t);self.clear=True
        self.advance(t,angle)
        self.teacher.arm_joints=np.array([],int);self.teacher.arm_q=np.array([],int);self.teacher.arm_v=np.array([],int)
        self.left.contact_force=self.normal_feedforward_N
        for name in ('lh_LFJ5','rh_LFJ5'):
            self.teacher.path[-1,self.teacher.names.index(name)]=joints[name]
        if hasattr(self.left,'hybrid_normal_target'):
            raise ValueError('This declared position-path comparison must not inherit a hybrid projection')

    def advance(self,t,angle):
        if self.started is None:return
        reference,velocity,acceleration=self.phase.update(t,angle)
        s=float(np.clip((reference-self.plan['initial_leaf_angle_rad'])/self.delta,0,1))
        coordinate=self.path.spline(s)
        derivative=self.path.spline(s,1)*velocity/self.delta
        second=self.path.spline(s,2)*(velocity/self.delta)**2+self.path.spline(s,1)*acceleration/self.delta
        if (max(abs(derivative[6:]))>1.2 or max(abs(second[6:]))>3.
                or np.linalg.norm(derivative[:3])>.02 or np.linalg.norm(derivative[3:6])>.03):
            raise ValueError('The consumed panel reference exceeded its declared motion envelope')
        self.latest=dict(time_s=float(t),coordinate=coordinate,velocity=derivative,
                         acceleration=second,aperture=reference,progress=s)

    def body_goal(self,t):
        if self.latest is None or abs(self.latest['time_s']-t)>1e-8:
            raise ValueError('Body and hand targets require the same current panel clock')
        x=self.latest['coordinate']
        return dict(position=self.initial_root[:3]+x[:3],rotation=(Rotation.from_rotvec(x[3:6])*self.root_rotation).as_matrix(),joints=dict(zip(self.names,x[6:])),progress=self.latest['progress'])

    def update(self,t,root,joints,leaf_pose,palm_load,angle,right_clear=True):
        if self.started is None or right_clear is not True:
            raise ValueError('Require actual complete release before screened panel control')
        if not np.isfinite(np.r_[t,palm_load,angle,leaf_pose]).all() or palm_load<0:
            raise ValueError('Require finite current physical measurements')
        self.advance(t,angle);self.last_update=float(t)
        x=self.latest['coordinate'];self.previous=x[6:].copy()
        self.teacher.path[-1,self.indices]=self.previous
        left_indices=[self.names.index(n) for n in self.left.names[1:]]
        self.left.target=self.previous[left_indices].copy()
        self.left.target_velocity=self.latest['velocity'][6:][left_indices].copy()
        self.left.normal=Rotation.from_quat([*leaf_pose[4:7],leaf_pose[3]]).as_matrix()[:,1]
        self.left.progress=1.
        self.left._read(root,joints)
        # FK of the screened desired pose uses a separate unstepped data object.
        d=self.reference_data;m=self.teacher.m;goal=self.body_goal(t)
        d.qpos[:3]=goal['position'];quat=Rotation.from_matrix(goal['rotation']).as_quat();d.qpos[3:7]=np.r_[quat[3],quat[:3]]
        for name,value in self.plan['initial_robot_joints'].items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        d.qpos[self.qa]=self.previous;mujoco.mj_kinematics(m,d)
        position_error=float(np.linalg.norm(self.left.d.site_xpos[self.left.palm]-d.site_xpos[self.left.palm]))
        if palm_load>=2.:
            if self.loaded_since is None:self.loaded_since=float(t)
        else:self.loaded_since=None
        self.left.info=dict(phase='screened_whole_body_panel',left_progress=1.,left_tracking_error_m=position_error,left_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(d.site_xmat[self.left.palm].reshape(3,3)@self.left.d.site_xmat[self.left.palm].reshape(3,3).T).as_rotvec())),left_panel_load_N=float(palm_load),left_offset_m=0.,left_ik_residual=position_error*100.,left_loaded_duration_s=0. if self.loaded_since is None else t-self.loaded_since,right_release_clear=True,reference_aperture_rad=self.latest['aperture'],measured_aperture_rad=float(angle),target_maximum_joint_speed_rad_s=float(max(abs(self.latest['velocity'][6:]))),target_maximum_joint_acceleration_rad_s2=float(max(abs(self.latest['acceleration'][6:]))),target_root_speed_m_s=float(np.linalg.norm(self.latest['velocity'][:3])),target_root_rotvec_speed_rad_s=float(np.linalg.norm(self.latest['velocity'][3:6])),target_time_s=float(t),normal_feedforward_N=self.normal_feedforward_N,tracking_lead_rad=self.phase.lead_at(angle),profile='screened-position-v1; original doubled damping and cup feedback retained; no hybrid normal projection')
