"""Privileged force-driven lever return after qualified standing transfer.

A source-bound dense geometric route is followed through the existing stance QP
and original arm motors. No physical-state setters or simulator handles enter.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .operation_teacher import smooth_phase
from .attained_arm_tracking import AttainedArmTracking
from .motor_handoff import MotorHandoff


def validate_return_targets(plan,model):
    root=int(model.joint('robot/free_base').qposadr[0])
    for row in plan['rows']:
        q=np.asarray(row['qpos'],float)
        if q.shape!=(model.nq,) or not np.isfinite(q).all() or row['root_qpos_address']!=root:raise ValueError('Return scene coordinate contract changed')
        if not np.isclose(np.linalg.norm(q[root+3:root+7]),1.,atol=1e-8,rtol=0):raise ValueError('Return root rotation must be normalized')
        for name,value in row['joints'].items():
            if not np.isfinite(value) or abs(q[model.joint('robot/'+name).qposadr[0]]-value)>1e-12:raise ValueError('Consumed return targets differ from audited scene coordinates')
    if not np.allclose(plan['rows'][0]['qpos'],plan['source_state']['qpos'],atol=1e-10,rtol=0):raise ValueError('Return must begin at its recorded attained state')


def load_return_route(path,motors):
    config=json.loads(Path(path).read_text())
    if config.get('schema')!='doorbench.standing-return.v1':raise ValueError('Explicit standing return route required')
    plan_path=Path(config['plan_path']);audit_path=Path(config['audit_path'])
    for p,key in [(plan_path,'plan_sha256'),(audit_path,'audit_sha256')]:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=config[key]:raise ValueError('Return route evidence changed')
    if config['robot_xml_sha256']!=motors['source_xml_sha256']:raise ValueError('Return route robot differs')
    plan=json.loads(plan_path.read_text());audit=json.loads(audit_path.read_text())
    if audit.get('passed') is not True or audit.get('samples')!=401 or audit.get('physics_steps')!=0:raise ValueError('Independent dense return screen required')
    if audit['input_sha256'].get(str(plan_path))!=config['plan_sha256']:raise ValueError('Return audit belongs to a different plan')
    # Bind every original source and the independently authored screen itself.
    for mapping in (plan['source_sha256'],audit['input_sha256']):
        for name,digest in mapping.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Return source bytes changed: '+name)
    from .landed_left_planner import LandedLeftScene
    scene=LandedLeftScene(Path(config['robot_path']),Path(config['door_path']))
    validate_return_targets(plan,scene.m)
    rows=plan['rows']
    if len(rows)!=41 or not np.allclose([r['progress'] for r in rows],np.linspace(0,1,41),atol=1e-12):raise ValueError('Complete monotonic return path required')
    if not np.isfinite(audit['duration_s']) or audit['duration_s']<=0:raise ValueError('Positive audited return duration required')
    return config,plan,audit


class StandingReturnTeacher:
    def __init__(self,transfer,motors,path,*,hold_finger_posture=False,support_load_target=None):
        if not transfer.attained_arm_tracking:raise ValueError('Return requires the qualified attained-arm transfer mode')
        if type(hold_finger_posture) is not bool:raise ValueError('Explicit attained hand posture option required')
        if support_load_target is not None and (not np.isfinite(support_load_target) or not 2<support_load_target<=4):raise ValueError('Return support target must remain above the original 2 N gate and at most 4 N')
        self.support_load_target=support_load_target;self.initial_support_target=transfer.left.support_load_target
        self.hold_finger_posture=hold_finger_posture;self.hand=None
        self.transfer=transfer;self.operation=transfer.operation;self.acquisition=transfer.acquisition
        self.config,self.plan,self.audit=load_return_route(path,motors)
        self.rows=self.plan['rows'];self.names=list(self.rows[0]['joints'])
        self.joints=np.array([[r['joints'][n] for n in self.names] for r in self.rows])
        self.roots=np.array([r['qpos'][r['root_qpos_address']:r['root_qpos_address']+7] for r in self.rows])
        if not np.isfinite(np.r_[self.joints.ravel(),self.roots.ravel()]).all():raise ValueError('Finite return targets required')
        self.rotations=Slerp(np.linspace(0,1,41),Rotation.from_quat(self.roots[:,[4,5,6,3]]))
        self.start_time=float(self.plan['source_state']['time_s']);self.duration=self.audit['duration_s']
        self.return_started=None;self.arm=None;self.handoff=None;self.info={}

    @property
    def started(self):return self.transfer.started

    def force(self,t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,*,grasp_qualified,left_panel_load):
        teacher=self.acquisition
        if self.return_started is None and t>=self.start_time-1e-8:
            if not grasp_qualified or left_panel_load<2 or self.transfer.left.progress<.999:raise ValueError('Loaded opposed grasp and left support required before return')
            if not np.allclose(root[:7],self.roots[0],atol=1e-5,rtol=0) or not np.allclose([joints[n] for n in self.names],self.joints[0],atol=1e-5,rtol=0):raise ValueError('Return requires its exact attained start; no pose replacement')
            self.previous_force=teacher.last_force.copy();self.arm=AttainedArmTracking(teacher,joints,self.previous_force);self.return_started=t
            if self.hold_finger_posture:
                from .attained_hand_tracking import AttainedHandTracking
                self.hand=AttainedHandTracking(teacher,joints,self.previous_force)
        if self.return_started is None:
            force,info=self.transfer.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified,left_panel_load=left_panel_load)
            self.info=info;return force,info
        elapsed=t-self.return_started;u=float(smooth_phase(elapsed/self.duration));coordinate=u*40;i=min(int(coordinate),39);f=coordinate-i
        position=(1-f)*self.roots[i,:3]+f*self.roots[i+1,:3];q=(1-f)*self.joints[i]+f*self.joints[i+1];targets=dict(zip(self.names,q))
        teacher.stance.target_root[:]=position;teacher.stance.target_rotation=self.rotations(u).as_matrix()
        teacher.stance.joint_target[:]=[targets[teacher.m.joint(int(j)).name] for j in teacher.stance.joints]
        if self.support_load_target is not None:self.transfer.left.support_load_target=self.initial_support_target+float(smooth_phase(elapsed))*(self.support_load_target-self.initial_support_target)
        self.transfer.left.update_targets(t,root,joints,leaf_pose,left_panel_load,handle_pose)
        forces,info=self.operation.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified)
        forces=self.transfer.left.apply_forces(forces,joints,velocities)
        forces,arm_info=self.arm.force(forces,t,targets,joints,velocities)
        hand_info={}
        if self.hand:forces,hand_info=self.hand.force(forces,joints,velocities)
        if self.handoff is None:self.handoff=MotorHandoff(self.previous_force,forces,teacher.caps,1.)
        forces=self.handoff.force(forces,elapsed);teacher.last_force=forces.copy()
        info={**info,**self.transfer.left.info,**arm_info,**hand_info,'phase':'standing_lever_return','standing_transfer_started_s':self.transfer.started,'return_started_s':self.return_started,'return_progress':u,'return_goal_operator_rad':self.rows[0]['goal_operator_rad']*(1-u),'return_route_duration_s':self.duration,'controller_scope':'Privileged screened torso/right-arm targets; actual left support and original motors'}
        self.info=info;return forces,info
