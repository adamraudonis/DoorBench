"""Privileged coordinated standing transfer through original robot motors.

Consumes an independently screened, attained-state geometric route. The right
hand operation stays active while the actual feet support a small root shift and
left-palm approach. A geometric plan never substitutes for a physical test.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .bimanual_transfer import LeftPalmContact
from .operation_teacher import smooth_phase
from .transfer_preload import PROFILES, transfer_preload


def validate_route_geometry(config):
    """Bind the consumed numeric targets to the separately audited scene path."""
    import mujoco
    from .landed_left_planner import LandedLeftScene,JOINT_NAMES
    c=config;robot=Path(c['robot_path']);door=Path(c['door_path']);source=Path(c['scene_path_source']);proof=Path(c['dense_audit_path'])
    for path,key in ((robot,'robot_xml_sha256'),(door,'door_xml_sha256'),(source,'scene_path_sha256'),(proof,'dense_audit_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=c[key]:raise ValueError('Standing route input bytes changed: '+str(path))
    audit=json.loads(proof.read_text())
    if audit.get('passed') is not True or audit.get('samples')!=1001 or audit.get('physics_steps')!=0:raise ValueError('Complete independent geometry audit required')
    if audit.get('receiving_palm_geometry',{}).get('passed') is not True:raise ValueError('Independent receiving-palm capture-distance audit required')
    for p in (robot,door,source):
        if audit['input_sha256'].get(str(p))!=hashlib.sha256(p.read_bytes()).hexdigest():raise ValueError('Route and audit inputs differ')
    scene=LandedLeftScene(robot,door);m,d=scene.m,scene.d;path=np.asarray(json.loads(source.read_text())['path_qpos'],float)
    qa=[m.joint('robot/'+n).qposadr[0] for n in c['joint_names']]
    if not np.array_equal(path[:,qa],c['joint_path']) or not np.array_equal(path[:,scene.root:scene.root+7],c['root_path']):raise ValueError('Consumed root/joint targets differ from audited path')
    if c['left_joint_names']!=JOINT_NAMES or len(c['left_targets'])!=len(path):raise ValueError('Left target contract changed')
    for q,row in zip(path,c['left_targets']):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d);r=d.xmat[scene.leaf].reshape(3,3)
        expected=dict(position=r.T@(d.site_xpos[scene.palm]-d.xpos[scene.leaf]),normal=r.T@d.site_xmat[scene.palm].reshape(3,3)[:,2],nominal=[q[m.joint('robot/'+n).qposadr[0]] for n in JOINT_NAMES])
        if row['phase']!='left_reach' or abs(row['leaf_rad']-q[m.joint('leaf_hinge').qposadr[0]])>1e-12 or any(not np.allclose(row[k],v,atol=1e-12,rtol=0) for k,v in expected.items()):raise ValueError('Left Cartesian targets differ from independently screened FK')


class StandingTransferTeacher:
    def __init__(self,operation,motors,path,*,start_seconds=22.,preload_profile='maintain',grasp_shift=(0.,0.,0.),hold_route=False,handoff_seconds=0.,fixed_pad_tracking=True,attained_arm_tracking=False,handle_relative_arm=False,leaf_relative_arm=False):
        if not np.isfinite(start_seconds) or start_seconds<=0:raise ValueError('Transfer start must be finite and positive')
        if preload_profile not in PROFILES:raise ValueError('Unknown transfer preload profile')
        self.attained_arm_tracking=attained_arm_tracking;self.arm_tracker=None
        if type(handle_relative_arm) is not bool or (handle_relative_arm and not attained_arm_tracking):
            raise ValueError('Handle-relative targets require explicit attained-arm tracking')
        if type(leaf_relative_arm) is not bool or (leaf_relative_arm and (not attained_arm_tracking or handle_relative_arm)):
            raise ValueError('Choose one relative arm frame with explicit attained-arm tracking')
        self.leaf_relative_arm=leaf_relative_arm
        self.handle_relative_arm=handle_relative_arm;self.handle_target=None
        if type(attained_arm_tracking) is not bool:raise ValueError('Explicit attained-arm tracking flag required')
        if attained_arm_tracking and fixed_pad_tracking:raise ValueError('Isolate attained-arm tracking from added fixed-pad feedback')
        if type(fixed_pad_tracking) is not bool:raise ValueError('Explicit fixed-pad tracking flag required')
        self.fixed_pad_tracking=fixed_pad_tracking
        if type(hold_route) is not bool:raise ValueError('Explicit diagnostic hold flag required')
        if not np.isfinite(handoff_seconds) or not 0<=handoff_seconds<=2:raise ValueError('Handoff duration must be in [0,2] seconds')
        self.handoff_seconds=handoff_seconds;self.motor_handoff=None
        self.hold_route=hold_route
        self.preload_profile=preload_profile
        self.grasp_shift=np.asarray(grasp_shift,float)
        if self.grasp_shift.shape!=(3,) or not np.isfinite(self.grasp_shift).all() or np.linalg.norm(self.grasp_shift)>.003:
            raise ValueError('Transfer grasp shift must be finite and within 3 mm')
        self.original_grasp_offset=operation.grasp_offset.copy()
        if np.linalg.norm(self.original_grasp_offset+self.grasp_shift)>.01:raise ValueError('Total grasp offset exceeds 10 mm')
        self.operation=operation;self.acquisition=operation.acquisition
        self.path=Path(path);self.config=json.loads(self.path.read_text())
        c=self.config
        if c.get('schema')!='doorbench.standing-transfer.v1' or c.get('geometric_screen_passed') is not True:
            raise ValueError('Explicit screened standing-transfer route required')
        if c['robot_xml_sha256']!=motors['source_xml_sha256']:
            raise ValueError('Standing route robot differs from controller')
        validate_route_geometry(c)
        self.names=self.acquisition.names
        if c['joint_names']!=self.names:raise ValueError('Standing route joint order changed')
        self.roots=np.asarray(c['root_path'],float);self.joints=np.asarray(c['joint_path'],float)
        if self.roots.shape!=(101,7) or self.joints.shape!=(101,len(self.names)) or not np.isfinite(np.r_[self.roots.ravel(),self.joints.ravel()]).all():raise ValueError('Expected complete finite 101-node route')
        if not np.allclose(np.linalg.norm(self.roots[:,3:],axis=1),1,atol=1e-6):raise ValueError('Unit root orientations required')
        proof=Path(c['dense_audit_path'])
        if hashlib.sha256(proof.read_bytes()).hexdigest()!=c['dense_audit_sha256'] or json.loads(proof.read_text()).get('passed') is not True:raise ValueError('Independent route audit changed')
        leftnames=c['left_joint_names'];rows=[]
        for row in c['left_targets']:
            rows.append({**row,**{k:np.asarray(row[k],float) for k in ('position','normal','nominal')}})
        self.left=LeftPalmContact(self.acquisition,motors,(leftnames,rows),fixed_waist=attained_arm_tracking,track_fixed_pads=fixed_pad_tracking,reach_seconds=8.,contact_force=8.,maximum_normal_offset=.008)
        self.start_seconds=start_seconds;self.started=None;self.info={}
        self.rotations=Slerp(np.linspace(0,1,101),Rotation.from_quat(self.roots[:,[4,5,6,3]]))

    def force(self,t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,*,grasp_qualified,left_panel_load):
        teacher=self.acquisition
        if self.started is None and t>=self.start_seconds-1e-8 and grasp_qualified:
            root=np.asarray(root,float);actual=np.array([joints[n] for n in self.names])
            delta=Rotation.from_quat(root[[4,5,6,3]])*Rotation.from_quat(self.roots[0,[4,5,6,3]]).inv()
            if np.linalg.norm(root[:3]-self.roots[0,:3])>.003 or delta.magnitude()>.01 or np.max(abs(actual-self.joints[0]))>.02:
                raise ValueError('Actual standing state differs from the independently screened route start')
            if self.operation.open_started is None or not .075<=angles['leaf']<=.10:
                raise ValueError('Qualified partial opening required before standing transfer')
            self.previous_motors=teacher.last_force.copy()
            if self.attained_arm_tracking:
                from .attained_arm_tracking import AttainedArmTracking
                self.arm_tracker=AttainedArmTracking(teacher,joints,self.previous_motors)
            self.left.begin(t,root,joints,leaf_pose,handle_pose);self.started=t
            if self.handle_relative_arm or self.leaf_relative_arm:
                from .handle_relative_arm import HandleRelativeArmTarget
                self.handle_target=HandleRelativeArmTarget(teacher.m,self.names,root,joints,leaf_pose if self.leaf_relative_arm else handle_pose,reference_frame="leaf" if self.leaf_relative_arm else "handle")
            self.initial_digit_forces=dict(teacher.digit_forces)
        if self.started is not None:
            pressure_blend=float(smooth_phase(t-self.started))
            teacher.digit_forces=transfer_preload(self.initial_digit_forces,t-self.started,self.preload_profile)
            self.operation.grasp_offset=self.original_grasp_offset+pressure_blend*self.grasp_shift
            self.left.update_targets(self.started if self.hold_route else t,root,joints,leaf_pose,left_panel_load,handle_pose)
            u=float(smooth_phase(self.left.progress));coordinate=u*100;i=min(int(coordinate),99);f=coordinate-i
            root_goal=(1-f)*self.roots[i,:3]+f*self.roots[i+1,:3]
            q=(1-f)*self.joints[i]+f*self.joints[i+1]
            teacher.stance.target_root[:]=root_goal
            teacher.stance.target_rotation=self.rotations(u).as_matrix()
            teacher.stance.joint_target[:]=[q[self.names.index(teacher.m.joint(int(j)).name)] for j in teacher.stance.joints]
        forces,info=self.operation.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified)
        if self.started is not None:
            forces=self.left.apply_forces(forces,joints,velocities)
            if self.arm_tracker:
                arm_target=dict(zip(self.names,q))
                if self.handle_target is not None:
                    arm_target,compensation_info=self.handle_target.target(t,root,joints,leaf_pose if self.leaf_relative_arm else handle_pose,arm_target)
                    info={**info,**compensation_info}
                forces,arm_info=self.arm_tracker.force(forces,t,arm_target,joints,velocities,reference_velocity=None if self.handle_target is None else self.handle_target.target_velocity)
                info={**info,**arm_info}
            if self.handoff_seconds:
                from .motor_handoff import MotorHandoff
                if self.motor_handoff is None:self.motor_handoff=MotorHandoff(self.previous_motors,forces,teacher.caps,self.handoff_seconds)
                forces=self.motor_handoff.force(forces,t-self.started)
                teacher.last_force=forces.copy()
                info={**info,'motor_handoff_seconds':self.handoff_seconds,'motor_handoff_initial_offset_Nm':self.motor_handoff.offset.tolist()}
            info={**info,**self.left.info, 'standing_transfer_started_s':self.started,'preload_profile':self.preload_profile,'diagnostic_route_held':self.hold_route,'fixed_pad_tracking':self.fixed_pad_tracking,'requested_digit_preloads_N':dict(teacher.digit_forces),'transfer_grasp_shift_m':self.grasp_shift.tolist()}
        self.info=info
        return forces,info
