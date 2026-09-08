"""Privileged bounded-motor left-palm contact primitive for the H1 v2 fixture.

The physical acquisition/operation controller remains active. This primitive
changes its nominal targets and adds left-arm gravity/contact feedforward only
through original motors. Analytic FK is never stepped and plant poses are never
written here. The first diagnostic intentionally stops before right release.
"""
import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from doorbench.dexterous.environment import DexterousDoorEnv


def load_screen_targets(report_path, robot, door):
    report=json.loads(Path(report_path).read_text())
    if not report.get('passed'):raise ValueError('Use a passed sampled bimanual screen')
    if 'compiled_robot_identity' in report:
        from doorbench.dexterous.robot_identity import verify_robot_identity
        verify_robot_identity(robot,report['compiled_robot_identity'])
    elif report.get('robot_sha256')!=hashlib.sha256(Path(robot).read_bytes()).hexdigest():
        raise ValueError('Static route belongs to a different robot model')
    if report.get('schema')=='doorbench.left-palm-targets.v1':
        names=report['joint_names'];rows=[]
        if names!=['torso']+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']:
            raise ValueError('Unexpected left-arm target joint order')
        for row in report['targets']:
            clean=dict(row)
            for key,shape in (('position',(3,)),('normal',(3,)),('nominal',(8,))):
                clean[key]=np.asarray(row[key],float)
                if clean[key].shape!=shape or not np.isfinite(clean[key]).all():raise ValueError('Invalid finite left target')
            if not np.isclose(np.linalg.norm(clean['normal']),1.,atol=1e-6):raise ValueError('Invalid left target normal')
            rows.append(clean)
        return names,rows
    sim=DexterousDoorEnv(door,robot,json.loads(Path(robot).with_suffix('.audit.json').read_text()))
    try:
        sim.reset(images=False,randomize=False);m,d=sim.m,sim.d
        names=['torso']+['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1']
        qa=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in names]
        palm=m.site('robot/lh_palm_touch').id;leaf=m.body('leaf').id
        result=[]
        for row in report['rows']:
            d.qpos[:]=row['qpos'];mujoco.mj_kinematics(m,d)
            rotation=d.xmat[leaf].reshape(3,3)
            result.append(dict(phase=row['phase'],leaf_rad=row['leaf_rad'],
                position=rotation.T@(d.site_xpos[palm]-d.xpos[leaf]),
                normal=rotation.T@d.site_xmat[palm].reshape(3,3)[:,2],
                nominal=d.qpos[qa].copy()))
        return names,result
    finally:sim.close()


class LeftPalmContact:
    def __init__(self, teacher, motors, targets, *, reach_seconds=5., contact_force=8., fixed_waist=False, track_fixed_pads=False):
        self.teacher=teacher;self.m=teacher.m;self.d=mujoco.MjData(self.m)
        self.fixed_waist=bool(fixed_waist)
        self.track_fixed_pads=bool(track_fixed_pads)
        self.names,self.rows=targets;self.path=[row for row in self.rows if row['phase']=='left_reach']
        self.reach_seconds=reach_seconds;self.contact_force=contact_force
        if not np.isfinite([reach_seconds,contact_force]).all() or reach_seconds<=0 or not 0<contact_force<=20:
            raise ValueError('Invalid left contact settings')
        if len(self.path)<2:raise ValueError('Left approach requires at least two screened samples')
        m=self.m
        self.joints=np.array([m.joint(n).id for n in self.names[1:]])
        self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([next(i for i,a in enumerate(motors['actuators']) if set(a['terms'])=={name}) for name in self.names[1:]])
        self.palm=m.site('lh_palm_touch').id
        self.right_palm=m.site('rh_palm_touch').id
        self.solve_names=self.names+['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']
        if self.fixed_waist:self.solve_names=self.names[1:]
        self.solve_joints=np.array([m.joint(n).id for n in self.solve_names])
        self.solve_qa=m.jnt_qposadr[self.solve_joints]
        self.solve_indices=[teacher.names.index(n) for n in self.solve_names]
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()
        self.started=None;self.last_update=None;self.progress=0.;self.tracking=0.
        self.previous=None;self.target=None;self.loaded_since=None;self.info={}
        self.waist_index=teacher.names.index('torso')
        self.left_indices=[teacher.names.index(n) for n in self.names[1:]]
        self.lower=m.jnt_range[self.solve_joints,0]+.01;self.upper=m.jnt_range[self.solve_joints,1]-.01
        self.offset=0.

    def begin(self,t,root,joints,leaf_pose,handle_pose):
        self.started=t
        self._read(root,joints)
        rotation=Rotation.from_quat([*leaf_pose[4:7],leaf_pose[3]]).as_matrix()
        actual=rotation.T@(self.d.site_xpos[self.palm]-leaf_pose[:3])
        self.initial_position_delta=actual-self.path[0]['position']
        self.waist_delta=joints['torso']-self.path[0]['nominal'][0]
        self.previous=self.d.qpos[self.solve_qa].copy()
        if self.fixed_waist:return
        hr=Rotation.from_quat([*handle_pose[4:7],handle_pose[3]]).as_matrix()
        self.m.body_pos[self.teacher.lever_body]=handle_pose[:3]+hr@np.array([-.06,-.077,0.])
        self.m.body_quat[self.teacher.lever_body]=handle_pose[3:7]
        mujoco.mj_forward(self.m,self.d)
        distal={digit:[g for g in geoms if self.m.body(self.m.geom_bodyid[g]).name.endswith('distal')] for digit,geoms in self.teacher.digit_geoms.items()}
        if self.track_fixed_pads:
            from doorbench.dexterous.pad_tracking import PadTracker
            self.pads=PadTracker(self.m,self.d,distal,self.teacher.lever,digits=('ff','mf','rf','lf','th'),stiffness=800.,damping=2.,maximum_force=6.)
            self.pad_targets_handle={digit:hr.T@(position-handle_pose[:3]) for digit,position in self.pads.positions(self.d).items()}
        # Keep the waist target explicit during bimanual work. Right-arm IK
        # still compensates through its seven actual joints as the waist moves.
        keep=np.array([self.m.joint(int(j)).name!='torso' for j in self.teacher.arm_joints])
        self.teacher.arm_joints=self.teacher.arm_joints[keep]
        self.teacher.arm_q=self.teacher.arm_q[keep];self.teacher.arm_v=self.teacher.arm_v[keep]

    def _read(self,root,joints):
        root=np.asarray(root,float)
        if root.shape!=(13,) or set(joints)!=set(self.teacher.names) or not np.isfinite(np.r_[root,[joints[n] for n in self.teacher.names]]).all() or np.linalg.norm(root[3:7])<1e-8:
            raise ValueError('Expected complete finite measured robot state with wxyz root quaternion')
        d=self.d;m=self.m;d.qpos[:7]=root[:7]
        for name,value in joints.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
        mujoco.mj_kinematics(m,d)

    def update_targets(self,t,root,joints,leaf_pose,left_panel_load,handle_pose):
        if self.started is None:return
        if not np.isfinite([t,left_panel_load]).all() or left_panel_load<0 or (self.last_update is not None and t<self.last_update-1e-9):raise ValueError('Invalid left contact clock/load')
        leaf_pose=np.asarray(leaf_pose,float)
        if leaf_pose.shape!=(7,) or not np.isfinite(leaf_pose).all() or np.linalg.norm(leaf_pose[3:])<1e-8:raise ValueError('Expected finite leaf pose with wxyz quaternion')
        self.handle_pose=np.asarray(handle_pose).copy()
        m,d=self.m,self.d;self._read(root,joints)
        rotation=Rotation.from_quat([*leaf_pose[4:7],leaf_pose[3]]).as_matrix();normal=rotation[:,1]
        if self.last_update is None or t-self.last_update>=.01-1e-8:
            dt=0. if self.last_update is None else t-self.last_update;self.last_update=t
            self.progress=min(1.,self.progress+dt/self.reach_seconds*np.clip((.025-self.tracking)/.015,0.,1.))
            u=self.progress**3*(10+self.progress*(-15+6*self.progress));coordinate=u*(len(self.path)-1)
            i=min(int(coordinate),len(self.path)-2);f=coordinate-i
            local=(1-f)*self.path[i]['position']+f*self.path[i+1]['position']
            local=local+(1-min(1.,u/.4))*self.initial_position_delta
            desired_z=(1-f)*self.path[i]['normal']+f*self.path[i+1]['normal'];desired_z=rotation@(desired_z/np.linalg.norm(desired_z))
            nominal=(1-f)*self.path[i]['nominal']+f*self.path[i+1]['nominal']
            if u>.98:self.offset=float(np.clip(self.offset+dt*.002*np.clip((4.-left_panel_load)/4.,-1.,1.),0.,.004))
            goal=np.asarray(leaf_pose[:3])+rotation@local+normal*self.offset
            self.tracking=float(np.linalg.norm(goal-d.site_xpos[self.palm]))
            desired_waist=nominal[0]+self.waist_delta
            seed=d.qpos[self.solve_qa].copy()
            if self.fixed_waist:seed[:]=nominal[1:]
            else:seed[:8]=nominal;seed[0]=desired_waist
            right_goal=self.teacher.positions[-1].copy();right_rotation=self.teacher.rotations[-1].copy()
            def residual(q):
                d.qpos[self.solve_qa]=q;mujoco.mj_kinematics(m,d)
                right=np.r_[100*(d.site_xpos[self.right_palm]-right_goal),10*Rotation.from_matrix(right_rotation@d.site_xmat[self.right_palm].reshape(3,3).T).as_rotvec()] if not self.fixed_waist else np.zeros(0)
                return np.r_[100*(d.site_xpos[self.palm]-goal),10*(d.site_xmat[self.palm].reshape(3,3)[:,2]-desired_z),
                    right,.03*(q-seed)]
            fit=least_squares(residual,np.clip(self.previous,self.lower,self.upper),bounds=(self.lower,self.upper),max_nfev=140)
            self.target=(fit.x if self.fixed_waist else fit.x[1:8]).copy();self.previous=fit.x.copy()
            self.teacher.path[-1,self.solve_indices]=fit.x
            if left_panel_load>=2. and u>.99:
                if self.loaded_since is None:self.loaded_since=t
            else:self.loaded_since=None
            self.info=dict(phase='left_panel_contact',left_progress=float(u),left_tracking_error_m=self.tracking,
                left_panel_load_N=float(left_panel_load),left_offset_m=self.offset,
                left_ik_residual=float(np.linalg.norm(fit.fun[:6])),
                left_loaded_duration_s=0. if self.loaded_since is None else t-self.loaded_since)
        self.normal=normal

    def apply_forces(self,forces,joints,velocities):
        if self.started is None or self.target is None:return forces
        teacher=self.teacher;m,d=self.m,teacher.d
        forces=np.asarray(forces,float)
        if forces.shape!=(len(teacher.act),) or not np.isfinite(forces).all():raise ValueError('Expected finite native motor forces')
        q=np.array([joints[n] for n in self.names[1:]]);v=np.array([velocities[n] for n in self.names[1:]])
        kp=teacher.kp[self.act];bias=teacher.bias[self.act]
        damping=np.array([.8 if 'WRJ' in n else 10. for n in self.names[1:]])
        left_force=kp*self.target+bias[:,0]+bias[:,1]*q+bias[:,2]*v+kp*9*(self.target-q)-damping*v+d.qfrc_bias[self.va]
        mujoco.mj_jacSite(m,d,self.jp,self.jr,self.palm)
        push=self.contact_force*np.clip((self.progress-.94)/.06,0.,1.)
        left_force+=self.jp[:,self.va].T@(self.normal*push)
        result=forces.copy();result[self.act]=np.clip(left_force,teacher.caps[self.act,0],teacher.caps[self.act,1])
        if self.fixed_waist or not self.track_fixed_pads:return result
        hr=Rotation.from_quat([*self.handle_pose[4:7],self.handle_pose[3]]).as_matrix()
        pad_targets={digit:self.handle_pose[:3]+hr@local for digit,local in self.pad_targets_handle.items()}
        generalized,errors=self.pads.generalized_force(d,pad_targets)
        result[teacher.fingers]+=teacher.finger_inverse@generalized[teacher.va]
        result[teacher.arm_motors]+=teacher.arm_inverse@generalized[teacher.va]
        self.info['right_fixed_pad_error_m']=errors
        return np.clip(result,teacher.caps[:,0],teacher.caps[:,1])
