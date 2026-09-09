"""Privileged closed-loop acquisition through original bounded robot motors.

This is a control adapter, not a sensor-only actor. Its MuJoCo model provides
FK, mass/bias terms and geometry queries only: it is never stepped. Measured
robot state and contact loads come from the active plant. Only motor forces are
returned; no support, pose command, grasp constraint or door effort is emitted.
"""
from types import SimpleNamespace
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_verification import scalar_transmission_matrix
from .stance import StanceController,validate_stance_solver_settings


class AcquisitionTeacher:
    def __init__(self, robot_xml, motors, reference, *, reach_seconds=6.6,
                 grip_force=6., finger_grip_scale=1/3, grip_start=.995,
                 palm_integral=1., torso_impedance=10., middle_finger_force=None,
                 index_finger_force=None, stance_solver_settings=None, stance_profile=None,
                 pressure_segment='nearest'):
        if pressure_segment not in ('nearest','distal'):raise ValueError('Unknown pressure segment')
        self.pressure_segment=pressure_segment
        if stance_profile not in (None,'landed-foot-v1'):raise ValueError('Unknown declared stance profile')
        self.stance_profile=stance_profile
        self.stance_solver_settings=validate_stance_solver_settings(stance_solver_settings)
        if stance_profile is not None and self.stance_solver_settings:raise ValueError('Landed-foot profile already fixes its solver settings')
        if motors.get('source_xml_sha256')!=hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest():
            raise ValueError('Acquisition model does not match imported motor contract')
        if motors.get('hand_mechanics_profile')!='shadow-loopback-v2':
            raise ValueError('Acquisition qualification requires the corrected hand model')
        values=[reach_seconds,grip_force,finger_grip_scale,grip_start,palm_integral,torso_impedance]
        if not np.isfinite(values).all() or reach_seconds<=0 or min(values[1:])<0 or grip_start>1:
            raise ValueError('Invalid acquisition controller configuration')
        if middle_finger_force is not None and (not np.isfinite(middle_finger_force) or middle_finger_force<0):raise ValueError('Invalid middle-finger pressure')
        if index_finger_force is not None and (not np.isfinite(index_finger_force) or index_finger_force<0):raise ValueError('Invalid index-finger pressure')
        spec=mujoco.MjSpec.from_file(str(robot_xml))
        spec.worldbody.add_body(name='analytic_lever').add_geom(name='analytic_lever_capsule',
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.007,.053,0],
            quat=[2**-.5,0,2**-.5,0],contype=0,conaffinity=0)
        self.m=spec.compile();self.d=mujoco.MjData(self.m);self.work=mujoco.MjData(self.m)
        m=self.m;self.names=motors['joint_names'];self.joints=np.array([m.joint(n).id for n in self.names])
        self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([m.actuator(v['name']).id for v in motors['actuators']])
        self.matrix=scalar_transmission_matrix(m,self.act,self.joints)
        self.kp=np.array([v['kp'] for v in motors['actuators']]);self.bias=np.array([v['bias'] for v in motors['actuators']])
        self.caps=np.array([v['force_range'] for v in motors['actuators']]);self.control_range=np.array([v['control_range'] for v in motors['actuators']])
        names=[v['name'] for v in motors['actuators']]
        self.arm=np.array([(n.startswith('right_') and not any(k in n for k in ('hip','knee','ankle'))) or n.startswith('rh_A_WRJ') for n in names])
        self.fingers=np.array([i for i,n in enumerate(names) if n.startswith('rh_') and 'WRJ' not in n])
        self.arm_motors=np.flatnonzero(self.arm)
        self.finger_inverse=np.linalg.pinv(self.matrix[self.fingers].T)
        self.arm_inverse=np.linalg.pinv(self.matrix[self.arm_motors].T)
        self.gain=self.arm.astype(float)*9
        self.damping=np.array([(.8 if 'WRJ' in n else 10.) if arm else 20. if n=='torso' else 0. for n,arm in zip(names,self.arm)])
        self.gain[names.index('torso')]=torso_impedance-1
        arm_names=reference['workspace_fit']['joint_names']
        self.arm_joints=np.array([m.joint(n).id for n in arm_names]);self.arm_q=m.jnt_qposadr[self.arm_joints];self.arm_v=m.jnt_dofadr[self.arm_joints]
        acquisition=reference['acquisition'];order=[acquisition['joint_names'].index(n) for n in self.names]
        self.path=np.asarray(acquisition['path_qpos'],float)[:,order]
        if 'recorded_root_path' in acquisition:raise ValueError('This first adapter supports screened geometric paths only')
        if len(self.path)<2 or not np.isfinite(self.path).all():raise ValueError('Invalid acquisition path')
        self.initial_root=np.asarray(reference['initial_root'],float)
        self.work.qpos[:7]=self.initial_root
        self.palm=m.site('rh_palm_touch').id;self.positions=[];self.rotations=[]
        for q in self.path:
            self.work.qpos[self.qa]=q;mujoco.mj_kinematics(m,self.work)
            self.positions.append(self.work.site_xpos[self.palm].copy())
            self.rotations.append(self.work.site_xmat[self.palm].reshape(3,3).copy())
        self.positions=np.asarray(self.positions);self.rotations=np.asarray(self.rotations)
        self.lever_body=m.body('analytic_lever').id;self.lever=m.geom('analytic_lever_capsule').id
        self.digit_geoms={digit:[g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('rh_'+digit)] for digit in ('ff','mf','rf','lf','th')}
        if pressure_segment=='distal':
            self.digit_geoms={digit:[g for g in geoms if m.body(m.geom_bodyid[g]).name=='rh_'+digit+'distal'] for digit,geoms in self.digit_geoms.items()}
            if any(not geoms for geoms in self.digit_geoms.values()):raise ValueError('Missing distal collision geometry')
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()
        self.reach_seconds=reach_seconds;self.grip_force=grip_force;self.finger_grip_scale=finger_grip_scale
        self.grip_start=grip_start;self.integral_gain=palm_integral
        self.digit_forces={digit:grip_force*(1. if digit=='th' else finger_grip_scale) for digit in self.digit_geoms}
        if middle_finger_force is not None:self.digit_forces['mf']=middle_finger_force
        if index_finger_force is not None:self.digit_forces['ff']=index_finger_force
        self.position_integral=np.zeros(3);self.rotation_integral=np.zeros(3)
        self.progress=0.;self.tracking_error=0.;self.last_update=None;self.stance=None
        self.target=self.matrix@self.path[0];self.last_force=np.zeros(len(self.act));self.stance_targets=None
        self.info={};self.ticks=0

    def force(self,t,root,joints,velocities,handle_pose,hand_forces):
        """Read measured state; return 61 finite capped motor forces in contract order.

        Root angular velocity is world-frame, matching Isaac's root_state_w.
        Hand forces map body names to world force vectors. Contact moments are
        approximated about body origins here and must be tested in the plant.
        """
        m,d,w=self.m,self.d,self.work
        root=np.asarray(root,float);q=np.array([joints[n] for n in self.names]);v=np.array([velocities[n] for n in self.names])
        if root.shape!=(13,) or not np.isfinite(np.r_[t,root,q,v]).all():raise ValueError('Invalid measured acquisition state')
        if self.last_update is not None and t<self.last_update-1e-9:raise ValueError('Controller clock went backwards')
        d.qpos[:7]=root[:7];d.qpos[self.qa]=q;d.qvel[:3]=root[7:10]
        rotation=Rotation.from_quat([*root[4:7],root[3]]).as_matrix()
        d.qvel[3:6]=rotation.T@root[10:13];d.qvel[self.va]=v
        handle_pose=np.asarray(handle_pose,float);hr=Rotation.from_quat([*handle_pose[4:7],handle_pose[3]]).as_matrix()
        m.body_pos[self.lever_body]=handle_pose[:3]+hr@np.array([-.06,-.077,0.])
        m.body_quat[self.lever_body]=handle_pose[3:7]
        mujoco.mj_forward(m,d)
        if self.last_update is None or t-self.last_update>=.01-1e-8:
            elapsed=0. if self.last_update is None else t-self.last_update;self.last_update=t
            if t>1.:self.progress=min(1.,self.progress+elapsed/self.reach_seconds*np.clip((.015-self.tracking_error)/.010,0.,1.))
            u=self.progress**3*(10+self.progress*(-15+6*self.progress));coordinate=u*(len(self.path)-1)
            i=min(int(coordinate),len(self.path)-2);f=coordinate-i
            desired=self.path[i]*(1-f)+self.path[i+1]*f
            pos=self.positions[i]*(1-f)+self.positions[i+1]*f
            rot=Rotation.from_rotvec(f*Rotation.from_matrix(self.rotations[i+1]@self.rotations[i].T).as_rotvec()).as_matrix()@self.rotations[i]
            self.tracking_error=float(np.linalg.norm(pos-d.site_xpos[self.palm]))
            if self.integral_gain and u>.99 and not np.any(np.abs(self.last_force[self.arm_motors])>=.98*np.abs(self.caps[self.arm_motors]).max(1)):
                self.position_integral+=self.integral_gain*elapsed*(pos-d.site_xpos[self.palm])
                self.rotation_integral+=self.integral_gain*elapsed*Rotation.from_matrix(rot@d.site_xmat[self.palm].reshape(3,3).T).as_rotvec()
            self.position_integral*=min(1.,.02/max(1e-12,np.linalg.norm(self.position_integral)))
            self.rotation_integral*=min(1.,.05/max(1e-12,np.linalg.norm(self.rotation_integral)))
            pos+=self.position_integral;rot=Rotation.from_rotvec(self.rotation_integral).as_matrix()@rot
            w.qpos[:]=d.qpos;w.qpos[self.qa]=desired
            for _ in range(25):
                mujoco.mj_kinematics(m,w);mujoco.mj_comPos(m,w)
                error=np.r_[5*(pos-w.site_xpos[self.palm]),Rotation.from_matrix(rot@w.site_xmat[self.palm].reshape(3,3).T).as_rotvec()]
                mujoco.mj_jacSite(m,w,self.jp,self.jr,self.palm)
                jac=np.vstack([5*self.jp[:,self.arm_v],self.jr[:,self.arm_v]])
                change=jac.T@np.linalg.solve(jac@jac.T+.003*np.eye(6),error)
                w.qpos[self.arm_q]=np.clip(w.qpos[self.arm_q]+np.clip(change,-.04,.04),m.jnt_range[self.arm_joints,0],m.jnt_range[self.arm_joints,1])
                if np.linalg.norm(error)<1e-4:break
            self.target=np.clip(self.matrix@w.qpos[self.qa],self.control_range[:,0],self.control_range[:,1])
            external=np.zeros(m.nv)
            for name,load in hand_forces.items():
                body=m.body(name.rsplit('/',1)[-1]).id;mujoco.mj_jacBody(m,d,self.jp,self.jr,body)
                external+=self.jp.T@np.asarray(load)
            if self.stance is None:
                stance_sim=SimpleNamespace(m=m,d=d,joint_prefix='',actuators=self.act,root_qadr=0,root_vadr=0,pelvis=m.body('pelvis').id,external_generalized_force=external)
                if self.stance_profile=='landed-foot-v1':
                    from .locomotion_manipulation import LandedFootStanceController
                    stance_sim.stance_solver_settings={'max_iter':50000,'rho':.001,'adaptive_rho_interval':25}
                    self.stance=LandedFootStanceController(stance_sim)
                else:self.stance=StanceController(stance_sim,solver_settings=self.stance_solver_settings)
            self.stance.sim.external_generalized_force=external
            self.stance_targets,status=self.stance.command()
            self.info=dict(phase='acquisition',path_fraction=float(u),tracking_error_m=self.tracking_error,stance_status=status)
            if self.stance_solver_settings or self.stance_profile:self.info['stance_solver']=dict(self.stance.last_solver_metadata)
            self.info['stance_profile']=self.stance_profile
        length=self.matrix@q;speed=self.matrix@v
        force=self.kp*self.target+self.bias[:,0]+self.bias[:,1]*length+self.bias[:,2]*speed+self.kp*self.gain*(self.target-length)-self.damping*speed
        for local in self.arm_motors:
            aid=self.act[local]
            if m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:force[local]+=d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]
        if self.info['path_fraction']>self.grip_start:
            generalized=np.zeros(m.nv)
            for digit,geoms in self.digit_geoms.items():
                nearest=None
                for geom in geoms:
                    pair=np.zeros(6);gap=mujoco.mj_geomDistance(m,d,geom,self.lever,.08,pair)
                    if nearest is None or gap<nearest[0]:nearest=(gap,geom,pair.copy())
                gap,geom,pair=nearest;delta=pair[3:]-pair[:3];distance=np.linalg.norm(delta)
                if distance<1e-7 or gap>.05:continue
                direction=delta/distance*(1 if gap>=0 else -1)
                mujoco.mj_jac(m,d,self.jp,self.jr,pair[:3],int(m.geom_bodyid[geom]))
                generalized+=self.jp.T@direction*self.digit_forces[digit]
            force[self.fingers]+=self.finger_inverse@generalized[self.va]
            force[self.arm_motors]+=self.arm_inverse@generalized[self.va]
        if self.stance_targets is not None:
            local=self.stance.local
            force[local]=self.kp[local]*self.stance_targets+self.bias[local,0]+self.bias[local,1]*length[local]+self.bias[local,2]*speed[local]
        self.last_force=np.clip(force,self.caps[:,0],self.caps[:,1]);self.ticks+=1
        if not np.isfinite(self.last_force).all():raise ValueError('Nonfinite acquisition motor command')
        return self.last_force.copy(),dict(self.info)
