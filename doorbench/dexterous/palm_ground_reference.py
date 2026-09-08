"""Offline palm path and bounded robot-only IK in an initial-yaw/XY ground gauge.

The static path is projected before runtime. The IK calculator receives only
robot encoders, an existing sensor-derived pelvis estimate and scripted goals.
It has no scene, object transform, simulator handle or independent IMU integrator.
"""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation, Slerp

ARM_NAMES=('right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw',
           'right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1')
SCHEMA='doorbench.static-ground-palm-reference.v1'
FIELDS={'schema','frame','source_reference_sha256','robot_xml_sha256','source_pelvis_height_m',
        'fractions','position_m','quaternion_xyzw','timing','scope'}
FRAME='initial pelvis yaw and XY removed; original ground Z retained'
SCOPE='Offline fixed scripted palm poses; no root XY/yaw, door pose or runtime world-state input'


def project_palm_reference(robot_xml,reference):
    """Offline input projection only. Full source reset never enters runtime."""
    path=Path(robot_xml);raw=Path(reference).read_bytes();ref=json.loads(raw)
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    names=ref['acquisition']['joint_names'];q=np.asarray(ref['acquisition']['path_qpos'],float)
    root=np.asarray(ref['initial_root'],float)
    if m.nq!=76 or m.nv!=75 or m.nu!=61 or q.shape!=(401,69) or root.shape!=(7,) or not np.isfinite(np.r_[q.ravel(),root]).all():
        raise ValueError('Require the frozen full401-pose H1 reference')
    if tuple(m.joint(i).name for i in range(1,m.njnt))!=tuple(names):raise ValueError('Original named source order required')
    R=Rotation.from_quat(root[[4,5,6,3]]).as_matrix()
    if np.linalg.norm(R[:,2]-[0,0,1])>1e-10:raise ValueError('Initial source pelvis must be upright; remove only yaw')
    qa=np.array([m.joint(n).qposadr[0] for n in names]);site=m.site('rh_palm_touch').id
    fixed=[i for i,n in enumerate(names) if not (n=='torso' or n.startswith('rh_') or n in ARM_NAMES)]
    if not np.array_equal(q[:,fixed],np.broadcast_to(q[0,fixed],q[:,fixed].shape)):
        raise ValueError('Static route cannot smuggle leg or left-hand motion')
    positions=[];rotations=[]
    for angles in q:
        # Setting a detached FK calculator is projection, never active dynamics.
        d.qpos[:7]=[0,0,float(root[2]),1,0,0,0];d.qpos[qa]=angles;mujoco.mj_kinematics(m,d)
        positions.append(d.site_xpos[site].tolist());rotations.append(Rotation.from_matrix(d.site_xmat[site].reshape(3,3)).as_quat().tolist())
    return dict(schema=SCHEMA,frame=FRAME,source_reference_sha256=hashlib.sha256(raw).hexdigest(),
        robot_xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_pelvis_height_m=float(root[2]),
        fractions=np.linspace(0.,1.,401).tolist(),position_m=positions,quaternion_xyzw=rotations,
        timing=dict(start_s=1.,reach_seconds=16.,settle_seconds=2.,duration_s=19.),
        scope=SCOPE)


class GroundPalmReference:
    def __init__(self,value,robot_sha):
        if type(value) is not dict or set(value)!=FIELDS or value.get('schema')!=SCHEMA or value.get('frame')!=FRAME:
            raise ValueError('Exact static ground-palm schema required')
        if value['robot_xml_sha256']!=robot_sha:raise ValueError('Palm reference robot identity differs')
        if value['scope']!=SCOPE or any(type(value[k]) is not str or len(value[k])!=64 or any(c not in '0123456789abcdef' for c in value[k]) for k in ('robot_xml_sha256','source_reference_sha256')):
            raise ValueError('Exact offline scope and source fingerprints required')
        self.fractions=np.asarray(value['fractions'],float);self.positions=np.asarray(value['position_m'],float)
        quats=np.asarray(value['quaternion_xyzw'],float)
        if (self.fractions.shape!=(401,) or not np.array_equal(self.fractions,np.linspace(0.,1.,401)) or
                self.positions.shape!=(401,3) or quats.shape!=(401,4) or not np.isfinite(np.r_[self.positions.ravel(),quats.ravel()]).all() or
                not np.allclose(np.linalg.norm(quats,axis=1),1.,atol=1e-10,rtol=0)):
            raise ValueError('Require finite401-sample positions and unit quaternion rotations')
        if value['timing']!=dict(start_s=1.,reach_seconds=16.,settle_seconds=2.,duration_s=19.):
            raise ValueError('Frozen nineteen-second acquisition timing required')
        self.rotation=Slerp(self.fractions,Rotation.from_quat(quats))
        self.source_pelvis_height_m=float(value['source_pelvis_height_m'])
        if not np.isfinite(self.source_pelvis_height_m) or not .8<self.source_pelvis_height_m<.95:
            raise ValueError('Retain the declared source ground-height convention')

    def target(self,time_s):
        if type(time_s) not in (int,float) or not np.isfinite(time_s) or not 0<=time_s<=19.:
            raise ValueError('Require finite acquisition-local time')
        u=float(np.clip((time_s-1.)/16.,0.,1.));f=u**3*(10.+u*(-15.+6.*u))
        return np.array([np.interp(f,self.fractions,self.positions[:,i]) for i in range(3)]),self.rotation(f).as_matrix()


class RobotPalmReferenceIK:
    def __init__(self,robot_model,joint_names):
        self.m=m=robot_model;self.names=tuple(joint_names);self.d=mujoco.MjData(m)
        if m.nq!=76 or m.nv!=75 or m.nu!=61 or tuple(m.joint(i).name for i in range(1,m.njnt))!=self.names:
            raise ValueError('Original robot-only kinematic calculator required')
        self.qa=np.array([m.joint(n).qposadr[0] for n in self.names])
        ids=np.array([m.joint(n).id for n in ARM_NAMES]);self.armqa=m.jnt_qposadr[ids];self.armva=m.jnt_dofadr[ids]
        self.limits=m.jnt_range[ids].copy();self.site=m.site('rh_palm_touch').id
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()

    def fk(self):mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)

    def goals(self,nominal,joints,estimated_root,target_position,target_rotation,*,weight=1.,maximum_joint_correction=.05):
        q=np.asarray(joints,float);root=np.asarray(estimated_root,float);p=np.asarray(target_position,float);R=np.asarray(target_rotation,float)
        if (q.shape!=(69,) or root.shape!=(7,) or p.shape!=(3,) or R.shape!=(3,3) or
                not np.isfinite(np.r_[q,root,p,R.ravel(),weight,maximum_joint_correction]).all() or
                not np.isclose(np.linalg.norm(root[3:]),1.,atol=1e-6,rtol=0) or not 0<=weight<=1 or maximum_joint_correction!=.05):
            raise ValueError('Bounded numeric robot-only palm inputs required')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(R),1.,atol=1e-10,rtol=0):
            raise ValueError('Target must be a proper rotation')
        initial=np.array([nominal[n] for n in ARM_NAMES],float)
        if not np.isfinite(initial).all() or np.any(initial<self.limits[:,0]) or np.any(initial>self.limits[:,1]):
            raise ValueError('Original nominal arm limits required')
        d,m=self.d,self.m;d.qpos[:7]=root;d.qpos[self.qa]=q;d.qpos[self.armqa]=initial;self.fk()
        p0=d.site_xpos[self.site].copy();R0=d.site_xmat[self.site].reshape(3,3).copy()
        error_p=p-p0;error_R=Rotation.from_matrix(R@R0.T).as_rotvec()
        if np.linalg.norm(error_p)>.010 or np.linalg.norm(error_R)>.05:
            raise ValueError('Palm correction exceeds ten-millimeter/fifty-milliradian admission envelope')
        goal=p0+weight*error_p;goalR=Rotation.from_rotvec(weight*error_R).as_matrix()@R0
        low=np.maximum(self.limits[:,0],initial-.05);high=np.minimum(self.limits[:,1],initial+.05)
        for iteration in range(8):
            dp=goal-d.site_xpos[self.site];dr=Rotation.from_matrix(goalR@d.site_xmat[self.site].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(dp)<1e-8 and np.linalg.norm(dr)<1e-7:break
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.site)
            A=np.vstack([100*self.jp[:,self.armva],10*self.jr[:,self.armva],.001*np.eye(7)])
            fit=lsq_linear(A,np.r_[100*dp,10*dr,np.zeros(7)],bounds=(low-d.qpos[self.armqa],high-d.qpos[self.armqa]),
                method='bvls',tol=1e-12,max_iter=50)
            if not fit.success:raise ValueError('Bounded seven-joint palm IK did not converge')
            d.qpos[self.armqa]=np.clip(d.qpos[self.armqa]+fit.x,low,high);self.fk()
        ep=float(np.linalg.norm(goal-d.site_xpos[self.site]));er=float(Rotation.from_matrix(goalR@d.site_xmat[self.site].reshape(3,3).T).magnitude())
        if ep>5e-6 or er>5e-5:raise ValueError('Bounded palm correction cannot attain its six-dimensional target')
        result=dict(nominal);result.update(zip(ARM_NAMES,d.qpos[self.armqa].tolist()))
        return result,dict(position_correction_m=error_p.tolist(),rotation_correction_rad=error_R.tolist(),weight=float(weight),
            position_error_m=ep,orientation_error_rad=er,maximum_joint_correction_rad=float(np.max(abs(d.qpos[self.armqa]-initial))),
            iterations=iteration+1,calculator_time_s=float(d.time))
