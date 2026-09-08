"""Small palm translation using only an unstepped robot kinematic calculator."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


ARM_NAMES=('torso','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw',
           'right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1')


class RobotPalmTranslation:
    def __init__(self,robot_only_model,joint_names):
        self.m=m=robot_only_model
        self.names=tuple(joint_names)
        if m.nq!=76 or m.nv!=75 or m.nu!=61 or len(self.names)!=69:
            raise ValueError('A calibrated robot-only model is required; no active door scene')
        if tuple(m.joint(i).name for i in range(1,m.njnt))!=self.names:
            raise ValueError('Original named encoder order required')
        self.d=mujoco.MjData(m)
        self.qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in self.names])
        ids=np.array([m.joint(n).id for n in ARM_NAMES]);self.armqa=m.jnt_qposadr[ids];self.armva=m.jnt_dofadr[ids]
        self.limits=m.jnt_range[ids].copy()
        self.palm=m.site('rh_palm_touch').id
        self.fingers=[m.body('rh_'+digit+'distal').id for digit in ('ff','mf','rf','lf')]
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros((3,m.nv))

    def _fk(self):
        mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)

    def goals(self,nominal_goals,joint_positions,offset_m):
        """Translate along the four fingertips' robot-local palmar direction.

        Pelvis is an arbitrary fixed gauge, other robot joints are encoders, and
        the eight nominal arm angles come from the declared static motor route.
        No object geometry, environment collision, or dynamic model is queried.
        """
        result=dict(nominal_goals)
        if isinstance(offset_m,(bool,np.bool_)) or not np.isfinite(offset_m) or not 0<=offset_m<=.00025:
            raise ValueError('Original quarter-millimeter translation envelope required')
        q=np.asarray(joint_positions,dtype=float)
        if q.shape!=(69,) or not np.isfinite(q).all():raise ValueError('Exact numeric encoder vector required')
        initial=np.array([result[n] for n in ARM_NAMES])
        if not np.isfinite(initial).all() or np.any(initial<self.limits[:,0]) or np.any(initial>self.limits[:,1]):
            raise ValueError('Nominal arm goals exceed original joint limits')
        if offset_m==0.:
            return result,dict(offset_m=0.,position_error_m=0.,orientation_error_rad=0.,maximum_joint_correction_rad=0.,iterations=0)
        m,d=self.m,self.d
        d.qpos[:7]=[0,0,1,1,0,0,0];d.qpos[self.qa]=q;d.qpos[self.armqa]=initial
        self._fk();R0=d.site_xmat[self.palm].reshape(3,3).copy();p0=d.site_xpos[self.palm].copy()
        direction=np.mean([d.xmat[b].reshape(3,3)@np.array([0.,-1.,0.]) for b in self.fingers],axis=0)
        length=float(np.linalg.norm(direction))
        if length<.5:raise ValueError('Finger palmar normals do not define a stable shared direction')
        direction/=length;target=p0+offset_m*direction
        for iteration in range(6):
            position=target-d.site_xpos[self.palm]
            rotation=Rotation.from_matrix(R0@d.site_xmat[self.palm].reshape(3,3).T).as_rotvec()
            if np.linalg.norm(position)<1e-8 and np.linalg.norm(rotation)<1e-7:break
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.palm)
            A=np.vstack((100*self.jp[:,self.armva],10*self.jr[:,self.armva],.001*np.eye(8)))
            rhs=np.r_[100*position,10*rotation,np.zeros(8)]
            current=d.qpos[self.armqa].copy()
            fit=lsq_linear(A,rhs,bounds=(self.limits[:,0]-current,self.limits[:,1]-current),method='bvls',tol=1e-12,max_iter=50)
            if not fit.success:raise ValueError('Bounded robot-only palm solve did not converge')
            d.qpos[self.armqa]=np.clip(current+fit.x,self.limits[:,0],self.limits[:,1]);self._fk()
        pos_error=float(np.linalg.norm(target-d.site_xpos[self.palm]))
        rot_error=float(Rotation.from_matrix(R0@d.site_xmat[self.palm].reshape(3,3).T).magnitude())
        corrected=d.qpos[self.armqa].copy();max_delta=float(np.max(abs(corrected-initial)))
        if pos_error>2e-6 or rot_error>2e-5 or max_delta>.025:
            raise ValueError('Small robot-only palm translation failed its geometric accuracy envelope')
        result.update(dict(zip(ARM_NAMES,corrected.tolist())))
        return result,dict(offset_m=float(offset_m),direction_palm_local=(R0.T@direction).tolist(),
            position_error_m=pos_error,orientation_error_rad=rot_error,maximum_joint_correction_rad=max_delta,iterations=iteration+1,
            input_scope='robot encoders and static nominal arm goals; fixed arbitrary pelvis gauge; no object geometry')
