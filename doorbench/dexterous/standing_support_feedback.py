"""Measured normal-force feedback through the existing left-arm motors.

Uses the same hybrid normal projection as panel continuation. It supplies
numeric targets to LeftPalmContact; it never receives an active plant object.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .operation_teacher import pose_components, smooth_phase


class StandingSupportFeedback:
    def __init__(self, left):
        self.left=left;self.previous=None;self.started=None
        m,d=left.m,left.d;body=m.site_bodyid[left.palm]
        site_R=d.site_xmat[left.palm].reshape(3,3);cloud=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
            mesh=int(m.geom_dataid[g])
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or mesh<0:raise ValueError('Original palm collision meshes required')
            start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
            world=m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            cloud.extend((world-d.site_xpos[left.palm])@site_R)
        self.surface=np.asarray(cloud)
        if self.surface.ndim!=2 or not len(self.surface):raise ValueError('Palm collision surface required')

    def update(self,t,leaf_pose,palm_load,target):
        if not np.isfinite([t,palm_load,target]).all() or palm_load<0 or not 2<target<=4:raise ValueError('Finite bounded support measurements required')
        p,R=pose_components(leaf_pose);l=self.left;d=l.d
        if self.started is None:self.started=t
        velocity=np.zeros(3);dt=0.
        if self.previous is not None:
            oldt,oldp,oldR=self.previous;dt=t-oldt
            if not 0<dt<=.05:raise ValueError('Monotonic physical support clock required')
            omega=Rotation.from_matrix(R@oldR.T).as_rotvec()/dt
            velocity=(p-oldp)/dt+np.cross(omega,d.site_xpos[l.palm]-p)
        l.surface_velocity_world=velocity;l.normal=R[:,1]
        prior=getattr(l,'filtered_palm_load',palm_load)
        l.filtered_palm_load=prior+(0. if dt==0 else dt/(.02+dt))*(palm_load-prior)
        l.hybrid_normal_target=float(target)
        l.hybrid_blend=float(smooth_phase((t-self.started)/2.))
        support=self.surface@(d.site_xmat[l.palm].reshape(3,3).T@l.normal)
        l.normal_contact_point_local=self.surface[support>=support.max()-.001].mean(axis=0)
        self.previous=(float(t),p.copy(),R.copy())
