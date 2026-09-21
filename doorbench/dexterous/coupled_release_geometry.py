"""Unstepped release targets indexed by progress and measured mechanism state.

This module never receives or writes the plant data. All qpos writes below are
to its private kinematic model; consumers receive only motor/body references.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.transform import Rotation,Slerp

from .environment import DexterousDoorEnv
from .operation_teacher import smooth_phase


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class CoupledReleaseGeometry:
    def __init__(self,path):
        self.path=Path(path);self.plan=json.loads(self.path.read_text());p=self.plan
        if p.get('schema')!='doorbench.coupled-release-envelope.v1':raise ValueError('Explicit coupled measured-angle envelope required')
        for name,digest in p['input_sha256'].items():
            if sha(name)!=digest:raise ValueError('Coupled envelope source changed: '+name)
        self.c=json.loads(Path(p['source_candidate']).read_text());c=self.c
        self.sim=DexterousDoorEnv(Path(c['door_path']),Path(c['robot_path']),json.loads(Path(c['robot_path']).with_suffix('.audit.json').read_text()))
        self.m=self.sim.m;self.d=self.sim.d;m=self.m;d=self.d
        self.initial=np.asarray(c['initial_qpos']);self.rq=c['root_qpos_address'];self.names=c['joint_names']
        self.qa=np.array([m.joint('robot/'+n).qposadr[0] for n in self.names]);self.joints=np.array([m.joint('robot/'+n).id for n in self.names])
        self.lq=m.joint('leaf_hinge').qposadr[0];self.oq=m.joint('leaf_handle_hinge').qposadr[0];self.bq=m.joint('leaf_latch_bolt_slide').qposadr[0]
        self.leaf=m.body('leaf').id;self.handle=m.body('leaf_handle').id;self.rh=m.site('robot/rh_palm_touch').id;self.lh=m.site('robot/lh_palm_touch').id
        self.initial_rotation=Rotation.from_quat(self.initial[self.rq+3:self.rq+7][[1,2,3,0]])
        self.elapsed=np.asarray(p['elapsed_s']);self.angles=np.asarray(p['leaf_rad']);coordinates=np.asarray(p['coordinates'])
        if coordinates.shape!=(len(self.elapsed),len(self.angles),6+len(self.names)) or not np.isfinite(coordinates).all() or not np.all(np.diff(self.elapsed)>0) or not np.all(np.diff(self.angles)>0):raise ValueError('Finite ordered measured-angle map required')
        if self.elapsed[0]!=0 or self.elapsed[-1]!=p['duration_s'] or self.angles[0]>.08 or self.angles[-1]<.4:raise ValueError('Full declared progress/aperture envelope required')
        self.splines=[RectBivariateSpline(self.elapsed,self.angles,coordinates[:,:,i],kx=3,ky=3,s=0) for i in range(coordinates.shape[-1])]
        ref=json.loads(Path(c['configuration']['right_hand_route']).read_text())
        d.qpos[:]=self.initial;mujoco.mj_kinematics(m,d)
        initial=dict(time_s=0.,qpos=self.initial.tolist(),palm_position=d.site_xpos[self.rh].tolist(),palm_rotation=d.site_xmat[self.rh].reshape(3,3).tolist())
        refs=[initial]+[{**r,'time_s':r['time_s']+.5} for r in ref['trials'][0]['rows']]
        self.times=np.array([r['time_s'] for r in refs]);self.qs=np.array([r['qpos'] for r in refs]);self.positions=np.array([r['palm_position'] for r in refs]);self.rotations=Slerp(self.times,Rotation.from_matrix([r['palm_rotation'] for r in refs]))
        self.through=c['configuration']['follow_handle_through_route_seconds']
        self.localp=np.array(c['left_palm_in_leaf_position']);self.localr=np.array(c['left_palm_in_leaf_rotation'])
        self.arm={}
        for side,hand,site in [('right','rh',self.rh),('left','lh',self.lh)]:
            names=[side+'_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+[hand+'_'+n for n in ('WRJ2','WRJ1')]
            js=np.array([m.joint('robot/'+n).id for n in names])
            self.arm[side]=(site,m.jnt_qposadr[js],m.jnt_dofadr[js],m.jnt_range[js,0]+.01,m.jnt_range[js,1]-.01)
        self.jp=np.zeros((3,m.nv));self.jr=np.zeros_like(self.jp)

    def coordinates(self,elapsed,angle):
        if not np.isfinite([elapsed,angle]).all() or not self.elapsed[0]<=elapsed<=self.elapsed[-1] or not self.angles[0]<=angle<=self.upper_angle(elapsed):
            raise ValueError('Measured progress/aperture outside screened coupled envelope')
        return np.array([s(elapsed,angle,grid=False).item() for s in self.splines])

    def upper_angle(self,elapsed):
        nodes=self.plan.get('admitted_leaf_upper_nodes')
        if nodes is None:return float(self.angles[-1])
        nodes=np.asarray(nodes,float)
        if nodes.ndim!=2 or nodes.shape[1]!=2 or len(nodes)<2 or not np.isfinite(nodes).all() or not np.all(np.diff(nodes[:,0])>0) or nodes[0,0]!=0 or nodes[-1,0]!=self.plan['duration_s'] or np.any(nodes[:,1]<self.angles[0]) or np.any(nodes[:,1]>self.angles[-1]):
            raise ValueError('Finite explicitly bounded progress/aperture domain required')
        return float(np.interp(elapsed,nodes[:,0],nodes[:,1]))

    def _correct_arm(self,side,position,rotation):
        m,d=self.m,self.d;site,qa,va,lower,upper=self.arm[side];nominal=d.qpos[qa].copy()
        for _ in range(8):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            error=np.r_[100*(position-d.site_xpos[site]),10*Rotation.from_matrix(rotation@d.site_xmat[site].reshape(3,3).T).as_rotvec()]
            if np.linalg.norm(error)<1e-6:break
            mujoco.mj_jacSite(m,d,self.jp,self.jr,site)
            jac=np.r_[100*self.jp[:,va],10*self.jr[:,va]]
            step=np.linalg.lstsq(np.r_[jac,.001*np.eye(7)],np.r_[error,.001*(nominal-d.qpos[qa])],rcond=None)[0]
            d.qpos[qa]=np.clip(d.qpos[qa]+np.clip(step,-.03,.03),np.minimum(lower,nominal),np.maximum(upper,nominal))
        mujoco.mj_kinematics(m,d)

    def evaluate(self,elapsed,angle,operator,latch,*,correct=True):
        p=self.plan;m,d=self.m,self.d;x=self.coordinates(elapsed,angle)
        if not np.isfinite([operator,latch]).all() or not p['operator_envelope_rad'][0]<=operator<=p['operator_envelope_rad'][1] or abs(latch)>.001:
            raise ValueError('Measured operator/latch outside screened coupled envelope')
        clock=float(smooth_phase(elapsed/p['duration_s']))*self.times[-1]
        i=min(len(self.times)-2,max(0,int(np.searchsorted(self.times,clock,side='right')-1)));f=(clock-self.times[i])/(self.times[i+1]-self.times[i])
        reference=(1-f)*self.qs[i]+f*self.qs[i+1];rhp=(1-f)*self.positions[i]+f*self.positions[i+1];rhr=self.rotations(clock).as_matrix()
        d.qpos[:]=reference;mujoco.mj_kinematics(m,d);hp=d.xpos[self.handle].copy();hr=d.xmat[self.handle].reshape(3,3).copy()
        d.qpos[self.lq]=angle;d.qpos[self.oq]=operator;d.qpos[self.bq]=latch;mujoco.mj_kinematics(m,d)
        follow=1.-float(smooth_phase((max(0.,clock-.5)-self.through)/(8.-self.through)))
        transformedp=d.xpos[self.handle]+d.xmat[self.handle].reshape(3,3)@hr.T@(rhp-hp);transformedr=d.xmat[self.handle].reshape(3,3)@hr.T@rhr
        rhp=rhp+follow*(transformedp-rhp);rhr=Rotation.from_rotvec(follow*Rotation.from_matrix(transformedr@rhr.T).as_rotvec()).as_matrix()@rhr
        lhp=d.xpos[self.leaf]+d.xmat[self.leaf].reshape(3,3)@self.localp;lhr=d.xmat[self.leaf].reshape(3,3)@self.localr
        d.qpos[self.rq:self.rq+3]=self.initial[self.rq:self.rq+3]+x[:3]
        q=(Rotation.from_rotvec(x[3:6])*self.initial_rotation).as_quat();d.qpos[self.rq+3:self.rq+7]=q[[3,0,1,2]];d.qpos[self.qa]=x[6:]
        exact=(elapsed==0 and angle==self.initial[self.lq] and operator==self.initial[self.oq] and latch==self.initial[self.bq])
        if exact:d.qpos[:]=self.initial
        elif correct:
            self._correct_arm('right',rhp,rhr);self._correct_arm('left',lhp,lhr)
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        x[6:]=d.qpos[self.qa]
        return dict(coordinates=x,qpos=d.qpos.copy(),release_clock_s=clock,right_follow_weight=follow,right_palm_position=rhp,right_palm_rotation=rhr,left_palm_position=lhp,left_palm_rotation=lhr)

    def close(self):self.sim.close()
