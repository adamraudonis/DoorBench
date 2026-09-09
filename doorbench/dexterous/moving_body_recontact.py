"""Privileged moving-leaf targets, supplied only through original motor owners.

The caller records the preceding command and prepares each current target before
the stance QP. Repeated consumption at that same epoch must use identical input.
This experiment still requires independent actual and commanded-geometry audits.
"""
import copy
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .palm_recontact_teacher import TimedPalmRecontact
from .whole_body_contact_targets import WholeBodyContactTargets


class MovingBodyPalmRecontact(TimedPalmRecontact):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.body_planner=WholeBodyContactTargets(self.teacher.m,self.names)
        self.preceding=[]
        self.consumed=None
        self.body_motor=[]
        m=self.teacher.m
        for name in self.names:
            joint=m.joint(name).id
            motors=[i for i,a in enumerate(self.teacher.act)
                    if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a,0]==joint]
            if len(motors)!=1 or not np.array_equal(m.actuator_gear[self.teacher.act[motors[0]]],[1,0,0,0,0,0]):
                raise ValueError('Require original direct body motors')
            self.body_motor.append(motors[0])

    def remember_output(self,t,body_goal):
        if self.started is not None:
            raise ValueError('Cannot replace the preceding command after the transition')
        goal=copy.deepcopy(body_goal)
        target=self.teacher.target[self.body_motor].copy()
        for i,n in enumerate(self.names):
            if any(k in n for k in ('hip_','knee','ankle')):target[i]=goal['joints'][n]
        for n,v in zip(self.left.names[1:],self.left.target):target[self.names.index(n)]=v
        record=dict(time_s=float(t),position=np.asarray(goal['position'],float),
                    rotation=np.asarray(goal['rotation'],float),joints=target)
        if not np.isfinite(np.r_[t,record['position'],record['rotation'].ravel(),target]).all():
            raise ValueError('Finite preceding command required')
        if self.preceding and abs(t-self.preceding[-1]['time_s']-.002)>1e-8:
            raise ValueError('Require consecutive preceding command epochs')
        self.preceding=(self.preceding+[record])[-2:]

    def begin(self,t,root,joints,leaf_pose,angle):
        if len(self.preceding)!=2 or abs(t-self.preceding[-1]['time_s']-.002)>1e-8:
            raise ValueError('Require the two immediately preceding motor command samples')
        rotation=Rotation.from_quat([*root[4:7],root[3]])
        coordinates=[np.r_[r['position']-root[:3],
                      (Rotation.from_matrix(r['rotation'])*rotation.inv()).as_rotvec(),r['joints']]
                     for r in self.preceding]
        velocity=(coordinates[1]-coordinates[0])/.002
        self.body_planner.begin(t,root,joints,coordinate=coordinates[1],velocity=velocity)
        super().begin(t,root,joints,leaf_pose,angle)

    def adapt_body_targets(self,t,root,joints,position,rotation):
        x,v,a,info=self.body_planner.update(t,joints,position,rotation,self.latest['coordinate'])
        self.latest.update(coordinate=x,velocity=v,acceleration=a,whole_body_adaptation=info)
        return True

    def update(self,t,root,joints,leaf_pose,palm_load,angle,right_clear=True):
        if set(joints)!=set(self.body_planner.scalar) or right_clear is not True:
            raise ValueError('Complete current joints and actual right-hand release required')
        observation=np.r_[t,root,[joints[n] for n in self.body_planner.scalar],leaf_pose,palm_load,angle,float(right_clear)]
        if self.consumed is not None and self.consumed[0]==t:
            if not np.array_equal(observation,self.consumed):
                raise ValueError('Repeated target consumption changed the current observation')
            return
        super().update(t,root,joints,leaf_pose,palm_load,angle,right_clear)
        self.consumed=observation.copy()
        self.left.info['whole_body_adaptation']=dict(self.latest['whole_body_adaptation'])
