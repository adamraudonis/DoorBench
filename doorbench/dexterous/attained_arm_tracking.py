"""Experimental motor tracking of a screened arm route from a loaded pose.

The initial motor preload is retained as feedforward, adjusted for changing
modeled gravity. All actuation uses the original motors and force limits.
This privileged controller has no access to physical state setters.
"""
import numpy as np


class AttainedArmTracking:
    def __init__(self, teacher, joints, previous_forces):
        self.teacher=teacher
        torso=[i for i,act in enumerate(teacher.act) if teacher.m.joint(int(teacher.m.actuator_trnid[act,0])).name=='torso']
        self.act=np.unique(np.r_[teacher.arm_motors,np.asarray(torso,dtype=int)])
        self.names=[];self.va=[];self.qa=[]
        for i in self.act:
            actuator=teacher.act[i]
            joint=teacher.m.actuator_trnid[actuator,0]
            self.names.append(teacher.m.joint(int(joint)).name)
            self.va.append(int(teacher.m.jnt_dofadr[joint]))
            self.qa.append(int(teacher.m.jnt_qposadr[joint]))
        self.va=np.asarray(self.va);self.qa=np.asarray(self.qa)
        self.initial=np.asarray([joints[n] for n in self.names],float)
        self.preload=np.asarray(previous_forces,float)[self.act].copy()-teacher.d.qfrc_bias[self.va]
        self.kp=teacher.kp[self.act]*(1.+teacher.gain[self.act])
        self.kd=teacher.damping[self.act]-teacher.bias[self.act,2]
        if not np.isfinite(np.r_[self.initial,self.preload,self.kp,self.kd]).all():raise ValueError('Finite attained-arm motor contract required')
        self.previous_target=None;self.previous_time=None;self.last_call_time=None;self.reference_velocity=np.zeros_like(self.initial)

    def force(self, forces, t, target, joints, velocities, *, reference_velocity=None):
        target=np.asarray([target[n] for n in self.names],float)
        q=np.asarray([joints[n] for n in self.names]);v=np.asarray([velocities[n] for n in self.names])
        if not np.isfinite(np.r_[target,q,v,t]).all():raise ValueError('Finite arm target/state required')
        if self.last_call_time is not None and t<self.last_call_time:raise ValueError('Monotonic arm clock required')
        changed=self.previous_target is None or not np.array_equal(target,self.previous_target)
        velocity=self.reference_velocity.copy()
        if changed:
            velocity=np.zeros_like(target) if self.previous_time is None else (target-self.previous_target)/max(t-self.previous_time,1e-12)
        elif self.previous_time is not None and t-self.previous_time>.02:
            velocity=np.zeros_like(target)
        if reference_velocity is not None:
            # Caller differentiates independently clocked target components.
            # This changes only target feedforward, never the measured state.
            velocity=np.array([reference_velocity[n] for n in self.names],float)
            if not np.isfinite(velocity).all():raise ValueError('Finite arm reference velocity required')
        # A screened path is slow, but reject an accidental target/clock jump.
        if np.max(abs(velocity))>2.:raise ValueError('Screened arm reference exceeds 2 rad/s')
        result=np.asarray(forces,float).copy()
        requested=self.preload+self.teacher.d.qfrc_bias[self.va]+self.kp*(target-q)+self.kd*(velocity-v)
        result[self.act]=np.clip(requested,self.teacher.caps[self.act,0],self.teacher.caps[self.act,1])
        if changed:self.previous_target=target.copy();self.previous_time=t
        self.reference_velocity=velocity.copy();self.last_call_time=t
        return result,dict(attained_arm_tracking=True,arm_reference_velocity_rad_s=velocity.tolist(),maximum_arm_target_error_rad=float(np.max(abs(target-q))),arm_clipped_motors=int(np.count_nonzero(result[self.act]!=requested)))
