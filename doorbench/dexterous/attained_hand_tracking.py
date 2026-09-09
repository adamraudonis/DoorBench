"""Experimental attained finger posture through original coupled hand motors."""
import numpy as np


class AttainedHandTracking:
    def __init__(self,teacher,joints,previous_forces,*,stiffness_scale=5.):
        if not np.isfinite(stiffness_scale) or not 1<=stiffness_scale<=10:raise ValueError('Bounded hand stiffness scale required')
        self.teacher=teacher;self.indices=teacher.fingers.copy();self.scale=float(stiffness_scale)
        q=np.asarray([joints[n] for n in teacher.names],float)
        self.targets=(teacher.matrix@q)[self.indices]
        gravity=teacher.finger_inverse@teacher.d.qfrc_bias[teacher.va]
        self.preload=np.asarray(previous_forces,float)[self.indices]-gravity
        self.kp=teacher.kp[self.indices]*self.scale
        self.kd=(teacher.damping[self.indices]-teacher.bias[self.indices,2])*np.sqrt(self.scale)
        if not np.isfinite(np.r_[self.targets,self.preload,self.kp,self.kd]).all():raise ValueError('Finite coupled hand contract required')

    def force(self,forces,joints,velocities):
        teacher=self.teacher;q=np.asarray([joints[n] for n in teacher.names]);v=np.asarray([velocities[n] for n in teacher.names])
        if not np.isfinite(np.r_[q,v]).all():raise ValueError('Finite measured finger states required')
        length=(teacher.matrix@q)[self.indices];speed=(teacher.matrix@v)[self.indices]
        gravity=teacher.finger_inverse@teacher.d.qfrc_bias[teacher.va]
        requested=self.preload+gravity+self.kp*(self.targets-length)-self.kd*speed
        result=np.asarray(forces,float).copy();result[self.indices]=np.clip(requested,teacher.caps[self.indices,0],teacher.caps[self.indices,1])
        return result,dict(attained_finger_posture=True,finger_stiffness_scale=self.scale,maximum_finger_transmission_error_rad=float(np.max(abs(self.targets-length))),finger_clipped_motors=int(np.count_nonzero(result[self.indices]!=requested)))
