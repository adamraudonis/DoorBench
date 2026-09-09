"""Sensor-encoder realization of the original affine motor feedback law.

Static motor transmissions, gains, target limits and force caps are preserved.
This is an explicit alternative action space, never a reinterpretation of an
existing force checkpoint. No root state, task geometry or simulator is used.
"""
import numpy as np

MOTOR_TARGET_SCHEMA='doorbench.sensor-motor-target-actor.v1'


class MotorTargetControl:
    def __init__(self,motors):
        self.names=tuple(motors['joint_names']);actuators=motors['actuators']
        self.matrix=np.array([[a['terms'].get(n,0.) for n in self.names] for a in actuators],float)
        self.kp=np.array([a['kp'] for a in actuators],float)
        self.bias=np.array([a['bias'] for a in actuators],float)
        self.controls=np.array([a['control_range'] for a in actuators],float)
        self.caps=np.array([a['force_range'] for a in actuators],float)
        if (self.matrix.shape!=(61,69) or self.bias.shape!=(61,3) or
                not np.isfinite(np.r_[self.matrix.ravel(),self.kp,self.bias.ravel(),self.controls.ravel(),self.caps.ravel()]).all()
                or np.any(self.kp<=0) or np.any(np.diff(self.controls,axis=1)<=0) or np.any(np.diff(self.caps,axis=1)<=0)):
            raise ValueError('Finite original positive-gain motor contract required')

    def state_bias(self,q,dq):
        q,dq=np.asarray(q),np.asarray(dq)
        if q.shape!=dq.shape or q.shape[-1]!=69 or not np.isfinite(np.r_[q.ravel(),dq.ravel()]).all():
            raise ValueError('Matching finite original joint encoders required')
        return self.bias[:,0]+(q@self.matrix.T)*self.bias[:,1]+(dq@self.matrix.T)*self.bias[:,2]

    def forces(self,normalized,q,dq):
        normalized=np.asarray(normalized)
        if normalized.shape!=np.asarray(q).shape[:-1]+(61,) or not np.isfinite(normalized).all():
            raise ValueError('Finite target action required')
        u=self.controls[:,0]+(np.clip(normalized,-1.,1.)+1.)*.5*np.diff(self.controls,axis=1)[:,0]
        return np.clip(self.kp*u+self.state_bias(q,dq),self.caps[:,0],self.caps[:,1])

    def targets_for_forces(self,forces,q,dq,*,maximum_error_Nm=1e-4):
        forces=np.asarray(forces)
        u=np.clip((forces-self.state_bias(q,dq))/self.kp,self.controls[:,0],self.controls[:,1])
        target=2*(u-self.controls[:,0])/np.diff(self.controls,axis=1)[:,0]-1.
        actual=self.forces(target,q,dq)
        error=float(np.max(abs(actual-forces)))
        if not np.isfinite(error) or error>maximum_error_Nm:
            raise ValueError(f'Teacher forces not realizable through original motor targets: {error:.6g} Nm')
        return target.astype(np.float32)


class MotorTargetDemonstration:
    """Convert labels only; actual previous-action input remains a motor force."""
    def __init__(self,source,motors):
        self.source=source;self.control=MotorTargetControl(motors)
        for key in ('dimensions','layout','numeric','times','motor_contract_sha256'):
            setattr(self,key,getattr(source,key))
        self.metadata=dict(source.metadata,action_semantics='original_motor_target_v1')

    def __len__(self):return len(self.source)

    def sequence(self,start,length):
        inputs,normalized_force=self.source.sequence(start,length)
        caps=self.control.caps
        force=caps[:,0]+(normalized_force+1.)*.5*np.diff(caps,axis=1)[:,0]
        q=self.numeric['joint_position'][start:start+length]
        dq=self.numeric['joint_velocity'][start:start+length]
        target=self.control.targets_for_forces(force,q,dq)
        return inputs,target
