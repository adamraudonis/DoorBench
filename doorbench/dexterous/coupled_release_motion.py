"""Reference motion limits for an opt-in measured-angle release primitive."""
import numpy as np


class CoupledReferenceMotion:
    """Keep original body/joint reference limits before any motor is commanded.

    The state contains root displacement/rotvec followed by scalar joints.
    Geometry must be checked *after* this limit is applied: independently
    limiting coordinates does not by itself preserve a contact/foot constraint.
    """
    def __init__(self,initial):
        self.value=np.asarray(initial,float).copy()
        if self.value.ndim!=1 or len(self.value)<7 or not np.isfinite(self.value).all():raise ValueError('Finite coupled reference coordinates required')
        self.velocity=np.zeros_like(self.value);self.time=None

    def update(self,t,desired):
        desired=np.asarray(desired,float)
        if desired.shape!=self.value.shape or not np.isfinite(np.r_[t,desired]).all():raise ValueError('Finite complete coupled reference required')
        if self.time is None:
            if not np.allclose(desired,self.value,atol=1e-12,rtol=0):raise ValueError('Coupled motion must begin at exact attained reference')
            self.time=float(t)
            return self.value.copy(),dict(limited=False,joint_speed_rad_s=0.,joint_acceleration_rad_s2=0.,root_speed_m_s=0.,root_rotation_speed_rad_s=0.)
        dt=t-self.time
        if not 0<dt<=.00200001:raise ValueError('Coupled reference requires consecutive500Hz observations')
        requested=(desired-self.value)/dt;velocity=requested.copy()
        for sl,limit in [(slice(0,3),.02),(slice(3,6),.03)]:
            speed=np.linalg.norm(velocity[sl])
            if speed>limit:velocity[sl]*=limit/speed
        # Brake early enough to stop at a stationary target under the same
        # acceleration bound; clipping acceleration alone overshoots reversals.
        distance=desired[6:]-self.value[6:]
        # The discrete stopping-distance upper bound includes one eighth of
        # a*dt² beyond its continuous counterpart.
        stopping_speed=np.maximum(0.,np.sqrt(6.*abs(distance))-1.5*dt)
        velocity[6:]=np.sign(distance)*np.minimum(abs(velocity[6:]),stopping_speed)
        close=(abs(distance)<=3.*dt*dt)&(abs(self.velocity[6:])<=3.*dt)&(self.velocity[6:]*distance>=0.)
        velocity[6:]=np.where(close,distance/dt,velocity[6:])
        velocity[6:]=np.clip(velocity[6:],self.velocity[6:]-3.*dt,self.velocity[6:]+3.*dt)
        velocity[6:]=np.clip(velocity[6:],-1.2,1.2)
        acceleration=(velocity[6:]-self.velocity[6:])/dt
        result=self.value+velocity*dt
        info=dict(limited=not np.allclose(result,desired,atol=1e-12,rtol=0),joint_speed_rad_s=float(np.max(abs(velocity[6:]))),joint_acceleration_rad_s2=float(np.max(abs(acceleration))),root_speed_m_s=float(np.linalg.norm(velocity[:3])),root_rotation_speed_rad_s=float(np.linalg.norm(velocity[3:6])))
        self.value=result;self.velocity=velocity;self.time=float(t)
        return result.copy(),info
