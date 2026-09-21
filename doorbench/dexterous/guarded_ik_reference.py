"""Opt-in reference experiment primitive, with no controller or plant access.

A common minimum-jerk coordinate interpolates each complete supplied IK vector.
This preserves the segment's convex joint box, not nonlinear palm/foot/contact
geometry. Every 500 Hz sample therefore requires a fresh caller-owned guard.
No force, state reset, physics step or acceptance threshold is provided here.
"""
from dataclasses import dataclass
import math

import numpy as np


def _vector(value,size,name):
    result=np.asarray(value,float)
    if result.shape!=(size,) or not np.isfinite(result).all():
        raise ValueError('Complete finite reference vector required: '+name)
    return result.copy()


def _readonly(value):
    result=np.asarray(value,float).copy();result.flags.writeable=False
    return result


@dataclass(frozen=True)
class IkReferenceSample:
    time_s: float
    names: tuple
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    goal_time_s: float
    latest_target_time_s: float
    segment_duration_s: float
    queued_target: bool


@dataclass(frozen=True)
class _Segment:
    start: float
    duration: float
    origin: np.ndarray
    goal: np.ndarray
    goal_time: float


class GuardedIkReference:
    """Causal rest-to-rest interpolation; newest queued IK goal replaces older.

    ``sample`` must run at exactly the declared motor cadence. ``target`` is
    provided only at a real IK refresh; no future endpoint is guessed. Updates
    arriving during a segment queue without breaking position, velocity or
    acceleration continuity. This introduces explicit lag, which the live
    geometry/contact guard may reject. Any rejection is terminal.

    Rate/acceleration/maximum-duration limits are mandatory prospective inputs;
    this utility does not assert that any values are physically qualified.
    """
    def __init__(self,names,initial,*,start_time_s,dt,lower,upper,
                 maximum_velocity,maximum_acceleration,minimum_segment_s,
                 maximum_segment_s):
        self.names=tuple(names);size=len(self.names)
        if not size or len(set(self.names))!=size or any(not isinstance(n,str) or not n for n in self.names):
            raise ValueError('Unique ordered reference coordinate names required')
        numbers=[start_time_s,dt,minimum_segment_s,maximum_segment_s]
        if not np.isfinite(numbers).all() or dt<=0 or minimum_segment_s<dt or maximum_segment_s<minimum_segment_s:
            raise ValueError('Finite explicit clock and segment bounds required')
        self.lower=_vector(lower,size,'lower');self.upper=_vector(upper,size,'upper')
        self.vmax=_vector(maximum_velocity,size,'velocity')
        self.amax=_vector(maximum_acceleration,size,'acceleration')
        self._position=_vector(initial,size,'initial')
        if (np.any(self.lower>=self.upper) or np.any(self.vmax<=0) or np.any(self.amax<=0)
            or np.any(self._position<self.lower) or np.any(self._position>self.upper)):
            raise ValueError('Initial joint box and positive motion bounds required')
        self.start=float(start_time_s);self.dt=float(dt)
        self.minimum=float(minimum_segment_s);self.maximum=float(maximum_segment_s)
        self.last_time=None;self.latest_target_time=self.start
        self._active=None;self._pending=None;self.failure=None

    def _duration(self,delta):
        # max s'(u)=15/8; max |s''(u)|=10/sqrt(3), u in [0,1].
        wanted=max(self.minimum,float(np.max(1.875*abs(delta)/self.vmax)),
            float(np.max(np.sqrt((10/math.sqrt(3))*abs(delta)/self.amax))))
        duration=max(1,math.ceil(wanted/self.dt-1e-12))*self.dt
        if duration>self.maximum+1e-12:
            raise ValueError('Requested IK change exceeds declared segment duration')
        return duration

    def sample(self,time_s,*,validate,target=None):
        """Return references only after the complete current-state guard passes.

        ``validate(sample)`` must return the literal bool True after checking
        the full reconstructed body reference against current measured root,
        feet, mechanism and contact geometry. A dummy guard in a mathematical
        test does not confer runtime admission. Feedback and motor caps stay
        the future caller's responsibility and must remain unchanged.
        """
        if self.failure is not None:raise ValueError('Reference interpolation is terminal after rejection')
        try:
            return self._sample(time_s,validate=validate,target=target)
        except Exception as error:
            stamp=float(time_s) if isinstance(time_s,(int,float,np.number)) else None
            self.failure=dict(time_s=stamp if stamp is not None and np.isfinite(stamp) else None,reason=str(error))
            raise

    def _sample(self,time_s,*,validate,target):
        t=float(time_s);expected=self.start if self.last_time is None else self.last_time+self.dt
        if not np.isfinite(t) or abs(t-expected)>1e-9:
            raise ValueError('Continuous motor-cadence reference clock required')
        if not callable(validate):raise ValueError('Fresh full-reference guard required')
        pending=self._pending;latest=self.latest_target_time
        if target is not None:
            goal=_vector(target,len(self.names),'new IK target')
            if np.any(goal<self.lower) or np.any(goal>self.upper):
                raise ValueError('IK endpoint outside original joint box')
            pending=(goal,t);latest=t
        active=self._active;origin=self._position
        if active is not None and t>=active.start+active.duration-1e-9:
            origin=active.goal;active=None
        if active is None and pending is not None:
            goal,goal_time=pending;pending=None
            if not np.array_equal(goal,origin):
                active=_Segment(t,self._duration(goal-origin),origin.copy(),goal.copy(),goal_time)
        if active is None:
            q=origin.copy();v=np.zeros_like(q);a=v.copy();goal_time=latest;duration=0.
        else:
            u=float(np.clip((t-active.start)/active.duration,0.,1.));delta=active.goal-active.origin
            s=u**3*(10+u*(-15+6*u));sd=30*u*u*(1-u)**2
            sdd=60*u*(1-u)*(1-2*u)
            q=active.origin+s*delta;v=sd/active.duration*delta;a=sdd/active.duration**2*delta
            goal_time=active.goal_time;duration=active.duration
        # Authoritative arithmetic checks after interpolation, never clipping.
        if (np.any(q<self.lower-1e-12) or np.any(q>self.upper+1e-12)
            or np.any(abs(v)>self.vmax+1e-12) or np.any(abs(a)>self.amax+1e-12)):
            raise ValueError('Interpolated reference violates declared motion bounds')
        result=IkReferenceSample(t,self.names,_readonly(q),_readonly(v),_readonly(a),
            goal_time,latest,duration,pending is not None)
        if validate(result) is not True:
            raise ValueError('Current full-reference geometry guard rejected interpolation')
        self._position=q.copy();self._active=active;self._pending=pending
        self.last_time=t;self.latest_target_time=latest
        return result
