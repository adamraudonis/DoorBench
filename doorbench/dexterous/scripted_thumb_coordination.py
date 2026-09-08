"""Instance-specific frozen thumb goals; no offline scene data enters runtime."""
import re
import numpy as np
from .sensor_thumb_flexion_force import SensorThumbFlexionForceController


THUMB_NAMES=('rh_THJ5','rh_THJ4','rh_THJ3','rh_THJ2','rh_THJ1')
FIXED={
    'schema':'doorbench.scripted-thumb-coordination.v1','start_s':23.,'ramp_s':1.,'duration_s':36.,
    'blend':'quintic blend from current pre-cap thumb goals to frozen attained-state candidate',
    'maximum_source_goal_change_rad':.03,
    'scope':'instance-specific scripted thumb goals; balance and pressure consume permitted encoders/IMU/local touch only',
    'pressure_axes':'THJ1/THJ2 unchanged; THJ3/4/5 are coordinated posture targets',
}
EXTRA=('source_goal_at_start','terminal_goal','robot_xml_sha256','source_candidate_sha256','source_trial_provenance_sha256')


def validate_coordination_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(FIXED)|set(EXTRA):raise ValueError('Exact frozen scripted-thumb profile required')
    for key,wanted in FIXED.items():
        value=protocol[key]
        if type(wanted) is float:
            if type(value) not in (int,float) or not np.isfinite(value) or value!=wanted:raise ValueError('Unsupported thumb schedule: '+key)
        elif type(value) is not type(wanted) or value!=wanted:raise ValueError('Unsupported thumb schedule: '+key)
    for key in ('source_goal_at_start','terminal_goal'):
        v=protocol[key]
        if type(v) is not dict or set(v)!=set(THUMB_NAMES) or any(type(x) not in (float,int) or not np.isfinite(x) for x in v.values()):
            raise ValueError('Exactly five finite named thumb goals required')
    if max(abs(protocol['terminal_goal'][n]-protocol['source_goal_at_start'][n]) for n in THUMB_NAMES)>.03:
        raise ValueError('Frozen thumb adjustment exceeds30mrad')
    for key in EXTRA[2:]:
        if type(protocol[key]) is not str or not re.fullmatch('[0-9a-f]{64}',protocol[key]):raise ValueError('Bound source hashes required')
    return {**protocol,'source_goal_at_start':dict(protocol['source_goal_at_start']),'terminal_goal':dict(protocol['terminal_goal'])}


class ScriptedThumbGoalCoordinator:
    def __init__(self,protocol):self.protocol=validate_coordination_protocol(protocol);self.reset()
    def reset(self):self.started=False;self.info={}
    def apply(self,goals,*,now_s):
        if isinstance(now_s,(bool,np.bool_)) or type(now_s) not in (int,float,np.float32,np.float64) or not np.isfinite(now_s) or now_s<0:
            raise ValueError('Finite numeric local schedule clock required')
        if type(goals) is not dict or not set(THUMB_NAMES)<=set(goals) or any(type(x) not in (int,float,np.float32,np.float64) or not np.isfinite(x) for x in goals.values()):
            raise ValueError('Finite named joint goals required')
        t=float(now_s);result=dict(goals);blend=0.
        if t>=23.-1e-9:
            if not self.started:
                if abs(t-23.)>1e-8 or max(abs(goals[n]-self.protocol['source_goal_at_start'][n]) for n in THUMB_NAMES)>1e-8:
                    raise ValueError('Actual23s starting goals differ from the frozen source; requalification required')
                self.started=True
            u=float(np.clip((t-23.)/1.,0,1));blend=u**3*(10+u*(-15+6*u))
            for n in THUMB_NAMES:result[n]=(1-blend)*goals[n]+blend*self.protocol['terminal_goal'][n]
        self.info=dict(scripted_thumb_coordination_blend=blend,scripted_thumb_goal={n:float(result[n]) for n in THUMB_NAMES},
            thumb_goals_before_coordination={n:float(goals[n]) for n in THUMB_NAMES},
            scripted_thumb_scope='frozen23s instance-specific goal coordination; no runtime scene/centroid/normal input')
        return result,dict(self.info)


class _CoordinatedArm:
    def __init__(self,inner,coordinator):self.inner=inner;self.coordinator=coordinator
    def __getattr__(self,name):return getattr(self.inner,name)
    def reset_episode(self):self.coordinator.reset();self.inner.reset_episode()
    def force(self,packet,*,now_s,joint_goals=None):
        goals,extra=self.coordinator.apply(joint_goals,now_s=now_s)
        # The normal-pressure projector must see the modified goals too. The
        # original capped force owner therefore remains inside this wrapper.
        force,info=self.inner.force(packet,now_s=now_s,joint_goals=goals)
        return force,dict(info,**extra)


class SensorScriptedThumbCoordinationController(SensorThumbFlexionForceController):
    def __init__(self,*args,coordination_protocol,**kwargs):
        self.coordination_protocol=validate_coordination_protocol(coordination_protocol)
        layout=args[2] if len(args)>2 else kwargs.get('sensor_layout')
        if type(layout) is not dict or layout.get('robot_xml_sha256')!=self.coordination_protocol['robot_xml_sha256']:
            raise ValueError('Scripted thumb goals are bound to the admitted robot design')
        super().__init__(*args,**kwargs)
        b=self.arm.balance
        for mapping in ('source_goal_at_start','terminal_goal'):
            for n,v in self.coordination_protocol[mapping].items():
                bounds=b.m.jnt_range[b.m.joint(n).id]
                if not bounds[0]<=v<=bounds[1]:raise ValueError('Frozen thumb goal exceeds original authored range')
        self.coordinator=ScriptedThumbGoalCoordinator(self.coordination_protocol)
        self.arm=_CoordinatedArm(self.arm,self.coordinator)
    def force(self,packet,*,now_s):
        force,info=super().force(packet,now_s=now_s)
        self.info=dict(info,high_level_controller='scripted_thumb_coordination_v1')
        return force,dict(self.info)
