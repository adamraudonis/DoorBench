"""Contact-normal effort control with posture retained in the other motor directions.

Own-robot encoders and local tactile input only. This is an effort-space
hierarchy, not an assertion of dynamically decoupled Cartesian motion.
"""
import numpy as np
from .sensor_digit_force_control import SensorDigitForceController


HIERARCHICAL_PROTOCOL={
    'schema':'doorbench.sensor-hierarchical-digit-force.v1',
    'start_after_s':19.,'ramp_seconds':2.,
    'projection':'Euclidean orthogonal projection in original finger motor effort space',
    'transfer':'capture initial projected position effort as a fixed virtual-normal scalar',
    'initial_scalar_range_N':[0.,2.],
    'total_virtual_normal_range_N':[0.,4.],
    'retained':'orthogonal motor-space posture, all damping, robot bias dynamics, original caps',
    'inner_force_profile':'doorbench.sensor-digit-force.v1',
    'duration_s':36.,
}


def validate_hierarchical_protocol(protocol):
    if type(protocol) is not dict or set(protocol)!=set(HIERARCHICAL_PROTOCOL):raise ValueError('Exact declared hierarchical profile required')
    for key,expected in HIERARCHICAL_PROTOCOL.items():
        value=protocol[key]
        if isinstance(expected,float):
            if type(value) not in (int,float) or not np.isfinite(value) or value!=expected:raise ValueError('Unsupported hierarchical parameter: '+key)
        elif isinstance(expected,list):
            if type(value) is not list or len(value)!=len(expected) or any(type(x) not in (int,float) for x in value) or value!=expected:raise ValueError('Unsupported normal effort range')
        elif type(value) is not type(expected) or value!=expected:raise ValueError('Unsupported hierarchical parameter: '+key)
    return dict(protocol)


def normal_posture_transfer(position,unit_bias,groups,baseline,correction,alpha):
    """Return a pre-cap bias adjustment; unactuated joint moments never appear."""
    position=np.asarray(position,float);unit_bias=np.asarray(unit_bias,float)
    baseline=np.asarray(baseline,float);correction=np.asarray(correction,float);alpha=np.asarray(alpha,float)
    if (position.shape!=(61,) or unit_bias.shape!=(61,) or any(x.shape!=(5,) for x in [baseline,correction,alpha]) or
            not np.isfinite(np.r_[position,unit_bias,baseline,correction,alpha]).all() or np.any(alpha<0) or np.any(alpha>1) or
            np.any(baseline<0) or np.any(baseline>2) or np.any(abs(correction)>2.+1e-12)):
        raise ValueError('Finite bounded motor-space transfer required')
    adjustment=np.zeros(61);details={}
    for k,(digit,(_,rows,_,_)) in enumerate(groups.items()):
        p=unit_bias[rows];den=float(p@p)
        if den<1e-10:raise ValueError('Degenerate contact-normal actuation direction')
        equivalent=float(p@position[rows]/den);projected=p*equivalent
        total=float(np.clip(baseline[k]+correction[k],0.,4.))
        # The outer feedback has already added alpha*p*correction. Remove only
        # the current position-effort projection, retaining a fixed captured
        # effort to avoid erasing the established preload during handover.
        adjustment[rows]=alpha[k]*(p*(total-correction[k])-projected)
        details[digit]=dict(alpha=float(alpha[k]),initial_position_effort_equivalent_N=float(baseline[k]),
            current_position_effort_equivalent_N=equivalent,total_virtual_normal_effort_N=total,
            removed_position_effort_Nm=(alpha[k]*projected).tolist(),motor_bias_adjustment_Nm=adjustment[rows].tolist(),
            original_motor_names=None)
    return adjustment,details


class _ProjectedArm:
    """Intercept fully assembled joint goals before the unchanged capped adapter."""
    def __init__(self,arm):self.original=arm;self.owner=None;self.initial_normal=None;self.info={}
    def __getattr__(self,name):return getattr(self.original,name)
    def reset_episode(self):
        self.initial_normal=None;self.info={};self.original.reset_episode()
    def force(self,packet,*,now_s,joint_goals=None):
        arm=self.original;b=arm.balance;owner=self.owner;now=float(now_s)
        self.info={}
        if now>=19.-1e-9:
            if type(joint_goals) is not dict or set(joint_goals)!=set(arm.goal_names):raise ValueError('Complete named goals required before normal projection')
            if any(type(v) not in (int,float,np.float32,np.float64) for v in joint_goals.values()):raise ValueError('Numeric goals required before projection')
            goal=b.desired.copy()
            for n,v in joint_goals.items():goal[owner.joint_names.index(n)]=v
            q=packet['joint_position'].astype(float);target=b.matrix@goal;length=b.matrix@q
            position=b.kp*target+b.bias[:,1]*length
            unit,_=owner.pad_force.motor_bias(q,np.ones(5))
            groups=owner.pad_force.groups
            if self.initial_normal is None:
                self.initial_normal=np.array([unit[rows]@position[rows]/(unit[rows]@unit[rows]) for _,rows,_,_ in groups.values()])
                if np.any(self.initial_normal<0) or np.any(self.initial_normal>2):raise ValueError('Initial equivalent normal effort is outside frozen transfer range')
            weight=np.clip(owner.local_distal_loads(packet)/.2,0.,1.)
            t=float(np.clip((now-19.)/2.,0.,1.));alpha=weight*t**3*(10+t*(-15+6*t))
            adjustment,details=normal_posture_transfer(position,unit,groups,self.initial_normal,owner.virtual_force,alpha)
            # Outer controller reset this array to its original + current
            # virtual-force bias. Add the projection before caps/history.
            arm.original_constant_bias[:]+=adjustment
            for digit,(_,rows,_,_) in groups.items():details[digit]['original_motor_names']=[owner.action_names[r] for r in rows]
            self.info=dict(hierarchical_motor_bias_adjustment_Nm=adjustment.tolist(),normal_posture_transfer=details,
                additional_bias_after_projection_Nm=(arm.original_constant_bias-owner._base_motor_bias).tolist(),
                normal_control_scope='original finger motor effort span; not Cartesian dynamic decoupling')
        force,info=arm.force(packet,now_s=now_s,joint_goals=joint_goals)
        return force,dict(info,**self.info)


class SensorHierarchicalDigitForceController(SensorDigitForceController):
    def __init__(self,arm_controller,operation_schedule,sensor_layout,impedance_protocol,
                 motor_contract,index_protocol,force_protocol,hierarchical_protocol):
        self.hierarchical_protocol=validate_hierarchical_protocol(hierarchical_protocol)
        projected=_ProjectedArm(arm_controller)
        super().__init__(projected,operation_schedule,sensor_layout,impedance_protocol,motor_contract,index_protocol,force_protocol)
        projected.owner=self
    def force(self,packet,*,now_s):
        force,info=super().force(packet,now_s=now_s)
        self.info=dict(info,high_level_controller='hierarchical_local_digit_force_v1')
        return force,dict(self.info)
