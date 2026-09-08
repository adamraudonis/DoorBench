"""Opt-in fixed palm-reference correction over the existing sensor estimator."""
import json
from pathlib import Path
import numpy as np
from .palm_ground_reference import ARM_NAMES, GroundPalmReference, RobotPalmReferenceIK
from .sensor_contract import validate_actor_packet

PROTOCOL=dict(schema='doorbench.sensor-palm-reference.v1',start_s=.2,ramp_seconds=.8,
    maximum_position_correction_m=.010,maximum_orientation_correction_rad=.05,
    maximum_joint_correction_rad=.05,maximum_correction_rate_radps=.25,
    root_estimate_lag_s=.002,physics_dt_s=.002,duration_s=19.,
    frame='initial pelvis yaw and XY removed; original ground Z retained',
    corrected_joints=list(ARM_NAMES),scope='Scripted ground-frame palm path; existing sensor estimator only; no object or active world pose input')


def corrected_goals(solver,reference,nominal,joints,previous_root,time_s,previous_correction):
    """Shared numeric correction for runtime and detached recorded-state screen."""
    if type(time_s) not in (int,float) or not np.isfinite(time_s) or not 0<=time_s<19.:
        raise ValueError('Declared nineteen-second decision clock required')
    previous=np.asarray(previous_correction,float)
    if previous.shape!=(7,) or not np.isfinite(previous).all():raise ValueError('Require seven previous correction values')
    u=float(np.clip((time_s-.2)/.8,0.,1.));weight=u**3*(10+u*(-15+6*u))
    if weight==0.:
        return dict(nominal),np.zeros(7),dict(weight=0.,position_error_m=0.,orientation_error_rad=0.,maximum_joint_correction_rad=0.,correction_rate_limited=False)
    position,rotation=reference.target(float(time_s))
    solved,info=solver.goals(nominal,joints,previous_root,position,rotation,weight=weight)
    initial=np.array([nominal[n] for n in ARM_NAMES]);wanted=np.array([solved[n] for n in ARM_NAMES])-initial
    correction=previous+np.clip(wanted-previous,-.25*.002,.25*.002)
    # The bounded solve lies inside exact original joint limits. Slew-limiting
    # toward it must also account for the separately moving nominal target.
    correction=np.clip(correction,solver.limits[:,0]-initial,solver.limits[:,1]-initial)
    if np.max(abs(correction-previous))>.25*.002+1e-8:
        raise ValueError('Moving nominal bound conflicts with correction rate limit')
    result=dict(nominal);result.update(zip(ARM_NAMES,(initial+correction).tolist()))
    return result,correction,dict(info,correction_rate_limited=bool(np.max(abs(correction-wanted))>1e-10),
        applied_correction_rad=correction.tolist(),target_position_ground_m=position.tolist())


class SensorPalmReferenceController:
    """Only packet/clock/static joint goals in; original capped force out.

    The previous decision's balance estimate is read before the single base
    force call. The base alone advances IMU/foot estimation and action history.
    The deliberately mixed epochs (previous root/current encoders) are logged.
    """
    def __init__(self,arm_controller,palm_reference,robot_sha,protocol=None):
        self.arm=arm_controller
        value=json.loads(Path(palm_reference).read_text()) if not isinstance(palm_reference,dict) else palm_reference
        if protocol is not None and protocol!=PROTOCOL:raise ValueError('Exact opt-in palm protocol required')
        self.reference=GroundPalmReference(value,robot_sha)
        self.solver=RobotPalmReferenceIK(self.arm.balance.m,self.arm.joint_names)
        self.reset_episode()

    def __getattr__(self,name):return getattr(self.arm,name)

    def reset_episode(self):
        self.arm.reset_episode();self.correction=np.zeros(7);self.failed_reason=None;self.info={}

    @property
    def last_info(self):return dict(self.info)

    def force(self,packet,*,now_s,joint_goals=None):
        if self.failed_reason is not None:raise RuntimeError('Palm correction requires reset_episode: '+self.failed_reason)
        try:
            validate_actor_packet(packet,self.arm.shapes,61)
            if type(now_s) not in (int,float) or not np.isfinite(now_s) or not 0<=now_s<19.:
                raise ValueError('Declared nineteen-second clock required')
            if type(joint_goals) is not dict or set(joint_goals)!=set(self.arm.goal_names):raise ValueError('Complete static joint goals required')
            b=self.arm.balance;root_epoch=b.last_time
            if now_s>.2:
                if root_epoch is None or abs(now_s-root_epoch-.002)>1e-8:raise ValueError('Require exactly previous-decision root estimate')
                if not packet['sensor_valid'][0] or not 0<=now_s-packet['sensor_time_s'][0]<=.006+1e-9:raise ValueError('Fresh encoder packet required')
            previous_root=b.last_info.get('estimated_root_local')
            goals,correction,detail=corrected_goals(self.solver,self.reference,joint_goals,packet['joint_position'],
                previous_root,now_s,self.correction)
            force,info=self.arm.force(packet,now_s=now_s,joint_goals=goals)
            self.correction=correction
            self.info=dict(info,high_level_controller='sensor_palm_reference_v1',palm_reference_correction=detail,
                correction_root_estimate_epoch_s=root_epoch,correction_encoder_epoch_s=float(packet['sensor_time_s'][0]),
                correction_uses_only_previous_estimate=True,source_ground_pelvis_height_m=self.reference.source_pelvis_height_m,
                estimator_initial_pelvis_height_m=float(b.calibration_root[2]),
                nominal_scripted_joint_goals=dict(joint_goals))
            return force,dict(self.info)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc);raise
