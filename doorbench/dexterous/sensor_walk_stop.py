"""Development walking-to-stance transition driven by own sensors only.

A fixed elapsed-time stop request is a component experiment, not visual approach.
No plant object enters this controller. Touch selects a supported handoff; only
private estimator state and motor targets are initialized at that handoff.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .sensor_balance import SensorBalanceController
from .sensor_locomotion import SensorLocomotionController
from .locomotion_manipulation import LandedFootStanceController


class SensorWalkStopController:
    @classmethod
    def from_calibration(cls,robot,motors,layout,calibration,checkpoint,stop_after_s):
        import json
        # Validate the exact same signed robot-only calibration before adding
        # the separately declared stop request; no task geometry is admitted.
        SensorLocomotionController.from_calibration(robot,motors,layout,calibration,checkpoint)
        c=json.loads(__import__('pathlib').Path(calibration).read_text())
        return cls(robot,motors,layout,c['motor_posture'],checkpoint,c['body_command'],stop_after_s)

    def __init__(self,robot,motors,layout,posture,checkpoint,command,stop_after_s,lower_to_m=None,stance_yaw_weight=2.,hip_spread=0.,leg_path=False):
        if not np.isfinite(stop_after_s) or stop_after_s<1:raise ValueError('Explicit stop request after at least one second required')
        if lower_to_m is not None and (not np.isfinite(lower_to_m) or not .8<=lower_to_m<=1.05):raise ValueError('Bounded robot stance height required')
        if not np.isfinite(stance_yaw_weight) or not 2<=stance_yaw_weight<=300:raise ValueError('Bounded stance yaw objective required')
        if leg_path and lower_to_m is None:raise ValueError('Leg path requires a lowering task')
        self.leg_path=bool(leg_path);self.lowering_nodes=None
        self.stance_yaw_weight=float(stance_yaw_weight)
        self.lower_to_m=lower_to_m
        self.walk=SensorLocomotionController(robot,motors,layout,posture,checkpoint,command)
        if not np.isfinite(hip_spread) or not 0<=hip_spread<=.12:raise ValueError('Bounded hip-spread calibration required')
        self.walk.hip_spread=float(hip_spread)
        self.initial_command=np.asarray(command,float).copy();self.inputs=(robot,motors,layout);self.stop_after=float(stop_after_s)
        offset=0;self.foot_slices=[]
        for row in layout['sensors']:
            size=row['dimension']
            if row['name'] in ('left_ankle_touch','right_ankle_touch'):self.foot_slices.append(slice(offset,offset+size))
            offset+=size
        if len(self.foot_slices)!=2:raise ValueError('Both calibrated foot sensors required')
        self.action_semantics='sensor_walk_supported_stop_v1';self.checkpoint_sha256=self.walk.checkpoint_sha256
        self.reset_episode()

    def reset_episode(self):
        self.walk.reset_episode();self.walk.command_motion(self.initial_command);self.brake_started=None;self.balance=None;self.lowering_nodes=None;self.support_since=None;self.handoff=None;self.last_info={}

    @property
    def previous_action(self):
        if self.balance is None:return self.walk.previous_action
        b=self.balance
        return (b.last_force/np.maximum(abs(b.caps[:,0]),abs(b.caps[:,1]))).astype(np.float32)

    def force(self,packet,now_s):
        t=float(now_s)
        if self.balance is not None:
            if self.lower_to_m is not None:
                u=float(np.clip((t-self.handoff['time_s']-1.)/4.,0.,1.));blend=u*u*u*(10.+u*(-15.+6.*u))
                if self.lowering_nodes is not None:
                    index=min(int(blend*40),39);weight=blend*40-index;node=(1-weight)*self.lowering_nodes[index]+weight*self.lowering_nodes[index+1]
                    self.balance.stance.joint_target=node[:10].copy();self.balance.stance.target_root=node[10:13].copy()
                    self.balance.stance.target_rotation=Rotation.from_euler('xyz',[*node[13:15],self.handoff['stance_heading_rad']]).as_matrix()
                else:self.balance.stance.target_root[2]=self.handoff['estimated_height_m']+blend*(self.lower_to_m-self.handoff['estimated_height_m'])
            force,info=self.balance.force(packet,now_s=t)
            self.last_info=dict(stage='stance',handoff=self.handoff,lower_to_m=self.lower_to_m,target_height_m=float(self.balance.stance.target_root[2]),**info);return force
        if self.brake_started is None and t>=self.stop_after and abs((t%.8)-.2)<.0011:self.brake_started=t
        if self.brake_started is not None:self.walk.command_motion([0.,0.,0.],phase_amplitude=max(0.,1.-(t-self.brake_started)))
        previous_force=self.walk.last_force.copy()
        force=self.walk.force(packet,t)
        loads=[float(np.linalg.norm(packet['tactile'][s].reshape(3,-1).sum(axis=1))) for s in self.foot_slices]
        w=self.walk;m,d=w.m,w.d
        d.qpos[3:7]=Rotation.from_matrix(w.orientation).as_quat()[[3,0,1,2]]
        d.qpos[w.qa]=packet['joint_position'];d.qvel[:]=0.;d.qvel[w.va]=packet['joint_velocity'];d.qvel[3:6]=w.gyro
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        estimates=[]
        for side in ('left','right'):
            mujoco.mj_jacBody(m,d,w.jp,w.jr,m.body(side+'_ankle_link').id)
            estimates.append(-(w.jp[:,3:]@d.qvel[3:]))
        speed=float(np.linalg.norm(np.average(estimates,axis=0,weights=np.maximum(loads,1.))[:2]))
        supported=self.brake_started is not None and t-self.brake_started>=3. and packet['sensor_valid'][4] and min(loads)>100 and speed<.02
        if not supported:self.support_since=None
        elif self.support_since is None:self.support_since=t
        if self.support_since is not None and t-self.support_since>=.02-1e-9:
            robot,motors,layout=self.inputs;w=self.walk
            q=packet['joint_position'].astype(float).copy()
            # Desired targets must be authored feasible even when measured joints
            # exhibit small solver stop penetration. This never moves the plant.
            original=q.copy()
            for i,name in enumerate(w.motor.names):
                joint=w.m.joint(name)
                if joint.limited:q[i]=np.clip(q[i],*joint.range)
            for side in ('lh','rh'):
                for digit in ('FF','MF','RF','LF'):
                    i=w.motor.names.index(f'{side}_{digit}J1');j=w.motor.names.index(f'{side}_{digit}J2');q[i]=min(q[i],q[j])
            desired=dict(zip(w.motor.names,q))
            b=SensorBalanceController(robot,motors,layout,desired,gravity_correction=0.,solver_profile='fixed-rho-interval25-v1')
            # Transfer the prior sensor-derived orientation in the same local
            # gauge. All arrays below belong to b's unstepped robot calculator.
            b.d.qpos[3:7]=Rotation.from_matrix(w.orientation).as_quat()[[3,0,1,2]]
            mujoco.mj_forward(b.m,b.d)
            b.imu_rotation=b.d.site_xmat[b.imu].reshape(3,3).copy()
            b.sim.stance_weights=np.r_[[200,200,1000,300,300,self.stance_yaw_weight],np.full(10,.01)]
            b.stance=LandedFootStanceController(b.sim)
            b.last_gyro_time=float(packet['sensor_time_s'][2]);b.last_time=t-.002
            b.last_force=previous_force;b.last_sensor_times=packet['sensor_time_s'].copy()
            self.handoff=dict(time_s=t,foot_touch_norms_N=loads,supported_for_s=t-self.support_since,estimated_speed_m_s=speed,
                estimated_height_m=float(b.stance.target_root[2]),stance_heading_rad=float(Rotation.from_matrix(b.stance.target_rotation).as_euler('xyz')[2]),desired_joint_projection_max_rad=float(np.max(abs(q-original))),source='own encoders, mounted IMU estimate and foot tactile grids',plant_pose_writes=0)
            if self.leg_path:
                from .sensor_stance_path import plan_lowering
                self.lowering_nodes,path_info=plan_lowering(b,self.lower_to_m);self.handoff['leg_path']=path_info
                b.sim.stance_weights[6:]=100.
            self.balance=b
            force,info=b.force(packet,now_s=t)
            self.last_info=dict(stage='stance',handoff=self.handoff,**info)
        else:self.last_info=dict(stage='walking',stop_requested=t>=self.stop_after,brake_started=self.brake_started,gait_amplitude=self.walk.phase_amplitude,foot_touch_norms_N=loads,estimated_speed_m_s=speed,**self.walk.last_info)
        return force
