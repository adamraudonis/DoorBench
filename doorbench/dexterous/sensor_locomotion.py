"""Pinned H1 locomotion from encoders and the robot's own mounted IMU.

Upright startup is a declared calibration assumption; yaw/XY are arbitrary.
No live root pose, base velocity, door geometry, contact labels or task phase is
accepted. A constant body command is a baseline, not a learned door policy.
"""
from pathlib import Path
import hashlib
import json

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .locomotion import H1WalkingPolicy, JOINT_NAMES, DEFAULT_ANGLES
from .motor_target_control import MotorTargetControl
from .sensor_contract import ActorDimensions, SENSOR_KEYS, validate_actor_packet
from .grasp_verification import scalar_transmission_matrix
from .motor_contract_identity import motor_contract_fingerprint


def pelvis_rotation_increment(imu_before_pelvis,imu_after_pelvis,gyro_interval,dt):
    """Remove articulated IMU mount motion from its measured finite rotation."""
    return imu_before_pelvis@Rotation.from_rotvec(np.asarray(gyro_interval)*dt).as_matrix()@imu_after_pelvis.T


class SensorLocomotionController:
    @classmethod
    def from_calibration(cls,robot_xml,motors,layout,calibration,checkpoint):
        c=json.loads(Path(calibration).read_text())
        fields={'schema','motor_posture','body_command','robot_xml_sha256','motor_contract_sha256','checkpoint_sha256','initial_orientation'}
        if (set(c)!=fields or c['schema']!='doorbench.sensor-locomotion-calibration.v1'
                or c['initial_orientation']!='upright; yaw and XY arbitrary'
                or c['robot_xml_sha256']!=hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest()
                or c['motor_contract_sha256']!=motor_contract_fingerprint(motors)
                or c['checkpoint_sha256']!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()):
            raise ValueError('Bound robot-only locomotion calibration required')
        return cls(robot_xml,motors,layout,c['motor_posture'],checkpoint,c['body_command'])

    def __init__(self,robot_xml,motors,layout,motor_posture,checkpoint,body_command):
        if hashlib.sha256(Path(robot_xml).read_bytes()).hexdigest()!=motors['source_xml_sha256']:
            raise ValueError('Robot mechanics differ from calibration')
        self.motor=MotorTargetControl(motors)
        if (layout['joint_order']!=motors['joint_names'] or
                layout['action_order']!=[a['name'] for a in motors['actuators']] or
                layout['robot_xml_sha256']!=motors['source_xml_sha256']):
            raise ValueError('Sensor and motor calibration differ')
        self.m=mujoco.MjModel.from_xml_path(str(robot_xml));self.d=mujoco.MjData(self.m)
        m=self.m;self.qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in self.motor.names])
        self.va=np.array([m.jnt_dofadr[m.joint(n).id] for n in self.motor.names])
        aids=np.array([m.actuator(a['name']).id for a in motors['actuators']])
        jids=np.array([m.joint(n).id for n in self.motor.names])
        pairs=((self.motor.matrix,scalar_transmission_matrix(m,aids,jids)),
            (self.motor.kp,m.actuator_gainprm[aids,0]),(self.motor.bias,m.actuator_biasprm[aids,:3]),
            (self.motor.controls,m.actuator_ctrlrange[aids]),(self.motor.caps,m.actuator_forcerange[aids]))
        if m.opt.timestep!=.002 or any(not np.array_equal(x,y) for x,y in pairs):
            raise ValueError('Original robot timing, transmissions or motor coefficients differ')
        self.imu=m.site('imu').id;imu=layout['imu']
        self.delta_gyro=imu.get('gyro_profile')=='pose-delta-angle-v1'
        if imu.get('gyro_profile') not in (None,'pose-delta-angle-v1','backend-angular-velocity-v1'):
            raise ValueError('Unsupported mounted gyro semantics')
        if (m.body(m.site_bodyid[self.imu]).name!=imu['body_name'] or
                not np.array_equal(m.site_quat[self.imu],imu['quaternion_wxyz_body']) or
                not np.array_equal(m.site_pos[self.imu],imu['position_body_m'])):
            raise ValueError('IMU mount differs from static robot calibration')
        self.names=[a['name'] for a in motors['actuators']]
        if set(motor_posture)!=set(self.names):raise ValueError('Require only the original named motor posture')
        self.posture=np.array([motor_posture[n] for n in self.names])
        if not np.isfinite(self.posture).all() or np.any(self.posture<self.motor.controls[:,0]) or np.any(self.posture>self.motor.controls[:,1]):
            raise ValueError('Motor posture exceeds original target ranges')
        self.command=np.asarray(body_command,float)
        if self.command.shape!=(3,) or not np.isfinite(self.command).all() or np.any(abs(self.command)>[.3,.12,.4]):
            raise ValueError('Require a bounded constant body velocity command')
        self.legs=np.array([self.names.index(n) for n in JOINT_NAMES]);self.joints=np.array([self.motor.names.index(n) for n in JOINT_NAMES])
        self.phase_amplitude=1.
        self.walk=H1WalkingPolicy(checkpoint);self.shapes=ActorDimensions(tactile=layout['tactile_dimension']).shapes
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()
        self.action_semantics='pinned_h1_sensor_locomotion_constant_command_v1'
        self.checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        self._active=False

    def command_motion(self,command,*,phase_amplitude=1.):
        command=np.asarray(command,float)
        if command.shape!=(3,) or not np.isfinite(command).all() or np.any(abs(command)>[.3,.12,.4]) or not np.isfinite(phase_amplitude) or not 0<=phase_amplitude<=1:
            raise ValueError('Bounded body command and gait amplitude required')
        self.command=command.copy();self.phase_amplitude=float(phase_amplitude)

    def reset_episode(self):
        self.walk.reset();self.orientation=np.eye(3);self.last_time=None;self.history={}
        self.target=DEFAULT_ANGLES.copy();self.gyro=np.zeros(3);self.last_info={};self._active=True
        self.last_force=np.zeros(61)

    @property
    def previous_action(self):
        return (2*(self.last_force-self.motor.caps[:,0])/np.diff(self.motor.caps,axis=1)[:,0]-1).astype(np.float32)

    def force(self,packet,now_s):
        if not self._active:raise ValueError('Explicit reset required')
        try:return self._force(packet,now_s)
        except Exception:
            self._active=False;raise

    def _force(self,packet,now_s):
        validate_actor_packet(packet,self.shapes,61);t=float(now_s)
        if not np.isfinite(t) or (self.last_time is None and t!=0.) or (self.last_time is not None and abs(t-self.last_time-.002)>1e-8):
            raise ValueError('Require a consecutive 2 ms clock from reset')
        times=packet['sensor_time_s'];valid=packet['sensor_valid']
        if np.any(times>t+1e-8):raise ValueError('No future measurements allowed')
        if not np.allclose(packet['previous_action'],self.previous_action,atol=2e-7,rtol=0):
            raise ValueError('Previous action must be the actually submitted force')
        for k in ('joint_position','joint_velocity'):
            i=SENSOR_KEYS.index(k)
            if not valid[i] or abs(times[i]-t)>1e-8:raise ValueError('Current encoder measurements required')
        step=round(t/.002);q=packet['joint_position'];dq=packet['joint_velocity']
        self.history[step]=(q.copy(),dq.copy())
        if step:
            i=SENSOR_KEYS.index('imu_gyro');stamp=float(times[i])
            expected_stamp=t if self.delta_gyro else t-.002
            if not valid[i] or abs(stamp-expected_stamp)>1e-8:raise ValueError('Actual declared-epoch IMU sample required')
            old_q,old_dq=self.history[step-1]
            # Remove measured torso-joint motion at the IMU's actual epoch.
            # This private robot has no translation/world-pose knowledge.
            m,d=self.m,self.d;d.qpos[:]=m.qpos0;d.qpos[:3]=0.;d.qpos[3:7]=[1.,0.,0.,0.]
            d.qpos[self.qa]=old_q;d.qvel[:]=0.;d.qvel[self.va]=old_dq
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.imu)
            if self.delta_gyro:
                before=d.site_xmat[self.imu].reshape(3,3).copy()
                d.qpos[self.qa]=q;mujoco.mj_kinematics(m,d)
                after=d.site_xmat[self.imu].reshape(3,3).copy()
                # Exact finite rotation composition removes torso-joint motion
                # without treating an interval delta as endpoint angular rate.
                delta=pelvis_rotation_increment(before,after,packet['imu_gyro'],.002)
                self.gyro=Rotation.from_matrix(delta).as_rotvec()/.002
                self.orientation=self.orientation@delta
            else:
                self.gyro=d.site_xmat[self.imu].reshape(3,3)@packet['imu_gyro']-self.jr@d.qvel
                self.orientation=self.orientation@Rotation.from_rotvec(self.gyro*.002).as_matrix()
            del self.history[step-1]
        if step%10==0:
            self.target=self.walk.step(q[self.joints],dq[self.joints],self.gyro,
                self.orientation.T@np.array([0.,0.,-1.]),self.command,t,phase_amplitude=self.phase_amplitude)
        desired=H1WalkingPolicy.torques(self.target,q[self.joints],dq[self.joints])
        desired=np.clip(desired,self.motor.caps[self.legs,0],self.motor.caps[self.legs,1])
        bias=self.motor.state_bias(q,dq);u=self.posture.copy()
        u[self.legs]=np.clip((desired-bias[self.legs])/self.motor.kp[self.legs],
            self.motor.controls[self.legs,0],self.motor.controls[self.legs,1])
        forces=np.clip(self.motor.kp*u+bias,self.motor.caps[:,0],self.motor.caps[:,1])
        self.last_time=t
        self.last_force=forces.copy()
        self.last_info=dict(estimated_projected_gravity=(self.orientation.T@np.array([0.,0.,-1.])).tolist(),
            estimated_pelvis_gyro=self.gyro.tolist(),private_model_physics_steps=0,
            body_command=self.command.tolist(),upright_startup_calibration=True)
        return forces
