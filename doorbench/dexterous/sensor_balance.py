"""Opt-in robot-only balance from encoders, IMU and local foot touch.

The calculator is a separate, unstepped robot model. Its floating state is an
estimate in an arbitrary local gauge, never the active simulator's state. The
fixed desired posture contains joint angles only. No door, world pose, teacher,
reference motion, object labels, or active-plant handle enters this interface.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_verification import scalar_transmission_matrix
from .locomotion_manipulation import LandedFootStanceController
from .sensor_contract import ActorObservationBuilder, SENSOR_KEYS, validate_actor_packet


class SensorBalanceController:
    """Five-second stationary support feasibility component, not a door policy.

    Both feet must remain supported. Initial orientation is the declared upright
    calibration; yaw and XY have no global meaning. RGB is validated but unused.
    The first cold packet uses only constant calibration/posture feedforward.
    """
    def __init__(self, robot_xml, motor_contract, sensor_layout, desired_posture,
                 *, physics_dt_s=.002, image_shape=(128,128,3), gravity_correction=.2,
                 solver_profile=None):
        if solver_profile not in (None,'fixed-rho-interval25-v1'):raise ValueError('Unknown explicit balance solver profile')
        self.solver_profile=solver_profile
        path=Path(robot_xml);motors=motor_contract;layout=sensor_layout
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if motors.get('source_xml_sha256')!=sha or layout.get('robot_xml_sha256')!=sha:
            raise ValueError('Robot bytes must match motor and sensor calibration')
        if motors.get('hand_mechanics_profile')!='shadow-loopback-v2':
            raise ValueError('Explicit corrected hand mechanics required')
        self.names=list(motors['joint_names']);self.actions=[a['name'] for a in motors['actuators']]
        if len(self.names)!=69 or len(set(self.names))!=69 or len(self.actions)!=61 or len(set(self.actions))!=61:
            raise ValueError('Expected original unique 69-joint / 61-motor contract')
        if layout.get('joint_order')!=self.names or layout.get('action_order')!=self.actions or layout.get('channel_order')!=['z','x','y'] or layout.get('interface_version')!='doorbench.sensors.v2':
            raise ValueError('Sensor ordering or interface differs from calibrated robot')
        if type(desired_posture) is not dict or set(desired_posture)!=set(self.names):
            raise ValueError('Desired posture must contain exactly the robot joint angles')
        self.desired=np.array([desired_posture[n] for n in self.names],float)
        self.dt=float(physics_dt_s);self.gravity_correction=float(gravity_correction)
        if not np.isfinite(self.desired).all() or not np.isfinite([self.dt,self.gravity_correction]).all() or self.dt!=.002 or not 0<=self.gravity_correction<=2:
            raise ValueError('Invalid constant posture or controller timing')
        self.m=m=mujoco.MjModel.from_xml_path(str(path));self.d=mujoco.MjData(m)
        if abs(m.opt.timestep-self.dt)>1e-12 or m.nu!=61:
            raise ValueError('Native timestep or actuator count changed')
        native_names=[m.joint(i).name for i in range(m.njnt) if m.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE]
        if native_names!=self.names or [m.actuator(i).name for i in range(m.nu)]!=self.actions or m.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE or m.nq!=76 or m.nv!=75:
            raise ValueError('Authored joint, root or motor ordering changed')
        self.joints=np.array([m.joint(n).id for n in self.names]);self.qa=m.jnt_qposadr[self.joints];self.va=m.jnt_dofadr[self.joints]
        self.act=np.array([m.actuator(n).id for n in self.actions]);self.matrix=scalar_transmission_matrix(m,self.act,self.joints)
        self.caps=np.array([a['force_range'] for a in motors['actuators']],float)
        if self.caps.shape!=(61,2) or not np.isfinite(self.caps).all() or not np.array_equal(self.caps,m.actuator_forcerange[self.act]):
            raise ValueError('Original native motor limits differ')
        self.kp=np.array([a['kp'] for a in motors['actuators']]);self.bias=np.array([a['bias'] for a in motors['actuators']])
        self.limits=np.array([a['control_range'] for a in motors['actuators']])
        if not np.array_equal(self.kp,m.actuator_gainprm[self.act,0]) or not np.array_equal(self.bias,m.actuator_biasprm[self.act,:3]) or not np.array_equal(self.limits,m.actuator_ctrlrange[self.act]):
            raise ValueError('Motor gains, affine bias or native controls differ')
        limited=m.jnt_limited[self.joints].astype(bool)
        if np.any(self.desired[limited]<m.jnt_range[self.joints,0][limited]-1e-7) or np.any(self.desired[limited]>m.jnt_range[self.joints,1][limited]+1e-7):
            raise ValueError('Constant posture exceeds authored joint bounds')
        for side in ('rh','lh'):
            for digit in ('FF','MF','RF','LF'):
                if self.desired[self.names.index(f'{side}_{digit}J1')]>self.desired[self.names.index(f'{side}_{digit}J2')]+1e-7:
                    raise ValueError('Constant posture violates passive loopback anatomy')
        # Only the calculator's motor command representation is normalized.
        m.actuator_gainprm[self.act,0]=1.;m.actuator_biasprm[self.act,:3]=0.;m.actuator_ctrlrange[self.act]=self.caps
        self.pelvis=m.body('pelvis').id;self.feet=[m.body(s+'_ankle_link').id for s in ('left','right')]
        self.imu=m.site('imu').id
        imu=layout['imu'];imu_body=m.body(m.site_bodyid[self.imu]).name
        if imu['body_name']!=imu_body or not np.allclose(imu['position_body_m'],m.site_pos[self.imu],atol=1e-12,rtol=0) or not np.allclose(imu['quaternion_wxyz_body'],m.site_quat[self.imu],atol=1e-12,rtol=0):
            raise ValueError('Torso IMU calibration mismatch')
        native_touch=[m.sensor(i).name for i in range(m.nsensor) if m.sensor(i).name.endswith('_touch')]
        if [s['name'] for s in layout['sensors']]!=native_touch:
            raise ValueError('Tactile sensor order differs from native calibration')
        plugin_nodes={n.get('name'):n for n in ET.fromstring(mujoco.MjSpec.from_file(str(path)).to_xml()).findall('./sensor/plugin')}
        self.foot_taxels=[];offset=0
        for row in layout['sensors']:
            dim=int(row['dimension']);sensor=m.sensor(row['name']).id
            if dim!=m.sensor_dim[sensor] or dim!=3*row['width']*row['height']:
                raise ValueError('Tactile bin dimensions differ from native calibration')
            site=int(m.sensor_objid[sensor]);quat=m.site_quat[site]
            native_config={x.get('key'):x.get('value') for x in plugin_nodes[row['name']].findall('config')}
            if (row['body_name']!=m.body(int(m.site_bodyid[site])).name
                    or not np.allclose(row['position_body_m'],m.site_pos[site],atol=1e-12,rtol=0)
                    or not np.allclose(row['quaternion_xyzw_body'],quat[[1,2,3,0]],atol=1e-12,rtol=0)
                    or [row['width'],row['height']]!=list(map(int,native_config['size'].split()))
                    or row['fov_degrees']!=list(map(float,native_config['fov'].split()))
                    or row['gamma']!=float(native_config.get('gamma',0))
                    or int(native_config.get('nchannel',1))!=3):
                raise ValueError('Tactile site, frame or bin calibration mismatch')
            if row['name'] in ('left_ankle_touch','right_ankle_touch'):
                self.foot_taxels.append((row['name'],slice(offset,offset+dim),dim//3))
            offset+=dim
        self.foot_taxels.sort(key=lambda row:row[0])
        if len(self.foot_taxels)!=2 or offset!=layout['tactile_dimension']:
            raise ValueError('Both calibrated ankle tactile grids required')
        self.shapes=ActorObservationBuilder(joint_count=69,action_count=61,tactile_dimension=offset,image_shape=image_shape).shapes
        self.upper=np.array([not any(k in n for k in ('hip_','knee','ankle')) for n in self.actions])
        self.high_impedance=np.array([n=='torso' or n.startswith(('left_','right_')) or 'WRJ' in n for n in self.actions]) & self.upper
        self.target=np.clip(self.matrix@self.desired,self.limits[:,0],self.limits[:,1])
        self.jp=np.zeros((3,m.nv));self.jr=self.jp.copy()
        self.reset_episode()

    def reset_episode(self):
        m,d=self.m,self.d;mujoco.mj_resetData(m,d)
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[self.qa]=self.desired
        mujoco.mj_forward(m,d)
        # Height follows the actual calibrated foot meshes, not a task root pose.
        sole=[]
        for body in self.feet:
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid]!=body or not (m.geom_contype[gid] or m.geom_conaffinity[gid]):continue
                if m.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_MESH:raise ValueError('Unsupported calibrated foot geometry')
                mesh=m.geom_dataid[gid];verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
                sole.extend((verts@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid])[:,2])
        d.qpos[2]=-min(sole);mujoco.mj_forward(m,d)
        self.imu_rotation=d.site_xmat[self.imu].reshape(3,3).copy()
        self.calibration_root=d.qpos[:3].copy()
        self.sim=SimpleNamespace(m=m,d=d,root_qadr=0,root_vadr=0,pelvis=self.pelvis,
            joint_prefix='',actuators=self.act,external_generalized_force=np.zeros(m.nv),
            stance_solver_settings={'max_iter':50000,'rho':.001})
        if self.solver_profile=='fixed-rho-interval25-v1':self.sim.stance_solver_settings['adaptive_rho_interval']=25
        self.stance=LandedFootStanceController(self.sim)
        self.failed_reason=None
        self.last_time=None;self.last_gyro_time=None;self.last_solver_time=-np.inf
        self.last_force=np.zeros(61);self.leg_force=None;self.status='uninitialized'
        self.qp_failures=0;self.ticks=0;self.last_info={};self.last_sensor_times=np.full(7,-1.)
        self.support_loads=np.zeros(2)

    def _estimate(self,packet):
        m,d=self.m,self.d
        q=packet['joint_position'].astype(float);v=packet['joint_velocity'].astype(float)
        gyro=packet['imu_gyro'].astype(float);accel=packet['imu_accelerometer'].astype(float)
        stamp=float(packet['sensor_time_s'][2])
        elapsed=0. if self.last_gyro_time is None else stamp-self.last_gyro_time
        if elapsed<0 or elapsed>.01+1e-10:raise ValueError('Unsupported IMU sample interval')
        if self.last_gyro_time is not None:
            self.imu_rotation=self.imu_rotation@Rotation.from_rotvec(gyro*elapsed).as_matrix()
            norm=np.linalg.norm(accel)
            if abs(norm-9.81)<1.:
                up=self.imu_rotation.T@np.array([0.,0.,1.])
                error=np.cross(accel/norm,up)
                self.imu_rotation=self.imu_rotation@Rotation.from_rotvec(self.gravity_correction*elapsed*error).as_matrix()
        self.last_gyro_time=stamp
        # Robot-only FK supplies the torso-to-pelvis transform from encoders.
        d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[self.qa]=q;d.qvel[:]=0;d.qvel[self.va]=v
        mujoco.mj_kinematics(m,d)
        relative_imu=d.site_xmat[self.imu].reshape(3,3).copy()
        pelvis_rotation=self.imu_rotation@relative_imu.T
        quat=Rotation.from_matrix(pelvis_rotation).as_quat();d.qpos[3:7]=quat[[3,0,1,2]]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        mujoco.mj_jacSite(m,d,self.jp,self.jr,self.imu)
        world_omega=self.imu_rotation@gyro-self.jr[:,self.va]@v
        d.qvel[3:6]=pelvis_rotation.T@world_omega
        # Foot positions are inferred support landmarks. They never constrain the
        # active plant. The feet can and must physically slip/fall in evaluation.
        for i,(_,indices,n) in enumerate(self.foot_taxels):
            channels=packet['tactile'][indices].reshape(3,n).sum(axis=1)
            self.support_loads[i]=np.linalg.norm(channels)
        weights=np.clip(self.support_loads,0.,500.)
        if weights.sum()<1.:weights=np.ones(2)
        weights=weights/weights.sum()
        relative_feet=d.xpos[self.feet].copy()
        d.qpos[:3]=np.sum(weights[:,None]*(self.stance.foot_positions-relative_feet),axis=0)
        velocities=[]
        for b in self.feet:
            mujoco.mj_jacBody(m,d,self.jp,self.jr,b)
            velocities.append(-(self.jp[:,3:]@d.qvel[3:]))
        d.qvel[:3]=np.sum(weights[:,None]*velocities,axis=0)
        mujoco.mj_forward(m,d)

    def force(self, packet, *, now_s):
        """Accept exact numeric packet+clock; any rejection requires episode reset."""
        if self.failed_reason is not None:
            raise RuntimeError('Rejected inference requires reset_episode: '+self.failed_reason)
        try:
            return self._force(packet,now_s=now_s)
        except Exception as exc:
            self.failed_reason=type(exc).__name__+': '+str(exc)
            raise

    def _force(self, packet, *, now_s):
        """Internal calculation; failures may mutate only the local estimator."""
        validate_actor_packet(packet,self.shapes,61)
        now=float(now_s)
        if not np.isfinite(now) or now<0 or (self.last_time is None and now!=0.) or (self.last_time is not None and abs(now-self.last_time-self.dt)>1e-8):
            raise ValueError('Reset at t0 and advance exactly one physics interval')
        times=packet['sensor_time_s'];valid=packet['sensor_valid']
        if valid.dtype!=np.bool_ or np.any(times>now+1e-10) or np.any(times< -1.) or np.any(times<self.last_sensor_times) or np.any(valid & (times<0)):
            raise ValueError('Invalid or noncausal sensor metadata')
        expected=self.last_force/np.maximum(abs(self.caps[:,0]),abs(self.caps[:,1]))
        if not np.allclose(packet['previous_action'],expected,atol=2e-7,rtol=0):
            raise ValueError('Previous action must be the last submitted motor force')
        cold=now==0 and not np.any(valid)
        if not cold:
            if not np.all(valid[:5]) or np.any(now-times[:5]>.006+1e-9):
                raise ValueError('Balance requires fresh encoders, IMU and foot touch')
            self._estimate(packet)
            if now>.05 and min(self.support_loads)<5.:
                raise RuntimeError('Stationary two-foot support hypothesis is unsupported by touch')
        m,d=self.m,self.d
        if now-self.last_solver_time>=.01-1e-9:
            leg,status=self.stance.command();self.status=status;self.last_solver_time=now
            if leg is None:self.qp_failures+=1
            else:self.leg_force=leg.copy()
        if self.leg_force is None:raise RuntimeError('No feasible calibrated support command')
        q=d.qpos[self.qa];v=d.qvel[self.va];length=self.matrix@q;speed=self.matrix@v
        force=self.kp*self.target+self.bias[:,0]+self.bias[:,1]*length+self.bias[:,2]*speed
        force+=self.high_impedance*(9*self.kp*(self.target-length)-np.array([.8 if 'WRJ' in n else 20. if n=='torso' else 10. for n in self.actions])*speed)
        for local in np.flatnonzero(self.upper):
            aid=self.act[local]
            if m.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
                force[local]+=d.qfrc_bias[m.jnt_dofadr[m.actuator_trnid[aid,0]]]
        force[self.stance.local]=self.leg_force
        self.last_force=np.clip(force,self.caps[:,0],self.caps[:,1])
        if not np.isfinite(self.last_force).all():raise RuntimeError('Nonfinite balance force')
        self.last_time=now;self.last_sensor_times=times.copy();self.ticks+=1
        self.last_info=dict(controller='sensor_balance_v1',scope='stationary support component',cold_start=cold,
            qp_status=self.status,qp_failures=self.qp_failures,estimated_root_local=d.qpos[:7].tolist(),
            qp_solver=dict(self.stance.last_solver_metadata),
            estimated_velocity_local=d.qvel[:6].tolist(),foot_tactile_force_norm_N=self.support_loads.tolist(),
            calculator_time_s=float(d.time),input_streams=list(SENSOR_KEYS[:5]))
        return self.last_force.copy(),dict(self.last_info)
