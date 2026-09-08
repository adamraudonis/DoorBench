"""Privileged task-space teacher: FK/IK only, never steps a second simulator.

Isaac is the plant. MuJoCo provides independent analytic robot kinematics.
Only bounded motor targets leave this module. Not a vision/tactile policy.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


class HandleTeacher:
    def __init__(self,robot_xml,motors,reference,stance_qp=False,grip_rotation_fraction=1.,grip_force=0.):
        spec=mujoco.MjSpec.from_file(robot_xml)
        spec.worldbody.add_body(name='analytic_lever').add_geom(name='analytic_lever_capsule',type=mujoco.mjtGeom.mjGEOM_CAPSULE,size=[.007,.053,0],quat=[2**-.5,0,2**-.5,0])
        self.m=spec.compile();self.d=mujoco.MjData(self.m)
        self.motors=motors;self.joint_names=motors['joint_names']
        self.initial=np.array(reference['controls'][min(40,len(reference['controls'])-1)]);self.control=self.initial.copy()
        names=reference.get('workspace_fit',{}).get('joint_names',['right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_yaw','rh_WRJ2','rh_WRJ1'])
        self.joints=[self.m.joint(n).id for n in names];self.qa=self.m.jnt_qposadr[self.joints];self.va=self.m.jnt_dofadr[self.joints]
        self.act=[next(i for i,m in enumerate(motors['actuators']) if set(m['terms'])=={n}) for n in names]
        self.fingers=[i for i,m in enumerate(motors['actuators']) if m['name'].startswith('rh_') and 'WRJ' not in m['name']]
        self.low=self.m.jnt_range[self.joints,0]+.015;self.high=self.m.jnt_range[self.joints,1]-.015
        self.palm=self.m.site('rh_palm_touch').id
        self.relative_position=None;self.release=None;self.push_start=None;self.retract_start=None;self.clear_steps=0
        self.jp=np.zeros((3,self.m.nv));self.jr=np.zeros_like(self.jp)
        self.stance_qp=stance_qp;self.stance=None;self.stance_status=None
        self.feedforward=np.zeros(len(motors['actuators']))
        self.workspace=reference.get('workspace_fit')
        self.workspace_limit=max(p[1] for p in self.workspace['poses']) if self.workspace else .12
        self.release_angle=max(.07,self.workspace_limit-.1)
        self.grip_rotation_fraction=grip_rotation_fraction
        self.stance_debug={}
        self.grip_force=grip_force;self.grip_hold=0;self.press_start=None;self.grip_debug={}
        self.lever_body=self.m.body('analytic_lever').id;self.lever_geom=self.m.geom('analytic_lever_capsule').id
        self.finger_geoms={digit:[g for g in range(self.m.ngeom) if self.m.geom_contype[g] and self.m.body(self.m.geom_bodyid[g]).name.startswith('rh_'+digit)] for digit in ('ff','mf','rf','lf','th')}
        self.motor_map=np.zeros((self.m.nv,len(motors['actuators'])))
        for i,motor in enumerate(motors['actuators']):
            for n,c in motor['terms'].items():self.motor_map[self.m.jnt_dofadr[self.m.joint(n).id],i]=c
        self.finger_inverse=np.linalg.pinv(self.motor_map[:,self.fingers])

    def command(self,t,root,joints,leaf,handle,handle_angle,door_angle,velocities=None,hand_forces=None,grasp=None):
        m,d=self.m,self.d;d.qpos[:7]=root[:7]
        for n,q in joints.items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=q
        mujoco.mj_kinematics(m,d)
        measured_finger_lengths=np.array([sum(c*joints[n] for n,c in self.motors['actuators'][i]['terms'].items()) for i in self.fingers])
        arm_bias=np.zeros(len(self.joints))
        if self.stance_qp:
            from types import SimpleNamespace
            from doorbench.dexterous.stance import StanceController
            d.qvel[:3]=root[7:10]
            rotation=Rotation.from_quat([*root[4:7],root[3]]).as_matrix()
            d.qvel[3:6]=rotation.T@root[10:13]
            if velocities:
                for n,v in velocities.items():d.qvel[m.jnt_dofadr[m.joint(n).id]]=v
            mujoco.mj_forward(m,d)
            arm_bias=d.qfrc_bias[self.va].copy()
            if self.stance is None:
                self.stance=StanceController(SimpleNamespace(m=m,d=d,joint_prefix='',actuators=np.arange(m.nu),root_qadr=0,root_vadr=0,pelvis=m.body('pelvis').id,
                    external_generalized_force=np.zeros(m.nv),stance_weights=np.r_[[40,40,60,100,100,2],np.ones(10)]))
            external=np.zeros(m.nv)
            for path,force in (hand_forces or {}).items():
                bid=m.body(path.rsplit('/',1)[-1]).id
                mujoco.mj_jacBody(m,d,self.jp,self.jr,bid);external+=self.jp.T@np.asarray(force)
            self.stance.sim.external_generalized_force=external
            command,self.stance_status=self.stance.command()
            if command is not None:self.control[self.stance.local]=command
            self.stance_debug=dict(target_root=self.stance.target_root.tolist(),root=root[:3].tolist(),
                acceleration=self.stance.last[:6].tolist() if self.stance.last is not None else None,
                planned_foot_wrenches=self.stance.last[-12:].tolist() if self.stance.last is not None else None)
        leaf_pos=np.array(leaf[:3]);leaf_rot=Rotation.from_quat([*leaf[4:7],leaf[3]]).as_matrix()
        hpos=np.array(handle[:3]);hrot=Rotation.from_quat([*handle[4:7],handle[3]]).as_matrix()
        reaction=sum((np.asarray(f) for f in (hand_forces or {}).values()),np.zeros(3))
        measured_push=max(0.,float(-reaction@leaf_rot[:,1]))
        push_scale=float(np.clip((30.-measured_push)/15.,0.,1.))
        grip_torque=np.zeros(m.nv)
        if self.grip_force:
            # A geometry query in the analytic model, never a force or pose write
            # to the Isaac plant. Only bounded finger motor torques leave here.
            m.body_pos[self.lever_body]=hpos+hrot@np.array([-.06,-.077,0.])
            m.body_quat[self.lever_body]=handle[3:7]
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            self.grip_debug={}
            for digit,geoms in self.finger_geoms.items():
                nearest=None
                for g in geoms:
                    pair=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,g,self.lever_geom,.1,pair)
                    if nearest is None or distance<nearest[0]:nearest=(distance,g,pair)
                distance,g,pair=nearest;delta=pair[3:]-pair[:3];length=np.linalg.norm(delta)
                self.grip_debug[digit]=dict(distance_m=float(distance),body=m.body(m.geom_bodyid[g]).name)
                if length<1e-7 or distance>.025:continue
                # When intersecting, closest-point displacement reverses sign.
                inward=delta/length*(1 if distance>=0 else -1)
                mujoco.mj_jac(m,d,self.jp,self.jr,pair[:3],int(m.geom_bodyid[g]))
                grip_torque+=self.jp.T@inward*self.grip_force
            self.grip_hold=self.grip_hold+1 if (grasp or {}).get('opposed') else 0
            if self.press_start is None and self.grip_hold>=5:self.press_start=t
        motion_time=t if not self.grip_force else 0. if self.press_start is None else t-self.press_start+1.
        if self.relative_position is None:
            self.relative_position=hrot.T@(d.site_xpos[self.palm]-hpos)
            self.relative_rotation=hrot.T@d.site_xmat[self.palm].reshape(3,3)
            self.handle_local=leaf_rot.T@(hpos-leaf_pos)
        if self.release is None and door_angle>self.release_angle:
            self.release=t;self.release_control=self.control.copy()
            # The lever spring returns after unloading. Keep the released hand
            # in its own leaf-relative pose instead of chasing that spring motion.
            self.release_position=leaf_rot.T@(d.site_xpos[self.palm]-leaf_pos)
            self.release_rotation=leaf_rot.T@d.site_xmat[self.palm].reshape(3,3)
        releasing=self.release is not None
        if releasing:
            loaded=sum(np.linalg.norm(f) for f in (hand_forces or {}).values())
            self.clear_steps=self.clear_steps+1 if loaded<.2 else 0
            if self.retract_start is None and (self.clear_steps>=3 or t-self.release>.6):
                self.retract_start=t
                self.withdraw_position=d.site_xpos[self.palm].copy()
                self.withdraw_rotation=d.site_xmat[self.palm].reshape(3,3).copy()
                self.withdraw_normal=leaf_rot[:,1].copy()
                self.withdraw_fingers=measured_finger_lengths.copy()
        retracting=self.retract_start is not None
        goal=min(.87*smooth((motion_time-1.)/2.),handle_angle+.06)
        # Release the physical bolt before pushing, otherwise strike friction
        # can bind a partly retracted latch under the opening load.
        if self.push_start is None and (handle_angle>.82 or door_angle>.015):self.push_start=t
        lead=.008*push_scale*smooth((t-self.push_start)/.3) if self.push_start is not None else 0.
        if self.push_start is not None:goal=.87
        if releasing:goal=handle_angle;lead=.005*push_scale
        lead=min(lead,max(0.,self.workspace_limit-.005-door_angle))
        future_leaf=leaf_rot@Rotation.from_rotvec([0,0,lead]).as_matrix()
        origin=leaf_pos+future_leaf@self.handle_local
        operator_rotation=future_leaf@Rotation.from_rotvec([0,-goal,0]).as_matrix()
        position=origin+operator_rotation@self.relative_position
        rotation=future_leaf@Rotation.from_rotvec([0,-goal*self.grip_rotation_fraction,0]).as_matrix()@self.relative_rotation
        if releasing:
            position=leaf_pos+future_leaf@self.release_position
            rotation=future_leaf@self.release_rotation
        if retracting:
            position=self.withdraw_position-.2*smooth((t-self.retract_start)/.5)*self.withdraw_normal
            rotation=self.withdraw_rotation
        tracking_vector=position-d.site_xpos[self.palm]
        tracking_error=float(np.linalg.norm(tracking_vector))
        tracking_angle=float(np.linalg.norm(Rotation.from_matrix(rotation@d.site_xmat[self.palm].reshape(3,3).T).as_rotvec()))
        mujoco.mj_jacSite(m,d,self.jp,self.jr,self.palm)
        measured_jac=self.jp[:,self.va].copy();measured_rot_jac=self.jr[:,self.va].copy()
        nominal=None
        if self.workspace and not releasing:
            poses=np.asarray(self.workspace['poses']);angles=np.asarray(self.workspace['arm_qpos'])
            pushing=self.push_start is not None
            mask=poses[:,0]>.86 if pushing else poses[:,1]==0
            coordinate=poses[mask,1] if pushing else poses[mask,0]
            query=door_angle+lead if pushing else goal
            nominal=np.array([np.interp(query,coordinate,angles[mask,i]) for i in range(len(self.joints))])
        for _ in range(35):
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            error=np.r_[(position-d.site_xpos[self.palm])*5,Rotation.from_matrix(rotation@d.site_xmat[self.palm].reshape(3,3).T).as_rotvec()]
            mujoco.mj_jacSite(m,d,self.jp,self.jr,self.palm)
            jac=np.vstack([self.jp[:,self.va]*5,self.jr[:,self.va]])
            inverse=jac.T@np.linalg.inv(jac@jac.T+.003*np.eye(6))
            change=inverse@error
            if nominal is not None:change+=(np.eye(len(self.joints))-inverse@jac)@(.15*(nominal-d.qpos[self.qa]))
            d.qpos[self.qa]=np.clip(d.qpos[self.qa]+np.clip(change,-.08,.08),self.low,self.high)
            if np.linalg.norm(error)<1e-4:break
        mujoco.mj_kinematics(m,d)
        self.control[self.act]=d.qpos[self.qa]
        # Apply pressure through the arm motors, with the native motor force caps
        # enforced by the plant adapter. This is not an external hand/door force.
        down=12.*smooth((motion_time-1.)/.8)
        push=18.*push_scale if self.push_start is not None else 0.
        force=leaf_rot@np.array([0.,push,-down])
        self.feedforward[:]=0.
        moment=leaf_rot@np.array([0.,-1.2*smooth((motion_time-1.)/.8),0.])
        self.feedforward[self.act]=arm_bias+measured_jac.T@force+measured_rot_jac.T@moment+grip_torque[self.va]
        self.feedforward[self.fingers]=self.finger_inverse@grip_torque
        if releasing:
            # Release along each digit's actual clearance direction before
            # retracting the wrist. A simultaneous all-zero hand pose can hook
            # the lever while the finger spread joints sweep sideways.
            self.control[self.fingers]=measured_finger_lengths
            self.release_control=self.control.copy()
            self.feedforward[:]=0.
            self.feedforward[self.fingers]=-self.finger_inverse@grip_torque
            self.feedforward[self.act]=arm_bias+measured_jac.T@(leaf_rot@np.array([0.,10.*push_scale,0.]))
        if retracting:
            self.control[self.fingers]=self.withdraw_fingers*(1-smooth((t-self.retract_start-.5)/.3))
            self.feedforward[:]=0.;self.feedforward[self.act]=arm_bias
        return self.control.copy(),dict(phase='withdraw' if retracting else 'release' if releasing else 'acquire' if self.grip_force and self.press_start is None else 'push' if self.push_start is not None else 'press',ik_error_m=float(np.linalg.norm(position-d.site_xpos[self.palm])),tracking_error_m=tracking_error,tracking_vector_m=tracking_vector.tolist(),tracking_angle_rad=tracking_angle,goal_handle_rad=float(goal),measured_push_N=measured_push,push_scale=push_scale,stance=self.stance_status,stance_debug=self.stance_debug,grip_debug=self.grip_debug)
