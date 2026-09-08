"""Measured-state privileged walking, readiness, grasp and partial-opening teacher.

Every native model here is an unstepped dynamics/geometry calculator. The active
simulator supplies measured states, contacts and mechanism poses; only original
capped robot motor forces are returned. This is not a sensor-only actor.
"""
import copy
from pathlib import Path

import mujoco
import numpy as np

from .acquisition_teacher import AcquisitionTeacher
from .approach_teacher import ApproachBodyTeacher
from .locomotion_approach import wrap_angle
from .operation_teacher import DoorOperationTeacher, pose_components


class ReadinessCollisionScreen:
    """Unstepped scene geometry with the same authoring frames as the active asset."""
    def __init__(self, door_xml, robot_xml):
        scene = mujoco.MjSpec.from_file(str(door_xml))
        scene.memory = 128*1024*1024
        robot = mujoco.MjSpec.from_file(str(robot_xml))
        frame = scene.worldbody.add_frame(pos=[0.,-1.5,0.])
        scene.attach(robot,prefix='robot/',frame=frame)
        self.m = scene.compile()
        self.root = int(self.m.jnt_qposadr[self.m.joint('robot/free_base').id])

    def check(self, proposal, handle_pose, leaf_pose, angles):
        m = self.m
        d = mujoco.MjData(m)
        names = proposal['acquisition']['joint_names']
        ids = np.array([m.joint('robot/'+n).id for n in names])
        qa = m.jnt_qposadr[ids]
        d.qpos[self.root:self.root+7] = proposal['initial_root']
        for key,name in [('operator','leaf_handle_hinge'),('leaf','leaf_hinge'),('latch','leaf_latch_bolt_slide')]:
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = float(angles[key])
        mujoco.mj_kinematics(m,d)
        frame_errors = {}
        for name,pose in [('leaf_handle',handle_pose),('leaf',leaf_pose)]:
            pos,rot = pose_components(pose)
            body = m.body(name).id
            frame_errors[name] = dict(position_m=float(np.linalg.norm(d.xpos[body]-pos)),
                                     rotation_matrix_error=float(np.linalg.norm(d.xmat[body].reshape(3,3)-rot)))
        if any(v['position_m']>.003 or v['rotation_matrix_error']>.02 for v in frame_errors.values()):
            return dict(passed=False,reason='Actual asset pose differs from readiness screen',frame_errors=frame_errors)
        bad = []
        for index,q in enumerate(proposal['acquisition']['path_qpos']):
            d.qpos[qa] = q
            mujoco.mj_kinematics(m,d)
            mujoco.mj_collision(m,d)
            contacts = []
            for contact in d.contact[:d.ncon]:
                bodies = [m.body(m.geom_bodyid[g]).name for g in contact.geom]
                if any(name.startswith('robot/rh_') for name in bodies):
                    contacts.append(dict(bodies=bodies,distance_m=float(contact.dist)))
            if contacts:
                bad.append(dict(sample=index,contacts=contacts))
        return dict(passed=not bad,samples=len(proposal['acquisition']['path_qpos']),
                    bad_samples=bad,frame_errors=frame_errors,native_mirror_steps=0)


class FullSequenceTeacher:
    """One force-only interface; measured hand loads inform the landed-foot QP."""
    def __init__(self,robot_xml,motors,reference,preparation,reset,checkpoint,joint_geometry,
                 *,door_xml,prepare_seconds=8.,operation_options=None):
        if not np.isfinite(prepare_seconds) or prepare_seconds<=0:
            raise ValueError('Preparation duration must be finite and positive')
        self.robot_xml = Path(robot_xml)
        self.motors = motors
        self.reference = reference
        self.preparation = preparation
        self.prepare_seconds = float(prepare_seconds)
        self.body = ApproachBodyTeacher(robot_xml,motors,reset,checkpoint,
            height=reference['initial_root'][2],handoff_delay=3.,yaw_weight=2.)
        self.acquisition = AcquisitionTeacher(robot_xml,motors,reference)
        self.operation = DoorOperationTeacher(self.acquisition,joint_geometry,**(operation_options or {}))
        self.screen = ReadinessCollisionScreen(door_xml,robot_xml)
        self.names = self.acquisition.names
        self.caps = self.acquisition.caps
        self.leg_local = np.array([i for i,v in enumerate(motors['actuators'])
            if any(k in v['name'] for k in ('hip_','knee','ankle'))])
        if len(self.leg_local)!=10:
            raise ValueError('Expected ten H1 leg motors')
        self.prep = self.prep_started = self.acquisition_started = self.quiet_since = None
        self.actual_preparation = self.readiness_screen = None
        self.handoffs = {}
        self.blocked_reason = None
        self.info = dict(phase='initial')

    def force(self,t,root,joints,velocities,foot_loads,handle_pose,leaf_pose,angles,hand_forces,
              *,grasp_qualified,hand_contact_count):
        """Consume actual state once per 2 ms; return 61 native-capped forces.

        Root is xyz+wxyz+world linear/world angular velocity. Poses are world
        xyz+wxyz. Angles contain operator/leaf radians and latch metres. Hand
        forces map body names to world vectors; contact count and strict pad
        qualification must come from the active plant. No reference contact is
        substituted for measured contact.
        """
        if not isinstance(hand_contact_count,(int,np.integer)) or hand_contact_count<0:
            raise ValueError('Actual nonnegative hand contact count required')
        if not isinstance(grasp_qualified,(bool,np.bool_)):
            raise ValueError('Actual grasp qualification must be explicit')
        root = np.asarray(root,float)
        body_force,body_info = self.body.force(t,root,joints,velocities,foot_loads,hand_forces=hand_forces)
        controller = self.body.controller
        force = body_force.copy()
        info = dict(phase=controller.stage)
        if self.prep is None and self.blocked_reason is None and controller.stage=='low stance hold':
            ready = (t-controller.stance_started>=5. and min(foot_loads)>30 and
                     abs(root[2]-self.reference['initial_root'][2])<.01 and
                     np.linalg.norm(root[7:10])<.02 and hand_contact_count==0)
            self.quiet_since = t if ready and self.quiet_since is None else self.quiet_since if ready else None
            if self.quiet_since is not None and t-self.quiet_since>=.5:
                proposal = copy.deepcopy(self.preparation)
                names = proposal['acquisition']['joint_names']
                path = np.asarray(proposal['acquisition']['path_qpos'],float)
                for index,name in enumerate(names):
                    if any(k in name for k in ('hip_','knee','ankle')):
                        path[:,index] = joints[name]
                path[0] = [joints[name] for name in names]
                proposal['acquisition']['path_qpos'] = path.tolist()
                proposal['initial_root'] = root[:7].tolist()
                self.actual_preparation = proposal
                self.readiness_screen = self.screen.check(proposal,handle_pose,leaf_pose,angles)
                if not self.readiness_screen['passed']:
                    self.blocked_reason = 'readiness_screen_failed'
                else:
                    self.prep = AcquisitionTeacher(self.robot_xml,self.motors,proposal,
                        reach_seconds=self.prepare_seconds,grip_force=0.,palm_integral=0.)
                    self.prep_started = t
                    _,rotation = pose_components(root[:7])
                    yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
                    self.handoffs['lowered'] = dict(time_s=t,root=root[:7].tolist(),
                        original_heading_error_deg=abs(float(np.rad2deg(wrap_angle(yaw-controller.waypoint.yaw)))))
        if self.prep is not None and self.acquisition_started is None:
            force,info = self.prep.force(t-self.prep_started,root,joints,velocities,handle_pose,hand_forces)
            info['phase'] = 'arm preparation'
            if info.get('path_fraction',0)>=.999 and t-self.prep_started>=self.prepare_seconds+2 and info['tracking_error_m']<.005 and hand_contact_count==0:
                self.acquisition_started = t
                self.handoffs['acquisition'] = dict(time_s=t,root=root[:7].tolist(),
                    palm_error_m=info['tracking_error_m'],hand_contact_count=hand_contact_count)
        if self.acquisition_started is not None:
            force,info = self.operation.force(t-self.acquisition_started,root,joints,velocities,
                handle_pose,leaf_pose,angles,hand_forces,grasp_qualified=grasp_qualified)
        # Retain the same actual landed-body controller through arm loading.
        force[self.leg_local] = body_force[self.leg_local]
        force = np.clip(force,self.caps[:,0],self.caps[:,1])
        if not np.isfinite(force).all():
            raise ValueError('Nonfinite full-sequence motor command')
        self.info = {**info,'body_stance_solver':body_info['stance_solver'],
            'body_stance_solver_failures':body_info['stance_solver_failures'],
            'native_mirror_steps':0,'blocked_reason':self.blocked_reason,
            'prep_started_s':self.prep_started,'acquisition_started_s':self.acquisition_started}
        return force,self.info.copy()
