"""Privileged force-only composition of actual approach and bimanual opening.

This controller is a development candidate. Component results do not qualify the
combined motion. The active simulator provides every state/contact measurement;
the original landed-foot balance controller continues through manipulation.
"""
import numpy as np
from pathlib import Path

from .full_sequence_teacher import FullSequenceTeacher
from .full_opening_teacher import FullOpeningTeacher


class WalkingOpeningTeacher(FullSequenceTeacher):
    def __init__(self,robot_xml,motors,reference,preparation,reset,checkpoint,
                 joint_geometry,*,door_xml,left_targets,release_screen,
                 runtime_screen=None,opening_options=None,prepare_seconds=8.):
        door_xml=Path(door_xml)
        if door_xml.is_dir():door_xml=door_xml/'door.xml'
        super().__init__(robot_xml,motors,reference,preparation,reset,checkpoint,
                         joint_geometry,door_xml=door_xml,prepare_seconds=prepare_seconds)
        self.opening=FullOpeningTeacher(robot_xml,motors,reference,joint_geometry,
            door_xml=door_xml,left_targets=left_targets,release_screen=release_screen,
            runtime_screen=runtime_screen,**(opening_options or {}))
        # Use exactly the opening teacher's acquisition state/targets. This
        # replaces only the post-preparation primitive, never landed leg control.
        if self.names!=self.opening.names or not np.array_equal(self.caps,self.opening.caps):
            raise ValueError('Approach and opening motor contracts must match')
        self.acquisition=self.opening.acquisition
        self.operation=self.opening
        self._measurement=None

    def force(self,t,root,joints,velocities,foot_loads,handle_pose,leaf_pose,angles,
              hand_forces,*,evidence,right_palm_pose,pose_time_s,contact_interval_s):
        interval=np.asarray(contact_interval_s,float)
        dt=self.opening.physics_dt
        if not np.isfinite([t,pose_time_s]).all() or t<0 or abs(pose_time_s-t)>1e-7:
            raise ValueError('Current joint and body poses must share the episode clock')
        if interval.shape!=(2,) or not np.isfinite(interval).all() or not np.allclose(
                interval,[max(0.,t-dt),t],atol=1e-7,rtol=0.):
            raise ValueError('Actual contact interval must precede the current episode state')
        self._measurement=dict(episode_time_s=float(t),evidence=evidence,
            right_palm_pose=right_palm_pose,pose_time_s=pose_time_s,
            contact_interval_s=interval.copy())
        try:
            if self.opening.whole_body_return_path is not None and self.opening.release.started is not None:
                from .whole_body_return import apply_stance_goal
                # Preserve the same ordering as the independently checked native
                # intervention: attained-foot targets before this tick's leg QP.
                goal=self.opening.release.body_goal(t-self.acquisition_started)
                apply_stance_goal(self.body.controller,goal)
            force,info=super().force(t,root,joints,velocities,foot_loads,handle_pose,
                leaf_pose,angles,hand_forces,grasp_qualified=evidence['grasp_qualified'],
                hand_contact_count=evidence['hand_contact_count'])
            info['opening_clock_offset_s']=self.acquisition_started
            return force,info
        finally:
            self._measurement=None

    def _manipulation_force(self,t,root,joints,velocities,handle_pose,leaf_pose,
                            angles,hand_forces,grasp_qualified):
        measured=self._measurement
        if measured is None or self.acquisition_started is None:
            raise ValueError('Opening requires a current measured preparation handoff')
        offset=self.acquisition_started
        if abs(measured['episode_time_s']-offset-t)>1e-7:
            raise ValueError('Opening clock cannot skip or repeat the physical handoff')
        # At the contact-free handoff the preceding physical interval belongs to
        # preparation. Preserve that episode interval in diagnostics; a clipped
        # empty local interval does not count toward a manipulation contact hold.
        local_interval=np.maximum(0.,measured['contact_interval_s']-offset)
        force,info=self.opening.force(t,root,joints,velocities,handle_pose,leaf_pose,
            angles,hand_forces,evidence=measured['evidence'],
            right_palm_pose=measured['right_palm_pose'],
            pose_time_s=measured['pose_time_s']-offset,contact_interval_s=local_interval,
            episode_pose_time_s=measured['pose_time_s'])
        return force,{**info,'episode_pose_time_s':measured['pose_time_s'],
                      'episode_contact_interval_s':measured['contact_interval_s'].tolist(),
                      'opening_clock_offset_s':float(offset)}
