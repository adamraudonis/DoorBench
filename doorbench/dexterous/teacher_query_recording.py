"""Evaluator-only pre-action measurements for offline teacher corrections.

This recorder accepts copies of simulator measurements. It neither constructs a
teacher nor proposes actions. Its privileged arrays must never enter actor
observations or imitation labels without a separately audited teacher query.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from .isaac_sensor_recording import _atomic_npz, _atomic_json


class TeacherQueryRecorder:
    def __init__(self, output, *, joint_names, hand_body_names, dt=.002):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.joint_names=tuple(joint_names);self.hand_body_names=tuple(hand_body_names)
        if len(self.joint_names)!=69 or len(set(self.joint_names))!=69:
            raise ValueError('Expected all 69 named robot joints')
        if not self.hand_body_names or len(set(self.hand_body_names))!=len(self.hand_body_names):
            raise ValueError('Expected unique measured hand body names')
        if dt!=.002:raise ValueError('This evidence contract requires the actual 500 Hz control cadence')
        self.dt=dt;self.rows={name:[] for name in ('time_s','root_state','joint_position','joint_velocity',
            'handle_pose','leaf_pose','door_position','right_hand_forces_world')}
        _atomic_json(self.output/'contract.json',dict(schema='doorbench.teacher-query-evidence.v1',
            scope=__doc__,joint_order=list(self.joint_names),hand_body_order=list(self.hand_body_names),
            door_position_order=['operator','leaf','latch'],pose_convention='xyz+wxyz; root13 includes world linear and angular velocity',
            force_semantics='Actual right-hand body normal-contact force vectors summed over handle and leaf filters, matching AcquisitionTeacher Isaac adapter; tangential forces and other counterparts are not included',
            clock='Measured before each executed actor decision, including reset t=0; no future state or interpolated impulses',
            contains_teacher_labels=False,actor_input=False,physics_dt_s=dt))

    def record(self, *, time_s, root_state, joint_position, joint_velocity,
               handle_pose, leaf_pose, door_position, right_hand_forces_world):
        values=locals();expected=len(self.rows['time_s'])*self.dt
        if not np.isfinite(time_s) or abs(time_s-expected)>1e-8:
            raise ValueError('Teacher query must record each causal pre-action measurement exactly once')
        shapes=dict(root_state=(13,),joint_position=(69,),joint_velocity=(69,),handle_pose=(7,),
            leaf_pose=(7,),door_position=(3,),right_hand_forces_world=(len(self.hand_body_names),3))
        copied={}
        for key,shape in shapes.items():
            value=np.array(values[key],dtype=np.float32,copy=True)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid measured '+key)
            copied[key]=value
        for key in ('root_state','handle_pose','leaf_pose'):
            if not np.isclose(np.linalg.norm(copied[key][3:7]),1.,atol=1e-5):raise ValueError('Measured quaternion must be normalized')
        self.rows['time_s'].append(float(time_s))
        for key,value in copied.items():self.rows[key].append(value)

    def finish(self, *, complete, executed_steps=None):
        executed_steps=len(self.rows['time_s']) if executed_steps is None else int(executed_steps)
        if not 0<=executed_steps<=len(self.rows['time_s']):raise ValueError('Executed steps differ from pre-action evidence')
        if complete and executed_steps!=len(self.rows['time_s']):raise ValueError('Complete capture cannot contain an unexecuted query')
        _atomic_npz(self.output/'pre-action-measurements.npz',**{k:np.asarray(v) for k,v in self.rows.items()})
        _atomic_json(self.output/'report.json',dict(capture_complete=bool(complete),samples=len(self.rows['time_s']),executed_steps=executed_steps,
            file_sha256={name:hashlib.sha256((self.output/name).read_bytes()).hexdigest() for name in ('contract.json','pre-action-measurements.npz')},
            contains_teacher_labels=False,actor_input=False,scope='Privileged measured-state evidence only; failed actor actions are not expert labels; an unexecuted final query in an interrupted capture is diagnostic only'))
