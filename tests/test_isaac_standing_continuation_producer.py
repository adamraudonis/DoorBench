import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from doorbench.dexterous.isaac_standing_continuation_measurements import BODY_NAMES
from test_isaac_transfer_prefix_wiring import MAIN,Tensor,execute,names


def test_capture_receives_actual_poststep_body_origins_and_independent_buffers():
    block=next(node for node in ast.walk(MAIN) if isinstance(node,ast.If)
        and 'standing_continuation_steps' in names(node.test)
        and 'pack_standing_continuation' in names(node))
    robot_names=list(BODY_NAMES[:4]);door_names=list(BODY_NAMES[4:])
    robot_poses=np.array([[float(i),0.,0.,1.,0.,0.,0.] for i in range(4)])
    door_poses=np.array([[float(i+4),0.,0.,1.,0.,0.,0.] for i in range(2)])
    normal_buffers=tuple(np.array([float(i)]) for i in range(6))
    calls=[]
    def pack(**kwargs):calls.append(kwargs);return dict(foot_loads_N=[211.,212.])
    states=dict(continuation_body_poses=[],actual_foot_loads=[])
    rows=[]
    scope=dict(np=np,standing_continuation_steps=rows,BODY_NAMES=BODY_NAMES,pack_standing_continuation=pack,
        audit_contacts=SimpleNamespace(get_contact_data=lambda dt:tuple(Tensor(v) for v in normal_buffers)),
        robot=SimpleNamespace(body_names=robot_names,data=SimpleNamespace(body_state_w=Tensor(robot_poses[None]))),
        door=SimpleNamespace(body_names=door_names,data=SimpleNamespace(body_state_w=Tensor(door_poses[None]))),
        step=49,dt=.002,audit_paths=['captured paths'],audit_filters=['captured filters'],normal=np.array([1.]),
        transfer_friction=np.array([2.]),points=np.array([3.]),counts=np.array([4]),starts=np.array([5]),
        delivery_failure=None,acquisition_states=states)
    execute([block],scope)
    assert len(calls)==len(rows)==len(states['actual_foot_loads'])==1
    actual=calls[0]
    assert actual['time_s']==actual['pose_time_s']==.1 and actual['physics_qualified']
    assert tuple(actual['body_poses'])==BODY_NAMES
    np.testing.assert_array_equal(states['continuation_body_poses'][0],np.vstack([robot_poses,door_poses]))
    np.testing.assert_array_equal(states['actual_foot_loads'][0],[211.,212.])
    for original,recorded in zip(normal_buffers,actual['normal_buffers']):
        np.testing.assert_array_equal(original,recorded)
        assert not np.shares_memory(original,recorded)
    scope['delivery_failure']='failed submitted motor input'
    execute([block],scope)
    assert not calls[-1]['physics_qualified']


def test_capture_only_changes_explicit_withdrawal_mode_and_declares_helpers():
    assign=next(node for node in ast.walk(MAIN) if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='record_standing_continuation' for target in node.targets))
    for route,expected in ((None,False),('admitted-route.json',True)):
        scope=dict(a=SimpleNamespace(standing_withdrawal_route=route));execute([assign],scope)
        assert scope['record_standing_continuation'] is expected
    from scripts.isaac.run_local_operation import runtime_source_paths
    sources={path.name for path in runtime_source_paths(['--standing-withdrawal-route','route.json'])}
    assert {'isaac_standing_continuation_measurements.py','isaac_post_opening_measurements.py'}<=sources
