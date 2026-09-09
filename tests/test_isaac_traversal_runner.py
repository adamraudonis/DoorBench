"""CPU contracts for the opt-in Isaac adapter; no engine execution is claimed."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1]/'scripts/dexterous/isaac_opening.py'
TREE = ast.parse(SOURCE.read_text())
HELPERS = {}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and
    n.name in ('validate_traversal_mode','actual_motor_delivery','independent_traversal_checks','controller_root_state')],
    type_ignores=[]), str(SOURCE), 'exec'), {'np':np,'math':math}, HELPERS)


def run_parser(monkeypatch, extra):
    module=ModuleType('isaaclab.app')
    module.AppLauncher=SimpleNamespace(add_app_launcher_args=lambda parser:None)
    monkeypatch.setitem(sys.modules,'isaaclab.app',module)
    monkeypatch.setattr(sys,'argv',[str(SOURCE),'--robot-usd','robot.usda',
        '--door-usd','door.usda','--motors','motors.json','--reference','reference.json',
        '--output','fresh-output',*extra])
    nodes=[]
    for node in TREE.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='launcher' for t in node.targets):
            break
        nodes.append(node)
    result={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),result)
    return result['a']


TRAVERSE_ARGS=['--traverse','--full-sequence-reset','body-reset.json','--full-opening',
    '--acquisition','--operate-after-acquisition','--preparation-reference','preparation.json',
    '--locomotion-checkpoint','h1.pt','--native-door','door.xml','--native-robot','robot.xml',
    '--left-palm-targets','left.json','--right-release-screen','release.json']


def test_real_parser_accepts_explicit_traversal_and_transfer_target(monkeypatch):
    args=run_parser(monkeypatch,[*TRAVERSE_ARGS,'--transfer-load-target','6'])
    assert args.traverse and args.transfer_load_target==6.
    assert args.time_scale==1. and args.target_aperture==1.2


def test_legacy_default_has_no_traversal_and_keeps_original_transfer_load(monkeypatch):
    args=run_parser(monkeypatch,[])
    assert not args.traverse and args.transfer_load_target==4.


def test_native_handoff_and_stow_settings_are_explicit(monkeypatch):
    options=['--opening-handoff-policy','loaded-hold-v2','--traversal-stow-phase-seconds','4']
    args=run_parser(monkeypatch,[*TRAVERSE_ARGS,*options])
    assert args.opening_handoff_policy=='loaded-hold-v2' and args.traversal_stow_phase_seconds==4.
    with pytest.raises(SystemExit):run_parser(monkeypatch,options)


@pytest.mark.parametrize('option,value',[
    ('--target-aperture','1.1'),('--time-scale','2'),
    ('--transfer-load-target','nan'),('--transfer-load-target','inf'),
    ('--transfer-load-target','2'),('--transfer-load-target','10.01'),
])
def test_parser_rejects_unsupported_clock_aperture_or_load(monkeypatch, option, value):
    with pytest.raises(SystemExit):run_parser(monkeypatch,[*TRAVERSE_ARGS,option,value])


@pytest.mark.parametrize('extra',[
    ['--traverse'],['--traverse','--full-opening'],['--transfer-load-target','6'],
    [*TRAVERSE_ARGS,'--sensor-policy-checkpoint','actor.pt'],
    [*TRAVERSE_ARGS,'--mechanism-test'],[*TRAVERSE_ARGS,'--panel-push'],
])
def test_mode_boundary_is_validated_before_engine_launch(monkeypatch, extra):
    with pytest.raises(SystemExit):run_parser(monkeypatch,extra)


def transmission():
    matrix=np.eye(61,69)
    for index in range(8):matrix[53+index,61+index]=1.
    return matrix,np.linalg.pinv(matrix.T)


def test_submitted_input_readback_inverse_uses_matching_pre_step_passive_terms():
    rng=np.random.default_rng(19);matrix,inverse=transmission()
    velocity=rng.normal(0,.3,69);damping=np.linspace(.1,1.,69);friction=np.full(69,.02)
    requested=rng.normal(0,20.,61)
    delivered=(matrix.T@requested-damping*velocity-friction*np.tanh(velocity/.001)).astype(np.float32)
    actual,residual=HELPERS['actual_motor_delivery'](delivered,velocity,matrix,inverse,damping,friction)
    np.testing.assert_allclose(actual,requested,atol=4e-6,rtol=0)
    assert residual<1e-5
    with pytest.raises(ValueError,match='transmission'):
        HELPERS['actual_motor_delivery'](delivered,-velocity,matrix,inverse,damping,friction)


def test_unactuated_differential_effort_cannot_be_hidden_by_inverse():
    matrix,inverse=transmission();delivered=np.zeros(69);delivered[-1]=.01
    with pytest.raises(ValueError,match='transmission'):
        HELPERS['actual_motor_delivery'](delivered,np.zeros(69),matrix,inverse,np.zeros(69),np.zeros(69))


def test_traversal_selects_actor_origin_velocity_and_preserves_legacy_mode():
    mixed=np.array([[.1,-.8,1.,1.,0.,0.,0.,.2,.1,0.,.3,.5,.7]])
    link=mixed.copy();com_offset=np.array([-.0002,.00004,-.04522])
    link[:,7:10]+=np.cross(mixed[:,10:13],-com_offset)
    data=SimpleNamespace(root_state_w=mixed,root_link_state_w=link)
    np.testing.assert_array_equal(HELPERS['controller_root_state'](data,traverse=False),mixed)
    result=HELPERS['controller_root_state'](data,traverse=True)
    np.testing.assert_array_equal(result,link)
    np.testing.assert_array_equal(result[:,:7],mixed[:,:7])
    np.testing.assert_array_equal(result[:,10:13],mixed[:,10:13])
    assert not np.array_equal(result[:,7:10],mixed[:,7:10])


def trial_rows():
    return [dict(time_s=i*.002,pose_time_s=i*.002,
        contact_interval_s=[max(0.,(i-1)*.002),i*.002],
        root=[.05,1.,1.04,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        post_started=i>=500,minimum_body_y_m=.93 if i>=500 else None,
        passage_completed=i>=900,foot_loads_N=[240.,240.],right_pad_patches_valid=True,
        continuation_evidence=dict(physics_qualified=True,left_hand_contacts=0,
            left_hand_load_N=0.,right_environment_contacts=0)) for i in range(1501)]


def evaluate(rows, *, opening=True, done=True, failure=None, physics=True):
    return HELPERS['independent_traversal_checks']({'joint_stops':physics,'sustained_pad_grasp':False},
        {'passed':opening},rows,dt=.002,end_s=3.,maximum_seconds=5.,controller_done=done,failure=failure)


def test_final_audit_qualifies_quiet_passage_without_demanding_released_grasp():
    checks=evaluate(trial_rows())
    assert all(checks.values())
    assert 'sustained_pad_grasp' not in checks


@pytest.mark.parametrize('change',[
    lambda rows:rows.pop(17),
    lambda rows:rows[44].update(right_pad_patches_valid=False),
    lambda rows:rows[44]['continuation_evidence'].update(physics_qualified=False),
    lambda rows:rows[800].update(pose_time_s=1.598),
    lambda rows:rows[800].update(contact_interval_s=[1.600,1.602]),
    lambda rows:rows[-1].update(minimum_body_y_m=.1),
    lambda rows:rows[-1].update(passage_completed=False),
    lambda rows:rows[-1]['root'].__setitem__(7,.031),
    lambda rows:rows[-1]['foot_loads_N'].__setitem__(1,9.),
    lambda rows:rows[-1]['continuation_evidence'].update(left_hand_contacts=1),
    lambda rows:rows[-1]['continuation_evidence'].update(left_hand_load_N=.2),
    lambda rows:rows[-1]['continuation_evidence'].update(right_environment_contacts=1),
    lambda rows:rows[-1]['root'].__setitem__(0,.1),
])
def test_independent_final_audit_rejects_prefix_gaps_and_bad_endpoints(change):
    rows=trial_rows();change(rows)
    assert not all(evaluate(rows).values())


@pytest.mark.parametrize('kwargs',[{'opening':False},{'done':False},{'failure':'stow failed'},{'physics':False}])
def test_good_endpoint_never_redeems_failed_opening_or_physics(kwargs):
    assert not all(evaluate(trial_rows(),**kwargs).values())


@pytest.mark.parametrize('change',[
    lambda rows:rows[-1].update(passage_completed='true'),
    lambda rows:rows[-1]['continuation_evidence'].update(left_hand_contacts=True),
    lambda rows:rows[-1]['continuation_evidence'].update(left_hand_load_N=float('nan')),
])
def test_bad_report_types_fail_closed(change):
    rows=trial_rows();change(rows)
    with pytest.raises(ValueError):evaluate(rows)


def test_loop_adds_no_runtime_robot_pose_or_joint_state_write():
    main=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loops=[n for n in ast.walk(main) if isinstance(n,ast.For) and
        isinstance(n.target,ast.Name) and n.target.id=='step']
    assert len(loops)==1
    forbidden={'write_root_pose_to_sim','write_root_velocity_to_sim','write_joint_state_to_sim'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
        n.func.attr in forbidden for n in ast.walk(loops[0]))
    calls=[n for n in ast.walk(loops[0]) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and
        n.func.value.id=='continuous' and n.func.attr=='force']
    assert len(calls)==1
    values={arg.arg:ast.unparse(arg.value) for arg in calls[0].keywords}
    assert values['applied_motor_forces']=='last_actual_motor_forces'
    assert values['body_poses']=="state['body_poses']"
    assert values['door_velocities']=="state['door_velocities']"


def test_source_preserves_independent_opening_prefix_and_actual_traversal_arrays():
    text=SOURCE.read_text()
    assert "if full_opening and (not continuous or frozen_opening_report is None):" in text
    assert "if not continuous:break" in text
    assert "if continuous.done:" in text
    assert "if sequence_reset or a.full_opening or sensor_control:" in text
    assert "acquisition_states['actual_motor_forces'].append(last_actual_motor_forces.copy())" in text
    assert "acquisition_states['actual_joint_effort'].append(delivered.copy())" in text
    assert "acquisition_states['legacy_root_state_w'].append(robot.data.root_state_w[0].cpu().numpy().copy())" in text
    assert "acquisition_states['root'].append(controller_root_state(robot.data,traverse=bool(continuous or a.sensor_locomotion_calibration))" in text
    assert "traversal-contacts.jsonl.gz" in text
