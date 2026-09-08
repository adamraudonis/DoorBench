"""Endpoint teacher computation cannot overwrite actual transition evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import mujoco,numpy as np

spec=importlib.util.spec_from_file_location('post_opening_probe',Path(__file__).resolve().parents[1]/'scripts/dexterous/probe_post_opening.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_teacher_dynamics_data_are_separate_from_active_solver_buffers():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <geom type="plane" size="2 2 .1"/><body name="robot/lh_palm" pos="0 0 .099">
    <freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qvel[2]=-1.;mujoco.mj_forward(m,d);mujoco.mj_step(m,d)
    s=SimpleNamespace(m=m,d=d,joints=[],qadr=[],vadr=[],root_qadr=0,root_vadr=0,pelvis=1,actuators=[])
    fields=('qpos','qvel','qfrc_constraint','efc_force','xpos','xmat','actuator_force')
    before={k:getattr(d,k).copy() for k in fields}
    planner=probe.planning_state(s)
    assert planner.d is not d and not hasattr(planner,'plant')
    for k in fields:np.testing.assert_array_equal(getattr(d,k),before[k])
    assert planner.d.time==d.time
    assert not np.array_equal(planner.d.xpos,d.xpos)
    planner.d.qvel[2]=2.;mujoco.mj_forward(m,planner.d)
    for k in fields:np.testing.assert_array_equal(getattr(d,k),before[k])


def test_hand_release_uses_actual_archived_interval_contacts():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="robot/lh_palm"><geom type="sphere" size=".1"/></body>
    <body name="robot/rh_palm"><geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    raw=dict(contacts=[dict(body=[0,1],distance_m=-.001,wrench_contact_frame=[3,0,0,0,0,0]),
                       dict(body=[0,1],distance_m=.0002,wrench_contact_frame=[2,0,0,0,0,0]),
                       dict(body=[0,2],distance_m=-.001,wrench_contact_frame=[7,0,0,0,0,0])])
    assert probe.interval_hand_loads(m,raw)==(1,5.)
