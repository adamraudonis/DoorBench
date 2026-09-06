"""Source consistency cannot mutate mechanisms or claim task completion."""
import copy
import math
import mujoco
import numpy as np
import pytest
from doorbench.task_qa import run_task_declarations


def fixture(family='swing_single', hi=1.6):
    m=mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><worldbody><body name="leaf"><joint name="leaf_hinge" range="0 {hi}"/><geom type="box" size=".5 .02 1" mass="30"/></body></worldbody><equality><weld name="hold" body1="leaf"/></equality></mujoco>')
    s={'family':family,'kinematics':{'type':'hinge_vertical','max_open_deg':90},'lock':{'model':'none','engaged':False},'robot':{'approach_side':'-y'},'benchmark':{'scenarios':[{'name':'open_only','thresholds':{'open_rad':.4,'clear_rad':1.}}]}}
    return s,{'primary_joint':'leaf_hinge'},m


def test_read_only_and_no_native_completion_claim(tmp_path):
    s,meta,m=fixture();before=np.empty(mujoco.mj_sizeModel(m),dtype=np.uint8);mujoco.mj_saveModel(m,buffer=before)
    r=run_task_declarations(s,str(tmp_path),meta,m)
    after=np.empty_like(before);mujoco.mj_saveModel(m,buffer=after)
    assert np.array_equal(before,after)
    assert r['ok'] and not r['mechanically_verified']
    assert r['task_completion']=='requires_source_bound_native_episode'
    # A physically holding weld remains present; declarations alone cannot
    # establish a release or successful movement.
    assert bool(m.eq_active0[0])


def test_truncated_range_and_open_only_threshold_are_rejected(tmp_path):
    s,meta,m=fixture(hi=.03);r=run_task_declarations(s,str(tmp_path),meta,m)
    assert {f['rule'] for f in r['failures']}=={'travel','threshold_range'}


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,True])
def test_invalid_thresholds_fail(tmp_path,bad):
    s,meta,m=fixture();s['benchmark']['scenarios'][0]['thresholds']['open_rad']=bad
    assert not run_task_declarations(s,str(tmp_path),meta,m)['ok']


def test_lift_progress_is_not_compared_to_drum_angle(tmp_path):
    s,meta,m=fixture('rollup',hi=.1);s['kinematics']={'type':'slide_vertical','travel_m':3.}
    s['benchmark']['scenarios'][0]['thresholds']={'open_m':.3,'clear_m':1.9}
    meta['rollup_curtain']={'progress':{'closed_z_m':.02,'open_z_m':3.02}}
    r=run_task_declarations(s,str(tmp_path),meta,m)
    assert r['ok'] and r['coordinate']=='declared_bottom_height' and not r['mechanically_verified']
    s['benchmark']['scenarios'][0]['thresholds']['clear_m']=3.2
    assert not run_task_declarations(s,str(tmp_path),meta,m)['ok']


def test_download_only_pet_doors_have_no_evaluation_tasks(tmp_path):
    s,meta,m=fixture('pet_door');assert not run_task_declarations(s,str(tmp_path),meta,m)['ok']
    s['benchmark']['scenarios']=[];assert run_task_declarations(s,str(tmp_path),meta,m)['ok']


def test_inside_free_egress_uses_actual_trim_face(tmp_path):
    s,meta,m=fixture();s['lock']={'model':'keyed_lever','engaged':True,'robot_side_release':False}
    # The independent inside trim still releases without an outside key.
    meta['rotary_locksets']=[{'inside_face':-1.}]
    assert run_task_declarations(s,str(tmp_path),meta,m)['release_path']=='robot'
