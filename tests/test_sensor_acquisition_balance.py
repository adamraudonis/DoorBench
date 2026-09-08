"""Contact-enabled scope and detached original distal-pad reconstruction."""
import copy
import inspect
import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from doorbench.dexterous.sensor_acquisition_schedule import ScriptedAcquisitionSchedule
from doorbench.dexterous.reach_balance_schedule import ReferenceReachSchedule
from scripts.dexterous.audit_sensor_acquisition_contacts import raw_pad_evidence


def test_full_contact_enabled_route_is_separate_from_qualified_prefix():
    names=['torso','wrist'];path=[[-.9,-.15],[-.2,-.15]]
    full=ScriptedAcquisitionSchedule(names,names,path)
    prefix=ReferenceReachSchedule(names,names,path)
    assert full.goals(19.)==pytest.approx({'torso':-.2,'wrist':-.15})
    assert prefix.goals(11.)['torso']==pytest.approx(-.9+.45*.7)
    assert full.goals(0.)==dict(zip(names,path[0]))
    assert full.duration_s==19.
    for t in np.arange(0,19.002,.002):assert full.goals(float(t))['wrist']==-.15
    assert list(inspect.signature(full.goals).parameters)==['t']
    with pytest.raises(ValueError):full.goals(float('nan'))


def fixture():
    names=['ff','mf','rf','lf','th']
    xml='<mujoco><worldbody><body name="lever"><geom name="lever" type="capsule" size=".01 .1"/></body>'
    for name in names:xml+=f'<body name="robot/rh_{name}distal"><geom name="{name}" type="box" size=".01 .01 .02"/></body>'
    m=mujoco.MjModel.from_xml_string(xml+'</worldbody></mujoco>');lever=m.geom('lever').id
    raw=dict(contacts=[],body_ids=[m.body('lever').id],body_positions_world_m=[[0.,0,0]],body_rotations_world=[np.eye(3).tolist()])
    for digit in names:
        side=-1. if digit=='th' else 1.;R=np.diag([side,side,1.])
        raw['body_ids'].append(m.body('robot/rh_'+digit+'distal').id)
        raw['body_positions_world_m'].append([0,.02*side,-.02]);raw['body_rotations_world'].append(R.tolist())
        raw['contacts'].append(dict(geom=[lever,m.geom(digit).id],position_world_m=[0,.01*side,0.],frame_world=[[0,side,0],[1.,0,0],[0,0,-side]],wrench_contact_frame=[1.,0,0,0,0,0]))
    return m,raw,lever


def test_actual_body_frame_and_opposed_contact_force_positive_control():
    m,r,lever=fixture();score,patches=raw_pad_evidence(m,r,lever)
    assert score['valid_pad_grasp'] and all(c['pad_qualified'] for c in patches)
    assert score['minimum_pairwise_finger_alignment']==pytest.approx(1.)
    assert score['maximum_thumb_finger_dot']==pytest.approx(-1.)
    assert score['qualified_pad_forces_N']=={k:1. for k in ('ff','mf','rf','lf','th')}


@pytest.mark.parametrize('corruption',['dorsal','endcap','missing_force','wrong_thumb_frame'])
def test_raw_contact_negative_controls_cannot_supply_valid_pad_grasp(corruption):
    m,r,lever=fixture()
    if corruption=='dorsal':r['contacts'][0]['frame_world'][0]=[0,-1.,0]
    if corruption=='endcap':r['contacts'][0]['position_world_m'][2]=.1
    if corruption=='missing_force':r['contacts'][0]['wrench_contact_frame'][0]=0.
    if corruption=='wrong_thumb_frame':r['body_rotations_world'][-1]=np.eye(3).tolist()
    score,_=raw_pad_evidence(m,r,lever)
    assert not score['valid_pad_grasp']
