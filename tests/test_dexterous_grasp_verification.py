"""Adversarial cases missed by centroid and 50 Hz grasp checks."""
import mujoco
import numpy as np
import pytest

from doorbench.dexterous.contact_audit import opposition
from doorbench.dexterous.grasp_verification import (
    audit_grasp_steps, pad_opposition, scalar_transmission_matrix,
    shadow_lever_pad_grasp,
)


def contacts():
    return [dict(digit=d,position=[i*.01,1. if d!='th' else -1.,0.],
                 normal_force_N=1.,pad_qualified=True) for i,d in enumerate(('ff','mf','rf','lf','th'))]


def test_opposite_loaded_patch_cannot_hide_behind_digit_centroid():
    rows=contacts();rows[0]['normal_force_N']=2.
    rows.append(dict(rows[0],position=[0.,-1.,0.],normal_force_N=.9))
    assert opposition(rows,[0,0,0],[1,0,0])['opposed']
    report=pad_opposition(rows,[0,0,0],[1,0,0])
    assert not report['valid_pad_grasp']
    assert report['misplaced_force_N']['ff']==.9


def test_dorsal_or_nonpad_touch_is_not_an_opposed_grasp():
    rows=contacts();rows[-1]['pad_qualified']=False
    assert opposition(rows,[0,0,0],[1,0,0])['opposed']
    assert not pad_opposition(rows,[0,0,0],[1,0,0])['valid_pad_grasp']


def test_thumb_must_oppose_each_finger_not_only_average():
    rows=contacts();angle=np.deg2rad(55)
    rows[0]['position']=[0,np.cos(angle),np.sin(angle)]
    # Thumb is opposite three fingers, but only 95 degrees from the fourth.
    rows[-1]['position']=[0,np.cos(np.deg2rad(150)),np.sin(np.deg2rad(150))]
    assert opposition(rows,[0,0,0],[1,0,0])['opposed']
    assert not pad_opposition(rows,[0,0,0],[1,0,0])['valid_pad_grasp']


def test_all_five_distal_pad_loads_pass_and_missing_evidence_fails():
    rows=contacts();assert pad_opposition(rows,[0,0,0],[1,0,0])['valid_pad_grasp']
    rows[0]['normal_force_N']=.19
    assert not pad_opposition(rows,[0,0,0],[1,0,0])['valid_pad_grasp']
    rows[0]['normal_force_N']=float('nan')
    with pytest.raises(ValueError):pad_opposition(rows,[0,0,0],[1,0,0])


def timeline():
    return [dict(sim_time_s=i*.002,door_q=0.,handle_angle_rad=0.,
                 root_height_m=1.,torso_tilt_deg=0.,max_joint_limit_violation_rad=0.,
                 max_nonfoot_penetration_m=0.,external_wrench_max=0.,
                 applied_generalized_force_max=0.,hand_contact_count=int(i>2),
                 finite=True,numerical_warnings=0,native_motor_limits=True,
                 pad_grasp=dict(valid_pad_grasp=i>=5)) for i in range(31)]


def audit(rows):
    return audit_grasp_steps(rows,physics_dt=.002,expected_duration=.06,required_hold=.02)


def test_every_physics_tick_required_even_when_sparse_frames_all_pass():
    rows=timeline();assert audit(rows)['passed']
    sparse=rows[::10]
    assert not audit(sparse)['checks']['complete_physics_step_evidence']
    assert not audit(sparse)['checks']['sustained_pad_grasp']


@pytest.mark.parametrize('field,bad,check',[
    ('max_joint_limit_violation_rad',.021,'physical_joint_limits'),
    ('max_nonfoot_penetration_m',.0031,'nonfoot_penetration'),
    ('native_motor_limits',False,'native_motor_limits'),
    ('external_wrench_max',.01,'no_external_assistance'),
])
def test_single_hidden_substep_spike_fails(field,bad,check):
    rows=timeline();rows[23][field]=bad
    assert not audit(rows)['checks'][check]


def test_transient_lost_grasp_and_already_moved_reset_fail():
    rows=timeline();rows[23]['pad_grasp']['valid_pad_grasp']=False
    assert not audit(rows)['checks']['sustained_pad_grasp']
    rows=timeline();rows[0]['handle_angle_rad']=.28
    assert not audit(rows)['checks']['resting_operator_start']
    rows=timeline();rows[0]['door_q']=.4
    assert not audit(rows)['checks']['closed_leaf_start']
    rows=timeline();rows[0]['hand_contact_count']=1
    assert not audit(rows)['checks']['contact_free_start']


def test_empty_partial_nonfinite_or_missing_records_never_pass():
    assert not audit([])['passed']
    assert not audit(timeline()[:-1])['passed']
    rows=timeline();rows[20]['finite']='true'
    assert not audit(rows)['passed']
    rows=timeline();rows[20]['torso_tilt_deg']=float('nan')
    assert not audit(rows)['passed']
    rows=timeline();del rows[20]['native_motor_limits']
    assert not audit(rows)['passed']


def tendon_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="a" axis="0 0 1"/><geom size=".01"/>
        <body pos=".1 0 0"><joint name="b" axis="0 1 0"/><geom size=".01"/></body>
      </body></worldbody><tendon><fixed name="pair"><joint joint="a" coef="1"/>
      <joint joint="b" coef=".7"/></fixed></tendon>
      <actuator><motor name="coupled" tendon="pair" gear="2.5"/>
      <motor name="direct" joint="a" gear="-1.3"/></actuator></mujoco>''')


def test_native_tendon_gear_and_derivatives_match_real_transmission():
    model=tendon_model();data=mujoco.MjData(model)
    matrix=scalar_transmission_matrix(model,[0,1],[0,1])
    for pose in ([.2,.6],[-.3,.1],[1.,-.4]):
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        np.testing.assert_allclose(matrix@data.qpos,data.actuator_length,atol=1e-12)
        for j in range(2):
            plus=np.array(pose);minus=plus.copy();plus[j]+=1e-6;minus[j]-=1e-6
            data.qpos[:]=plus;mujoco.mj_forward(model,data);hi=data.actuator_length.copy()
            data.qpos[:]=minus;mujoco.mj_forward(model,data);lo=data.actuator_length.copy()
            np.testing.assert_allclose((hi-lo)/2e-6,matrix[:,j],atol=1e-9)
    # The old sum-only rule silently ignores 2.5x motor gearing.
    assert not np.allclose(matrix[0],[1.,.7])


def test_incomplete_transmission_ordering_fails_explicitly():
    model=tendon_model()
    with pytest.raises(ValueError,match='missing'):scalar_transmission_matrix(model,[0],[0])
    with pytest.raises(ValueError,match='Duplicate'):scalar_transmission_matrix(model,[0],[0,0])


def test_empty_native_capsule_is_not_pad_grasp():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="lever" type="capsule" size=".007 .053"/></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    result=shadow_lever_pad_grasp(model,data,'lever')
    assert not result['valid_pad_grasp'] and result['contacts']==[]


@pytest.mark.parametrize('cap',[False,True])
def test_native_volar_contact_on_endcap_is_rejected(cap):
    # Same distal volar surface in both fixtures; only lever contact region differs.
    pose='pos="0 .02 .075" quat=".70710678 .70710678 0 0"' if cap else 'pos="0 .022 -.02"'
    axis='0 0 1' if cap else '0 1 0'
    gravity='0 0 -9.81' if cap else '0 -9.81 0'
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="{gravity}"/>
      <worldbody><geom name="lever" type="capsule" size=".007 .053"/>
      <body name="robot/rh_thdistal" {pose}><joint type="slide" axis="{axis}"/>
      <geom type="sphere" pos="0 -.012 .02" size=".004" mass=".1"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    report=shadow_lever_pad_grasp(model,data,'lever')
    assert len(report['contacts'])==1
    contact=report['contacts'][0]
    assert contact['normal_force_N']>0
    assert contact['body_position_m'][1]<-.001
    assert -contact['hand_outward_normal_body'][1]>.99
    assert contact['pad_qualified']==(not cap)
    assert (contact['axial_clearance_m']<0)==cap


def test_step_wrapper_preserves_wrench_that_doorenv_clears(monkeypatch):
    from types import SimpleNamespace
    from doorbench.dexterous import grasp_verification as module
    data=SimpleNamespace(xfrc_applied=np.zeros((2,6)))
    data.xfrc_applied[1,0]=7.
    def step():data.xfrc_applied[:]=0.
    sim=SimpleNamespace(d=data,plant=SimpleNamespace(step=step))
    def capture(sim,lever_geom,**kwargs):
        assert np.max(np.abs(sim.d.xfrc_applied))==0
        return kwargs
    monkeypatch.setattr(module,'native_grasp_sample',capture)
    record=module.audited_native_step(sim,'lever',handle_joint='handle')
    assert record['pre_step_external_wrench_max']==7.
