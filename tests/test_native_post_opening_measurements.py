"""The numeric adapter does not write poses or misclassify self contact as a door."""
from types import SimpleNamespace
import mujoco,numpy as np
import pytest
from doorbench.dexterous.native_post_opening_measurements import measured_contacts,measured_state,outward_release_normal,qualified_physics_row


def fixture():
    m=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
    <body name="leaf"><joint name="leaf_hinge" axis="0 0 1"/><geom type="sphere" size=".1"/></body>
    <body name="robot/pelvis" pos="0 -1 1"><freejoint name="robot/free_base"/><geom type="sphere" size=".1"/>
      <body name="robot/lh_palm" pos="0 0 .3"><joint name="robot/elbow"/><geom type="sphere" size=".1"/></body>
      <body name="robot/rh_palm" pos="0 .2 .3"><geom type="sphere" size=".1"/></body>
      <body name="robot/rh_ffdistal" pos="0 .3 .3"><geom type="sphere" size=".1"/></body>
    </body></worldbody></mujoco>''');d=mujoco.MjData(m);j=m.joint('robot/free_base').id
    s=SimpleNamespace(m=m,d=d,root_qadr=int(m.jnt_qposadr[j]),root_vadr=int(m.jnt_dofadr[j]),pelvis=m.body('robot/pelvis').id)
    mujoco.mj_forward(m,d);return s


def contact(m,a,b,force=1):
    return dict(body=[m.body(a).id,m.body(b).id],frame_world=[[0,1,0],[1,0,0],[0,0,-1]],wrench_contact_frame=[force,0,0,0,0,0],distance_m=-.0001)


def test_right_self_contact_does_not_falsely_block_environment_release():
    s=fixture();m=s.m;raw=dict(contacts=[contact(m,'robot/rh_palm','robot/rh_ffdistal'),contact(m,'leaf','robot/lh_palm',3)])
    _,forces,e=measured_contacts(m,raw,physics_qualified=True)
    assert e['right_environment_contacts']==0 and e['left_hand_contacts']==1 and e['left_hand_load_N']==3
    np.testing.assert_array_equal(forces['lh_palm'],[0,3,0])
    np.testing.assert_array_equal(outward_release_normal(m,raw),[0,1,0])
    raw['contacts'].append(contact(m,'leaf','robot/rh_palm'))
    assert measured_contacts(m,raw,physics_qualified=False)[2]['right_environment_contacts']==1
    assert not measured_contacts(m,raw,physics_qualified=False)[2]['physics_qualified']
    c=contact(m,'leaf','robot/rh_palm');c['distance_m']=.0001
    assert measured_contacts(m,dict(contacts=[c]),physics_qualified=True)[2]['right_environment_contacts']==1
    c['wrench_contact_frame'][0]=0.
    assert measured_contacts(m,dict(contacts=[c]),physics_qualified=True)[2]['right_environment_contacts']==0
    left=contact(m,'leaf','robot/lh_palm',.01);left['distance_m']=.0001
    assert measured_contacts(m,dict(contacts=[left]),physics_qualified=True)[2]['left_hand_contacts']==1
    left['wrench_contact_frame'][0]=0.
    assert measured_contacts(m,dict(contacts=[left]),physics_qualified=True)[2]['left_hand_contacts']==0


def test_current_state_reader_preserves_every_active_state_and_force_array():
    s=fixture();d=s.d;r=s.root_qadr;v=s.root_vadr
    d.qpos[r+3:r+7]=[np.sqrt(.5),0,0,np.sqrt(.5)];d.qvel[v+3:v+6]=[1,2,3];mujoco.mj_forward(s.m,d)
    before={name:getattr(d,name).copy() for name in ('qpos','qvel','ctrl','xpos','xmat','qfrc_constraint','actuator_force')}
    root,joints,velocities,other=measured_state(s,['elbow'],['leaf_hinge'],['pelvis','leaf'])
    np.testing.assert_allclose(root[10:13],[-2,1,3],atol=1e-15)
    assert set(joints)==set(velocities)=={'elbow'} and set(other['door_positions'])=={'leaf_hinge'}
    for name,value in before.items():np.testing.assert_array_equal(getattr(d,name),value)


def test_missing_release_contact_and_missing_actual_physics_result_fail():
    s=fixture()
    with pytest.raises(ValueError):outward_release_normal(s.m,dict(contacts=[]))
    with pytest.raises(TypeError):measured_contacts(s.m,dict(contacts=[]))
    with pytest.raises(ValueError):measured_contacts(s.m,dict(contacts=[]),physics_qualified='yes')


def test_physical_gate_remains_strict_for_limits_and_external_assistance():
    r=dict(finite=True,native_motor_limits=True,numerical_warnings=0,torso_tilt_deg=1,root_height_m=1.,max_joint_limit_violation_rad=0.,max_nonfoot_penetration_m=0.,max_shadow_loopback_violation_rad=0.,external_wrench_max=0.,applied_generalized_force_max=0.)
    assert qualified_physics_row(r)
    for key,value in [('max_joint_limit_violation_rad',.0201),('external_wrench_max',1e-9),('finite','yes'),('torso_tilt_deg',12.),('motor_delivery_error_Nm',np.nan)]:
        bad=dict(r);bad[key]=value;assert not qualified_physics_row(bad)
