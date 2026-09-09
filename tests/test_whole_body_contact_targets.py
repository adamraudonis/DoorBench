import mujoco
import numpy as np
import pytest

from doorbench.dexterous.whole_body_contact_targets import WholeBodyContactTargets


def fixture():
    names = [f'joint_{i}' for i in range(25)]
    bodies = []
    for body,indices,position,site in (
        ('lh',range(0,6),'.3 -.3 0','lh_palm_touch'),
        ('rh',range(6,12),'.3 .3 0','rh_palm_touch'),
        ('left_ankle_link',range(12,17),'0 -.1 -1',None),
        ('right_ankle_link',range(17,22),'0 .1 -1',None),
        ('spare',range(22,25),'0 0 .1',None)):
        joints=''
        for i,j in enumerate(indices):
            axis=('1 0 0','0 1 0','0 0 1')[i%3]
            kind='slide' if i<3 else 'hinge'
            joints+=f'<joint name="{names[j]}" type="{kind}" axis="{axis}" range="-.5 .5"/>'
        bodies.append(f'<body name="{body}" pos="{position}">{joints}<geom size=".02" mass=".01"/>'+
                      (f'<site name="{site}"/>' if site else '')+'</body>')
    model=mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody><body><freejoint/>'
            '<geom size=".03" mass="5"/>'+''.join(bodies)+'</body></worldbody></mujoco>')
    return model,names


def test_stationary_complete_goal_preserves_targets_without_touching_another_data():
    model,names=fixture();active=mujoco.MjData(model);before=active.qpos.copy()
    planner=WholeBodyContactTargets(model,names)
    joints=dict.fromkeys(names,0.);root=np.r_[0,0,1,1,0,0,0,np.zeros(6)]
    planner.begin(0.,root,joints)
    goal=planner.data.site_xpos[planner.palms[0]].copy()
    for t in np.arange(0.,.102,.002):
        q,v,a,info=planner.update(float(t),joints,goal,np.eye(3),np.zeros(31))
    np.testing.assert_allclose(q,0.,atol=1e-12)
    np.testing.assert_array_equal(active.qpos,before)
    assert info['maximum_commanded_foot_position_error_m']<1e-12


def test_moving_palm_goal_uses_the_whole_body_under_original_rate_bounds():
    model,names=fixture();planner=WholeBodyContactTargets(model,names)
    joints=dict.fromkeys(names,0.);root=np.r_[0,0,1,1,0,0,0,np.zeros(6)]
    planner.begin(0.,root,joints)
    initial=planner.data.site_xpos[planner.palms[0]].copy();previous=planner.target.copy();old_v=planner.velocity.copy()
    for t in np.arange(0.,.502,.002):
        goal=initial+np.array([.02*t,0,0])
        q,v,a,info=planner.update(float(t),joints,goal,np.eye(3),np.zeros(31))
        if t:
            np.testing.assert_allclose(q-previous,.001*(old_v+v),atol=1e-12)
        assert max(abs(v[6:]))<=1.2+1e-9
        assert max(abs(a[6:]))<=3.+1e-8
        assert np.linalg.norm(v[:3])<=.02+1e-9
        assert np.linalg.norm(v[3:6])<=.03+1e-9
        previous=q;old_v=v
    assert info['commanded_lh_position_error_m']<1e-4
    assert info['maximum_commanded_foot_position_error_m']<1e-4


@pytest.mark.parametrize('fault',['missing_joint','extra_joint','nonfinite','time_gap'])
def test_incomplete_measurements_and_clock_gaps_are_rejected(fault):
    model,names=fixture();planner=WholeBodyContactTargets(model,names)
    joints=dict.fromkeys(names,0.);root=np.r_[0,0,1,1,0,0,0,np.zeros(6)]
    planner.begin(0.,root,joints);t=.002;goal=np.array([.3,-.3,1.])
    if fault=='missing_joint':del joints[names[0]]
    if fault=='extra_joint':joints['unknown_joint']=0.
    if fault=='nonfinite':goal[0]=np.nan
    if fault=='time_gap':t=.004
    with pytest.raises(ValueError):planner.update(t,joints,goal,np.eye(3),np.zeros(31))
