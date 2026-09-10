import copy
import mujoco
import numpy as np
import pytest
from doorbench.dexterous.handle_relative_arm import ARM_NAMES,HandleRelativeArmTarget


def fixture():
    axes=['0 0 1','0 1 0','1 0 0','0 1 0','0 0 1','1 0 0','0 1 0']
    chain=''.join(f'<body pos=".08 .02 .03"><joint name="{n}" axis="{axis}" range="-3 3"/><geom size=".02"/>' for n,axis in zip(ARM_NAMES,axes))
    xml='<mujoco><compiler angle="radian"/><worldbody><body><freejoint/><geom size=".05"/>'+chain+'<site name="rh_palm_touch" pos=".06 0 0"/>'+'</body>'*8+'</worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml)
    joints=dict(zip(ARM_NAMES,[.2,.3,-.2,.4,.1,-.3,.2]));root=np.array([0,0,0,1,0,0,0,0,0,0,0,0,0.],float)
    handle=np.array([.5,.1,.2,1,0,0,0.]);return m,joints,root,handle


def test_anchor_continuity_correction_bound_and_no_physics(monkeypatch):
    m,joints,root,handle=fixture();before=m.qpos0.copy();original=copy.deepcopy(joints)
    def forbidden(*args,**kwargs):raise AssertionError('No physical stepping allowed')
    for name in ('mj_step','mj_forward','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)
    c=HandleRelativeArmTarget(m,list(ARM_NAMES),root,joints,handle)
    target,info=c.target(0,root,joints,handle,joints);assert target==joints
    root[0]+=.001
    last=np.zeros(7)
    for t in [.02,.04,.06]:
        target,info=c.target(t,root,joints,handle,joints)
        delta=np.array([target[n]-joints[n] for n in ARM_NAMES])
        assert np.max(abs(delta))<=.08+1e-12
        assert np.max(abs(delta-last))<=.01+1e-10
        last=delta
    assert np.max(abs(last))>1e-5
    assert joints==original;np.testing.assert_array_equal(m.qpos0,before)
    with pytest.raises(ValueError,match='Monotonic'):c.target(.01,root,joints,handle,joints)


def test_unreachable_handle_and_invalid_input_fail_closed():
    m,joints,root,handle=fixture();c=HandleRelativeArmTarget(m,list(ARM_NAMES),root,joints,handle)
    handle[0]+=1
    with pytest.raises(ValueError,match='bounded reachability'):c.target(0,root,joints,handle,joints)
    assert c.failure_snapshot['reference_pose']==handle.tolist()
    assert c.failure_snapshot['nominal']==joints
    assert c.failure_snapshot['starts']==2
    with pytest.raises(ValueError):HandleRelativeArmTarget(m,list(ARM_NAMES),root,{ARM_NAMES[0]:0},handle)


def test_route_and_compensation_velocities_use_separate_clocks():
    m,joints,root,handle=fixture();c=HandleRelativeArmTarget(m,list(ARM_NAMES),root,joints,handle)
    nominal=dict(joints);c.target(0,root,joints,handle,nominal)
    previous=np.zeros(7)
    # Root drift generates compensation at 2 ms intervals; the route changes
    # at 10 ms. Its .0005 rad step must contribute .05 rad/s, not .25.
    for step in range(1,21):
        t=step*.002;root[0]=t*.001
        if step%5==0:nominal[ARM_NAMES[0]]+=.0005
        c.target(t,root,joints,handle,nominal)
        compensation_velocity=(c.correction-previous)/.002
        expected=compensation_velocity.copy()
        if step>=5:expected[0]+=.05
        np.testing.assert_allclose([c.target_velocity[n] for n in ARM_NAMES],expected,atol=1e-11)
        previous=c.correction.copy()


def test_failed_warm_fit_retries_nominal_without_relaxing_limits(monkeypatch):
    import doorbench.dexterous.handle_relative_arm as module
    from types import SimpleNamespace
    original=module.least_squares;calls=[]
    def first_fit_stalls(fun,x0,**kwargs):
        calls.append(np.array(x0))
        if len(calls)==1:
            # A bounded but inaccurate stationary result must not be accepted.
            return SimpleNamespace(x=kwargs['bounds'][1].copy(),success=True,nfev=1,message='stalled test fit')
        return original(fun,x0,**kwargs)
    monkeypatch.setattr(module,'least_squares',first_fit_stalls)
    m,joints,root,handle=fixture();c=HandleRelativeArmTarget(m,list(ARM_NAMES),root,joints,handle)
    target,info=c.target(0,root,joints,handle,joints)
    assert len(calls)==2 and info['solve']['starts']==2
    assert target==joints
    assert info['solve']['position_error_m']<=.0005
    assert info['solve']['rotation_error_rad']<=.005
    assert info['correction_limit_rad']==.08
