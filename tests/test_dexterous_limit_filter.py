"""Native dynamics checks for the optional, privileged development filter."""
import mujoco
import numpy as np
from doorbench.dexterous.limit_filter import HandLimitFilter


def plant():
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
    <option timestep=".002" gravity="0 0 0"/><worldbody>
    <body><joint name="robot/rh_FFJ1" axis="0 0 1" range="0 1"/>
      <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" mass=".1" contype="0" conaffinity="0"/></body>
    <body pos="0 1 0"><joint name="robot/rh_FFJ2" axis="0 0 1" range="0 1"/>
      <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" mass=".1" contype="0" conaffinity="0"/></body>
    </worldbody><tendon><fixed name="sum"><joint joint="robot/rh_FFJ1" coef="1"/>
      <joint joint="robot/rh_FFJ2" coef="1"/></fixed></tendon>
    <actuator><motor name="robot/rh_A_FFJ0" tendon="sum" forcerange="-1 1" ctrlrange="-1 1"/></actuator></mujoco>''')
    data=mujoco.MjData(model)
    return model,data,HandLimitFilter(model,data,[0],[0,1],np.array([[1.,1.]]),[0],margin=.005)


def test_outward_command_is_redirected_through_native_tendon_without_state_writes():
    model,data,controller=plant();data.qpos[:]=.002;mujoco.mj_forward(model,data)
    original_ranges=model.jnt_range.copy();original_caps=model.actuator_forcerange.copy()
    for _ in range(100):
        before=data.qpos.copy();force,info=controller.apply(np.array([-1.]))
        np.testing.assert_array_equal(data.qpos,before)
        assert info['feasible'] and abs(force[0])<=1
        data.ctrl[:]=force;mujoco.mj_step(model,data)
        assert data.qpos.min()>=0
    assert data.qpos.min()>.004
    np.testing.assert_array_equal(model.jnt_range,original_ranges)
    np.testing.assert_array_equal(model.actuator_forcerange,original_caps)
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)


def test_incompatible_coupled_joint_demands_are_exposed_as_infeasible():
    model,data,controller=plant();data.qpos[:]=[.00001,.99999];mujoco.mj_forward(model,data)
    force,info=controller.apply(np.array([.2]))
    assert not info['feasible']
    # The diagnostic leaves the bounded nominal command visible; callers must
    # reject infeasibility, never interpret this fallback as enforced safety.
    np.testing.assert_array_equal(force,[.2])
