import numpy as np
import pytest
from doorbench.dexterous.panel_torso_target import original_torso_target_force


def test_complete_assembly_changes_only_the_original_torso_motor_and_caps_it():
    force=np.arange(61,dtype=float)/100;caps=np.tile([-200.,200.],(61,1))
    result,info=original_torso_target_force(force,10,.15,.1,.12,-.2,kp=100.,bias=[0,-100,-2],gain=9.,damping=20.,caps=caps)
    expected=100*.15-100*.12-2*(-.2)+900*(.15-.12)-20*(-.2-.1)
    assert result[10]==pytest.approx(expected)
    np.testing.assert_array_equal(np.delete(result,10),np.delete(force,10))
    np.testing.assert_array_equal(force,np.arange(61,dtype=float)/100)
    result,_=original_torso_target_force(force,10,1.,0.,0.,0.,kp=100.,bias=[0,-100,-2],gain=9.,damping=20.,caps=caps)
    assert result[10]==200.


def test_target_velocity_uses_existing_damping_not_a_new_gain():
    kwargs=dict(kp=100.,bias=[0,-100,-2],gain=9.,damping=20.,caps=np.tile([-200.,200.],(61,1)))
    a,_=original_torso_target_force(np.zeros(61),10,.1,0.,.1,0.,**kwargs)
    b,_=original_torso_target_force(np.zeros(61),10,.1,.02,.1,0.,**kwargs)
    assert b[10]-a[10]==pytest.approx(.4)


def test_nonfinite_force_cannot_be_hidden_by_final_clipping():
    force=np.zeros(61);force[5]=np.nan
    with pytest.raises(ValueError):original_torso_target_force(force,10,.1,0.,.1,0.,kp=100,bias=[0,-100,-2],gain=9,damping=20,caps=np.tile([-200.,200.],(61,1)))


def test_joint_coordinate_torso_adapter_rejects_a_changed_transmission():
    import mujoco
    from doorbench.dexterous.panel_torso_target import validate_unit_joint_motor
    for gear in (1.,2.):
        model=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body><joint name="torso"/><geom size=".1"/></body></worldbody><actuator><motor name="torso" joint="torso" gear="{gear}"/></actuator></mujoco>')
        if gear==1.:validate_unit_joint_motor(model,0,0)
        else:
            with pytest.raises(ValueError):validate_unit_joint_motor(model,0,0)


def test_boolean_motor_target_is_not_a_physical_measurement():
    with pytest.raises(ValueError):original_torso_target_force(np.zeros(61),10,True,0.,.1,0.,kp=100,bias=[0,-100,-2],gain=9,damping=20,caps=np.tile([-200.,200.],(61,1)))
