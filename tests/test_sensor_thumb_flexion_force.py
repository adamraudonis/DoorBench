import copy,json
from pathlib import Path
import numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_digit_force_control import make as old
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL
from doorbench.dexterous.sensor_hierarchical_digit_force import HIERARCHICAL_PROTOCOL,normal_posture_transfer
from doorbench.dexterous.sensor_smooth_contact_force import SensorSmoothContactForceController
from doorbench.dexterous.sensor_thumb_flexion_force import SensorThumbFlexionForceController,THUMB_PROTOCOL,validate_thumb_protocol
from doorbench.dexterous.tactile_contact_mode import MODE_PROTOCOL,THUMB_MODE_PROTOCOL


def make(authored,mode=None):
    c=old(authored)
    return SensorThumbFlexionForceController(c.arm,c._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],
        INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy(),
        contact_mode_protocol=THUMB_MODE_PROTOCOL.copy() if mode is None else mode,thumb_flexion_protocol=THUMB_PROTOCOL.copy())


def test_profile_allocation_scope_cannot_be_silently_swapped(authored):
    with pytest.raises(ValueError,match='scope'):make(authored,MODE_PROTOCOL.copy())
    c=old(authored)
    with pytest.raises(ValueError,match='scope'):
        SensorSmoothContactForceController(c.arm,c._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],
            INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy(),contact_mode_protocol=THUMB_MODE_PROTOCOL.copy())
    expected=json.loads((Path(__file__).parents[1]/'configs/dexterous/sensor-thumb-flexion-pressure-v1.json').read_text())
    assert validate_thumb_protocol(expected)==THUMB_PROTOCOL
    for key,value in [('start_after_s',18.),('original_transmission_and_caps',1),('thumb_pressure_joints',['rh_THJ5'])]:
        p=copy.deepcopy(THUMB_PROTOCOL);p[key]=value
        with pytest.raises(ValueError):validate_thumb_protocol(p)


def test_hierarchy_never_removes_opposition_posture(authored):
    c=make(authored);b=c.arm.balance;calc=c.pad_force
    unit,_=calc.motor_bias(b.desired,np.ones(5));position=np.linspace(-.3,.3,61)
    change,_=normal_posture_transfer(position,unit,calc.groups,np.ones(5)*.5,np.ones(5),np.ones(5))
    excluded=[c.action_names.index('rh_A_THJ'+str(k)) for k in (3,4,5)]
    np.testing.assert_array_equal(unit[excluded],0.);np.testing.assert_array_equal(change[excluded],0.)
    np.testing.assert_array_equal((position+unit+change)[excluded],position[excluded])
    assert calc.groups['th'][0].tolist()==[c.joint_names.index('rh_THJ2'),c.joint_names.index('rh_THJ1')]


def test_first_decisions_and_command_history_are_unchanged(authored):
    c=make(authored);baseline=old(authored);b=c.arm.balance
    for t in (0.,.002):
        packet=cold(b) if not t else valid(b,t)
        f,info=c.force(packet,now_s=t);other,_=baseline.force(copy.deepcopy(packet),now_s=t)
        np.testing.assert_array_equal(f,other);np.testing.assert_array_equal(f,b.last_force)
        assert not c.arm.info and np.all(f>=b.caps[:,0]) and np.all(f<=b.caps[:,1])
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();assert c.contact_mode.time is None
