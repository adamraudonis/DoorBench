import ast,copy,inspect,json
from pathlib import Path
import numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_digit_force_control import make as old
from test_sensor_distal_touch_control import packet
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL
from doorbench.dexterous.sensor_hierarchical_digit_force import HIERARCHICAL_PROTOCOL
from doorbench.dexterous.sensor_smooth_contact_force import SensorSmoothContactForceController
from doorbench.dexterous.tactile_contact_mode import MODE_PROTOCOL,TactileContactMode,validate_mode_protocol


def make(authored,stub=False):
    c=old(authored,stub=stub)
    return SensorSmoothContactForceController(c.arm,c._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],
        INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy(),contact_mode_protocol=MODE_PROTOCOL.copy())


def test_retention_decay_and_recovery_do_not_invent_touch():
    c=TactileContactMode(MODE_PROTOCOL.copy());c.update(np.ones(5),now_s=19.)
    for i in range(1,13):
        raw=np.ones(5);raw[0]=0;w,info=c.update(raw,now_s=19.+i*.002)
        assert not info['immediate_contact_progression_ready'] and not info['pressure_integration_allowed'][0]
        assert info['raw_contact_projection_weight'][0]==0
        if i<=10:assert w[0]==1.
        else:assert w[0]==pytest.approx(1.-.01*(i-10))
    before=c.weight.copy();w,info=c.update(np.ones(5),now_s=19.026)
    assert w[0]==pytest.approx(before[0]+.01) and info['zero_load_duration_s'][0]==0
    assert info['immediate_contact_progression_ready']


def test_recovery_has_strict_timeout_and_terminal_validation():
    c=TactileContactMode(MODE_PROTOCOL.copy());c.update(np.ones(5),now_s=19.)
    prior=c.weight.copy()
    for i in range(1,126):
        w,_=c.update(np.zeros(5),now_s=19.+i*.002)
        assert abs(w-prior).max()<=.01+1e-12;prior=w
    assert not np.any(w)
    with pytest.raises(ValueError,match='250ms'):c.update(np.zeros(5),now_s=19.252)
    with pytest.raises(RuntimeError):c.update(np.ones(5),now_s=19.252)
    c.reset()
    with pytest.raises(ValueError):c.update(np.ones(5),now_s=True)
    with pytest.raises(RuntimeError):c.update(np.ones(5),now_s=19.)


def test_exact_schema_and_sensor_only_boundary():
    import doorbench.dexterous.sensor_smooth_contact_force as source
    assert list(inspect.signature(SensorSmoothContactForceController.force).parameters)==['self','packet','now_s']
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment'} for n in ast.walk(ast.parse(inspect.getsource(source))))
    assert validate_mode_protocol(json.loads((Path(__file__).parents[1]/'configs/dexterous/sensor-contact-mode-v1.json').read_text()))==MODE_PROTOCOL
    for key,value in [('maximum_zero_load_recovery_s',1.),('maximum_weight_rate_per_s',True),('thumb_projection','restricted')]:
        p=copy.deepcopy(MODE_PROTOCOL);p[key]=value
        with pytest.raises(ValueError):validate_mode_protocol(p)


def test_real_initial_prefix_and_previous_force_ownership(authored):
    c=make(authored);base=old(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0 else valid(b,t)
        f,info=c.force(p,now_s=t);g,_=base.force(copy.deepcopy(p),now_s=t)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(f,b.last_force)
        assert c.contact_mode.time is None
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();assert c.contact_mode.time is None
    np.testing.assert_array_equal(c.arm.original_constant_bias,c._base_motor_bias)


def test_fresh_zero_contact_freezes_all_integrators_and_progression(authored):
    c=make(authored,stub=True)
    # The stub isolates schedule/mode logic; actual history/caps are separately
    # exercised against the real controller above. Its zero encoders do not
    # model an attained grasp, so inject an admitted fixed normal-equivalent
    # effort in this logic fixture only; the production admission is unchanged.
    c.arm.initial_normal=np.full(5,.5)
    for i in range(10600):
        t=i*.002;c.force(packet(c,t,[2.,2.,2.,2.,3.]),now_s=t)
    before=(c.force_integral.copy(),c.virtual_force.copy(),c.delta.copy(),c.index_delta,c.virtual_s)
    f,info=c.force(packet(c,21.2,[0.,0.,0.,0.,0.]),now_s=21.2)
    for value,prior in zip((c.force_integral,c.virtual_force,c.delta,c.index_delta,c.virtual_s),before):
        np.testing.assert_array_equal(value,prior)
    assert not info['local_touch_progression_ready'] and c.contact_mode.weight.tolist()==[1.]*5
    assert not any(info['pressure_integration_allowed'])
    p=packet(c,21.202,[2.,2.,2.,2.,3.]);p['sensor_time_s'][4]=0.
    with pytest.raises(ValueError,match='Fresh'):c.force(p,now_s=21.202)
    with pytest.raises(RuntimeError):c.force(packet(c,21.202,[2.,2.,2.,2.,3.]),now_s=21.202)
