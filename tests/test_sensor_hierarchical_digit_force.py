import ast,copy,inspect,json
from pathlib import Path
import numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_digit_force_control import make as old
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL
from doorbench.dexterous.sensor_hierarchical_digit_force import (HIERARCHICAL_PROTOCOL,
    SensorHierarchicalDigitForceController,normal_posture_transfer,validate_hierarchical_protocol)


def make(authored):
    c=old(authored)
    return SensorHierarchicalDigitForceController(c.arm,c._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy())


def test_normal_transfer_removes_only_actuated_normal_position_effort(authored):
    c=make(authored);calc=c.pad_force;unit,_=calc.motor_bias(c.arm.balance.desired,np.ones(5));groups=calc.groups
    position=np.linspace(-.2,.3,61);baseline=np.linspace(.2,1.2,5);correction=np.array([2.,-.8,.5,0.,-.3])
    adjustment,info=normal_posture_transfer(position,unit,groups,baseline,correction,np.ones(5))
    for k,(digit,(_,r,_,_)) in enumerate(groups.items()):
        p=unit[r];P=np.outer(p,p)/(p@p);old=position[r]+p*correction[k];new=old+adjustment[r]
        np.testing.assert_allclose((np.eye(len(r))-P)@new,(np.eye(len(r))-P)@position[r],atol=2e-16)
        assert p@new/(p@p)==pytest.approx(np.clip(baseline[k]+correction[k],0,4),abs=1e-14)
        assert info[digit]['total_virtual_normal_effort_N']>=0
    allrows=np.concatenate([g[1] for g in groups.values()]);assert np.all(adjustment[np.setdiff1d(np.arange(61),allrows)]==0)
    empty,_=normal_posture_transfer(position,unit,groups,baseline,correction,np.zeros(5));np.testing.assert_array_equal(empty,np.zeros(61))
    captured=np.array([unit[r]@position[r]/(unit[r]@unit[r]) for _,r,_,_ in groups.values()])
    # Synthetic bounded preload makes the force handover exactly bumpless.
    position=np.zeros(61)
    for k,(_,r,_,_) in enumerate(groups.values()):position[r]=unit[r]*baseline[k]
    transfer,_=normal_posture_transfer(position,unit,groups,baseline,np.zeros(5),np.ones(5));np.testing.assert_allclose(transfer,0.,atol=1e-16)


def test_schema_sensor_only_interface_and_bounds():
    import doorbench.dexterous.sensor_hierarchical_digit_force as source
    assert list(inspect.signature(SensorHierarchicalDigitForceController.force).parameters)==['self','packet','now_s']
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment'} for n in ast.walk(ast.parse(inspect.getsource(source))))
    assert validate_hierarchical_protocol(json.loads((Path(__file__).parents[1]/'configs/dexterous/sensor-hierarchical-digit-force-v1.json').read_text()))==HIERARCHICAL_PROTOCOL
    for key,value in [('start_after_s',18.),('initial_scalar_range_N',[0.,3.]),('duration_s',True)]:
        bad=copy.deepcopy(HIERARCHICAL_PROTOCOL);bad[key]=value
        with pytest.raises(ValueError):validate_hierarchical_protocol(bad)


def test_real_default_prefix_and_actual_command_ownership(authored):
    c=make(authored);baseline=old(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0 else valid(b,t);f,info=c.force(p,now_s=t);g,_=baseline.force(copy.deepcopy(p),now_s=t)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(f,b.last_force)
        assert not c.arm.info and c.arm.initial_normal is None
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();assert c.arm.initial_normal is None
    np.testing.assert_array_equal(c.arm.original_constant_bias,c._base_motor_bias)
