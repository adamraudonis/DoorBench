import ast,copy,inspect,json
from pathlib import Path
import numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_digit_force_control import make as old
from test_sensor_thumb_flexion_force import make as flexion
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL
from doorbench.dexterous.sensor_hierarchical_digit_force import HIERARCHICAL_PROTOCOL
from doorbench.dexterous.sensor_thumb_flexion_force import THUMB_PROTOCOL
from doorbench.dexterous.tactile_contact_mode import THUMB_MODE_PROTOCOL
from doorbench.dexterous.scripted_thumb_coordination import ScriptedThumbGoalCoordinator,SensorScriptedThumbCoordinationController,THUMB_NAMES,validate_coordination_protocol


def profile():return json.loads((Path(__file__).parents[1]/'configs/dexterous/sensor-thumb-coordination-v1.json').read_text())


def make(authored):
    c=old(authored)
    return SensorScriptedThumbCoordinationController(c.arm,c._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],
        INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy(),HIERARCHICAL_PROTOCOL.copy(),contact_mode_protocol=THUMB_MODE_PROTOCOL.copy(),
        thumb_flexion_protocol=THUMB_PROTOCOL.copy(),coordination_protocol=profile())


def test_named_goal_continuity_endpoints_and_no_alias():
    p=profile();c=ScriptedThumbGoalCoordinator(p);goals=dict(p['source_goal_at_start'],torso=.2);saved=goals.copy()
    previous=None;maximum_rate=0.
    for t in np.arange(0.,24.002,.002):
        actual,info=c.apply(goals,now_s=float(t))
        if t<=23.:assert actual==goals
        assert actual['torso']==goals['torso'] and goals==saved
        if previous is not None:maximum_rate=max(maximum_rate,max(abs(actual[n]-previous[n])/.002 for n in THUMB_NAMES))
        previous=actual
    assert maximum_rate<.05
    for n in THUMB_NAMES:assert actual[n]==p['terminal_goal'][n]
    c.reset();bad=goals.copy();bad['rh_THJ2']+=.001
    with pytest.raises(ValueError,match='23s starting goals'):c.apply(bad,now_s=23.)


def test_static_profile_rejects_oracle_fields_and_wrong_clock():
    p=profile();p['object_pose']=[0]*7
    with pytest.raises(ValueError):validate_coordination_protocol(p)
    p=profile();p['ramp_s']=True
    with pytest.raises(ValueError):validate_coordination_protocol(p)
    c=ScriptedThumbGoalCoordinator(profile())
    with pytest.raises(ValueError):c.apply(profile()['source_goal_at_start'],now_s=True)
    import doorbench.dexterous.scripted_thumb_coordination as source
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment','geom'} for n in ast.walk(ast.parse(inspect.getsource(source))))


def test_real_force_owner_and_reset_remain_inside_goal_wrapper(authored):
    c=make(authored);baseline=flexion(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0 else valid(b,t);f,info=c.force(p,now_s=t);g,_=baseline.force(copy.deepcopy(p),now_s=t)
        np.testing.assert_array_equal(f,g);np.testing.assert_array_equal(f,b.last_force)
        assert info['scripted_thumb_coordination_blend']==0.
    bad=valid(b,.004);bad['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(bad,now_s=.004)
    with pytest.raises(RuntimeError):c.force(bad,now_s=.004)
    c.reset_episode();assert not c.coordinator.started
    assert c.arm.inner.owner is c # modified goals enter the pressure projector
