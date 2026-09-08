"""Own-robot-only kinematics and causal local opposing-load feedback."""
import ast,copy,inspect,json
from pathlib import Path
import mujoco,numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_distal_touch_control import packet
from test_sensor_index_touch_control import make as index
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.sensor_palm_touch_control import SensorPalmTouchController,PALM_PROTOCOL,validate_palm_protocol
from doorbench.dexterous.robot_palm_translation import RobotPalmTranslation,ARM_NAMES


def make(authored,stub=False):
    old=index(authored,stub=stub)
    return SensorPalmTouchController(old.arm,old._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],INDEX_PROTOCOL.copy(),PALM_PROTOCOL.copy())


def test_exact_protocol_and_no_dynamic_or_scene_calls():
    import doorbench.dexterous.sensor_palm_touch_control as controller
    import doorbench.dexterous.robot_palm_translation as kinematics
    assert list(inspect.signature(SensorPalmTouchController.force).parameters)==['self','packet','now_s']
    for module in (controller,kinematics):
        tree=ast.parse(inspect.getsource(module))
        assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment'} for n in ast.walk(tree))
    path=Path(__file__).resolve().parents[1]/'configs/dexterous/sensor-palm-touch-v1.json'
    assert validate_palm_protocol(json.loads(path.read_text()))==PALM_PROTOCOL
    for key,value in [('maximum_translation_m',.001),('finger_normalization_N',1.),('start_after_s',18.),('maximum_translation_rate_mps',True)]:
        bad=dict(PALM_PROTOCOL);bad[key]=value
        with pytest.raises(ValueError):validate_palm_protocol(bad)


def test_translation_matches_independent_robot_fk_and_changes_only_eight_goals(authored):
    m=mujoco.MjModel.from_xml_path(str(authored[0]));names=[m.joint(i).name for i in range(1,m.njnt)]
    ref=json.loads((Path(__file__).resolve().parents[1]/'configs/dexterous/door55-precurl-v2/reference.json').read_text())['acquisition']
    q=np.asarray(ref['path_qpos'][-1]);assert names==ref['joint_names']
    goals=dict(zip(names,q));calc=RobotPalmTranslation(m,names)
    original={k:getattr(m,k).copy() for k in ('actuator_gainprm','actuator_biasprm','actuator_forcerange','jnt_range')}
    d=mujoco.MjData(m);d.qpos[:7]=[0,0,1,1,0,0,0];d.qpos[calc.qa]=q;mujoco.mj_kinematics(m,d)
    before=d.site_xpos[calc.palm].copy();R=d.site_xmat[calc.palm].reshape(3,3).copy()
    g,info=calc.goals(goals,q,.00025)
    for name in names:
        if name not in ARM_NAMES:assert g[name]==goals[name]
    d.qpos[calc.qa]=[g[n] for n in names];mujoco.mj_kinematics(m,d)
    np.testing.assert_allclose(d.site_xpos[calc.palm]-before,.00025*(R@info['direction_palm_local']),atol=2e-6,rtol=0)
    np.testing.assert_allclose(d.site_xmat[calc.palm].reshape(3,3),R,atol=2e-5,rtol=0)
    assert calc.d.time==0. and info['maximum_joint_correction_rad']<.025
    for key,value in original.items():np.testing.assert_array_equal(getattr(m,key),value)
    zero,_=calc.goals(goals,q,0.);assert zero==goals
    for bad in [True,-.0001,.000251]:
        with pytest.raises(ValueError):calc.goals(goals,q,bad)


def test_first19s_exact_then_bounded_causal_local_feedback_and_reset(authored):
    c=make(authored,stub=True);previous=0.
    for i in range(11000):
        t=i*.002;f,info=c.force(packet(c,t,[.5,.5,.5,.5,4.]),now_s=t)
        assert abs(c.palm_delta-previous)<=.0001*.002+1e-14 and 0<=c.palm_delta<=.00025
        if t<=19.:
            assert c.palm_delta==0.
            if t<19.:
                assert dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))==c._index_schedule.original.goals(t)
        else:
            assert info['palm_feedback_decision_s']==pytest.approx(t-.002)
            assert info['palm_opposing_normalized_load_error']==pytest.approx(4/3-.5/2)
        np.testing.assert_array_equal(f,c.arm.last_force);previous=c.palm_delta
    before=c.palm_delta
    for i in range(11000,12000):c.force(packet(c,i*.002,[2,2,2,2,.1]),now_s=i*.002)
    assert c.palm_delta<before and c.palm_translator.d.time==0.
    c.reset_episode();assert c.palm_delta==0. and c._palm_schedule.offset==0. and c.index_delta==0.
    np.testing.assert_array_equal(c.arm.balance.kp,c._initial_kp)


def test_real_force_owner_and_forbidden_pose_packet_are_terminal(authored):
    c=make(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0. else valid(b,t);force,_=c.force(p,now_s=t)
        np.testing.assert_array_equal(force,b.last_force)
        assert np.all(force>=b.caps[:,0]) and np.all(force<=b.caps[:,1])
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();p=cold(b);p['door_pose']=np.zeros(7)
    with pytest.raises(ValueError):c.force(p,now_s=0.)
    with pytest.raises(RuntimeError):c.force(cold(b),now_s=0.)
