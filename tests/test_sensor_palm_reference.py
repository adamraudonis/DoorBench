"""Frame, input and action-ownership checks for a rejected opt-in candidate."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_sensor_balance import authored, cold, valid
from doorbench.dexterous.palm_ground_reference import ARM_NAMES, GroundPalmReference, RobotPalmReferenceIK, project_palm_reference
from doorbench.dexterous.sensor_palm_reference import corrected_goals, SensorPalmReferenceController
from doorbench.dexterous.sensor_reach_balance import SensorReachBalanceController

SOURCE=Path(__file__).resolve().parents[1]/'configs/dexterous/door55-precurl-v2/reference.json'


@pytest.fixture(scope='module')
def projected(authored):
    return project_palm_reference(authored[0],SOURCE)


def test_projection_removes_xy_yaw_but_preserves_ground_height(authored,projected):
    source=json.loads(SOURCE.read_text());m=mujoco.MjModel.from_xml_path(str(authored[0]));d=mujoco.MjData(m)
    root=np.array(source['initial_root']);R=Rotation.from_quat(root[[4,5,6,3]]).as_matrix()
    qa=[m.joint(n).qposadr[0] for n in source['acquisition']['joint_names']];site=m.site('rh_palm_touch').id
    for i in (0,200,400):
        d.qpos[:7]=root;d.qpos[qa]=source['acquisition']['path_qpos'][i];mujoco.mj_kinematics(m,d)
        p=R.T@(d.site_xpos[site]-[root[0],root[1],0.])
        np.testing.assert_allclose(projected['position_m'][i],p,atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(projected['quaternion_xyzw'][i]).as_matrix(),R.T@d.site_xmat[site].reshape(3,3),atol=1e-14)
    assert projected['source_pelvis_height_m']==root[2]
    assert 'initial_root' not in projected


@pytest.mark.parametrize('bad',['extra','hash','scope','quaternion','timing'])
def test_static_path_contract_rejects_ambiguous_inputs(projected,bad):
    x=copy.deepcopy(projected)
    if bad=='extra':x['door_position']=[0,0,0]
    if bad=='hash':x['source_reference_sha256']='x'*64
    if bad=='scope':x['scope']='world-state feedback'
    if bad=='quaternion':x['quaternion_xyzw'][2]=[0,0,0,2]
    if bad=='timing':x['timing']['start_s']=0.
    with pytest.raises(ValueError):GroundPalmReference(x,projected['robot_xml_sha256'])


def test_ik_changes_only_seven_arm_goals_and_never_steps(authored,projected,monkeypatch):
    m=mujoco.MjModel.from_xml_path(str(authored[0]));names=authored[1]['joint_names'];ik=RobotPalmReferenceIK(m,names)
    source=json.loads(SOURCE.read_text());q=np.array(source['acquisition']['path_qpos'][0]);nominal=dict(zip(names,q))
    root=[0,0,source['initial_root'][2],1,0,0,0];p=np.array(projected['position_m'][0]);R=Rotation.from_quat(projected['quaternion_xyzw'][0]).as_matrix()
    monkeypatch.setattr(mujoco,'mj_step',lambda *a:pytest.fail('Detached IK must not step'))
    result,info=ik.goals(nominal,q,root,p+[.001,0,0],R)
    assert max(abs(result[n]-nominal[n]) for n in ARM_NAMES)<.05
    assert all(result[n]==nominal[n] for n in names if n not in ARM_NAMES)
    assert info['position_error_m']<5e-6 and ik.d.time==0
    with pytest.raises(ValueError,match='envelope'):ik.goals(nominal,q,root,p+[.011,0,0],R)
    with pytest.raises(ValueError,match='rotation'):ik.goals(nominal,q,root,p,np.diag([1,1,-1]))


def test_cold_window_ignores_no_estimator_and_later_correction_has_explicit_slew():
    nominal={n:0. for n in ARM_NAMES};ref=SimpleNamespace(target=lambda t:(np.zeros(3),np.eye(3)))
    def solve(n,*a,**kw):return {k:.02 for k in n},{}
    solver=SimpleNamespace(goals=solve,limits=np.tile([-1,1],(7,1)))
    g,d,_=corrected_goals(solver,ref,nominal,None,None,.2,np.zeros(7));assert g==nominal;assert not d.any()
    g,d,info=corrected_goals(solver,ref,nominal,None,None,1.,np.zeros(7))
    np.testing.assert_array_equal(d,np.full(7,.0005));assert info['correction_rate_limited']
    with pytest.raises(ValueError):corrected_goals(solver,ref,nominal,None,None,float('nan'),d)


def test_runtime_calls_base_once_and_logs_previous_root_epoch(authored,projected,monkeypatch):
    arm=SensorReachBalanceController(*copy.deepcopy(authored),image_shape=(8,8,3),allow_torso_yaw=True)
    c=SensorPalmReferenceController(arm,projected,projected['robot_xml_sha256']);b=arm.balance
    calls=[];original=arm.force
    def track(*a,**kw):calls.append(kw['now_s']);return original(*a,**kw)
    monkeypatch.setattr(arm,'force',track)
    g=dict(zip(arm.goal_names,arm.last_goals))
    c.force(cold(b),now_s=0.,joint_goals=g)
    for i in range(1,102):
        f,info=c.force(valid(b,i*.002),now_s=i*.002,joint_goals=g)
        np.testing.assert_array_equal(f,arm.last_force)
    assert calls==[i*.002 for i in range(102)]
    assert info['correction_root_estimate_epoch_s']==pytest.approx(.2)
    assert info['correction_encoder_epoch_s']==pytest.approx(.202)
    assert info['source_ground_pelvis_height_m']!=info['estimator_initial_pelvis_height_m']
    b.last_time=.198
    with pytest.raises(ValueError,match='previous-decision'):c.force(valid(b,.204),now_s=.204,joint_goals=g)
    with pytest.raises(RuntimeError):c.force(valid(b,.204),now_s=.204,joint_goals=g)
    assert len(calls)==102
