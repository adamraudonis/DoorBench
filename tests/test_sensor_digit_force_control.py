"""Virtual work respects underactuation; force feedback retains motor ownership."""
import ast,copy,inspect,json
from pathlib import Path
import mujoco,numpy as np,pytest
from test_sensor_balance import authored,cold,valid
from test_sensor_distal_touch_control import packet
from test_sensor_index_touch_control import make as index
from doorbench.dexterous.sensor_distal_touch_impedance import PROTOCOL
from doorbench.dexterous.sensor_index_touch_control import INDEX_PROTOCOL
from doorbench.dexterous.robot_digit_force import RobotDigitForce
from doorbench.dexterous.sensor_digit_force_control import FORCE_PROTOCOL,SensorDigitForceController,validate_force_protocol


def make(authored,stub=False):
    old=index(authored,stub=stub)
    if stub:old.arm.original_constant_bias=old.arm.balance.bias[:,0].copy()
    return SensorDigitForceController(old.arm,old._index_schedule.original,authored[2],PROTOCOL.copy(),authored[1],INDEX_PROTOCOL.copy(),FORCE_PROTOCOL.copy())


def test_exact_declared_protocol_and_no_active_dynamics():
    import doorbench.dexterous.robot_digit_force as mapping
    import doorbench.dexterous.sensor_digit_force_control as feedback
    path=Path(__file__).resolve().parents[1]/'configs/dexterous/sensor-digit-force-v1.json'
    assert validate_force_protocol(json.loads(path.read_text()))==FORCE_PROTOCOL
    for module in (mapping,feedback):
        assert not any(isinstance(n,ast.Attribute) and n.attr in {'mj_step','mj_forward','mj_collision','plant','environment'} for n in ast.walk(ast.parse(inspect.getsource(module))))
    assert list(inspect.signature(SensorDigitForceController.force).parameters)==['self','packet','now_s']
    for key,value in [('maximum_virtual_correction_N',3.),('start_after_s',18.),('proportional_gain',True),('targets_N',[1.,2.,2.,2.,3.])]:
        bad=copy.deepcopy(FORCE_PROTOCOL);bad[key]=value
        with pytest.raises(ValueError):validate_force_protocol(bad)


def test_projected_virtual_work_and_explicit_passive_difference(authored):
    c=make(authored);b=c.arm.balance;mapping=c.pad_force
    q=b.desired.copy();bias,info=mapping.motor_bias(q,np.array([1.,0.,0.,0.,0.]))
    columns,rows,A,site=mapping.groups['ff'];tau=np.array(info['ff']['requested_joint_moment_Nm']);realized=np.array(info['ff']['actuated_joint_moment_Nm']);residual=tau-realized
    np.testing.assert_allclose(A@residual,0.,atol=1e-14)
    j1=info['ff']['joint_names'].index('rh_FFJ1');j2=info['ff']['joint_names'].index('rh_FFJ2')
    assert realized[j1]==realized[j2] and abs(residual[j1])>1e-5
    v=np.array([.7,-.2,.3]);dq=A.T@v
    assert float(tau@dq)==pytest.approx(float(bias[rows]@(A@dq)),abs=1e-14)
    assert np.all(bias[np.setdiff1d(np.arange(61),rows)]==0.)
    negative,_=mapping.motor_bias(q,[-1.,0.,0.,0.,0.]);np.testing.assert_array_equal(negative,-bias)
    # Independent finite difference at the authored sensor force line.
    d=mujoco.MjData(mapping.m);d.qpos[:7]=[0,0,1,1,0,0,0];d.qpos[mapping.qa]=q;mujoco.mj_kinematics(mapping.m,d)
    normal=-d.site_xmat[site].reshape(3,3)[:,2].copy();eps=1e-6
    samples=[]
    for sign in [-1,1]:
        d.qpos[mapping.qa]=q;d.qpos[mapping.qa[columns]]+=sign*eps*dq;mujoco.mj_kinematics(mapping.m,d);samples.append(d.site_xpos[site].copy())
    assert float((samples[1]-samples[0])@normal/(2*eps))==pytest.approx(float(bias[rows]@(A@dq)),abs=1e-8)
    assert mapping.d.time==0.


def test_first19s_preserved_bounded_ramp_contact_gate_and_reset(authored):
    c=make(authored,stub=True);b=c.arm.balance
    native={k:getattr(b.m,k).copy() for k in ['actuator_gainprm','actuator_biasprm','actuator_forcerange']}
    prior=np.zeros(5)
    for i in range(11000):
        t=i*.002;f,info=c.force(packet(c,t,[.5,.5,.5,.5,4.]),now_s=t)
        if t<19.:
            np.testing.assert_array_equal(c.arm.original_constant_bias,c._base_motor_bias)
            assert dict(zip(info['goal_joint_names'],info['goal_joint_position_rad']))==c._index_schedule.original.goals(t)
        assert np.max(abs(c.virtual_force-prior))<=.004+1e-12
        assert np.max(np.abs(info['applied_virtual_digit_force_N']))<=2.
        np.testing.assert_array_equal(f,c.arm.last_force);prior=c.virtual_force.copy()
    assert c.virtual_force[0]>0 and c.virtual_force[4]<0
    no_contact=packet(c,22.,[0,0,0,0,0]);_,info=c.force(no_contact,now_s=22.)
    assert info['applied_virtual_digit_force_N']==[0.]*5
    np.testing.assert_array_equal(c.arm.original_constant_bias,c._base_motor_bias)
    for key,value in native.items():np.testing.assert_array_equal(getattr(b.m,key),value)
    c.reset_episode();assert np.all(c.virtual_force==0.) and np.all(c.force_integral==0.)
    np.testing.assert_array_equal(c.arm.original_constant_bias,c._base_motor_bias)


def test_real_force_caps_previous_action_and_oracle_rejection(authored):
    c=make(authored);b=c.arm.balance
    for t in (0.,.002):
        p=cold(b) if t==0. else valid(b,t);f,_=c.force(p,now_s=t)
        np.testing.assert_array_equal(f,b.last_force)
        assert np.all(f>=b.caps[:,0]) and np.all(f<=b.caps[:,1])
    p=valid(b,.004);p['previous_action'].fill(0.)
    with pytest.raises(ValueError,match='Previous action'):c.force(p,now_s=.004)
    with pytest.raises(RuntimeError):c.force(p,now_s=.004)
    c.reset_episode();p=cold(b);p['contact_object_id']=np.ones(5)
    with pytest.raises(ValueError):c.force(p,now_s=0.)
    with pytest.raises(RuntimeError):c.force(cold(b),now_s=0.)
