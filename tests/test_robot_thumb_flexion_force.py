import numpy as np,pytest,mujoco
from test_sensor_balance import authored
from test_sensor_digit_force_control import make
from doorbench.dexterous.robot_thumb_flexion_force import RobotThumbFlexionForce


def test_thumb_pressure_uses_only_original_flexion_motors(authored):
    c=make(authored);full=c.pad_force
    small=RobotThumbFlexionForce(full.m,full.names,full.actions,full.matrix)
    q=c.arm.balance.desired.copy();forces=np.array([1.,.7,-.4,.6,1.2])
    allbias,_=full.motor_bias(q,forces);bias,info=small.motor_bias(q,forces)
    keep=[full.actions.index('rh_A_THJ'+str(k)) for k in (1,2)]
    excluded=[full.actions.index('rh_A_THJ'+str(k)) for k in (3,4,5)]
    np.testing.assert_array_equal(bias[excluded],0.)
    np.testing.assert_allclose(bias[keep],allbias[keep],atol=1e-16)
    other=np.setdiff1d(np.arange(61),keep+excluded)
    np.testing.assert_array_equal(bias[other],allbias[other])
    assert info['th']['joint_names']==['rh_THJ2','rh_THJ1']
    np.testing.assert_allclose(info['th']['unavailable_passive_split_moment_Nm'],0.,atol=1e-16)
    # Independently differentiate the pad force line in the retained two-DOF
    # motion subspace. It must agree with the original transmission's work.
    columns,rows,A,site=small.groups['th'];dq=np.array([.6,-.2]);eps=1e-6
    d=mujoco.MjData(full.m);d.qpos[:7]=[0,0,1,1,0,0,0];d.qpos[small.qa]=q
    mujoco.mj_kinematics(full.m,d);normal=-d.site_xmat[site].reshape(3,3)[:,2].copy();positions=[]
    for sign in (-1,1):
        d.qpos[small.qa]=q;d.qpos[small.qa[columns]]+=sign*eps*dq;mujoco.mj_kinematics(full.m,d)
        positions.append(d.site_xpos[site].copy())
    assert float((positions[1]-positions[0])@normal*forces[4]/(2*eps))==pytest.approx(float(bias[rows]@(A@dq)),abs=1e-8)
    assert small.d.time==0.
