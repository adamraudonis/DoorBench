"""Keep articulated body measurements on the same clock as integrated joints."""
import mujoco
import numpy as np


def test_forward_refresh_aligns_body_pose_without_changing_integrated_state():
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body name="moving" pos="0 0 1"><freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    data.qvel[:]=[2.,0.,0.,0.,.5,0.]
    mujoco.mj_forward(model,data)
    mujoco.mj_step(model,data)
    body=model.body('moving').id
    # The dynamics-stage pose is explicitly not the new joint-state pose.
    assert abs(data.xpos[body,0]-data.qpos[0])>.003
    qpos=data.qpos.copy();qvel=data.qvel.copy();time=float(data.time)
    wrench=data.xfrc_applied.copy();generalized=data.qfrc_applied.copy()
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,qpos)
    np.testing.assert_array_equal(data.qvel,qvel)
    np.testing.assert_array_equal(data.xfrc_applied,wrench)
    np.testing.assert_array_equal(data.qfrc_applied,generalized)
    assert data.time==time
    np.testing.assert_allclose(data.xpos[body],qpos[:3],atol=1e-12)
    np.testing.assert_allclose(data.xquat[body],qpos[3:7],atol=1e-12)
