import mujoco
import numpy as np

from doorbench.dexterous.full_sequence_teacher import ReadinessCollisionScreen


def test_readiness_rejects_shallow_precontact_and_wrong_asset_frame(tmp_path):
    door = tmp_path/'door.xml'
    robot = tmp_path/'robot.xml'
    door.write_text('''<mujoco><worldbody>
      <body name="leaf" pos="0 0 .5"><joint name="leaf_hinge" axis="0 0 1"/>
        <geom type="box" size=".1 .1 .2"/>
        <body name="leaf_handle" pos=".2 .1 .2"><joint name="leaf_handle_hinge" axis="0 1 0"/>
          <geom type="sphere" size=".01"/></body>
        <body name="bolt" pos=".2 0 0"><joint name="leaf_latch_bolt_slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".01"/></body>
      </body></worldbody></mujoco>''')
    robot.write_text('''<mujoco><worldbody><body name="pelvis">
      <freejoint name="free_base"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="rh_ffdistal" pos=".1195 0 .5"><joint name="finger" axis="0 1 0"/>
        <geom type="sphere" size=".02"/></body>
      </body></worldbody></mujoco>''')
    screen = ReadinessCollisionScreen(door,robot)
    d = mujoco.MjData(screen.m)
    mujoco.mj_kinematics(screen.m,d)
    poses = [np.r_[d.xpos[screen.m.body(name).id],d.xquat[screen.m.body(name).id]]
             for name in ('leaf_handle','leaf')]
    angles = dict(operator=0.,leaf=0.,latch=0.)
    proposal = dict(initial_root=[1.,0.,0.,1.,0.,0.,0.],
                    acquisition=dict(joint_names=['finger'],path_qpos=[[0.],[0.]]))
    assert screen.check(proposal,*poses,angles)['passed']
    proposal['initial_root'][0] = 0.
    failed = screen.check(proposal,*poses,angles)
    assert not failed['passed']
    assert -.003 < failed['bad_samples'][0]['contacts'][0]['distance_m'] < 0.
    proposal['initial_root'][0] = 1.
    wrong_pose = poses[0].copy()
    wrong_pose[0] += .01
    mismatch = screen.check(proposal,wrong_pose,poses[1],angles)
    assert not mismatch['passed']
    assert mismatch['reason'] == 'Actual asset pose differs from readiness screen'
