from doorbench.dexterous.opening_audit import audit_opening


def evidence():
    rows=[]
    for i in range(201):
        t=i*.02
        rows.append(dict(time_s=t,sim_time_s=t,root=[0,0,.83],joints=[0.],torso_tilt_deg=1.,
            motor_forces=[.2],joint_torque_command=[.2],joint_torque_sent=[.2],
            door=dict(leaf_hinge=max(0.,min(1.,t-2)),leaf_handle_hinge=min(.87,t),leaf_latch_bolt_slide=min(.0127,t*.015)),
            grasp_opposition=dict(opposed=.1<t<2.)))
    return rows,dict(runtime_pose_writes=0,direct_door_commands=False,contact_material_audit=dict(backend_values_verified=True,backend_offsets_verified=True)),dict(actuators=[dict(force_range=[-1.,1.])])


def test_rejects_extra_render_steps_even_when_the_door_moves():
    rows,config,motors=evidence()
    assert audit_opening(rows,config,motors)['passed']
    for r in rows:r['sim_time_s']*=1.2
    result=audit_opening(rows,config,motors)
    assert not result['passed'] and not result['checks']['correct_physics_clock']


def test_door_motion_alone_is_not_robot_handle_opening():
    rows,config,motors=evidence()
    for r in rows:r['grasp_opposition']['opposed']=False
    result=audit_opening(rows,config,motors)
    assert not result['passed'] and not result['checks']['opposed_grasp_while_turning']
    rows,config,motors=evidence();rows[80]['motor_forces']=[1.1]
    assert not audit_opening(rows,config,motors)['checks']['native_motor_force_limits']
    rows,config,motors=evidence();config['direct_door_commands']=True
    assert not audit_opening(rows,config,motors)['passed']
