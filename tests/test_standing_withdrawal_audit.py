from doorbench.dexterous.standing_withdrawal_audit import withdrawal_checks


def rows():
    return [dict(sim_time_s=i/10,door_q=.08 if i<=10 else .3,handle_angle_rad=0.,bolt_slide_m=0.,
        pad_grasp=dict(valid_pad_grasp=i<=10,contacts=[]),
        left_surface=dict(palm_normal_load_N=3.),right_environment_clearance_m=.05 if i>=15 else 0.) for i in range(21)]


def audit(data):
    return withdrawal_checks({'sustained_pad_grasp':False,'native_motor_limits':True},data,
        dt=.1,duration=2.,started=.6,release_started=1.,completed=True)


def test_intentional_release_requires_prior_opposition_and_final_clearance():
    result=audit(rows())
    assert 'sustained_pad_grasp' not in result and all(result.values())
    data=rows();data[8]['pad_grasp']['valid_pad_grasp']=False
    assert not audit(data)['opposed_grip_before_intentional_release']
    data=rows();data[-2]['right_environment_clearance_m']=.03
    assert not audit(data)['final_hand_clear_of_environment']
    data=rows();data[8]['door_q']=.3
    assert not audit(data)['opening_bounded_before_intentional_release']


def test_release_does_not_exempt_wrong_loaded_surfaces_or_failed_motor_checks():
    data=rows();data[12]['pad_grasp']['contacts']=[{'pad_qualified':False}]
    assert not audit(data)['no_invalid_loaded_right_surfaces']
    result=withdrawal_checks({'native_motor_limits':False},rows(),dt=.1,duration=2.,started=.6,release_started=None,completed=False)
    assert not result['native_motor_limits'] and not result['withdrawal_route_completed']
    assert not result['opposed_grip_before_intentional_release']


def test_compressed_evidence_needs_only_one_pass():
    class Once:
        def __iter__(self):
            assert not getattr(self,'read',False), 'Full evidence reread'
            self.read=True
            yield from rows()
    assert audit(Once()) == audit(rows())


def test_missing_interval_and_outside_window_contact_still_fail():
    data=rows();data.pop(7)
    assert not audit(data)['opposed_grip_before_intentional_release']
    data=rows();data[13]['pad_grasp']['contacts']=[{'pad_qualified':False}]
    assert not audit(iter(data))['no_invalid_loaded_right_surfaces']


def test_coincident_windows_do_not_duplicate_interval_counts():
    data=rows()
    result=withdrawal_checks({},iter(data),dt=.1,duration=2.,started=1.,release_started=1.,completed=True)
    assert result['resting_grip_before_withdrawal']
    assert result['opposed_grip_before_intentional_release']


def test_clearance_keeps_pair_order_and_restores_ceiling_after_penetration(monkeypatch):
    import mujoco
    import pytest
    from doorbench.dexterous.standing_withdrawal_audit import environment_clearance
    calls=[];distances=[.2,.4,-.01,-.03]
    def exact(model,data,g,h,cap,segment):
        calls.append((g,h,cap));return min(distances[g],cap)
    monkeypatch.setattr(mujoco,'mj_geomDistance',exact)
    assert environment_clearance(None,None,iter([(i,10) for i in range(4)]))==-.03
    assert [(g,h) for g,h,_ in calls]==[(i,10) for i in range(4)]
    assert calls[0][2]==calls[3][2]==.5
    assert calls[1][2]==calls[2][2]==pytest.approx(.200000001)
    with pytest.raises(ValueError,match='nonempty'):
        environment_clearance(None,None,[])


def test_clearance_matches_original_exact_queries_across_real_poses():
    import mujoco
    import numpy as np
    from doorbench.dexterous.standing_withdrawal_audit import environment_clearance
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="10 10 .1"/>
      <geom type="box" size=".1 .2 .05" pos=".2 0 .3"/>
      <geom type="capsule" size=".03 .2" pos="0 .2 .3"/>
      <body><freejoint/><geom type="box" size=".05 .04 .03"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);g=int(m.body_geomadr[1]);pairs=[(g,h) for h in range(m.ngeom) if h!=g]
    rng=np.random.default_rng(901)
    for _ in range(100):
        d.qpos[:3]=rng.uniform([-.3,-.3,0],[.3,.3,.8])
        q=rng.normal(size=4);d.qpos[3:7]=q/np.linalg.norm(q);mujoco.mj_forward(m,d)
        expected=min(float(mujoco.mj_geomDistance(m,d,a,b,.5,None)) for a,b in pairs)
        assert environment_clearance(m,d,pairs)==expected
