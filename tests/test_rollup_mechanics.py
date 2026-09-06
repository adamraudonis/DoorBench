"""A rolling curtain must wind under native contact rather than translate away."""
import json
import math
import numpy as np
import mujoco
import pytest
from doorbench.spec import generate_all
from doorbench.build import export_door
from doorbench.geometry.rollup import curtain_dimensions,counterbalance_parameters

SPECS=[s for s in generate_all() if s['family']=='rollup']

@pytest.fixture(scope='module')
def exports(tmp_path_factory):
    root=tmp_path_factory.mktemp('rollup')
    for spec in SPECS:export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('json','mjcf'))
    return root

@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_all_rolling_curtains_have_real_material_reserve_intact_header_and_clear_closed_hardware(exports,spec):
    path=exports/'doors'/spec['id'];desc=json.loads((path/'model.json').read_text());mechanism=desc['meta']['rollup_curtain'];layout=curtain_dimensions(spec)
    leaves=[b for b in desc['bodies'] if b['semantic']=='leaf']
    assert len(leaves)==layout['slat_count']>30
    assert layout['physical_curtain_length_m']>spec['leaf']['height']+.5
    assert mechanism['counterbalance']['fraction']==spec['kinematics']['counterbalance_fraction']
    assert len({b['parent'] for b in leaves})==len(leaves)
    if spec['kinematics']['opener']=='chain_hoist':
        assert mechanism['drive']['mode']=='manual_chain' and mechanism['drive']['chain_hoist_supported'] is True
        assert desc['meta']['rollup_hoist']['free_root_joint']=='hoist_chain_free'
    for tier in ('full','simple','minimal'):
        m=mujoco.MjModel.from_xml_path(str(path/('door.xml' if tier=='full' else 'door_'+tier+'.xml')));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        assert m.opt.timestep<=.0005
        assert not any(m.body_gravcomp)
        assert not any(m.jnt_type[m.joint(n).id]!=mujoco.mjtJoint.mjJNT_HINGE for n in mechanism['slat_joints'])
        h=m.geom('wall_header').id
        assert m.geom_pos[h,2]-m.geom_size[h,2]==pytest.approx(spec['opening']['height'],abs=1e-6)
        # MJCF serializes each link position/quaternion to six decimals;
        # 40–60 composed links accumulate a few tens of micrometres.
        assert d.site_xpos[m.site('curtain_bottom').id,2]==pytest.approx(.02,abs=5e-5)
        assert not [(m.geom(c.geom1).name,m.geom(c.geom2).name,c.dist) for c in d.contact if c.dist<-.00005]
        assert m.body_mass[m.body('curtain_barrel').id]>10.
        for b in leaves:assert m.body_mass[m.body(b['name']).id]>.1
        if layout['grille']:
            assert not any('envelope' in g['name'] for b in leaves for g in b['geoms'])
            assert all(any('link_' in g['name'] for g in b['geoms']) for b in leaves)
        else:
            assert all(any(g['name'].endswith('_envelope') and g['collision'] and not g['visual'] for g in b['geoms']) for b in leaves)


def test_torsion_spring_has_real_open_preload_and_preserves_failed_fraction():
    for spec in SPECS:
        layout=curtain_dimensions(spec)
        healthy=counterbalance_parameters(layout,60.,4.,.9)
        assert 0<healthy['open_torque_Nm']<healthy['closed_torque_Nm']
        assert healthy['springref_rad']>healthy['estimated_open_angle_rad']>0
        assert healthy['stiffness_Nm_rad']*(healthy['springref_rad']-healthy['estimated_open_angle_rad'])==pytest.approx(healthy['open_torque_Nm'])
        weak=counterbalance_parameters(layout,60.,4.,.5)
        assert weak['closed_torque_Nm']/healthy['closed_torque_Nm']==pytest.approx(.5/.9)
        failed=counterbalance_parameters(layout,60.,4.,0.)
        assert failed['closed_torque_Nm']==failed['open_torque_Nm']==failed['stiffness_Nm_rad']==0.


def test_native_open_initialization_is_model_bound_force_limited_and_does_not_promote_timeout(exports):
    from doorbench.rollup import prepare_rollup_open
    path=exports/'doors'/'db0001_rollup';meta=json.loads((path/'model.json').read_text())['meta']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));original=m.qpos0.copy();mass=m.body_mass.copy();friction=m.geom_friction.copy()
    report=prepare_rollup_open(m,meta,time_limit_s=.02)
    assert not report['ok'] and report['reason']=='native_goal_not_reached'
    assert 'qpos' not in report and report['peak_force_N']<=120
    assert np.array_equal(m.qpos0,original) and np.array_equal(m.body_mass,mass) and np.array_equal(m.geom_friction,friction)
    cached=prepare_rollup_open(m,meta,time_limit_s=.02)
    assert cached['cache_hit'] and cached['compiled_model_sha256']==report['compiled_model_sha256']
    m.geom_friction[m.geom('curtain_barrel_shell').id,0]+=.01
    altered=prepare_rollup_open(m,meta,time_limit_s=.02)
    assert not altered['cache_hit'] and altered['compiled_model_sha256']!=report['compiled_model_sha256']
    with pytest.raises(ValueError):prepare_rollup_open(m,meta,np.zeros(m.nq+1),time_limit_s=.02)
    with pytest.raises(ValueError):prepare_rollup_open(m,meta,time_limit_s=float('nan'))


def test_physical_chain_cannot_be_initialized_by_substituting_a_bottom_handle(exports):
    from doorbench.rollup import prepare_rollup_open
    path=exports/'doors'/'db0258_rollup';meta=json.loads((path/'model.json').read_text())['meta'];m=mujoco.MjModel.from_xml_path(str(path/'door.xml'))
    report=prepare_rollup_open(m,meta)
    assert not report['ok'] and report['reason']=='chain_hoist_requires_material_link_initializer'
    assert report['elapsed_native_s']==0 and 'qpos' not in report


def test_shared_rollup_force_tracks_original_clock_and_never_increases_hand_limit():
    from doorbench.rollup import rollup_handle_force
    params=dict(start_m=.02,goal_m=2.54,mass_kg=75.,duration_s=12.,force_limit_N=120.)
    for t in np.linspace(0,16,65):
        result=rollup_handle_force(.02,0.,elapsed_s=float(t),**params)
        assert abs(result['force_N'])<=120
        assert result['damping_N_s_m']==pytest.approx(2*math.sqrt(3500*75))
    for t,z in [(0.,.02),(12.,2.54),(16.,2.54)]:
        result=rollup_handle_force(z,0.,elapsed_s=t,**params)
        assert result['target_z_m']==pytest.approx(z)
        assert result['target_speed_m_s']==result['force_N']==0.
    with pytest.raises(ValueError):rollup_handle_force(0.,0.,elapsed_s=0.,**{**params,'force_limit_N':121.})


def test_initializer_does_not_integrate_or_transplant_an_attached_robot(exports):
    from doorbench.rollup import prepare_rollup_open
    path=exports/'doors'/'db0718_rollup';meta=json.loads((path/'model.json').read_text())['meta']
    spec=mujoco.MjSpec.from_file(str(path/'door.xml'))
    robot=spec.worldbody.add_body(name='external_robot',pos=[0.,-2.,1.])
    robot.explicitinertial=True;robot.mass=10.;robot.inertia=[.04,.04,.04];robot.ipos=[0.,0.,0.]
    robot.add_freejoint(name='robot_free')
    robot.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.1,0.,0.],mass=10.)
    m=spec.compile();before=m.qpos0.copy();report=prepare_rollup_open(m,meta,time_limit_s=.02)
    assert not report['ok'] and report['reason']=='additional_dynamic_bodies_require_door_only_initialization'
    assert report['unsupported_joint_names']==['robot_free'] and report['elapsed_native_s']==0.
    assert 'qpos' not in report and np.array_equal(before,m.qpos0)


def test_full_mass_native_curtain_reaches_physical_up_stops_holds_without_hand_and_recloses(exports):
    from doorbench.rollup import prepare_rollup_open,rollup_grip_dynamics,rollup_handle_force
    path=exports/'doors'/'db0718_rollup';description=json.loads((path/'model.json').read_text());meta=description['meta'];mechanism=meta['rollup_curtain']
    m=mujoco.MjModel.from_xml_path(str(path/'door.xml'));d=mujoco.MjData(m);report=prepare_rollup_open(m,meta)
    assert report['ok'],report
    assert report['peak_force_N']<=120 and report['max_penetration_m']<.001
    assert not any(report['warnings'])
    d.qpos[:]=report['qpos'];d.qvel[:]=report['qvel'];mujoco.mj_forward(m,d)
    point=m.site('curtain_bottom').id;grip=m.site('lift_handle_grip').id;body=m.site_bodyid[grip]
    assert d.site_xpos[point,2]>2.51  # original H=2.44 m; full lintel clearance
    # The settled trajectory may end slightly below an unloaded stop. A
    # bounded upward proof load must meet the actual stop-angle contact.
    stop_contact=False;depth=0.
    for _ in range(round(.25/m.opt.timestep)):
        d.qfrc_applied[:]=0
        mujoco.mj_applyFT(m,d,np.array([0.,0.,120.]),np.zeros(3),d.site_xpos[grip],body,d.qfrc_applied)
        mujoco.mj_step(m,d)
        stop_contact |= any('curtain_stop_lug' in m.geom(c.geom1).name and 'curtain_up_stop' in m.geom(c.geom2).name or
                            'curtain_stop_lug' in m.geom(c.geom2).name and 'curtain_up_stop' in m.geom(c.geom1).name for c in d.contact)
        depth=max(depth,max((-float(c.dist) for c in d.contact),default=0.))
    assert stop_contact
    assert d.site_xpos[point,2]<2.57 and depth<.001
    # Support comes from real stop angles and the spring, not a stored hand
    # effort or a fabricated zero-velocity pose.
    d.qfrc_applied[:]=0
    for _ in range(round(.5/m.opt.timestep)):mujoco.mj_step(m,d)
    assert d.site_xpos[point,2]>2.51
    start=float(d.site_xpos[point,2])
    for i in range(round(15/m.opt.timestep)):
        mujoco.mj_forward(m,d);state=rollup_grip_dynamics(m,d,mechanism)
        force=rollup_handle_force(float(d.site_xpos[point,2]),state['grip_speed_m_s'],start_m=start,goal_m=.02,
            elapsed_s=i*m.opt.timestep,mass_kg=state['grip_effective_mass_kg'])['force_N']
        assert abs(force)<=120
        d.qfrc_applied[:]=0;mujoco.mj_applyFT(m,d,np.array([0.,0.,force]),np.zeros(3),d.site_xpos[grip],body,d.qfrc_applied);mujoco.mj_step(m,d)
        depth=max(depth,max((-float(c.dist) for c in d.contact),default=0.))
    assert d.site_xpos[point,2]<.03
    assert depth<.001 and not any(w.number for w in d.warning)
