"""Independent synthetic-domain and unstepped collision tests, not rollouts."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous import isaac_coupled_release_audit as audit
from test_isaac_coupled_release_geometry import toy_scene,map_fixture


def geometry_model(*,change=None):
    scene=toy_scene();m,d=scene.m,scene.d;initial=d.qpos.copy()
    plan,_,_=map_fixture()
    feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    rh,lh=[m.site('robot/'+hand+'_palm_touch').id for hand in ('rh','lh')]
    body=int(m.jnt_bodyid[m.joint('robot/free_base').id])
    model=SimpleNamespace(m=m,d=d,rq=scene.root,initial=initial,
        leaf=m.body('leaf').id,handle=m.body('leaf_handle').id,rh=rh,lh=lh,
        lq=int(m.joint('leaf_hinge').qposadr[0]),oq=int(m.joint('leaf_handle_hinge').qposadr[0]),
        bq=int(m.joint('leaf_latch_bolt_slide').qposadr[0]),through=3.5,plan=plan,
        elapsed=np.asarray(plan['elapsed_s']),angles=np.asarray(plan['leaf_rad']),
        source_data=dict(source_admission={'grasp_profile':'volar-phalange-v1'}),
        c=dict(initial_feet_positions=d.xpos[feet].copy().tolist(),
            initial_feet_rotations=d.xmat[feet].reshape(2,3,3).copy().tolist(),
            initial_com=d.subtree_com[body].copy().tolist()))
    model.upper_angle=lambda t:float(np.interp(t,[0,16],[.10,.4]))
    model.lower_angle=lambda t:.08
    def evaluate(t,angle,operator,latch):
        d.qpos[:]=initial;d.qpos[model.lq]=angle;d.qpos[model.oq]=operator;d.qpos[model.bq]=latch
        if change=='feet':d.qpos[scene.root]+=.001001
        if change=='root':d.qpos[scene.root]+=.030001
        if change=='joint':d.qpos[m.joint('robot/left_shoulder_pitch').qposadr[0]]=2.020001
        if change=='tilt':d.qpos[scene.root+3:scene.root+7]=[np.cos(np.radians(4.001)/2),np.sin(np.radians(4.001)/2),0,0]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        lp,rp=d.site_xpos[lh].copy(),d.site_xpos[rh].copy()
        if change=='palm_goal':lp[0]+=.001001
        return dict(qpos=d.qpos.copy(),coordinates=np.r_[np.zeros(6),np.zeros(25)],
            release_clock_s=8.5*t/16,right_follow_weight=1.,
            left_palm_position=lp,left_palm_rotation=d.site_xmat[lh].reshape(3,3).copy(),
            right_palm_position=rp,right_palm_rotation=d.site_xmat[rh].reshape(3,3).copy())
    model.evaluate=evaluate
    return model


def test_single_toy_configuration_recomputes_original_geometry_without_forces():
    model=geometry_model();inspector=audit.GeometryInspector(model)
    row=inspector.inspect(0.,model.initial[model.lq],0.,0.)
    assert row['failed']==[]
    assert max(row['values'].values())<1e-12
    final=inspector.inspect(16.,.4,0.,.001)
    assert not final['failed'] and final['final_gap_m']>=.04
    assert final['blend_gap_m']>=.004


@pytest.mark.parametrize('change,reason',[
    ('feet','foot_position_error_m'),('root','root_translation_m'),
    ('joint','original_collision_joint_loopback_limits'),('tilt','torso_tilt_deg'),
    ('palm_goal','left_position_error_m')])
def test_original_bounds_recomputed_instead_of_trusting_map_values(change,reason):
    model=geometry_model(change=change);inspector=audit.GeometryInspector(model)
    result=inspector.inspect(0.,model.initial[model.lq],0.,0.)
    assert reason in result['failed']


def test_receiving_only_collision_geometries_cannot_disappear_from_clearance():
    model=geometry_model();m,d=model.m,model.d
    geom=int(m.body_geomadr[m.site_bodyid[model.rh]])
    m.geom_contype[geom]=0;m.geom_conaffinity[geom]=1
    inspector=audit.GeometryInspector(model)
    assert geom in inspector.right


def test_raw_contact_recomputes_selected_anatomy_and_retains_distal_counter_score():
    model=geometry_model();m,d=model.m,model.d
    lever=m.geom('leaf_handle_lever_col_n').id
    # Deliberately collide the toy palm with the lever. A palm is never an
    # anatomical digit pad; no forces or source contact labels are invented.
    rotation=d.xmat[model.handle].reshape(3,3)
    m.geom_pos[lever]=rotation.T@(d.site_xpos[model.rh]-d.xpos[model.handle])
    m.geom_sameframe[lever]=0
    inspector=audit.GeometryInspector(model)
    result=inspector.inspect(0.,model.initial[model.lq],0.,0.)
    assert any('selected anatomy' in reason for reason in result['failed'])
    assert result['original_distal_invalid_geometry_patches']>0


def test_blend_and_final_clearance_require_literal_original_margins(monkeypatch):
    model=geometry_model();inspector=audit.GeometryInspector(model)
    original=inspector.gap
    def gap(first,second,limit):
        if first is inspector.right:
            return .003999 if second is inspector.handle else .039999
        return original(first,second,limit)
    monkeypatch.setattr(inspector,'gap',gap)
    result=inspector.inspect(16.,.4,0.,0.)
    assert 'literal4mm_all_handle_blend_clearance' in result['failed']
    assert 'original_final_right_hand_environment_clearance' in result['failed']


def test_dense_domain_contains_source_operator_bolt_extrema_and_actual_cell_centers():
    model=geometry_model();points=list(audit.iter_domain_samples(model))
    assert len(points)>50000
    assert {point[3] for point in points}=={-.001,0.,.001}
    assert {point[2] for point in points}=={-.01,0.,.01}
    assert (float((model.elapsed[20]+model.elapsed[21])/2),.08,0.,0.) in points
    midpoint=float((model.angles[2]+model.angles[3])/2)
    assert (16.,midpoint,0.,0.) in points
    assert all(.08<=a<=model.upper_angle(t) for t,a,_,_ in points)


@pytest.mark.parametrize('explicit_null',[False,True])
def test_missing_or_null_upper_nodes_use_the_full_declared_rectangular_domain(explicit_null):
    model=geometry_model()
    if explicit_null:model.plan['admitted_leaf_upper_nodes']=None
    else:model.plan.pop('admitted_leaf_upper_nodes')
    model.upper_angle=lambda t:float(model.angles[-1])
    points=list(audit.iter_domain_samples(model))
    assert (0.,.4,0.,0.) in points and (16.,.4,.01,.001) in points
    assert all(.08<=angle<=.4 for _,angle,_,_ in points)


def test_coarse_diagnostic_never_passes_even_without_a_geometric_failure():
    model=geometry_model()
    result=audit.audit_geometry(model,coarse=True)
    assert not result['passed'] and result['coarse_diagnostic']
    assert result['failed_samples']==0 and result['samples']==5131
    assert result['exact_initial_state'] and result['complete_original_geometry_coverage']
    assert result['limits']==audit.ORIGINAL_LIMITS


def test_truncated_coverage_cannot_pass_by_omitting_blend_or_final_clearance(monkeypatch):
    model=geometry_model()
    monkeypatch.setattr(audit,'iter_domain_samples',lambda *args,**kwargs:iter(()))
    result=audit.audit_geometry(model)
    assert not result['passed'] and not result['complete_original_geometry_coverage']
    assert result['minimum_blend_all_handle_clearance_m'] is None
    assert result['minimum_final_right_hand_environment_clearance_m'] is None


def test_receipt_remains_explicitly_unadmitted_even_after_static_success(monkeypatch,tmp_path):
    path=tmp_path/'synthetic-envelope.json';path.write_text('explicit synthetic fixture')
    model=geometry_model();closed=[];verified=[]
    model._file_hashes={str(path):audit.digest(path)}
    model.source_data.update(source_context_sha256='a'*64,source_state_sha256='b'*64,
        start_time_s=42.,source_archive_path='actual-fixture.npz',motor_contract_sha256='c'*64)
    model.verify_inputs=lambda:verified.append(True)
    model.close=lambda:closed.append(True)
    monkeypatch.setattr(audit,'IsaacCoupledReleaseGeometry',lambda *args,**kwargs:model)
    monkeypatch.setattr(audit,'audit_geometry',lambda *args,**kwargs:dict(passed=True))
    receipt=audit.audit_isaac_coupled_envelope(path,source_config='synthetic-config.json')
    assert receipt['schema']==audit.SCHEMA and receipt['source_engine']=='isaac-physx'
    assert receipt['passed'] and receipt['authorized_stages']==receipt['physics_steps']==0
    assert receipt['source_sample_playback']==receipt['active_state_writes']==0
    for key in ('runtime_route_exported','physical_admission','physical_contact_qualification',
        'delivered_motor_force_checked','dynamic_balance_qualification','motion_rate_qualification',
        'post_motion_limiter_geometry_checked'):
        assert receipt[key] is False
    assert verified==[True] and closed==[True]
    assert receipt['input_sha256'][str(path)]==audit.digest(path)


def test_source_change_at_end_of_audit_prevents_any_receipt(monkeypatch,tmp_path):
    model=geometry_model();closed=[]
    def fail():raise ValueError('Changed source during audit')
    model.verify_inputs=fail;model.close=lambda:closed.append(True)
    monkeypatch.setattr(audit,'IsaacCoupledReleaseGeometry',lambda *args,**kwargs:model)
    monkeypatch.setattr(audit,'audit_geometry',lambda *args,**kwargs:dict(passed=True))
    with pytest.raises(ValueError,match='Changed source'):
        audit.audit_isaac_coupled_envelope(tmp_path/'map.json',source_config='source.json')
    assert closed==[True]


def test_dense_sampler_honors_lower_curve_and_covers_its_knots_at_all_latch_extrema():
    model=geometry_model()
    nodes=[[0.,.08],[14.,.08],[15.5,.1],[16.,.1]]
    model.plan['admitted_leaf_lower_nodes']=nodes
    model.lower_angle=lambda t:float(np.interp(t,[n[0] for n in nodes],[n[1] for n in nodes]))
    points=list(audit.iter_domain_samples(model))
    assert len(points)>50000
    assert all(model.lower_angle(t)<=a<=model.upper_angle(t) for t,a,_,_ in points)
    for operator in (-.01,.01):
        for latch in (-.001,.001):
            assert (15.5,.1,operator,latch) in points
            assert (15.5,model.upper_angle(15.5),operator,latch) in points
    assert (16.,.08,0.,0.) not in points
    assert (14.,.08,0.,0.) in points and (16.,.1,0.,0.) in points
