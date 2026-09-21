"""Synthetic source and private-FK tests; no robot rollout or load evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.dexterous import isaac_panel_planning as planner
from doorbench.dexterous import isaac_panel_geometry_audit as audit
from doorbench.dexterous import isaac_panel_source as source_module
from doorbench.dexterous.qualified_isaac_grasp import digest


def toy_scene(floor_height=0.):
    """Complete named toy geometry tests source/solver wiring, never H1 success."""
    bodies=[]
    for name in planner.JOINT_NAMES:
        body='robot/torso_link' if name=='torso' else 'robot/'+name+'_body'
        hand='lh' if name=='lh_WRJ1' else 'rh' if name=='rh_WRJ1' else None
        hand_x='-.2' if hand=='lh' else '.2'
        palm=(f'<body name="robot/{hand}_palm" pos="{hand_x} -.3 0"><geom size=".01"/><site name="robot/{hand}_palm_touch"/></body>' if hand else '')
        bodies.append(f'<body name="{body}"><joint name="robot/{name}" range="-2 2"/><geom size=".001" contype="0" conaffinity="0"/>{palm}</body>')
    xml='''<mujoco><compiler angle="radian"/><asset><mesh name="cube" vertex="-.01 -.01 -.01 .01 -.01 -.01 -.01 .01 -.01 .01 .01 -.01 -.01 -.01 .01 .01 -.01 .01 -.01 .01 .01 .01 .01 .01"/></asset>
      <worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <body name="leaf" pos="2 0 1"><joint name="leaf_hinge" range="0 1.75"/><geom name="leaf_slab_main" type="box" size=".1 .02 .2"/>
        <body name="leaf_handle" pos=".1 0 0"><joint name="leaf_handle_hinge" range="-.1 1"/><geom size=".01"/>
          <body name="bolt"><joint name="leaf_latch_bolt_slide" type="slide" range="-.01 .01"/><geom size=".005"/></body>
        </body></body><body name="robot/base" pos="0 0 1"><freejoint name="robot/free_base"/><geom size=".01" contype="0" conaffinity="0"/>
      <body name="robot/left_ankle_link" pos="-.1 0 -.98"><geom size=".01" contype="0" conaffinity="0"/></body>
      <body name="robot/right_ankle_link" pos=".1 0 -.98"><geom size=".01" contype="0" conaffinity="0"/></body>
      <body name="robot/left_elbow_link" pos="-.4 -.4 0"><geom type="mesh" mesh="cube"/></body>
      <body name="robot/right_elbow_link" pos=".4 -.4 0"><geom type="mesh" mesh="cube"/></body>
      '''+''.join(bodies)+'''</body></worldbody></mujoco>'''
    xml=xml.replace('<geom name="floor"',f'<geom pos="0 0 {floor_height}" name="floor"')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qpos[m.joint('leaf_hinge').qposadr[0]]=.091
    mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
    return SimpleNamespace(m=m,d=d,root=int(m.joint('robot/free_base').qposadr[0]))


@pytest.fixture
def actual_fixture(tmp_path,monkeypatch):
    scene=toy_scene();file=tmp_path/'synthetic-source.npz';file.write_bytes(b'synthetic evidence only')
    admitted=dict(schema=planner.SOURCE_SCHEMA,source_engine='isaac-physx',source_run=str(tmp_path),source_time_s=60.,
        initial_qpos=scene.d.qpos.tolist(),initial_qvel=scene.d.qvel.tolist(),source_qualification={'passed':True},
        coordinate_admission={'passed':True},input_sha256={str(file):digest(file)},authorized_stages=0,
        robot_path=str(tmp_path/'robot.xml'),door_xml_path=str(tmp_path/'door.xml'),door_usd_path=str(tmp_path/'door.usd'))
    calls=[]
    def admit(*args,**kwargs):calls.append((args,kwargs));return copy.deepcopy(admitted)
    monkeypatch.setattr(source_module,'admit_isaac_panel_source',admit)
    monkeypatch.setattr(planner,'LandedLeftScene',lambda *args:toy_scene())
    context=planner.admit_isaac_panel_context(tmp_path,robot=admitted['robot_path'],door_xml=admitted['door_xml_path'],door_usd=admitted['door_usd_path'])
    return SimpleNamespace(context=context,admitted=admitted,calls=calls,file=file,tmp=tmp_path)


def candidate_without_solves(fixture,monkeypatch,*,target=1.62):
    """Deliberately frozen references; distant aperture must fail dense audit."""
    def frozen(m,d,base,options,progress=None):
        lq=int(m.joint('leaf_hinge').qposadr[0]);rows=[]
        for angle in np.linspace(base[lq],options.target_aperture_rad,options.nodes):
            q=base.copy();q[lq]=angle
            rows.append(dict(leaf_angle_rad=float(angle),qpos=q.tolist(),root_delta=[0.]*6,
                joint_targets={name:float(base[m.joint('robot/'+name).qposadr[0]]) for name in planner.JOINT_NAMES}))
        return rows,list(planner.JOINT_NAMES)
    monkeypatch.setattr(planner,'_solve_panel',frozen)
    return planner.generate_panel_candidate(fixture.context,{'nodes':21},target_aperture_rad=target)


def test_old_native_documents_supply_numbers_not_source_or_authority():
    old=dict(initial_qpos=[999.],rows=[{'qpos':[999.]}],source_run='native',passed=True,
        configuration={'radius_shift_m':-.01,'nodes':81,'target_aperture_rad':.2,
            'maximum_torso_tilt_deg':12.,'right_hand_relaxed_orientation':True,'source_run':'native'})
    original=copy.deepcopy(old);values=planner.numeric_panel_preferences(old)
    assert values['nodes']==81 and values['radius_shift_m']==-.01
    assert set(values)==set(planner.DEFAULT_PREFERENCES) and old==original
    assert 'target_aperture_rad' not in values and 'maximum_torso_tilt_deg' not in values


@pytest.mark.parametrize('key,value',[('nodes',20),('nodes',True),('nodes',402),('max_nfev',0),
    ('root_extent_m',.031),('root_rotation_rad',.051),('joint_margin_rad',0.),
    ('left_palm_rotation_fit_rad',.001),('height_drop_m',float('nan')),
    ('palm_twist_rad',float('inf')),('elbow_clearance_m',.002),('elbow_clearance_ramp_rad',.02),
    ('radius_shift_m',True),('foot_orientation_weight',101.)])
def test_preferences_cannot_expand_bounds_or_include_corrupt_numbers(key,value):
    with pytest.raises(ValueError):planner.numeric_panel_preferences({key:value})


def test_context_is_detached_and_changed_actual_evidence_rejected(actual_fixture):
    context=actual_fixture.context;original=context.qpos.copy();q=context.qpos;q[:]=99.
    payload=context.admission;payload['source_qualification']['passed']=False
    np.testing.assert_array_equal(context.qpos,original)
    assert context.admission['source_qualification']['passed']
    actual_fixture.file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):context.scene()


@pytest.mark.parametrize('field,value',[('schema','doorbench.native-source.v1'),('source_engine','native-mujoco'),
    ('source_qualification',{'passed':False}),('coordinate_admission',{'passed':False}),('authorized_stages',1)])
def test_source_context_rejects_wrong_engine_failed_or_promoted_source(actual_fixture,field,value):
    actual_fixture.admitted[field]=value
    with pytest.raises(ValueError):planner.admit_isaac_panel_context('source',robot='r',door_xml='d',door_usd='u')


def test_failed_source_rejects_before_creating_geometry(monkeypatch):
    def reject(*args,**kwargs):raise ValueError('Actual withdrawal failed original physical checks')
    monkeypatch.setattr(source_module,'admit_isaac_panel_source',reject)
    monkeypatch.setattr(planner,'LandedLeftScene',lambda *args:pytest.fail('Failed source created geometry'))
    with pytest.raises(ValueError,match='physical checks'):
        planner.admit_isaac_panel_context('failed',robot='r',door_xml='d',door_usd='u')


def test_candidate_keeps_exact_source_and_never_inherits_old_endpoint(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch)
    np.testing.assert_array_equal(candidate['rows'][0]['qpos'],actual_fixture.context.qpos)
    assert candidate['source_admission']==actual_fixture.context.admission
    assert candidate['target_aperture_rad']==1.62 and candidate['target_reaches_passage_aperture']
    assert candidate['authorized_stages']==candidate['physics_steps']==candidate['source_sample_playback']==0
    assert not candidate['geometric_admission'] and not candidate['physical_admission'] and not candidate['runtime_route_exported']
    assert candidate['configuration']['maximum_torso_tilt_deg']==4.
    assert candidate['input_sha256'][str(Path(planner.__file__).resolve())]==digest(planner.__file__)


def test_short_segment_does_not_claim_a_passage_aperture(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.3)
    assert not candidate['target_reaches_passage_aperture'] and candidate['passage_aperture_target_rad']==1.57


def test_generator_rejects_adjusted_source_knot(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch)
    rows=copy.deepcopy(candidate['rows']);rows[0]['qpos'][0]=np.nextafter(rows[0]['qpos'][0],1.)
    monkeypatch.setattr(planner,'_solve_panel',lambda *args:(rows,list(planner.JOINT_NAMES)))
    with pytest.raises(ValueError,match='exact admitted'):planner.generate_panel_candidate(actual_fixture.context)


@pytest.mark.parametrize('change',['schema','engine','source','context','epoch','source_qpos','source_qvel',
    'joint_order','authority','physical','runtime','fixed_limit','relaxed_hand','short_grid','nan',
    'first_qpos','first_reference','hidden_door','hidden_root','missing_hash','passage_claim'])
def test_auditor_rejects_relabelled_or_unbound_candidates(actual_fixture,monkeypatch,change):
    candidate=candidate_without_solves(actual_fixture,monkeypatch)
    replacements={'schema':('schema','doorbench.whole-body-panel-plan.v1'),'engine':('source_engine','native'),
        'context':('source_context_sha256','bad'),'epoch':('source_time_s',59.998),
        'authority':('authorized_stages',1),'physical':('physical_admission',True),'runtime':('runtime_route_exported',True),
        'passage_claim':('target_reaches_passage_aperture',False)}
    if change in replacements:key,value=replacements[change];candidate[key]=value
    if change=='source':candidate['source_admission']['source_qualification']['passed']=False
    if change=='source_qpos':candidate['initial_qpos'][0]+=1e-9
    if change=='source_qvel':candidate['initial_qvel'][0]+=1e-9
    if change=='joint_order':candidate['names'].reverse()
    if change=='fixed_limit':candidate['configuration']['maximum_torso_tilt_deg']=12.
    if change=='relaxed_hand':candidate['configuration']['right_hand_relaxed_orientation']=True
    if change=='short_grid':candidate['rows'].pop()
    if change=='nan':candidate['rows'][5]['root_delta'][0]=float('nan')
    if change=='first_qpos':candidate['rows'][0]['qpos'][0]=np.nextafter(candidate['rows'][0]['qpos'][0],1.)
    if change=='first_reference':candidate['rows'][0]['root_delta'][0]=1e-15
    if change=='hidden_door':candidate['rows'][3]['qpos'][1]=.02
    if change=='hidden_root':candidate['rows'][3]['root_delta'][0]=.001
    if change=='missing_hash':candidate['input_sha256'].pop(str(actual_fixture.file))
    with pytest.raises(ValueError):audit.validate_panel_candidate(candidate,actual_fixture.context,toy_scene())


def test_native_numerical_solver_runs_unstepped_and_retains_source(actual_fixture,monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *args:pytest.fail('Planner stepped physics'))
    candidate=planner.generate_panel_candidate(actual_fixture.context,{'nodes':21,'max_nfev':1},
        target_aperture_rad=.091001,solver_method='least-squares')
    assert len(candidate['rows'])==21
    np.testing.assert_array_equal(candidate['rows'][0]['qpos'],actual_fixture.context.qpos)
    assert candidate['rows'][0]['nfev']==0 and candidate['rows'][0]['root_delta']==[0.]*6
    assert all(np.isfinite(row['qpos']).all() for row in candidate['rows'])
    assert all(row['nfev']<=1 for row in candidate['rows'])
    assert candidate['physics_steps']==0


def test_constrained_solver_uses_original_interior_constraints(actual_fixture,monkeypatch):
    calls=[]
    def fit(fun,guess,*,method,bounds,constraints,options):
        values=constraints[0]['fun'](guess)
        assert method=='SLSQP' and len(values)==14 and np.isfinite(values).all()
        assert options['maxiter']==1 and np.isfinite(fun(guess))
        calls.append(guess.copy())
        return SimpleNamespace(x=guess.copy(),fun=fun(guess),status=9,message='Synthetic solver limit',nfev=1)
    monkeypatch.setattr(planner,'minimize',fit)
    candidate=planner.generate_panel_candidate(actual_fixture.context,{'nodes':21,'max_nfev':1},target_aperture_rad=.1)
    assert len(calls)==20 and all(r['solver_status']==9 for r in candidate['rows'][1:])
    assert not candidate['geometric_admission']


def test_nonfinite_solve_never_exports_candidate(actual_fixture,monkeypatch):
    monkeypatch.setattr(planner,'least_squares',lambda *args,**kwargs:SimpleNamespace(x=np.full(31,np.nan)))
    with pytest.raises(ValueError,match='Nonfinite'):
        planner.generate_panel_candidate(actual_fixture.context,{'nodes':21},solver_method='least-squares')


@pytest.mark.parametrize('settings',[{'samples':2000},{'samples':True},{'duration_s':float('nan')},
    {'duration_s':2.},{'aperture_speed_limit_rad_s':.15},{'aperture_acceleration_limit_rad_s2':.081},
    {'actual_leaf_lag_rad':.021}])
def test_audit_cannot_relax_density_or_rates(settings,tmp_path):
    with pytest.raises(ValueError):audit.audit_panel_candidate(tmp_path/'absent.json',**settings)


def test_independent_audit_re_admits_and_retains_real_geometry_failure(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    monkeypatch.setattr(mujoco,'mj_step',lambda *args:pytest.fail('Dense auditor stepped physics'))
    receipt,plan,traces=audit.audit_panel_candidate(path)
    assert len(actual_fixture.calls)==2 and receipt['samples']==2001 and traces.shape[0]==2001
    assert not receipt['passed'] and receipt['failures']
    assert receipt['maxima']['left_position_m']>.0001
    assert receipt['limits']['left_position_m']==.0001 and receipt['limits']['torso_tilt_deg']==4.
    assert receipt['minimum_all_rh_scene_clearance_capped_m']>=.04
    assert plan['schema']==audit.PLAN_SCHEMA and plan['authorized_stages']==0
    assert plan['screen_receipt']==receipt and not plan['physical_admission']


def test_passing_toy_static_screen_still_confers_no_runtime_or_passage_authority(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    receipt,plan,_=audit.audit_panel_candidate(path)
    assert receipt['passed'] and receipt['geometric_admission']
    assert not receipt['target_reaches_passage_aperture'] and not receipt['physical_admission']
    assert not plan['runtime_route_exported'] and plan['authorized_stages']==0
    np.testing.assert_array_equal(plan['initial_qpos'],actual_fixture.context.qpos)


def test_audit_refuses_source_change_after_geometry(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.1)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    def interrupted(*args):actual_fixture.file.write_bytes(b'changed');return {},{},np.zeros((1,1))
    monkeypatch.setattr(audit,'_dense_geometry',interrupted)
    with pytest.raises(ValueError,match='changed'):audit.audit_panel_candidate(path)


def test_reference_rate_violation_cannot_hide_in_a_short_aperture_segment(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.091001)
    scene=toy_scene();address=int(scene.m.joint('robot/left_shoulder_pitch').qposadr[0])
    for row,offset in zip(candidate['rows'],np.linspace(0,.001,len(candidate['rows']))):
        row['joint_targets']['left_shoulder_pitch']=float(offset)
        row['qpos'][address]=float(offset)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    receipt,_,_=audit.audit_panel_candidate(path)
    assert not receipt['passed'] and not receipt['reference_phase_envelope']['passed']
    assert receipt['reference_phase_envelope']['maximum_joint_speed_rad_s']>1.2
    assert any('measured_aperture_reference_envelope' in failure for failure in receipt['failures'])


def test_nonfinite_clearance_is_rejected_instead_of_counted_as_safe(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *args:float('nan'))
    with pytest.raises(ValueError,match='Nonfinite hand distance'):audit.audit_panel_candidate(path)


def test_changed_candidate_during_audit_rejects_export(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.1)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    def interrupted(*args):path.write_text('{}');return {},{},np.zeros((1,1))
    monkeypatch.setattr(audit,'_dense_geometry',interrupted)
    with pytest.raises(ValueError,match='changed during screen'):audit.audit_panel_candidate(path)


def test_explicit_lag_slice_remains_static_and_checks_original_joint_bounds(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    receipt,_,_=audit.audit_panel_candidate(path,actual_leaf_lag_rad=.005)
    assert receipt['actual_leaf_lag_rad']==.005 and 'not a guaranteed' in receipt['lag_scope']
    assert receipt['exact_initial_state'] and receipt['authorized_stages']==0
    assert receipt['limits']['joint_violation_increase_rad']==.000001


def test_receiving_only_hand_environment_and_elbow_shapes_remain_in_dense_clearance(actual_fixture,monkeypatch):
    candidate=candidate_without_solves(actual_fixture,monkeypatch,target=.09100001)
    path=actual_fixture.tmp/'candidate.json';path.write_text(json.dumps(candidate))
    def receiving_scene(*args):
        scene=toy_scene();active=(scene.m.geom_contype!=0)|(scene.m.geom_conaffinity!=0)
        scene.m.geom_contype[active]=0;scene.m.geom_conaffinity[active]=1
        return scene
    monkeypatch.setattr(planner,'LandedLeftScene',receiving_scene)
    exact=mujoco.mj_geomDistance;queries=[]
    def too_close(m,d,g,h,*args):
        queries.append((int(g),int(h)))
        assert m.geom_contype[g]==m.geom_contype[h]==0
        assert m.geom_conaffinity[g] and m.geom_conaffinity[h]
        return .001 if m.body(m.geom_bodyid[g]).name=='robot/rh_palm' else exact(m,d,g,h,*args)
    monkeypatch.setattr(mujoco,'mj_geomDistance',too_close)
    receipt,_,_=audit.audit_panel_candidate(path)
    assert queries and receipt['hand_shapes']==1 and receipt['elbow_shapes']==2 and receipt['scene_shapes']==4
    assert not receipt['passed'] and receipt['minimum_all_rh_scene_clearance_capped_m']==.001
    assert any(f.get('right_clearance_m')==.001 for f in receipt['failures'])


def test_receiving_only_shapes_remain_in_constrained_planner(actual_fixture,monkeypatch):
    def receiving_scene(*args):
        # Put the receiving plane inside the conservative broad-phase cap;
        # an exact-query mock must not pretend a distant pair is close.
        scene=toy_scene(floor_height=.985);active=(scene.m.geom_contype!=0)|(scene.m.geom_conaffinity!=0)
        scene.m.geom_contype[active]=0;scene.m.geom_conaffinity[active]=1
        return scene
    monkeypatch.setattr(planner,'LandedLeftScene',receiving_scene)
    seen=[]
    def fit(fun,guess,*,constraints,**kwargs):
        values=constraints[0]['fun'](guess);seen.append(values)
        return SimpleNamespace(x=guess.copy(),fun=fun(guess),status=9,message='Synthetic',nfev=1)
    monkeypatch.setattr(planner,'minimize',fit)
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *args:.001)
    candidate=planner.generate_panel_candidate(actual_fixture.context,{'nodes':21,'max_nfev':1},target_aperture_rad=.1)
    assert len(seen)==20 and all(values[-1]<0 for values in seen)
    assert all(row['right_hand_scene_clearance_capped_m']==.001 for row in candidate['rows'])
    assert not candidate['geometric_admission']
