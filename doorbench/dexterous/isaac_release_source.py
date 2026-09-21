"""Admit actual Isaac palm-supported rest for a new unstepped release plan.

This is an explicit PhysX source adapter. It never creates a native rollout
manifest, changes a plant state, or qualifies a release from geometry alone.
"""
from collections import deque
import copy
import gzip
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .destination_planner_admission import admit_destination_planner
from .grasp_verification import grasp_profile
from .json_record_stream import iter_json_object_array
from .isaac_pad_audit import shadow_physx_pad_grasp
from .landed_left_planner import LandedLeftScene
from .qualified_isaac_grasp import digest


def _finite(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value)


def _epoch(value,expected):
    return _finite(value) and abs(value-expected)<=1e-8


def _track(hashes,path):
    path=Path(path).resolve();actual=digest(path);prior=hashes.get(str(path))
    if prior is not None and prior!=actual:raise ValueError('Source evidence changed during release admission')
    hashes[str(path)]=actual
    return actual


def _same_contact_diagnostics(measured, recorded):
    """Preserve exact raw fields and labels; allow derived scalar roundoff only.

    NumPy/SciPy builds can differ in the final floating-point bit of these two
    recomputed diagnostics. Their physical thresholds and qualification labels
    are independently recomputed before this equality check and remain exact.
    """
    derived={'inward_radial_normal_alignment','axial_clearance_m'}
    if not isinstance(recorded,list) or len(measured)!=len(recorded):return False
    for actual,prior in zip(measured,recorded):
        if not isinstance(prior,dict) or set(actual)!=set(prior):return False
        for key,value in actual.items():
            if key in derived:
                if not (_finite(value) and _finite(prior[key]) and abs(value-prior[key])<=1e-12):return False
            elif isinstance(value,(bool,np.bool_)) and type(prior[key]) is not bool:return False
            elif value!=prior[key]:return False
    return True


def _actual_pad(row,t,profile):
    raw=row.get('raw_evidence',{})
    if (not _epoch(row.get('sim_time_s'),t) or not _epoch(row.get('physics_dt_s'),.002)
            or row.get('hand')!='rh' or row.get('grasp_profile')!=profile
            or row.get('valid_pad_grasp') is not True
            or raw.get('schema')!='doorbench.shadow-raw-pad-evidence.v1'
            or raw.get('clock')!='physx-interval-end' or raw.get('scope')!='complete-handle-body'
            or any(not _epoch(raw.get(k),v) for k,v in
                   [('interval_start_s',t-.002),('interval_end_s',t),('geometry_time_s',t)])):
        raise ValueError('Complete same-epoch selected raw RH grasp required')
    capacity,count=raw.get('contact_capacity'),raw.get('active_contact_count')
    if (type(capacity) is not int or type(count) is not int or not 0<=count<capacity
            or capacity!=row.get('contact_capacity') or count!=row.get('active_contact_count')):
        raise ValueError('Complete nontruncated contact buffer required')
    lever=raw['lever']
    measured=shadow_physx_pad_grasp(raw['contacts'],raw['body_transforms_xyzw'],lever['center'],lever['axis'],
        half_length=lever['half_length'],radius=lever['radius'],profile=profile)
    if (measured['valid_pad_grasp'] is not True or measured['non_digit_handle_force_N']!=0.
            or any(c['pad_qualified'] is not True for c in measured['contacts'])
            or not _same_contact_diagnostics(measured['contacts'],row.get('contacts'))):
        raise ValueError('Actual raw grasp/material patches disagree with qualified source')
    pairs=raw['handle_pair_forces_world_N']
    if any(c['body'] not in pairs for c in raw['contacts']):
        raise ValueError('Every recorded handle patch needs counterpart force accounting')
    for body,force in pairs.items():
        wanted=np.asarray(force,float)
        actual=sum((np.asarray(c['normal'])*c['normal_force_N'] for c in raw['contacts'] if c['body']==body),start=np.zeros(3))
        if wanted.shape!=(3,) or not np.isfinite(wanted).all() or np.linalg.norm(actual-wanted)>.001:
            raise ValueError('Actual handle pair-force accounting differs')
    for key in ('digit_forces_N','qualified_pad_forces_N'):
        declared=row.get(key,{})
        if set(declared)!=set(measured[key]) or any(not _finite(declared[d]) or abs(declared[d]-v)>1e-8 for d,v in measured[key].items()):
            raise ValueError('Actual selected digit loads disagree with source record')
    return measured


def _actual_palm(row,t):
    if not _epoch(row.get('time_s'),t):raise ValueError('Matched actual palm epoch required')
    pose=np.asarray(row['leaf_pose'],float)
    if pose.shape!=(7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1.)>2e-6:
        raise ValueError('Actual normalized leaf body pose required')
    normal=-Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()[:,1]
    surface=row['surface'];forces=surface['body_panel_forces_world_N']
    if 'lh_palm' not in forces:raise ValueError('Actual palm-only force vector required')
    loads={}
    for body,force in forces.items():
        vector=np.asarray(force,float)
        if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Finite actual panel force vector required')
        loads[body]=max(0.,float(normal@vector))
    reported=surface['palm_normal_load_N']
    if not _finite(reported) or abs(reported-loads['lh_palm'])>1e-8:
        raise ValueError('Actual palm-only force accounting differs')
    if loads['lh_palm']<2.:raise ValueError('Actual palm-only support must remain at least 2 N')
    return loads['lh_palm']


def validate_rest_window(times, doors, door_names, pads, surfaces, *, terminal, profile):
    """Join actual 500 Hz records by epoch, never interpolate or fill gaps."""
    times=np.asarray(times);doors=np.asarray(doors)
    grasp_profile(profile)
    expected=('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')
    if (times.ndim!=1 or times.shape!=(251,) or doors.shape!=(251,len(door_names))
            or len(set(door_names))!=len(door_names) or not set(expected)<=set(door_names)
            or not np.isfinite(times).all() or not np.isfinite(doors).all()
            or not _finite(terminal) or terminal<.5 or not _epoch(times[-1],terminal)
            or not np.allclose(np.diff(times),.002,atol=1e-8,rtol=0)
            or len(pads)!=251 or len(surfaces)!=251):
        raise ValueError('Complete actual 251-sample measured-rest window required')
    loads=[]
    for t,door,pad,surface in zip(times,doors,pads,surfaces):
        actual_pad=_actual_pad(pad,t,profile)
        load=_actual_palm(surface,t)
        leaf,operator,latch=[float(door[door_names.index(n)]) for n in expected]
        if not (.075<=leaf<=.10 and abs(operator)<=.05 and abs(latch)<=.001):
            raise ValueError('Actual source must remain within the original mechanism rest bounds')
        loads.append(load)
    contacts=copy.deepcopy(actual_pad['contacts'])
    if any(c.get('pad_qualified') is not True for c in contacts if c['normal_force_N']>0):
        raise ValueError('Every loaded terminal material patch must be qualified')
    if {c['digit'] for c in contacts if c['normal_force_N']>0}!={'ff','mf','rf','lf','th'}:
        raise ValueError('Actual loaded material patches for all five digits required')
    return dict(terminal_time_s=float(terminal),window_samples=251,
        minimum_palm_load_N=float(min(loads)),
        maximum_abs_operator_rad=float(np.max(abs(doors[:,door_names.index(expected[1])]))),
        maximum_abs_latch_m=float(np.max(abs(doors[:,door_names.index(expected[2])]))),
        endpoint_contacts=contacts,
        scope='Actual Isaac spring-rest transfer endpoint; no commanded-return milestone')


def _bind_planning_door(trial,configuration,report,provenance,robot,door_xml,hashes):
    """Use the actual captured transfer route when native_door was not loaded.

    Standalone Isaac runs record the USD, robot XML and transfer route, but not
    door.xml directly. The route binds the original planning XML; a subsequent
    measured-body coordinate admission proves its frame agreement. This is not
    an invented XML-to-USD export certificate or collision-model equivalence.
    """
    route_name=configuration['args'].get('standing_transfer_route')
    if not route_name:raise ValueError('Source-bound actual standing transfer route required')
    route_path=Path(route_name).resolve();captured=trial/'standing-transfer-route.json'
    files={Path(k).resolve():v for k,v in provenance.items()}
    expected=files.get(route_path)
    if (not expected or digest(captured)!=expected or digest(route_path)!=expected
            or Path(report['standing_transfer']['route']).resolve()!=route_path):
        raise ValueError('Original and captured transfer route must match run provenance')
    route=json.loads(captured.read_text())
    if (route.get('schema')!='doorbench.standing-transfer.v1' or route.get('geometric_screen_passed') is not True
            or route.get('robot_xml_sha256')!=digest(robot) or route.get('door_xml_sha256')!=digest(door_xml)):
        raise ValueError('Captured source route must bind the original planning XML assets')
    if door_xml in files and files[door_xml]!=digest(door_xml):
        raise ValueError('Recorded original door XML identity differs')
    for p in (route_path,captured):_track(hashes,p)
    return dict(route_path=str(route_path),captured_route_path=str(captured),route_sha256=expected,
        door_xml_sha256=digest(door_xml),direct_xml_provenance=door_xml in files,
        scope='Run-bound original planning XML via captured transfer route; actual USD and measured body-frame agreement are admitted separately')


def _tail(path):
    with gzip.open(path,'rt') as stream:
        return list(deque(iter_json_object_array(stream),maxlen=251))


def admit_isaac_release_source(source, *, robot, door_xml, door_usd,
                               profile='volar-phalange-v1'):
    from scripts.dexterous.plan_local_isaac_transfer import load_local_source
    from .isaac_prefix_witness import LiveIsaacPrefixWitness
    source=Path(source).resolve();trial=source/'trial'
    robot,door_xml,door_usd=[Path(p).resolve() for p in (robot,door_xml,door_usd)]
    grasp_profile(profile)
    extracted,motors,qualification=load_local_source(source,source/'independent-contact-audit.json')
    report=json.loads((trial/'operation-report.json').read_text())
    configuration=json.loads((trial/'configuration.json').read_text())
    if bool(configuration['args'].get('standing_transfer_stop_on_rest'))!=('standing_transfer_rest_stop' in report):
        raise ValueError('Prospective transfer stop declaration and report disagree')
    rest_stop_inputs={}
    if 'standing_transfer_rest_stop' in report:
        from .isaac_transfer_rest_audit import REST_FILES,validate_transfer_rest_evidence
        stop_audit=json.loads((source/'isaac-transfer-audit.json').read_text())
        validate_transfer_rest_evidence(report,stop_audit,trial)
        for name in REST_FILES:_track(rest_stop_inputs,trial/name)
        _track(rest_stop_inputs,Path(__file__).with_name('isaac_transfer_rest_audit.py'))
    if (not report.get('standing_transfer') or configuration['args'].get('grasp_profile')!=profile
            or report.get('grasp_profile')!=profile):
        raise ValueError('Qualified actual Isaac palm transfer with the selected profile required')
    if digest(robot)!=extracted['binding']['robot_source_sha256'] or digest(door_usd)!=extracted['binding']['door_source_sha256']:
        raise ValueError('Original physical robot and door assets required')
    # Admission only: no samples are fed to this witness and it cannot authorize
    # a stage. It verifies historical captured code without requiring old code
    # to replace today's controller implementation.
    witness=LiveIsaacPrefixWitness(source,expected_source_state_sha256=qualification['state_sha256'],
        stage_start_s=qualification['time_s'],runtime_configuration=configuration,runtime_motor_contract=motors)
    historical=witness.receipt()
    if historical['source_qualification']!=qualification or historical['stage_entry_authorized'] is not False or historical['intervals_verified']!=0:
        raise ValueError('Unchanged source-only qualification without stage authorization required')
    hashes={**historical['input_sha256'],**rest_stop_inputs}
    declaration=trial/'grasp-profile-definition.json'
    spec=json.loads(declaration.read_text())
    original=Path(configuration['args']['grasp_profile_definition'])
    if (spec.get('profile')!=profile or spec.get('robot_xml_sha256')!=digest(robot)
            or digest(original)!=digest(declaration)):
        raise ValueError('Original prospective grasp declaration required')
    recorded_provenance=json.loads((trial/'provenance.json').read_text())['files']
    if {Path(k).resolve():v for k,v in recorded_provenance.items()}.get(original.resolve())!=digest(declaration):
        raise ValueError('Prospective grasp declaration must be bound by the actual run')
    asset_binding=_bind_planning_door(trial,configuration,report,recorded_provenance,robot,door_xml,hashes)
    for p in (robot,door_xml,door_usd,declaration,original):_track(hashes,p)
    with np.load(trial/'acquisition-physics.npz',allow_pickle=False) as data:
        times=data['time_s'][-251:].copy();doors=data['door'][-251:].copy()
    pads=_tail(trial/'acquisition-pad-steps.json.gz')
    rest=validate_rest_window(times,doors,configuration['door_joint_names'],
        pads,_tail(trial/'standing-transfer-steps.json.gz'),
        terminal=qualification['time_s'],profile=profile)
    scene=LandedLeftScene(robot,door_xml)
    measured,coordinates=admit_destination_planner(scene.m,extracted,
        motor_contract=motors,door_source_sha256=digest(door_usd))
    scene.d.qpos[:]=measured.qpos;mujoco.mj_kinematics(scene.m,scene.d)
    maximum_point_error=maximum_body_position_error=maximum_body_rotation_error=0.
    for contact in rest['endpoint_contacts']:
        original_body=contact['body'];name=original_body.rsplit('/',1)[-1]
        match=re.fullmatch(r'rh_(ff|mf|rf|lf|th)(distal|middle|proximal)',name)
        if match is None or match.group(1)!=contact['digit']:
            raise ValueError('Explicit original finger body identity required')
        body=scene.m.body('robot/'+name).id
        point=np.asarray(contact['body_position_m'],float)
        position=np.asarray(contact['position'],float)
        pose=np.asarray(pads[-1]['raw_evidence']['body_transforms_xyzw'][original_body],float)
        if (point.shape!=(3,) or position.shape!=(3,) or pose.shape!=(7,)
                or not np.isfinite(np.r_[point,position,pose]).all() or abs(np.linalg.norm(pose[3:])-1.)>2e-6):
            raise ValueError('Finite normalized actual material/body frames required')
        rotation=scene.d.xmat[body].reshape(3,3)
        world=scene.d.xpos[body]+rotation@point
        point_error=float(np.linalg.norm(world-position))
        body_error=float(np.linalg.norm(scene.d.xpos[body]-pose[:3]))
        angle_error=float(Rotation.from_matrix(rotation@Rotation.from_quat(pose[3:]).as_matrix().T).magnitude())
        if max(point_error,body_error)>2e-6 or angle_error>2e-6:
            raise ValueError('Measured material patch does not match admitted original body frame')
        maximum_point_error=max(maximum_point_error,point_error)
        maximum_body_position_error=max(maximum_body_position_error,body_error)
        maximum_body_rotation_error=max(maximum_body_rotation_error,angle_error)
        contact['isaac_body_path']=original_body
        contact['body']='robot/'+name
    for module in (__file__,Path(__file__).with_name('isaac_pad_audit.py'),Path(__file__).with_name('grasp_verification.py')):
        _track(hashes,module)
    for name,sha in hashes.items():
        if digest(name)!=sha:raise ValueError('Actual source changed during release admission')
    return dict(schema='doorbench.isaac-release-source.v1',source_engine='isaac-physx',
        source_run=str(source),grasp_profile=profile,contact_audit_name='independent-contact-audit.json',
        source_qualification=qualification,coordinate_admission=coordinates,
        planning_asset_binding=asset_binding,
        material_frame_admission=dict(passed=True,maximum_position_error_m=maximum_point_error,
            maximum_body_position_error_m=maximum_body_position_error,maximum_body_rotation_error_rad=maximum_body_rotation_error,
            position_limit_m=2e-6,rotation_limit_rad=2e-6),
        measured_rest=rest,initial_qpos=measured.qpos.tolist(),
        robot_path=str(robot),door_xml_path=str(door_xml),door_usd_path=str(door_usd),
        input_sha256=hashes,physics_steps=0,source_sample_playback=0,authorized_stages=0,
        scope='Qualified recorded PhysX endpoint admitted to an unstepped original-model planning scene. No native rollout or release success is synthesized.')
