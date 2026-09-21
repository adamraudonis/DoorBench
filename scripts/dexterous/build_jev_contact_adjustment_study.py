#!/usr/bin/env python3
"""Build archived PhysX contact-history questions, without API calls or stepping.

Only model-inputs.jsonl is an inference payload. Source identities, selection
labels, current audit verdicts and future outcomes live in separate evidence.
No unexecuted correction is assigned a physically successful ground-truth label.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from doorbench.dexterous.json_record_stream import iter_json_object_array

DIGITS=('ff','mf','rf','lf','th')
DT=.002
HISTORY_OFFSETS=(-100,-80,-60,-40,-20,-10,0)
FORECAST_STEPS=50


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def unit(vector):
    value=np.asarray(vector,float);length=np.linalg.norm(value)
    return None if length<1e-10 else (value/length).tolist()


def reduce_row(row,index):
    if index==0:
        if abs(row['sim_time_s'])>1e-8:
            raise ValueError('Initial buffer row must be exactly at reset epoch')
        return dict(index=0,time_s=0.,source_valid_pad_grasp=False,
            source_note='Reset contact buffer, not a solved physical interval; excluded from every case.')
    t=row['sim_time_s'];raw=row['raw_evidence']
    if (abs(t-index*DT)>1e-8 or row['physics_dt_s']!=DT or raw['clock']!='physx-interval-end'
            or any(abs(raw[k]-v)>1e-8 for k,v in [('geometry_time_s',t),('interval_end_s',t),('interval_start_s',max(0.,t-DT))])):
        raise ValueError('Source contact and geometry clocks are not one exact actual interval')
    if row['hand']!='rh' or row['grasp_profile']!='volar-phalange-v1' or raw['scope']!='complete-handle-body':
        raise ValueError('Require declared original RH complete-handle contact evidence')
    center=np.array(raw['lever']['center']);axis=np.array(raw['lever']['axis']);axis=axis/np.linalg.norm(axis);digit_rows={}
    for digit in DIGITS:
        contacts=[c for c in raw['contacts'] if c['body'].rsplit('/',1)[-1].startswith('rh_'+digit)]
        loaded=[c for c in contacts if c['normal_force_N']>1e-6]
        total=sum(c['normal_force_N'] for c in loaded)
        if abs(total-row['digit_forces_N'][digit])>1e-5:raise ValueError('Raw digit force does not match independently audited reduction')
        centroid=radial=normal=centroid_radial=None
        if total>0:
            points=np.array([c['position'] for c in loaded]);weights=np.array([c['normal_force_N'] for c in loaded])
            relative=points-center;radials=relative-np.outer(relative@axis,axis)
            radial_lengths=np.linalg.norm(radials,axis=1)
            if np.any(radial_lengths<1e-8):raise ValueError('Loaded contact at undefined lever radial origin')
            radial=unit(np.sum(radials/radial_lengths[:,None]*weights[:,None],axis=0))
            normal=unit(np.sum(np.array([c['normal'] for c in loaded])*weights[:,None],axis=0))
            centroid=(np.sum(relative*weights[:,None],axis=0)/total).tolist()
            centroid_radial=unit(np.array(centroid)-np.dot(centroid,axis)*axis)
        digit_rows[digit]=dict(normal_load_N=float(total),positive_patch_count=len(loaded),
            zero_force_buffer_patch_count=len(contacts)-len(loaded),
            centroid_relative_to_lever_world_m=centroid,radial_direction_world=radial,
            centroid_radial_direction_world=centroid_radial,
            normal_force_direction_on_hand_world=normal,
            loaded_segments=sorted(set(c['body'].rsplit('/',1)[-1][len('rh_'+digit):] for c in loaded)))
    thumb=digit_rows['th']['radial_direction_world']
    dots={d:None if thumb is None or digit_rows[d]['radial_direction_world'] is None else
        float(np.dot(thumb,digit_rows[d]['radial_direction_world'])) for d in DIGITS[:-1]}
    if row.get('maximum_thumb_finger_dot') is not None and abs(max(dots.values())-row['maximum_thumb_finger_dot'])>1e-8:
        raise ValueError('Raw patch-radial reduction disagrees with independently audited strict score')
    # Keep both different reductions explicit: the historical live diagnostic
    # compares force-centroid directions to their mean, whereas the strict
    # selected profile uses normalized patch-radial averages and every pair.
    centroid_mean_dot=centroid_mean_alignment=None
    if all(digit_rows[d]['normal_load_N']>=.2 and digit_rows[d]['centroid_radial_direction_world'] is not None for d in DIGITS):
        mean=unit(np.sum([digit_rows[d]['centroid_radial_direction_world'] for d in DIGITS[:-1]],axis=0))
        if mean is not None:
            centroid_mean_dot=float(np.dot(digit_rows['th']['centroid_radial_direction_world'],mean))
            centroid_mean_alignment=min(float(np.dot(digit_rows[d]['centroid_radial_direction_world'],mean)) for d in DIGITS[:-1])
    # Raw coordinates are retained separately for inspecting arithmetic/frames.
    raw_loaded=[c for c in raw['contacts'] if c['normal_force_N']>1e-6]
    return dict(index=index,time_s=t,digits=digit_rows,thumb_finger_radial_dot=dots,
        thumb_vs_mean_finger_centroid_radial_dot=centroid_mean_dot,
        minimum_finger_centroid_dot_to_mean=centroid_mean_alignment,
        maximum_thumb_finger_dot=row.get('maximum_thumb_finger_dot'),
        minimum_finger_pair_dot=row.get('minimum_pairwise_finger_alignment'),
        source_valid_pad_grasp=row['valid_pad_grasp'],lever=raw['lever'],
        raw_loaded_contacts=raw_loaded,
        loaded_body_transforms_xyzw={name:raw['body_transforms_xyzw'][name] for name in set(c['body'] for c in raw_loaded)})


def load_run(run,robot_sha256):
    p=run/'trial';audit=json.loads((run/'independent-contact-audit.json').read_text())
    if not audit['accounting_passed'] or not audit['independent_raw_contact_audit_complete'] or not all(audit['checks'].values()):
        raise ValueError('Complete independent contact-accounting audit required')
    if audit['invalid_loaded_patches']!=0:
        raise ValueError('This study reduction requires all loaded raw surfaces to be admissible; never silently drop invalid patches')
    hashes={str(p/name):sha(p/name) for name in audit['input_sha256']}
    if any(hashes[str(p/name)]!=h for name,h in audit['input_sha256'].items()):raise ValueError('Original audit bindings changed')
    if json.loads((p/'motor-contract.json').read_text())['source_xml_sha256']!=robot_sha256:
        raise ValueError('Authored joint geometry is not bound to the original trial motor contract')
    for file in [run/'independent-contact-audit.json',p/'acquisition-physics.npz',p/'motor-contract.json',p/'provenance.json']:
        hashes[str(file)]=sha(file)
    rows=[]
    with gzip.open(p/'acquisition-pad-steps.json.gz','rt',encoding='utf-8') as stream:
        for index,row in enumerate(iter_json_object_array(stream)):
            rows.append(reduce_row(row,index))
    if len(rows)!=audit['physical_intervals']+1:raise ValueError('Missing physical contact intervals')
    with np.load(p/'acquisition-physics.npz',allow_pickle=False) as z:physics={k:z[k].copy() for k in z.files}
    if not np.allclose(physics['time_s'],np.arange(1,len(rows))*DT,atol=1e-10,rtol=0):raise ValueError('Physical and contact state clocks disagree')
    return dict(rows=rows,physics=physics,configuration=json.loads((p/'configuration.json').read_text()),
        report=json.loads((p/'operation-report.json').read_text()),hashes=hashes)


def patch_geometry(contact,body_pose,lever):
    """Exact frame arithmetic only; no contact-classification or expected answer."""
    pose=np.asarray(body_pose,float);rotation=Rotation.from_quat(pose[3:]).as_matrix()
    point=np.asarray(contact['position']);normal=np.asarray(contact['normal'])
    axis=np.asarray(unit(lever['axis']));relative=point-np.asarray(lever['center'])
    axial=float(relative@axis);radial=relative-axial*axis;radial_direction=unit(radial)
    if radial_direction is None:raise ValueError('Loaded patch has undefined lever radial')
    return dict(body=contact['body'].rsplit('/',1)[-1],position_world_m=point.tolist(),
        normal_on_hand_world=normal.tolist(),normal_load_N=contact['normal_force_N'],
        position_body_m=(rotation.T@(point-pose[:3])).tolist(),
        outward_normal_body=(rotation.T@-normal).tolist(),
        radial_direction_world=radial_direction,lever_axis_coordinate_m=axial,
        lever_side_axial_clearance_m=float(lever['half_length']-abs(axial)),
        inward_radial_normal_alignment=float((-normal)@(-np.asarray(radial_direction))))


def select(run):
    """Curated diagnostic pack, selected before querying; not random success rate."""
    rows=run['rows'];q=run['physics']['door'];names=run['configuration']['door_joint_names'];operator=q[:,names.index('leaf_handle_hinge')]
    start=round(run['report']['operation_reference']['operation_start_s']/DT)+1
    drops=[i for i in range(start+50,len(rows)-100) if rows[i-1]['source_valid_pad_grasp'] and not rows[i]['source_valid_pad_grasp']]
    if not drops:raise ValueError('Expected observed contact transition for diagnostic pack')
    drop=drops[0]
    # Five non-identical windows per run: loaded onset, warning, actual loss,
    # maximum lever excursion, and late outcome. Reasons never enter payload.
    selected=[('loaded_press_entry',start),('before_first_grasp_loss',drop-20),
        ('first_grasp_loss',drop),('near_maximum_operator',int(np.argmax(operator))+1),
        ('late_state',len(rows)-1-100)]
    if len(set(i for _,i in selected))!=5:raise ValueError('Case selection collided')
    return selected


def correction_candidates(model,configuration,physics,index):
    names=configuration['robot_joint_names'];current=physics['joints'][index-1]
    result={}
    for joint in ('rh_WRJ1','rh_WRJ2','rh_THJ5'):
        jid=model.joint(joint).id;q=float(current[names.index(joint)]);low,high=model.jnt_range[jid]
        for sign in (-1,1):
            delta=sign*.005;key=joint+('_minus' if sign<0 else '_plus')
            result[key]=dict(joint=joint,reference_delta_rad=delta,measured_joint_angle_rad=q,
                authored_joint_range_rad=[float(low),float(high)],joint_local_axis=model.jnt_axis[jid].tolist(),
                within_original_joint_range=bool(low<=q+delta<=high),
                interpolation_duration_s=.2,
                counterfactual_contact_force_N=None,counterfactual_opposition_score=None)
    result['hold_and_observe']=dict(reference_delta_rad=0.,description='Keep reference unchanged while original balance/grip feedback continues.')
    result['ask_astra_for_recovery']=dict(reference_delta_rad=0.,description='Ask for a new primitive or evidence; no angle adjustment.')
    return result


def material_point_sensitivity(model,configuration,physics,index,now,candidates):
    """Unstepped FK of measured material points; never predicts solved contact."""
    if configuration.get('root_state_convention')!='actor-origin pose and world actor-origin linear/angular velocity':
        raise ValueError('Explicit measured actor-origin root convention required')
    names=configuration['robot_joint_names']
    if len(names)!=69 or set(names)!={model.joint(i).name for i in range(model.njnt) if model.jnt_type[i]!=mujoco.mjtJoint.mjJNT_FREE}:
        raise ValueError('Original complete measured joint inventory required')
    d=mujoco.MjData(model);root=np.array(physics['root'][index-1,:7],float)
    if abs(np.linalg.norm(root[3:])-1)>1e-6:raise ValueError('Measured root quaternion is not normalized within numerical precision')
    root[3:]/=np.linalg.norm(root[3:]);d.qpos[:7]=root
    for name,q in zip(names,physics['joints'][index-1]):d.qpos[int(model.joint(name).qposadr[0])]=q
    mujoco.mj_kinematics(model,d);base_q=d.qpos.copy();patches=[];errors=[]
    for contact in now['raw_loaded_contacts']:
        short=contact['body'].rsplit('/',1)[-1];digit=short[3:5]
        if digit not in DIGITS:raise ValueError('Only actual identified digit material points supported')
        body=model.body(short).id;pose=np.asarray(now['loaded_body_transforms_xyzw'][contact['body']])
        actual_rotation=Rotation.from_quat(pose[3:]).as_matrix();rotation=d.xmat[body].reshape(3,3)
        local=actual_rotation.T@(np.array(contact['position'])-pose[:3])
        predicted=d.xpos[body]+rotation@local
        position_error=float(np.linalg.norm(predicted-contact['position']))
        angle_error=float(Rotation.from_matrix(rotation@actual_rotation.T).magnitude())
        errors.append((position_error,angle_error))
        if position_error>2e-6 or angle_error>2e-6:raise ValueError('Original-model FK does not reproduce actual same-epoch hand bodies')
        patches.append(dict(body=body,digit=digit,local=local,outward=actual_rotation.T@-np.asarray(contact['normal']),
            base_fk=predicted.copy(),weight=contact['normal_force_N']))
    center=np.asarray(now['lever']['center']);axis=np.asarray(unit(now['lever']['axis']))
    result={}
    for key,candidate in candidates.items():
        if 'joint' not in candidate:continue
        d.qpos[:]=base_q;d.qpos[int(model.joint(candidate['joint']).qposadr[0])]+=candidate['reference_delta_rad']
        mujoco.mj_kinematics(model,d)
        vectors={digit:np.zeros(3) for digit in DIGITS};weights={digit:0. for digit in DIGITS}
        shifts={digit:np.zeros(3) for digit in DIGITS};clearances=[];axial_clearances=[];alignments=[]
        for patch in patches:
            rotation=d.xmat[patch['body']].reshape(3,3);point=d.xpos[patch['body']]+rotation@patch['local']
            relative=point-center;axial=float(relative@axis);radial=relative-axial*axis
            length=float(np.linalg.norm(radial));direction=np.asarray(unit(radial))
            digit=patch['digit'];weight=patch['weight'];vectors[digit]+=weight*direction;weights[digit]+=weight
            shifts[digit]+=weight*(point-patch['base_fk'])
            clearances.append(length-now['lever']['radius']);axial_clearances.append(now['lever']['half_length']-abs(axial))
            alignments.append(float((rotation@patch['outward'])@-direction))
        directions={digit:unit(v) for digit,v in vectors.items()};thumb=directions['th']
        dots={digit:None if thumb is None or directions[digit] is None else float(np.dot(thumb,directions[digit])) for digit in DIGITS[:-1]}
        result[key]=dict(frozen_patch_thumb_finger_radial_dot=dots,
            frozen_patch_centroid_shift_world_m={digit:None if weights[digit]==0 else (shifts[digit]/weights[digit]).tolist() for digit in DIGITS},
            minimum_material_point_radial_clearance_m=min(clearances),maximum_material_point_radial_clearance_m=max(clearances),
            minimum_material_point_axial_clearance_m=min(axial_clearances),minimum_material_outward_inward_radial_alignment=min(alignments))
    d.qpos[:]=base_q
    return dict(scope='Zero-step original-model forward kinematics of the same recorded hand material points. The probe adds each candidate delta to the MEASURED JOINT COORDINATE, not to an archived motor reference. The proposed reference change need not produce that coordinate change. Contact identity, load weights, other joint angles, root and lever are frozen. Points may move inside the lever. These numbers are NOT predicted contact forces, collision admission, grasp qualification or evidence a correction succeeds.',
        material_point_reconstruction_max_error_m=max(v[0] for v in errors),
        measured_body_rotation_reconstruction_max_error_rad=max(v[1] for v in errors),
        candidates=result)


def questions():
    q={}
    for d in DIGITS:
        q['loaded_'+d]=dict(type='noul',instructions=f'At the last observed epoch, is digit {d} carrying at least 0.2 N of actual normal load against the identified handle? Buffer patches with zero force do not establish load.',
            criteria={'true':'Actual current normal load meets the specified minimum.','false':'Current normal load is below the specified minimum or the measurement is unavailable.'})
    q['all_thumb_pairs_opposed']=dict(type='noul',instructions='At the current epoch, is the thumb radial dot below -0.5 for EACH of the four individual finger directions? Inspect thumb_finger_radial_dot, not just thumb_vs_mean_finger_centroid_radial_dot. This asks only about current pair geometry, not load-floor or surface qualification or future success; any missing pair direction is insufficient evidence.',
        criteria={'true':'All four individual current thumb/finger radial dots are available and strictly below -0.5.','false':'At least one individual pair is missing or is -0.5 or greater.'})
    q['next_100ms_hold']=dict(type='noul',instructions='Based only on the observed 200 ms history, will all five digits maintain the stated loaded, anatomically admissible opposed grasp for every interval in the next 100 ms if the archived local controller continues? This is a forecast, not an assertion about unseen data. Consider contact variability, force direction, operator motion and opposition; no correction candidate has yet been applied.',
        criteria={'true':'Predict uninterrupted five-digit opposed loaded grasp in the next 100 ms.','false':'Predict at least one interval without that grasp.'})
    q['inspection_priority']=dict(type='choice',instructions='Which aspect most needs inspection before further lever progress, considering the short history and current patch geometry? This is advisory diagnosis, not permission to act.',criteria={
        'transient_digit_load':'Investigate inconsistent or missing force on one or more digits.',
        'thumb_finger_geometry':'Investigate radial orientation/opposition while contact force remains present.',
        'load_magnitude':'Investigate excessive or strongly uneven contact loading.',
        'observe_without_adjustment':'The observations favor maintaining the present bounded primitive and monitoring.',
        'insufficient_evidence':'The supplied evidence does not support a specific diagnosis.'})
    q['correction_candidate']=dict(type='choice',instructions='Choose one bounded candidate for a later independent geometric and motor trial, or choose hold/recovery when a direction is not justified. Do not claim an unexecuted correction will work. Joint axes are local to the original joint frame. The provided frozen-material forward kinematics estimates spatial sign effects only; moving a point inside the lever or improving a radial dot is not solved contact/force evidence. Any execution requires fresh same-epoch local joint/contact/collision/balance admission. This answer is a research proposal, never a control command.',
        criteria={k:(f'Prospectively test {joint} reference {sign}0.005 rad over 0.2 seconds; retain original motor caps and independent collision/contact checks.')
            for joint in ('rh_WRJ1','rh_WRJ2','rh_THJ5') for sign,k in [('-',joint+'_minus'),('+',joint+'_plus')]})
    q['correction_candidate']['criteria'].update(hold_and_observe='Keep angle reference unchanged and observe; no directional correction is justified now.',ask_astra_for_recovery='Request another primitive or additional geometric evidence; do not invent a correction direction.')
    return q


FORBIDDEN_KEYS={'source_valid_pad_grasp','valid_pad_grasp','passed','expected','label','future','selection_reason','source_run','source_row_index','maximum_thumb_finger_dot'}


def assert_payload_boundary(payload):
    def walk(value):
        if isinstance(value,dict):
            if FORBIDDEN_KEYS&set(value):raise ValueError('Evaluator/source answer leaked into model state')
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(payload)
    times=[frame['relative_time_s'] for frame in payload['state']['history']]
    if max(times)>0 or min(times)<-.200000001 or times[-1]!=0:raise ValueError('History contains future or wrong observation window')


def make_case(data,index,model):
    history=[];rows=data['rows'];now=rows[index];physics=data['physics'];configuration=data['configuration']
    if index<101 or index+FORECAST_STEPS>=len(rows):raise ValueError('Case must have complete solved history and withheld future')
    for offset in HISTORY_OFFSETS:
        row=rows[index+offset];i=index+offset-1
        history.append(dict(relative_time_s=offset*DT,
            digit_normal_loads_N={d:round(row['digits'][d]['normal_load_N'],5) for d in DIGITS},
            thumb_finger_radial_dot={d:None if v is None else round(v,6) for d,v in row['thumb_finger_radial_dot'].items()},
            thumb_vs_mean_finger_centroid_radial_dot=row['thumb_vs_mean_finger_centroid_radial_dot'],
            minimum_finger_centroid_dot_to_mean=row['minimum_finger_centroid_dot_to_mean'],
            operator_angle_rad=float(physics['door'][i,configuration['door_joint_names'].index('leaf_handle_hinge')]),
            operator_velocity_rad_s=float(physics['door_velocity'][i,configuration['door_joint_names'].index('leaf_handle_hinge')]),
            torso_tilt_deg=float(physics['torso_tilt_deg'][i])))
    recent=rows[index-99:index+1]
    # Arithmetic summaries are measurements, not classifications or decisions.
    variability={d:dict(minimum_N=min(r['digits'][d]['normal_load_N'] for r in recent),
        maximum_N=max(r['digits'][d]['normal_load_N'] for r in recent),
        standard_deviation_N=float(np.std([r['digits'][d]['normal_load_N'] for r in recent]))) for d in DIGITS}
    candidates=correction_candidates(model,configuration,physics,index)
    sensitivity=material_point_sensitivity(model,configuration,physics,index,now,candidates)
    payload=dict(model='jev-1.13.0',state=dict(
        scope='Archived object-identified PhysX contact telemetry, not vision or tactile-only perception. Forces act on the hand; positions, directions and opposition arithmetic are measured/computed in code. No live control occurs.',
        plan=dict(objective='Maintain a gentle opposed grasp while the existing original-motor lever/opening primitive continues; select only a bounded future diagnostic candidate when evidence supports it.',
            monitored_hand='right',digit_names=dict(ff='index',mf='middle',rf='ring',lf='little',th='thumb'),
            measured_force_floor_N=.2,excessive_load_watch_N=15.,opposition_convention='More negative thumb/finger radial dot means more opposed; original requirement is below -0.5 for every thumb/finger pair, finger-pair dot above 0.5. Surface admissibility also matters.',
            radial_reductions='thumb_finger_radial_dot uses each digit force-weighted unit patch radial, then normalizes. thumb_vs_mean_finger_centroid_radial_dot first projects each force-weighted contact centroid and normalizes, then compares the thumb with the normalized mean of four finger directions. These are distinct reductions; averaging can conceal an individual pair.',
            authored_surface_bands=dict(finger_segments=['proximal','middle','distal'],thumb_segments=['distal'],
                local_y_below_m=-.001,local_z_lower_m=.002,local_z_upper_m=dict(proximal=.045,middle=.025,distal=.040),
                outward_minus_local_y_above=.5,lever_side_axial_clearance_at_least_m=.001,inward_radial_normal_alignment_above=.8,
                maximum_misplaced_loaded_force_fraction=.05),
            maximum_candidate_reference_step_rad=.005),
        history=history,last_200ms_load_statistics=variability,
        current_digit_geometry={d:now['digits'][d] for d in DIGITS},
        current_lever_geometry_world=now['lever'],
        current_loaded_patches=[patch_geometry(c,now['loaded_body_transforms_xyzw'][c['body']],now['lever']) for c in now['raw_loaded_contacts']],
        correction_candidates=candidates,
        unstepped_material_point_sensitivity=sensitivity,
        unavailable=['measured tangential slip velocity','camera/tactile perception','dynamic response to unexecuted candidate','future control/contact measurements']),questions=questions())
    assert_payload_boundary(payload)
    following=rows[index+1:index+FORECAST_STEPS+1]
    valid=[r['source_valid_pad_grasp'] for r in following]
    label=dict(current_loaded={d:now['digits'][d]['normal_load_N']>=.2 for d in DIGITS},
        current_all_thumb_pairs_opposed=all(v is not None and v<-.5 for v in now['thumb_finger_radial_dot'].values()),
        current_recorded_opposed_grasp=now['source_valid_pad_grasp'],next_100ms_all_intervals_opposed_grasp=all(valid),
        current_legacy_mean_geometry_pass=bool(now['thumb_vs_mean_finger_centroid_radial_dot'] is not None and now['thumb_vs_mean_finger_centroid_radial_dot']<-.5 and now['minimum_finger_centroid_dot_to_mean']>.5),
        current_worst_thumb_finger_dot=max((v for v in now['thumb_finger_radial_dot'].values() if v is not None),default=None),
        future_failed_intervals=sum(not v for v in valid),
        first_future_failure_offset_s=next(((i+1)*DT for i,v in enumerate(valid) if not v),None),
        inspection_priority_expected=None,correction_candidate_expected=None,
        correction_label_limitation='No candidate was executed at this archived state. Admissibility is scoreable; causal success and a unique best angle are not.',
        comparator_persistence_prediction=now['source_valid_pad_grasp'],
        comparator_recent_all_valid_prediction=all(r['source_valid_pad_grasp'] for r in recent),
        admissible_joint_range_candidates=[k for k,v in candidates.items() if v.get('within_original_joint_range',True)],
        history_rows=[index+offset for offset in HISTORY_OFFSETS],history_dense_rows=[index-99,index],future_rows=[index+1,index+FORECAST_STEPS])
    evidence=dict(observed_rows=[rows[index+offset] for offset in HISTORY_OFFSETS],
        future_outcome_rows=[dict(index=r['index'],time_s=r['time_s'],source_valid_pad_grasp=r['source_valid_pad_grasp'],
            digit_loads_N={d:r['digits'][d]['normal_load_N'] for d in DIGITS},thumb_finger_radial_dot=r['thumb_finger_radial_dot']) for r in following])
    return payload,label,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Use a new study directory')
    a.output.mkdir(parents=True);(a.output/'evaluator-only').mkdir()
    robot=ROOT/'out/local-ready/h1-shadow-loopback-v2.xml';model=mujoco.MjModel.from_xml_path(str(robot))
    inputs={str(Path(__file__).resolve()):sha(__file__),str(robot):sha(robot)};allcases=[];sources={}
    for name in ('local-isaac-operation-002','local-isaac-jev-001'):
        run=ROOT/'out'/name;data=load_run(run,sha(robot));sources[name]=dict(task_passed=data['report']['passed'],contact_accounting_passed=True)
        inputs.update(data['hashes'])
        for reason,index in select(data):
            payload,label,evidence=make_case(data,index,model)
            allcases.append(dict(payload=payload,label=label,evidence=evidence,source=run,index=index,reason=reason))
        print(json.dumps(dict(source=name,rows=len(data['rows']),selected=[(reason,index*DT) for reason,index in select(data)])),flush=True)
    random.Random(2071).shuffle(allcases)
    labels=[];ledger=[]
    with (a.output/'model-inputs.jsonl').open('w',encoding='utf-8') as stream:
        for i,item in enumerate(allcases):
            case_id=f'case-{i+1:02d}';payload=item['payload']
            stream.write(json.dumps(dict(case_id=case_id,request=payload),allow_nan=False)+'\n')
            evidence_path=a.output/'evaluator-only'/(case_id+'-evidence.json');write(evidence_path,item['evidence'])
            labels.append(dict(case_id=case_id,**item['label'],source_run=str(item['source']),source_row_index=item['index'],
                selection_reason=item['reason'],source_epoch_s=item['index']*DT,evidence_sha256=sha(evidence_path),
                request_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()))
            ledger.append(dict(case_id=case_id,payload_bytes=len(json.dumps(payload)),history_samples=len(HISTORY_OFFSETS)))
    write(a.output/'evaluator-only/labels.json',labels)
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('Source bytes changed during study construction')
    write(a.output/'manifest.json',dict(schema='doorbench.jev-contact-adjustment-study.v1',case_count=len(allcases),
        model='jev-1.13.0',history_seconds=.2,forecast_seconds=.1,physics_steps=0,api_calls=0,
        source_task_results=sources,input_sha256=inputs,cases=ledger,
        model_inputs_sha256=sha(a.output/'model-inputs.jsonl'),evaluator_labels_sha256=sha(a.output/'evaluator-only/labels.json'),
        disclosure='Model payloads exclude source identities, selection reasons, recorded grasp validity and all future outcomes. This is a small curated diagnostic set, not independent trials or a learned-policy benchmark.',
        official_docs=['https://docs.typesafe.ai/api','https://docs.typesafe.ai/concepts/state','https://docs.typesafe.ai/model-jaggedness/jev-1.13','https://docs.typesafe.ai/models']))
    print(json.dumps(dict(cases=len(labels),future_hold_labels=collections.Counter(str(r['next_100ms_all_intervals_opposed_grasp']) for r in labels),output=str(a.output))))


if __name__=='__main__':main()
