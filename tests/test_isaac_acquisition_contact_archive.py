import copy,gzip,json
import numpy as np
from scipy.spatial.transform import Rotation
from doorbench.dexterous.isaac_pad_audit import shadow_physx_pad_grasp
from scripts.dexterous.audit_isaac_acquisition_contacts import audit


def record(tmp_path):
    patches=[];poses={}
    for digit,axial in zip(('ff','mf','rf','lf','th'),(-.03,-.01,.01,.03,0.)):
        thumb=digit=='th';point=np.array([axial,.007 if thumb else -.007,0.])
        rotation=Rotation.identity() if thumb else Rotation.from_euler('z',180,degrees=True)
        path='/World/H1/pelvis/rh_'+digit+'distal'
        poses[path]=np.r_[point-rotation.apply([0.,-.008,.020]),rotation.as_quat()].tolist()
        patches.append(dict(body=path,position=point.tolist(),normal=[0.,1. if thumb else -1.,0.],normal_force_N=1.))
    row=shadow_physx_pad_grasp(patches,poses,[0,0,0],[1,0,0],half_length=.053,radius=.007)
    row['grasp_profile']='distal-pad-v1';row['sim_time_s']=0.
    rows=[row]
    for i in range(1,252):
        s=copy.deepcopy(row);s['sim_time_s']=i*.002
        s['raw_evidence']=dict(schema='doorbench.shadow-raw-pad-evidence.v1',interval_start_s=(i-1)*.002,interval_end_s=i*.002,geometry_time_s=i*.002,contacts=copy.deepcopy(patches),body_transforms_xyzw=copy.deepcopy(poses),handle_pair_forces_world_N={c['body']:c['normal'] for c in patches},lever=dict(center=[0,0,0],axis=[1,0,0],half_length=.053,radius=.007))
        rows.append(s)
    for n,obj in [('acquisition-report.json',dict(passed=True,physics_dt_s=.002,duration_s=.502,checks=dict(sustained_pad_grasp=True))),('configuration.json',{}),('provenance.json',{})]:
        (tmp_path/n).write_text(json.dumps(obj))
    save(tmp_path,rows);return rows


def save(path,rows):
    with gzip.open(path/'acquisition-pad-steps.json.gz','wt') as f:json.dump(rows,f)


def test_raw_archive_detects_stale_frames_even_when_derived_labels_pass(tmp_path):
    rows=record(tmp_path);r=audit(tmp_path)
    assert r['independent_raw_contact_audit_complete']
    raw=rows[-1]['raw_evidence'];path=raw['contacts'][0]['body']
    raw['body_transforms_xyzw'][path][1]+=.02
    save(tmp_path,rows);r=audit(tmp_path)
    assert not r['independent_raw_contact_audit_complete']
    assert r['task_passed']  # Preserve what the original runtime reported.
    assert not r['checks']['raw_reductions_match']


def test_old_archive_is_never_promoted_to_raw_verification(tmp_path):
    rows=record(tmp_path)
    for row in rows:row.pop('raw_evidence',None)
    save(tmp_path,rows);r=audit(tmp_path)
    assert r['accounting_passed']
    assert not r['independent_raw_contact_audit_complete']


def test_raw_contact_epoch_and_pair_force_tampering_are_rejected(tmp_path):
    rows=record(tmp_path)
    rows[-1]['raw_evidence']['geometry_time_s']-=.002
    rows[-1]['raw_evidence']['contacts'][0]['normal_force_N']=2.
    save(tmp_path,rows);r=audit(tmp_path)
    assert not r['checks']['raw_contact_clocks_match']
    assert not r['checks']['raw_patch_loads_match_pair_forces']
