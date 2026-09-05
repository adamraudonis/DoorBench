#!/usr/bin/env python3
"""Export a read-only corpus snapshot for Motion Lab; only accepted clips play.

Writes content-addressed gzip JSON and audit files, then atomically replaces the
web index. No source inputs, native trajectories, or frozen generator files change.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np

SCHEMA='doorbench.planned-reference-web.v1'
INDEX_SCHEMA='doorbench.planned-reference-web-index.v1'
SOURCE_SCENARIOS={'open_and_traverse','unlock_and_traverse','locked_recognize'}
SCOPE='Accepted clips passed independent sampled kinematic and task-evidence checks. Task evidence uses the actor route and declared source outcome; mechanism semantics are not independently certified. Playback interpolation is illustrative; no dynamics, balance, causal control or personal visual approval is certified.'

def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
def read(path):return json.loads(Path(path).read_bytes())
def require(ok,message):
    if not ok:raise ValueError(message)

def checked_path(root,relative):
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==relative,'Unsafe corpus artifact path')
    path=root/p
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),'Artifact escapes corpus or is a symlink')
    return path

def put(out,relative,data):
    path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==data,f'Content-addressed artifact changed: {relative}');return relative
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(data);temporary=f.name
    os.replace(temporary,path);return relative

def audit(out,door_id,name,data):
    digest=sha(data);relative=f'audits/{door_id}/{Path(name).stem}.{digest}.json'
    put(out,relative,data);return {'path':relative,'sha256':digest,'bytes':len(data)}

def public_text(value):
    """Keep useful first-line errors, never local absolute paths or tracebacks."""
    if not isinstance(value,str):return value
    return re.sub(r'(?<![\w])(?:[A-Za-z]:[\\/]|/)[^\s\"\'<>]+','[local path]',value.splitlines()[0] if value else '')

def public_result(result,original):
    keys=['door_id','family','status','identity_sha256','created_at','run_id','source_outcome',
          'provenance','generator','new_completion','failure_counts','reason_code','runtime_s','scope']
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ['traceback','pid','command','environment']}
        if isinstance(value,list):return [clean(v) for v in value]
        return public_text(value)
    projected={key:clean(result[key]) for key in keys if key in result}
    error=result.get('error')
    if isinstance(error,dict):projected['error']={key:public_text(error[key]) for key in ['type','message'] if key in error}
    elif error:projected['error']=public_text(error)
    return {'schema':'doorbench.planned-reference-public-attempt.v1','projection':'Public summary; execution logs and traceback omitted, absolute paths redacted. Original result is identified by SHA-256.',
            'original_result_sha256':sha(original),**projected}

def verified_artifacts(directory,result):
    """Match the runner's closed per-attempt inventory, including failure files."""
    artifacts=result.get('artifacts');require(isinstance(artifacts,dict),'Missing attempt artifact inventory')
    paths=list(directory.rglob('*'));require(not any(p.is_symlink() for p in paths),'Attempt contains a symlink')
    actual={p.relative_to(directory).as_posix() for p in paths if p.is_file() and p!=directory/'result.json'}
    require(actual==set(artifacts),'Attempt artifact inventory changed (including stale failure files)')
    files={}
    for name,digest in artifacts.items():
        require(isinstance(name,str) and Path(name).name==name and name!='result.json','Unsafe attempt artifact name')
        data=checked_path(directory,name).read_bytes()
        require(isinstance(digest,str) and re.fullmatch(r'[a-f0-9]{64}',digest) is not None and sha(data)==digest,f'{result["door_id"]}: stale {name} artifact')
        files[name]=data
    return files

def pose_frames(arrays,prefix,n,names):
    pos=np.asarray(arrays[prefix+'pos']);quat=np.asarray(arrays[prefix+'quat'])
    require(pos.shape==(n,len(names),3) and quat.shape==(n,len(names),4),f'{prefix} pose shape mismatch')
    require(np.isfinite(pos).all() and np.isfinite(quat).all(),f'{prefix} nonfinite poses')
    require(np.all(np.abs(np.linalg.norm(quat,axis=-1)-1)<1e-5),f'{prefix} nonunit quaternion')
    return np.concatenate([pos,quat],axis=-1).reshape(n,-1).tolist()

def export_accepted(directory,result,source_assets,out,files):
    door_id=result['door_id']
    require('failure.json' not in files,'Accepted attempt contains failure.json')
    require({'clip.json','trajectory.npz','validation.json'}<=files.keys(),'Accepted attempt lacks required artifacts')
    clip=json.loads(files['clip.json']);validation=json.loads(files['validation.json'])
    require(clip.get('door_id')==door_id and validation.get('door_id')==door_id,'Mismatched door identity')
    require(clip.get('schema')=='doorbench.planned-reference.v1' and clip.get('complete_proposal') is True,'Incomplete/unsupported planned clip')
    require(validation.get('schema')=='doorbench.planned-reference-validation.v1' and validation.get('accepted') is True and validation.get('kinematic_accepted') is True and validation.get('status')=='accepted_kinematic' and not validation.get('failure_counts'),'Independent validation did not accept clip')
    require(validation.get('task_completion',{}).get('evidence_pass') is True and validation.get('task_completion',{}).get('complete_proposal') is True,'Task-evidence checks did not pass')
    require(validation.get('clip_sha256')==sha(files['clip.json']) and validation.get('trajectory_sha256')==sha(files['trajectory.npz']),'Validation is bound to different artifacts')
    require(clip.get('trajectory_sha256')==sha(files['trajectory.npz']),'Clip trajectory binding mismatch')
    outcome=result.get('source_outcome',{})
    require(outcome.get('success') is True and outcome.get('outcome')=='success' and not outcome.get('error') and clip.get('proposal',{}).get('source_outcome')==outcome,'Source success/proposal outcome binding mismatch')
    scenario=outcome.get('scenario')
    require(scenario in SOURCE_SCENARIOS and clip.get('proposal',{}).get('scenario')==scenario,'Missing or inconsistent bound source scenario')
    sources=result['provenance']['source_sha256']
    require(set(sources)=={'model.json','spec.json','door.xml'} and clip.get('source_sha256')==sources and validation.get('source_sha256')==sources,'Source bindings disagree')
    for name,digest in sources.items():require(sha((source_assets/'doors'/door_id/name).read_bytes())==digest,f'{door_id}: stale source {name}')
    for name,digest in result['provenance'].get('native_resources_sha256',{}).items():
        require(sha(checked_path(source_assets,name).read_bytes())==digest,f'{door_id}: stale native resource {name}')
    names=clip['actor']['body_names'];native_names=clip['native']['body_names']
    require(len(names)==16 and len(set(names))==16 and all(isinstance(x,str) for x in names),'Actor must declare 16 unique bodies')
    require(len(native_names)==len(set(native_names)) and all(isinstance(x,str) for x in native_names),'Invalid native body names')
    geometries=clip['actor']['geometries'];seen=set()
    for geom in geometries:
        require(geom['name'] not in seen and geom['body_name'] in names,'Duplicate geometry or missing actor body');seen.add(geom['name'])
        require(geom['type'] in ['box','sphere','capsule','cylinder'],'Unsupported actor primitive')
        require(np.shape(geom['size'])==(3,) and np.shape(geom['pos'])==(3,) and np.shape(geom['quat_wxyz'])==(4,),'Invalid actor primitive dimensions')
        require(np.isfinite([*geom['size'],*geom['pos'],*geom['quat_wxyz']]).all() and abs(np.linalg.norm(geom['quat_wxyz'])-1)<1e-5,'Invalid actor primitive transform')
        require(geom['size'][0]>0 and (geom['type']=='sphere' or geom['size'][1]>0) and (geom['type']!='box' or geom['size'][2]>0),'Nonpositive actor primitive size')
    model=read(source_assets/'doors'/door_id/'model.json')
    require(all(body['name'] in native_names or body['name']=='world_env' and 'world' in native_names for body in model['bodies']),'Native snapshots do not cover authored source bodies')
    with np.load(io.BytesIO(files['trajectory.npz']),allow_pickle=False) as arrays:
        times=arrays['actor_time'];n=len(times)
        require(times.shape==(n,) and n>=2 and n==clip['frames'] and np.isfinite(times).all() and times[0]==0 and np.all(np.diff(times)>0),'Invalid actor timeline')
        require(abs(float(times[-1])-clip['duration'])<1e-6,'Duration mismatch')
        source_time=arrays['native_time'];require(source_time.shape==(n,) and np.isfinite(source_time).all() and np.all(np.diff(source_time)>=0),'Invalid native timeline')
        require(len(clip['phases'])==n,'Phase count mismatch')
        for key in ['foot_contact','hand_contact']:require(arrays[key].shape==(n,2) and np.isin(arrays[key],[0,1]).all(),f'Invalid {key}')
        web={'schema':SCHEMA,'door_id':door_id,'status':'accepted_kinematic','scope':SCOPE,
             'source_scenario':scenario,
             'source_sha256':sources,'native_resources_sha256':result['provenance'].get('native_resources_sha256',{}),
             'identity_sha256':result['identity_sha256'],'trajectory_sha256':sha(files['trajectory.npz']),
             'validation_sha256':sha(files['validation.json']),'duration':clip['duration'],'times':times.tolist(),
             'native_time':source_time.tolist(),'phases':clip['phases'],'foot_contact':arrays['foot_contact'].astype(int).tolist(),
             'hand_contact':arrays['hand_contact'].astype(int).tolist(),
             'native':{'body_names':native_names,'poses':pose_frames(arrays,'body_',n,native_names)},
             'actor':{'body_names':names,'geometries':geometries,'poses':pose_frames(arrays,'actor_body_',n,names)}}
    # Check again after NPZ decoding; an active corpus may publish a replacement.
    verified_artifacts(directory,result)
    raw=encoded(web);packed=gzip.compress(raw,compresslevel=6,mtime=0);digest=sha(packed)
    path=put(out,f'clips/{door_id}.{digest}.json.gz',packed)
    audits={name:audit(out,door_id,name,files[name]) for name in ['clip.json','validation.json']}
    return {'path':path,'sha256':digest,'json_sha256':sha(raw),'bytes':len(packed),'frames':n,'duration':clip['duration']},audits

def export_corpus(corpus,out,assets='assets'):
    corpus=Path(corpus).resolve();out=Path(out).resolve();assets=Path(assets).resolve()
    require(not any(out==p or out.is_relative_to(p) or p.is_relative_to(out) for p in [corpus,assets]),'Output must be separate from corpus and source assets')
    index_bytes=(corpus/'index.json').read_bytes();source=json.loads(index_bytes)
    require(source.get('schema')=='doorbench.planned-reference-corpus.v1','Unsupported corpus index')
    manifest_bytes=(assets/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
    require(sha(manifest_bytes)==source['manifest_sha256'],'Corpus uses a different source manifest')
    require({r['door_id'] for r in source['doors']}=={r['id'] for r in manifest['doors']},'Corpus status rows do not cover the source manifest')
    rows=[];seen=set()
    for entry in source['doors']:
        door_id=entry['door_id'];require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',door_id) is not None and door_id not in seen,'Unsafe/duplicate door ID');seen.add(door_id)
        require(entry.get('status') in ['accepted_kinematic','rejected','unresolved'],'Unsupported attempt status')
        row={key:entry.get(key) for key in ['door_id','family','status','failure_counts','reason_code','source_outcome','identity_sha256']}
        row['reason']=public_text(entry.get('error',{}).get('message') if isinstance(entry.get('error'),dict) else entry.get('error'))
        row['source_scenario']=None
        row['audits']={};row['clip']=None
        if entry.get('result'):
            result_path=checked_path(corpus,entry['result']);result_bytes=result_path.read_bytes();result=json.loads(result_bytes)
            require(result.get('door_id')==door_id and result.get('status')==entry['status'] and result.get('identity_sha256')==entry['identity_sha256'],'Corpus result/index mismatch; retry after the current snapshot finishes')
            require(result.get('provenance',{}).get('generator_sha256')==source['generator']['sha256'] and result.get('provenance',{}).get('manifest_sha256')==source['manifest_sha256'],'Result belongs to a different corpus generator or manifest')
            outcome=result.get('source_outcome')
            if isinstance(outcome,dict):
                scenario=outcome.get('scenario');require(scenario is None or scenario in SOURCE_SCENARIOS,'Unsupported bound source scenario')
                row['source_scenario']=scenario
                row['source_outcome']={name:outcome.get(name) for name in ['scenario','success','outcome']}
            artifacts=verified_artifacts(result_path.parent,result)
            row['audits']['result.json']=audit(out,door_id,'result.json',encoded(public_result(result,result_bytes)))
            if entry['status']=='accepted_kinematic':
                require(all(result.get('new_completion',{}).get(k) is True for k in ['complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared']),'Incomplete acceptance contract')
                row['clip'],more=export_accepted(result_path.parent,result,assets,out,artifacts);row['audits'].update(more)
            elif 'validation.json' in result.get('artifacts',{}):
                data=checked_path(result_path.parent,'validation.json').read_bytes();require(sha(data)==result['artifacts']['validation.json'],'Stale audit artifact')
                row['audits']['validation.json']=audit(out,door_id,'validation.json',data)
            require(result_path.read_bytes()==result_bytes,'Attempt result changed while exporting; retry this snapshot')
        require(entry['status']!='accepted_kinematic' or row['clip'] is not None,'Accepted row has no verified playable clip')
        rows.append(row)
    index={'schema':INDEX_SCHEMA,'scope':SCOPE,'snapshot_id':source['snapshot_id'],'updated_at':source['updated_at'],
           'manifest_sha256':source['manifest_sha256'],'corpus_index_sha256':sha(index_bytes),'generator_sha256':source['generator']['sha256'],
           'counts':dict(Counter(r['status'] for r in rows)),'doors':rows}
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:f.write(encoded(index));temporary=f.name
    os.replace(temporary,out/'index.json');return index

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--corpus',default='out/reference-planned-corpus-v1')
    parser.add_argument('--out',default='out/planned-reference-web');parser.add_argument('--assets',default='assets');args=parser.parse_args()
    result=export_corpus(args.corpus,args.out,args.assets);print(json.dumps({'doors':len(result['doors']),'counts':result['counts'],'index':str(Path(args.out)/'index.json')}))

if __name__=='__main__':main()
