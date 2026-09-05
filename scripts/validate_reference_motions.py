#!/usr/bin/env python3
"""Validate all recorded source hashes, array dimensions and fixed limb lengths."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import mujoco


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(root:Path,assets:Path):
    index=json.loads((root/'index.json').read_text());manifest=json.loads((assets/'manifest.json').read_text())
    ids={d['id'] for d in manifest['doors']};rows=index['clips']
    assert index['schema']=='doorbench.reference-motion.v1'
    assert len(rows)==len(ids) and {r['door_id'] for r in rows}==ids
    assert index['manifest_sha256']==sha(assets/'manifest.json')
    reports=[]
    for row in rows:
        id=row['door_id'];assert 'error' not in row
        for key in ('clip','trajectory','web_clip'):
            path=root/row[key];assert path.resolve().is_relative_to(root.resolve())
            assert sha(path)==row[key+'_sha256'],(id,key,'checksum')
        assert gzip.decompress((root/row['web_clip']).read_bytes())==(root/row['clip']).read_bytes()
        c=json.loads((root/row['clip']).read_text());assert c['door_id']==id
        for name,value in row['source_sha256'].items():assert value==sha(assets/'doors'/id/name)
        assert c['source_sha256']==row['source_sha256']
        assert c['outcome']['success']==row['success'] and c['outcome']['outcome']==row['outcome']
        n=row['frames'];nf=row['physics_frames'];lead=round(c['lead_in_s']*c['fps'])
        assert n==nf+lead
        t=np.asarray(c['times']);assert len(t)==n and (np.diff(t)>0).all()
        with np.load(root/row['trajectory'],allow_pickle=False) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files),(id,'nonfinite')
            assert len(a['time'])==nf and len(a['actor_time'])==n
            assert a['actor_joints'].shape==(n,16,3)
            assert a['qpos'].shape[0]==nf and a['qvel'].shape==a['tau'].shape
            assert a['body_pos'].shape==(nf,len(c['native']['body_names']),3)
            assert a['body_quat'].shape==(nf,len(c['native']['body_names']),4)
            np.testing.assert_allclose(a['time']+c['lead_in_s'],t[lead:],atol=.0002)
            np.testing.assert_allclose(a['qpos'][:,c['native']['qpos_addresses']],np.array(c['door_q'])[lead:],atol=.000007,rtol=1e-5)
            np.testing.assert_allclose(a['actor_joints'],np.array(c['avatar']).reshape(n,16,3),atol=.00006)
            limbs=[(4,5,.30),(5,6,.28),(7,8,.30),(8,9,.28),(10,11,.43),(11,12,.43),(13,14,.43),(14,15,.43)]
            max_error=max(float(np.max(abs(np.linalg.norm(a['actor_joints'][:,i]-a['actor_joints'][:,j],axis=1)-length))) for i,j,length in limbs)
            assert max_error<.00001,(id,'limb lengths',max_error)
            assert np.min(a['actor_joints'][:,[12,15],2])>=-.001,(id,'foot below ground')
            model=mujoco.MjModel.from_xml_path(str(assets/'doors'/id/'door.xml')); pose=mujoco.MjData(model)
            for sample in np.unique(np.linspace(0,nf-1,min(8,nf)).astype(int)):
                pose.qpos[:]=a['qpos'][sample];mujoco.mj_kinematics(model,pose)
                np.testing.assert_allclose(a['body_pos'][sample],pose.xpos,atol=.00001,rtol=1e-6)
                np.testing.assert_allclose(a['body_quat'][sample],pose.xquat,atol=.00001,rtol=1e-6)
            active=np.asarray(c['hand_active'],bool)
            residual=np.linalg.norm(a['actor_joints'][:,9]-np.asarray(c['targets']),axis=1)
            np.testing.assert_allclose(residual[active],a['hand_target_error'][active],atol=.00015)
        reports.append({'door_id':id,'family':row['family'],'outcome':row['outcome'],'source_and_numeric_checks':'pass',
                        'max_limb_error_m':max_error,'unreachable_frames':row['unreachable_frames'],'max_hand_error_m':row['max_hand_error_m']})
    return {'schema':'doorbench.reference-validation.v1','doors':len(rows),'families':len({r['family'] for r in rows}),
            'index_sha256':sha(root/'index.json'),'outcomes':dict(Counter(r['outcome'] for r in rows)),
            'doors_with_unreachable_targets':sum(r['unreachable_frames']>0 for r in rows),
            'limitations':'Numeric integrity and source correspondence, not humanoid contact/balance certification.', 'checks':reports}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('out/reference-motions'));p.add_argument('--assets',type=Path,default=Path('assets'));p.add_argument('--out',type=Path,default=Path('out/reference-review/validation.json'));a=p.parse_args()
    report=validate(a.root,a.assets);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
