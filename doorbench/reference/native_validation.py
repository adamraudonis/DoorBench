"""Source and frame correspondence checks for native mechanism recordings."""
from __future__ import annotations
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .record import NATIVE_SCHEMA, digest
from ..benchmark_eligibility import is_benchmark_eligible


def validate_native(root:Path,assets:Path):
    import mujoco
    index=json.loads((root/'index.json').read_text())
    manifest=json.loads((assets/'manifest.json').read_text())
    ids={d['id'] for d in manifest['doors'] if is_benchmark_eligible(d)}
    rows=index['clips']
    assert index['schema']==NATIVE_SCHEMA
    assert len(rows)==len(ids) and {r['door_id'] for r in rows}==ids, 'Recording coverage does not match eligible manifest'
    assert index['manifest_sha256']==digest(assets/'manifest.json')
    reports=[]
    for row in rows:
        door_id=row['door_id'];assert 'error' not in row,row
        directory=assets/'doors'/door_id
        for key in ('clip','trajectory','web_clip'):
            path=root/row[key];assert path.resolve().is_relative_to(root.resolve())
            assert digest(path)==row[key+'_sha256'],(door_id,key,'checksum')
        assert gzip.decompress((root/row['web_clip']).read_bytes())==(root/row['clip']).read_bytes()
        clip=json.loads((root/row['clip']).read_text())
        assert clip['schema']==NATIVE_SCHEMA and clip['door_id']==door_id
        assert not any(k.startswith('avatar') for k in clip),'Native recording must not fabricate an actor'
        assert clip['source_sha256']==row['source_sha256']
        assert set(row['source_sha256'])=={'spec.json','model.json','door.xml'}
        for name,value in row['source_sha256'].items():assert digest(directory/name)==value,(door_id,name,'source mismatch')
        assert clip['outcome']['success']==row['success'] and clip['outcome']['outcome']==row['outcome']
        model=mujoco.MjModel.from_xml_path(str(directory/'door.xml'));pose=mujoco.MjData(model)
        metadata=json.loads((directory/'model.json').read_text())['meta']
        n=row['frames'];assert n==row['physics_frames'] and n>=2
        names=[model.joint(i).name for i in range(model.njnt)]
        assert clip['joint_names']==names
        assert clip['native']['qpos_addresses']==model.jnt_qposadr.tolist()
        assert clip['native']['body_names']==[model.body(i).name for i in range(model.nbody)]
        if 'joint_types' in clip['native']:
            assert clip['native']['joint_types']==[mujoco.mjtJoint(int(kind)).name for kind in model.jnt_type]
            assert clip['native']['qpos_widths']==np.diff(np.r_[model.jnt_qposadr,model.nq]).tolist()
            assert clip['native']['qvel_widths']==np.diff(np.r_[model.jnt_dofadr,model.nv]).tolist()
        assert len(clip['phases'])==len(clip['oracle_contacts'])==n
        with np.load(root/row['trajectory'],allow_pickle=False) as arrays:
            for key,shape in {'time':(n,),'qpos':(n,model.nq),'qvel':(n,model.nv),'tau':(n,model.nv),
                              'ctrl':(n,model.nu),'body_pos':(n,model.nbody,3),'body_quat':(n,model.nbody,4)}.items():
                assert arrays[key].shape==shape,(door_id,key,'shape')
            assert not any(k.startswith('actor') for k in arrays.files)
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            time=arrays['time'];assert time[0]==0 and (np.diff(time)>0).all()
            assert abs(time[-1]-clip['duration'])<1e-10
            np.testing.assert_array_equal(clip['times'],time)
            np.testing.assert_array_equal(clip['door_q'],arrays['qpos'][:,model.jnt_qposadr])
            if 'poses' in clip['native']:
                np.testing.assert_array_equal(clip['native']['poses'],np.concatenate((arrays['body_pos'],arrays['body_quat']),axis=2).reshape(n,-1))
            if any(int(t)==int(mujoco.mjtJoint.mjJNT_FREE) for t in model.jnt_type):
                assert 'poses' in clip['native'],'A free root requires complete native body poses'
                for j,kind in enumerate(model.jnt_type):
                    if int(kind)==int(mujoco.mjtJoint.mjJNT_FREE):
                        address=model.jnt_qposadr[j]
                        np.testing.assert_allclose(np.linalg.norm(arrays['qpos'][:,address+3:address+7],axis=1),1.,atol=1e-10)
            assert json.loads(arrays['oracle_contacts_json_utf8'].tobytes())==clip['oracle_contacts']
            if 'native_cables' in clip:
                assert len(clip['native_cables'])==n
                assert json.loads(arrays['native_cables_json_utf8'].tobytes())==clip['native_cables']
            # Inspect every recorded state and commanded contact, not just key poses.
            for i in range(n):
                pose.qpos[:]=arrays['qpos'][i];mujoco.mj_kinematics(model,pose)
                np.testing.assert_allclose(arrays['body_pos'][i],pose.xpos,atol=1e-11,rtol=0)
                np.testing.assert_allclose(arrays['body_quat'][i],pose.xquat,atol=1e-11,rtol=0)
                for contact in clip['oracle_contacts'][i]:
                    if contact['site'] is None:
                        assert contact['position'] is None
                        continue
                    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,contact['site']);assert sid>=0
                    np.testing.assert_allclose(contact['position'],pose.site_xpos[sid],atol=1e-11,rtol=0)
                    if 'force_N' in contact:
                        force=np.asarray(contact['force_N']);assert force.shape==(3,) and np.isfinite(force).all()
                        assert np.linalg.norm(force)<=120.+1e-8
                    if 'torque_Nm' in contact:
                        torque=np.asarray(contact['torque_Nm']);assert torque.shape==(3,) and np.isfinite(torque).all()
                        cap=metadata.get('site_wrench_limits_Nm',{}).get(contact['site'],0.)
                        assert np.linalg.norm(torque)<=cap+1e-8
        reports.append({'door_id':door_id,'outcome':row['outcome'],'frames_checked':n,'source_and_numeric_checks':'pass'})
    return {'schema':'doorbench.native-motion-validation.v1','doors':len(rows),'index_sha256':digest(root/'index.json'),
            'outcomes':dict(Counter(r['outcome'] for r in rows)),
            'limitations':'Source and state correspondence only; does not certify contact dynamics, human feasibility or all mechanisms.',
            'checks':reports}
