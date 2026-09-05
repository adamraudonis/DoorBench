"""Pure Python input validation; Blender integration is checked explicitly."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import blender_reference_motion as replay
from doorbench.appearance.pipeline import digest


@pytest.fixture
def inputs(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    model={'bodies':[{'name':'world_env'},{'name':'leaf'}]}
    for name,value in [('model.json',model),('spec.json',{'id':'tiny'}),('door.xml',{})]:
        (source/name).write_text(json.dumps(value))
    hashes={name:replay.sha256(source/name) for name in ('model.json','spec.json','door.xml')}
    job={'door_id':'tiny','door_dir':str(source),'hardware_dir':str(tmp_path),
         'source_sha256':hashes,'mesh_sha256':{},'renderer_sha256':{},
         'reference_state':{'body_aliases':{'world_env':'world'}}}
    job['job_sha256']=digest(job)
    actor=np.zeros((4,16,3)); actor[:,:,0]=np.arange(16)*.1; actor[:,3,2]=1.7
    arrays={'time':np.array([0,1,2.]),'body_pos':np.zeros((3,2,3)),
            'body_quat':np.tile([1.,0,0,0],(3,2,1)),
            'actor_time':np.array([0.,1,2,3]),'actor_joints':actor,'qpos':np.array([[0.],[1.],[2.]])}
    clip={'schema':'doorbench.reference-motion.v1','door_id':'tiny','source_sha256':hashes,
          'up_axis':'Z','units':'metres/radians/seconds','fps':20,'lead_in_s':1,
          'avatar_joint_names':replay.JOINTS,'avatar_bones':replay.BONES,
          'native':{'body_names':['world','leaf'],'qpos_addresses':[0]},
          'times':arrays['actor_time'].tolist(),'avatar':actor.reshape(4,48).tolist(),
          'door_q':[[0],[0],[1],[2]]}
    return tmp_path,job,clip,arrays


def load(fixture):
    root,job,clip,arrays=fixture
    (root/'job.json').write_text(json.dumps(job))
    (root/'clip.json').write_text(json.dumps(clip))
    np.savez(root/'motion.npz',**arrays)
    return replay.load_inputs(root/'job.json',root/'clip.json',root/'motion.npz')


def test_static_alias_and_native_body_mapping(inputs):
    assert load(inputs)[-1] == {'world_env':0,'leaf':1}


@pytest.mark.parametrize('damage,match',[
    ('source','Source changed'),('job','job checksum'),('door','door_id'),
    ('shape','body_pos'),('quaternion','unit WXYZ'),('finite','finite numeric'),
    ('time','strictly increase'),('actor','Clip avatar'),('qpos','door_q'),
    ('pickle','Object arrays'),('missingbody','No native body pose'),
    ('bone','nonzero length'),
])
def test_reject_unusable_or_mismatched_recordings(inputs,damage,match):
    root,job,clip,arrays=inputs
    if damage=='source': (Path(job['door_dir'])/'door.xml').write_text('changed')
    if damage=='job': job['door_id']='changed'
    if damage=='door': clip['door_id']='wrong'
    if damage=='shape': arrays['body_pos']=arrays['body_pos'][:,:1]
    if damage=='quaternion': arrays['body_quat'][1,1]=0
    if damage=='finite': arrays['actor_joints'][1,1,1]=float('nan')
    if damage=='time': arrays['time'][1]=0
    if damage=='actor': arrays['actor_joints'][1,1,1]=.1
    if damage=='qpos': arrays['qpos'][1,0]=.5
    if damage=='pickle': arrays['extra']=np.array([{'untrusted':True}],dtype=object)
    if damage=='missingbody': clip['native']['body_names'][1]='wrong'
    if damage=='bone': arrays['actor_joints'][:,1]=arrays['actor_joints'][:,0]
    with pytest.raises(ValueError,match=match): load(inputs)
