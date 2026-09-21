"""Execute the producer's actual prefix wiring with synthetic CPU observations."""
import ast
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_isaac_prefix_witness import source as source_fixture, witness, sample
from doorbench.dexterous.isaac_prefix_witness import PrefixDivergenceError, PREFIX_FIELDS
from doorbench.dexterous.standing_body_record import pack_standing_body_poses
from scripts.isaac.run_local_operation import runtime_source_paths

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/dexterous/isaac_opening.py'
TREE=ast.parse(SOURCE.read_text())
MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')

def names(node):return {n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
def calls(node,attribute):return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==attribute for n in ast.walk(node))
def execute(nodes,scope):exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)

ENTRY=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='standing_transfer' and 'transfer_prefix_authorized' in names(n))
RECORD=next(n for n in ast.walk(MAIN) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='physics_audit_enabled' and calls(n,'observe'))
OBSERVE_INDEX=next(i for i,n in enumerate(RECORD.body) if calls(n,'observe'))
FINAL=next(n.finalbody for n in ast.walk(MAIN) if isinstance(n,ast.Try) and any('transfer_prefix' in names(v) for v in n.finalbody))

class Tensor:
 def __init__(self,value):self.value=np.asarray(value)
 def __getitem__(self,key):return Tensor(self.value[key])
 def cpu(self):return self
 def numpy(self):return self.value
 def item(self):return self.value.item()


def setup(source_fixture,tmp_path,monkeypatch):
 value=witness(source_fixture);events=[]
 import doorbench.dexterous.isaac_opening_measurements as measures
 monkeypatch.setattr(measures,'panel_surface_loads',lambda *a:dict(total_normal_load_N=3.,palm_normal_load_N=3.))
 def force(*args,**kwargs):events.append(dict(step=scope['step'],authorized=scope['transfer_prefix_authorized'],verified=value.intervals_verified));return np.zeros(61),{}
 scope=dict(np=np,json=json,Path=Path,Rotation=None,transfer_prefix=value,transfer_prefix_authorized=False,
  a=SimpleNamespace(standing_transfer_start_seconds=.006,sensor_locomotion_calibration=False,acquisition_stance_profile='landed-foot-v1'),
  standing_transfer=SimpleNamespace(force=force),step=0,dt=.002,out=tmp_path,
  body=[np.zeros(7)],door=SimpleNamespace(body_names=['leaf','leaf_handle']),
  measured_args=(),angles={},loads={},pad_steps=[dict(valid_pad_grasp=True)],audit_paths=[],audit_filters=[],pairs=[],
  acquisition_states={k:[] for k in (*PREFIX_FIELDS,'torso_tilt_deg')},continuous=None,record_standing_continuation=False,
  record_standing_body_poses=True,standing_body_indices=np.arange(5),pack_standing_body_poses=pack_standing_body_poses,jev_gate=None)
 scope.update(standing_controller=scope['standing_transfer'],withdrawal_prefix=None,
              withdrawal_prefix_authorized=False,withdrawal_steps=None,transfer_rest_stop=None,standing_continuation_steps=None)
 def record(index,*,alter=None):
  row=sample(source_fixture,index)
  if alter:alter(row)
  scope['step']=index;scope['forces']=row['motor_forces']
  robot=SimpleNamespace(root=row['root'],joint_pos=Tensor(row['joints'][None]),joint_vel=Tensor(row['joint_velocity'][None]),body_state_w=Tensor(row['standing_body_poses'][None,:5]),projected_gravity_b=Tensor([[0.,0.,-1.]]))
  scope['robot']=SimpleNamespace(data=robot)
  # The first door body is unused here; the second is the exact recorded handle.
  doorposes=np.stack([row['standing_body_poses'][-1]]*2)[None]
  scope['door'].data=SimpleNamespace(joint_pos=Tensor(row['door'][None]),joint_vel=Tensor(row['door_velocity'][None]),body_state_w=Tensor(doorposes))
  scope['controller_root_state']=lambda data,**kw:Tensor(data.root[None])
  execute(RECORD.body[:OBSERVE_INDEX+1],scope)
 return value,scope,record,events


def test_last_poststep_record_is_observed_before_first_transfer_command(source_fixture,tmp_path,monkeypatch):
 value,scope,record,events=setup(source_fixture,tmp_path,monkeypatch)
 for i in range(3):
  scope['step']=i;execute([ENTRY],scope)
  assert events[-1]['authorized'] is False
  record(i)
  assert value.intervals_verified==i+1
  assert len(scope['acquisition_states']['standing_body_poses'])==i+1
 assert value.complete and not value.receipt()['passed']
 scope['step']=3;execute([ENTRY],scope)
 assert events[-1]==dict(step=3,authorized=True,verified=3)
 assert value.receipt()['passed']
 assert json.loads((tmp_path/'live-prefix-witness.json').read_text())['passed']
 scope['step']=4;execute([ENTRY],scope)
 assert events[-1]['authorized'] is True and value.intervals_verified==3


def test_missing_last_interval_blocks_force_and_preserves_failed_receipt(source_fixture,tmp_path,monkeypatch):
 value,scope,record,events=setup(source_fixture,tmp_path,monkeypatch)
 record(0);record(1);scope['step']=3
 with pytest.raises(PrefixDivergenceError):execute([ENTRY],scope)
 assert not events and not scope['transfer_prefix_authorized']
 execute(FINAL,scope)
 receipt=json.loads((tmp_path/'live-prefix-witness.json').read_text())
 assert receipt['intervals_verified']==2 and receipt['failure'] and not receipt['stage_entry_authorized']


@pytest.mark.parametrize('field',['root','motor_forces','standing_body_poses'])
def test_changed_recorded_state_or_command_never_reaches_stage(source_fixture,tmp_path,monkeypatch,field):
 value,scope,record,events=setup(source_fixture,tmp_path,monkeypatch)
 def alter(row):
  x=row[field];x.flat[0]=np.nextafter(x.flat[0],np.array(np.inf,dtype=x.dtype))
 with pytest.raises(PrefixDivergenceError):record(0,alter=alter)
 assert value.failed and value.intervals_verified==0
 scope['step']=3
 with pytest.raises(PrefixDivergenceError):execute([ENTRY],scope)
 assert not events
 execute(FINAL,scope)
 assert json.loads((tmp_path/'live-prefix-witness.json').read_text())['failure']['field']==field


def test_default_without_witness_calls_original_controller_without_admission(source_fixture,tmp_path,monkeypatch):
 value,scope,record,events=setup(source_fixture,tmp_path,monkeypatch)
 scope['transfer_prefix']=None;scope['step']=3
 execute([ENTRY],scope)
 assert events==[dict(step=3,authorized=False,verified=0)] and not value.failed
 execute(FINAL,scope)
 assert not (tmp_path/'live-prefix-witness.json').exists()


def test_optin_producer_and_launcher_capture_actual_binding_dependencies(tmp_path):
 start=next(i for i,n in enumerate(MAIN.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
 end=next(i for i in range(start,len(MAIN.body)) if isinstance(MAIN.body[i],ast.For) and isinstance(MAIN.body[i].target,ast.Name) and MAIN.body[i].target.id=='source')+1
 class Args:
  def __getattr__(self,name):return None
 a=Args();a.acquisition=True;a.operate_after_acquisition=True;a.standing_transfer_prefix_source=str(tmp_path/'original-source')
 for name in ('robot_usd','door_usd','motors','reference','standing_transfer_route'):
  p=tmp_path/name;p.write_text('{}');setattr(a,name,str(p))
 out=tmp_path/'new-run';out.mkdir()
 scope=dict(Path=Path,__file__=str(SOURCE),a=a,out=out,record_standing_body_poses=True,physics_audit_enabled=True,
  sequence=None,full_opening=None,continuous=None,sensor_actor=None,jev_plan=None,hashlib=hashlib,json=json,time=time)
 execute(MAIN.body[start:end],scope)
 files=json.loads((out/'provenance.json').read_text())['files']
 expected={str(p.resolve()) for p in runtime_source_paths(['--standing-transfer-route','--standing-transfer-prefix-source'])}
 assert {k for k in files if Path(k).suffix=='.py'}==expected
 for name in ('destination_state_binding.py','motor_contract_identity.py','isaac_attained_state.py','qualified_isaac_grasp.py','isaac_prefix_witness.py','plan_local_isaac_transfer.py'):
  path=next(Path(p) for p in expected if Path(p).name==name)
  assert files[str(path)]==hashlib.sha256(path.read_bytes()).hexdigest()
  assert (out/('source-'+name)).read_bytes()==path.read_bytes()
