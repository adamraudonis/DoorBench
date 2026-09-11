"""Exercise the actual provenance setup block without starting Isaac graphics."""
import ast,hashlib,json,time
from pathlib import Path


def test_transfer_route_is_copied_and_hashed_after_input_initialization(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/dexterous/isaac_opening.py'
    tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
    end=next(i for i in range(start,len(main.body)) if isinstance(main.body[i],ast.For) and isinstance(main.body[i].target,ast.Name) and main.body[i].target.id=='source')+1
    class Args:
        def __getattr__(self,name):return None
    args=Args()
    for name in ['robot_usd','door_usd','motors','reference','standing_transfer_route']:
        p=tmp_path/name;p.write_text('{}');setattr(args,name,str(p))
    output=tmp_path/'out';output.mkdir()
    scope=dict(Path=Path,__file__=str(source),a=args,out=output,record_standing_body_poses=False,physics_audit_enabled=True,
        sequence=None,full_opening=None,continuous=None,sensor_actor=None,hashlib=hashlib,json=json,time=time)
    exec(compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(source),'exec'),scope)
    route=Path(args.standing_transfer_route)
    assert (output/'standing-transfer-route.json').read_bytes()==route.read_bytes()
    assert json.loads((output/'provenance.json').read_text())['files'][str(route)]==hashlib.sha256(route.read_bytes()).hexdigest()


def test_transfer_recording_preserves_joint_friction_for_next_motor_step():
    from types import SimpleNamespace
    import numpy as np
    source=Path(__file__).resolve().parents[1]/'scripts/dexterous/isaac_opening.py'
    tree=ast.parse(source.read_text())
    block=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and isinstance(n.test,ast.Name) and n.test.id=='standing_transfer'
        and any(isinstance(c,ast.Attribute) and c.attr=='get_friction_data' for c in ast.walk(n)))
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def __getitem__(self,key):return Tensor(self.value[key])
        def cpu(self):return self
        def numpy(self):return self.value
    class Contacts:
        def get_contact_force_matrix(self,**kwargs):return Tensor(np.zeros((1,1,3)))
        def get_friction_data(self,dt):return [Tensor(np.zeros((16384,3))),Tensor(np.zeros((16384,3))),Tensor([[0]]),Tensor([[0]])]
    original=np.full(69,.1)
    scope=dict(np=np,door=SimpleNamespace(body_names=['leaf'],data=SimpleNamespace(body_state_w=Tensor([[[0,0,0,1,0,0,0]]]))),
        standing_transfer=SimpleNamespace(started=None),audit_contacts=Contacts(),dt=.002,
        contact_force_pairs=lambda *args,**kwargs:np.zeros((1,1,3)),
        panel_surface_loads=lambda *args:{},audit_paths=[],audit_filters=[],transfer_steps=[],
        teacher_info={'stance_status':'solved'},step=0,friction=original)
    exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),scope)
    assert scope['friction'] is original
    scope.update(matrix=np.eye(69),forces=np.zeros(69),damp=np.zeros(69),vel=np.ones(69))
    torque=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='torque' for t in n.targets)
        and any(isinstance(c,ast.Name) and c.id=='friction' for c in ast.walk(n)))
    exec(compile(ast.Module(body=[torque],type_ignores=[]),str(source),'exec'),scope)
    np.testing.assert_allclose(scope['torque'],-original)
