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
    scope=dict(Path=Path,__file__=str(source),a=args,out=output,record_standing_body_poses=False,
        sequence=None,full_opening=None,continuous=None,sensor_actor=None,hashlib=hashlib,json=json,time=time)
    exec(compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(source),'exec'),scope)
    route=Path(args.standing_transfer_route)
    assert (output/'standing-transfer-route.json').read_bytes()==route.read_bytes()
    assert json.loads((output/'provenance.json').read_text())['files'][str(route)]==hashlib.sha256(route.read_bytes()).hexdigest()
