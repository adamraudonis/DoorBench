#!/usr/bin/env python3
"""Import the free H1/Shadow articulation in Isaac Sim and record its real USD structure."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--mjcf',required=True)
p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
failed=False
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.mjcf')
    import omni.kit.commands
    import omni.usd
    from pxr import Usd,UsdPhysics,PhysxSchema,UsdGeom
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/World')
    ok,config=omni.kit.commands.execute('MJCFCreateImportConfig')
    assert ok
    config.set_fix_base(False)
    config.set_import_inertia_tensor(True)
    config.set_self_collision(True)
    config.set_import_sites(True)
    config.set_make_default_prim(True)
    config.set_merge_fixed_joints(False)
    config.set_create_physics_scene(True)
    config.set_default_drive_strength(0.)
    ok,result=omni.kit.commands.execute('MJCFCreateAsset',mjcf_path=str(Path(a.mjcf).resolve()),
        import_config=config,prim_path='/H1')
    assert ok, result
    for prim in stage.Traverse():
        if prim.GetName()=='worldBody' and prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    rows=[]
    for prim in stage.Traverse():
        row={'path':str(prim.GetPath()),'type':prim.GetTypeName(),'schemas':prim.GetAppliedSchemas()}
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or any('Tendon' in x for x in row['schemas']):
            row['attributes']={str(attr.GetName()):str(attr.Get()) for attr in prim.GetAttributes()}
            row['relationships']={str(rel.GetName()):[str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships()}
            rows.append(row)
    stage.GetRootLayer().Export(str((out/'robot.usda').resolve()))
    roots=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert len(roots)==1,roots
    assert len(joints)==69 and all(p.IsA(UsdPhysics.RevoluteJoint) for p in joints), 'Robot must have 69 hinges and a free root'
    assert all(p.GetRelationship('physics:body0').GetTargets() and p.GetRelationship('physics:body1').GetTargets() for p in joints), 'Robot joint attached to world'
    (out/'import-audit.json').write_text(json.dumps({'articulation_roots':roots,'prims':rows},indent=2)+'\n')
    print('IMPORT_AUDIT_OK '+json.dumps({'roots':roots,'bodies':sum('PhysicsRigidBodyAPI' in r['schemas'] for r in rows),'joints':sum(r['type'].endswith('Joint') for r in rows)}),flush=True)
except BaseException:
    failed=True
    import traceback
    traceback.print_exc()
    (out/'error.txt').write_text(traceback.format_exc())
finally:
    app.close()
if failed:raise SystemExit(1)
