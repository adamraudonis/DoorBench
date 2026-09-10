import importlib.util
from pathlib import Path
import pytest

Usd = pytest.importorskip('pxr.Usd')
from pxr import UsdGeom, UsdPhysics

spec = importlib.util.spec_from_file_location('inventory', Path(__file__).resolve().parents[1] / 'scripts/dexterous/audit_imported_collision_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('actual_type,actual_name,passed', [
    ('Cylinder', 'finger', True), ('Sphere', 'finger', False),
    ('Cylinder', 'wrong_finger', False),
])
def test_rejects_replacement_shape_and_wrong_identity(tmp_path, actual_type, actual_name, passed):
    source = tmp_path / 'source.xml'
    source.write_text('<mujoco><worldbody><body name="hand"><geom name="finger" type="cylinder" contype="1" conaffinity="1"/></body></worldbody></mujoco>')
    path = tmp_path / 'robot.usda'
    stage = Usd.Stage.CreateNew(str(path))
    prim = getattr(UsdGeom, actual_type).Define(stage, f'/collisions/hand/{actual_name}/{actual_name}').GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    stage.GetRootLayer().Save()
    result = module.audit(source, path)
    assert result['passed'] is passed
    assert bool(result['missing']) is (not passed)
    assert bool(result['unexpected']) is (not passed)
