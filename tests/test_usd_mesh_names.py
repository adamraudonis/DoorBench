"""Dimensions embedded in mesh filenames must not become USD property paths."""
import pytest
pytest.importorskip('pxr')
from pxr import Usd
import trimesh
from doorbench.export.usd import _write_mesh_usd,_safe

def test_decimal_mesh_identifier_is_valid_and_referenceable(tmp_path):
    name='section_astragal_2.440000_0.045000_0'
    path=str(tmp_path/(name+'.usdc'))
    _write_mesh_usd(trimesh.creation.box(),path,name)
    mesh=Usd.Stage.Open(path)
    assert str(mesh.GetDefaultPrim().GetPath())=='/'+_safe(name)
    stage=Usd.Stage.CreateInMemory()
    p=stage.DefinePrim('/holder')
    p.GetReferences().AddReference(path,'/'+_safe(name))
    assert stage.GetPrimAtPath('/holder/mesh').IsValid()
