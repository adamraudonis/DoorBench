"""Dimensions embedded in mesh filenames must not become USD property paths."""

import pytest

pytest.importorskip("pxr")
from pxr import Usd
import trimesh
from doorbench.export.usd import _ensure_mesh_usd, _safe


@pytest.mark.parametrize("partial_cache", [False, True])
def test_decimal_mesh_identifier_is_valid_and_referenceable(tmp_path, partial_cache):
    name = "section_astragal_2.440000_0.045000_0"
    path = str(tmp_path / (name + ".usdc"))
    if partial_cache:
        partial = Usd.Stage.CreateNew(path)
        partial.GetRootLayer().Save()
        del partial
    _ensure_mesh_usd(trimesh.creation.box(), path, name)
    mesh = Usd.Stage.Open(path)
    assert str(mesh.GetDefaultPrim().GetPath()) == "/" + _safe(name)
    stage = Usd.Stage.CreateInMemory()
    p = stage.DefinePrim("/holder")
    p.GetReferences().AddReference(path, "/" + _safe(name))
    assert stage.GetPrimAtPath("/holder/mesh").IsValid()
