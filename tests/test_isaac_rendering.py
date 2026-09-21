"""CPU USD composition checks: graphics corrections must preserve physics."""
import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from doorbench.dexterous.isaac_rendering import (
    configure_diagnostic_lighting, restore_robot_visual_materials,
)


def make_imported_stage():
    stage = Usd.Stage.CreateInMemory()
    for name in ('DefaultMaterial', 'material_black', 'material_white'):
        UsdShade.Material.Define(stage, '/H1/Looks/' + name)
    for name, material in [('body', 'material_black'), ('logo', 'material_white')]:
        parent = UsdGeom.Xform.Define(stage, '/VisualTemplate/' + name).GetPrim()
        mesh = UsdGeom.Mesh.Define(stage, str(parent.GetPath()) + '/' + name)
        UsdShade.MaterialBindingAPI.Apply(parent).Bind(
            UsdShade.Material(stage.GetPrimAtPath('/H1/Looks/' + material)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath('/H1/Looks/DefaultMaterial')))
    root = UsdGeom.Xform.Define(stage, '/H1/body').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(8.2)
    visuals = UsdGeom.Xform.Define(stage, '/H1/body/visuals').GetPrim()
    visuals.GetReferences().AddInternalReference('/VisualTemplate')
    visuals.SetInstanceable(True)
    collision = UsdGeom.Cube.Define(stage, '/H1/body/collisions/box')
    collision.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    physics = UsdShade.Material.Define(stage, '/H1/Contact')
    UsdPhysics.MaterialAPI.Apply(physics.GetPrim()).CreateStaticFrictionAttr(1.)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(physics, 'strongerThanDescendants', 'physics')
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1800.)
    return stage


def physical_state(stage):
    result = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        schemas = [s for s in prim.GetAppliedSchemas()
                   if 'physics' in s.lower() or 'physx' in s.lower()]
        properties = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                      if a.GetName().startswith(('physics:', 'physx'))}
        relationships = {r.GetName(): str(r.GetTargets()) for r in prim.GetRelationships()
                         if r.GetName().startswith(('physics:', 'physx', 'material:binding:physics'))}
        if schemas or properties or relationships:
            result[str(prim.GetPath())] = (schemas, properties, relationships)
    return result


def test_repairs_native_colors_but_preserves_physics_and_source_layer():
    stage = make_imported_stage()
    source_before = stage.GetRootLayer().ExportToString()
    physics_before = physical_state(stage)
    audit = restore_robot_visual_materials(stage, '/H1')
    assert audit['repaired_meshes'] == 2
    assert audit['expanded_visual_instances'] == 1
    for name, material in [('body', 'material_black'), ('logo', 'material_white')]:
        mesh = stage.GetPrimAtPath(f'/H1/body/visuals/{name}/{name}')
        actual, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        assert actual.GetPrim().GetName() == material
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/H1/body/collisions/box')).ComputeVisibility() == 'invisible'
    assert physical_state(stage) == physics_before
    assert stage.GetRootLayer().ExportToString() == source_before
    assert restore_robot_visual_materials(stage, '/H1')['repaired_meshes'] == 0


def test_visual_instance_with_physics_schema_is_rejected_before_mutation():
    stage = make_imported_stage()
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath('/VisualTemplate/body/body'))
    session_before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError, match='Physics schema'):
        restore_robot_visual_materials(stage, '/H1')
    assert stage.GetSessionLayer().ExportToString() == session_before


def test_lighting_preserves_physics_and_authored_source():
    stage = make_imported_stage()
    source_before = stage.GetRootLayer().ExportToString()
    physics_before = physical_state(stage)
    audit = configure_diagnostic_lighting(stage, '/World/Light')
    assert audit['previous_dome_intensity'] == 1800.
    assert UsdLux.DomeLight(stage.GetPrimAtPath('/World/Light')).GetIntensityAttr().Get() == 250.
    assert UsdLux.DistantLight(stage.GetPrimAtPath('/World/DiagnosticKey')).GetIntensityAttr().Get() == 600.
    assert physical_state(stage) == physics_before
    assert stage.GetRootLayer().ExportToString() == source_before
