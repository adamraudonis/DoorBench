from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.isaac_materials import native_contact_contract,bind_robot_contact_material,check_solver_materials,check_solver_offsets


def test_solver_offsets_must_be_verified_after_setting_instanced_shapes():
    assert check_solver_offsets([[.001,.001]],[[0.,0.]])['backend_offsets_verified']
    with pytest.raises(ValueError,match='different collision offsets'):check_solver_offsets([[.02,.02]],[[0.,0.]])


def test_material_contract_does_not_silently_flatten_another_robot():
    model=SimpleNamespace(geom_contype=np.array([1,1]),geom_conaffinity=np.array([1,1]),
        geom_friction=np.array([[1.,.005,.0001],[1.,.005,.0001]]),geom_priority=np.array([0,0]),geom_condim=np.array([3,3]))
    assert native_contact_contract(model)['dynamic_friction']==1.
    model.geom_friction[1,0]=.2
    with pytest.raises(ValueError,match='per-collider'):native_contact_contract(model)


def test_authored_coefficients_are_not_enough_if_solver_still_uses_defaults():
    contract=dict(static_friction=1.,dynamic_friction=1.)
    assert check_solver_materials([[1.,1.,0.]],contract)['backend_values_verified']
    with pytest.raises(ValueError,match='different robot contact'):check_solver_materials([[.5,.5,0.]],contract)


def test_physics_material_reaches_instanced_colliders_without_changing_appearance():
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/H1')
    UsdGeom.Xform.Define(stage,'/Source');cube=UsdGeom.Cube.Define(stage,'/Source/Shape')
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    visual=UsdShade.Material.Define(stage,'/Looks/Visual')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(visual)
    instance=UsdGeom.Xform.Define(stage,'/H1/Finger').GetPrim()
    instance.GetReferences().AddInternalReference('/Source');instance.SetInstanceable(True)
    result=bind_robot_contact_material(stage,'/H1',dict(static_friction=1.,dynamic_friction=1.,combine_mode='max'))
    assert result['colliders_checked']==1
    prim=stage.GetPrimAtPath('/H1/Finger/Shape')
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    assert UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Get()==1.
    appearance,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert appearance.GetPath()==visual.GetPath()
