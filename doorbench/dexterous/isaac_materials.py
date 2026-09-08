"""Explicit friction contract for the current uniform-material robot adapter."""
import numpy as np


def native_contact_contract(model):
    colliders=(model.geom_contype!=0)|(model.geom_conaffinity!=0)
    values=np.unique(model.geom_friction[colliders],axis=0)
    priorities=np.unique(model.geom_priority[colliders])
    dimensions=np.unique(model.geom_condim[colliders])
    if len(values)!=1 or list(priorities)!=[0] or max(dimensions)>3:
        raise ValueError('This robot needs a per-collider friction/priority adapter; uniform sliding friction cannot represent it')
    return dict(native_friction=values[0].tolist(),native_priority=0,native_contact_dimensions=dimensions.tolist(),
        static_friction=float(values[0,0]),dynamic_friction=float(values[0,0]),combine_mode='max',
        scope='Uniform robot sliding friction with equal-priority maximum combination. Contact solver dynamics are not asserted identical.')


def bind_robot_contact_material(stage,root_path,contract):
    from pxr import Usd,UsdPhysics,UsdShade,Sdf
    if not contract:raise ValueError('Missing native robot contact contract; regenerate the motor import contract')
    root=stage.GetPrimAtPath(root_path)
    material=UsdShade.Material.Define(stage,str(root.GetPath())+'/NativeContactMaterial')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(contract['static_friction'])
    api.CreateDynamicFrictionAttr(contract['dynamic_friction'])
    api.CreateRestitutionAttr(0.)
    material.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    material.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode',Sdf.ValueTypeNames.Token).Set(contract['combine_mode'])
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
    checked=0
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if bound.GetPath()!=material.GetPath():raise ValueError(f'Robot collider has the wrong physics material: {prim.GetPath()}')
        checked+=1
    if checked==0:raise ValueError('No robot colliders were checked')
    return dict(contract=contract,colliders_checked=checked,material_path=str(material.GetPath()))


def check_solver_materials(properties,contract):
    properties=np.asarray(properties)
    expected=np.array([contract['static_friction'],contract['dynamic_friction'],0.])
    if properties.ndim!=2 or properties.shape[0]==0 or properties.shape[1]!=3 or not np.isfinite(properties).all():
        raise ValueError('Missing or nonfinite solver material properties')
    error=float(np.max(np.abs(properties-expected)))
    if error>1e-6:raise ValueError(f'PhysX is using different robot contact coefficients (max error {error})')
    return dict(backend_values_verified=True,physx_shape_count=len(properties),max_coefficient_error=error)


def check_solver_offsets(contact,rest,expected_contact=.001):
    contact=np.asarray(contact);rest=np.asarray(rest)
    if contact.size==0 or contact.shape!=rest.shape or not np.isfinite(contact).all() or not np.isfinite(rest).all():
        raise ValueError('Missing or nonfinite solver collision offsets')
    if not np.allclose(contact,expected_contact,atol=1e-7,rtol=0) or not np.allclose(rest,0.,atol=1e-7,rtol=0):
        raise ValueError('PhysX is using different collision offsets')
    return dict(backend_offsets_verified=True,contact_offset_m=float(contact.max()),rest_offset_m=float(rest.max()))
