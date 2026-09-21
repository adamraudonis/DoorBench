"""Render-only corrections for the MJCF importer's fallback material bindings.

All changes live in the stage's session layer. No asset is saved, no collision
visibility is changed, and no physics material or simulation setting is edited.
The caller should retain the returned audit with its recorded frames.
"""


def restore_robot_visual_materials(stage, root_path):
    """Restore authored materials shadowed by an imported white fallback.

    The H1 importer binds ``material_black`` to a visual mesh's parent, then
    binds ``DefaultMaterial`` to the mesh itself. USD's normal descendant
    precedence makes the robot white. Strengthen the original parent binding,
    preserving the different torso/logo colors and the Shadow hand materials.
    Only affected visual instances are expanded so those bindings are editable.
    """
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"Missing robot root: {root_path}")
    repairs = []
    instance_paths = set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim) or '/visuals/' not in str(prim.GetPath()):
            continue
        actual, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not actual or actual.GetPrim().GetName() != 'DefaultMaterial':
            continue
        parent = prim.GetParent()
        targets = parent.GetRelationship('material:binding').GetTargets()
        if len(targets) != 1 or targets[0] == actual.GetPath():
            continue
        intended = UsdShade.Material(stage.GetPrimAtPath(targets[0]))
        if not intended:
            raise ValueError(f"Unresolved intended visual material: {targets[0]}")
        ancestor = parent
        while ancestor.IsInstanceProxy():
            ancestor = ancestor.GetParent()
        if ancestor.IsInstance():
            # Refuse to expand any subtree containing simulation schemas.
            if ancestor.GetName() != 'visuals':
                raise ValueError(f"Unexpected visual instance: {ancestor.GetPath()}")
            for child in Usd.PrimRange(ancestor, Usd.TraverseInstanceProxies()):
                if any('physics' in schema.lower() or 'physx' in schema.lower()
                       for schema in child.GetAppliedSchemas()):
                    raise ValueError(f"Physics schema inside visual instance: {child.GetPath()}")
            instance_paths.add(str(ancestor.GetPath()))
        repairs.append(dict(mesh_path=str(prim.GetPath()), binding_path=str(parent.GetPath()),
                            previous_material=str(actual.GetPath()), material=str(targets[0])))

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in sorted(instance_paths):
            stage.GetPrimAtPath(path).SetInstanceable(False)
        for repair in repairs:
            material = UsdShade.Material(stage.GetPrimAtPath(repair['material']))
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(repair['binding_path'])).Bind(
                material, UsdShade.Tokens.strongerThanDescendants)
        for repair in repairs:
            actual, _ = UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(repair['mesh_path'])).ComputeBoundMaterial()
            if str(actual.GetPath()) != repair['material']:
                raise ValueError(f"Visual material repair did not compose: {repair['mesh_path']}")
    return dict(scope='Session-layer visual binding correction only',
                repaired_meshes=len(repairs), expanded_visual_instances=len(instance_paths),
                repairs=repairs)


def configure_diagnostic_lighting(stage, dome_path, *, key_path='/World/DiagnosticKey',
                                  dome_intensity=250., key_intensity=600.):
    """Use a subdued fill and angled key, without changing surface colors.

    This is a presentation preset; its settings belong in frame provenance.
    Camera exposure and all physics properties remain untouched.
    """
    import math
    from pxr import Gf, Usd, UsdGeom, UsdLux

    if not all(math.isfinite(value) and value >= 0
               for value in (dome_intensity, key_intensity)):
        raise ValueError('Light intensities must be finite and nonnegative')
    dome = UsdLux.DomeLight(stage.GetPrimAtPath(dome_path))
    if not dome:
        raise ValueError(f'Missing dome light: {dome_path}')
    if stage.GetPrimAtPath(key_path):
        raise ValueError(f'Diagnostic key path already exists: {key_path}')
    previous = dome.GetIntensityAttr().Get()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome.CreateIntensityAttr(float(dome_intensity))
        key = UsdLux.DistantLight.Define(stage, key_path)
        key.CreateIntensityAttr(float(key_intensity))
        key.CreateAngleAttr(2.)
        key.CreateColorAttr(Gf.Vec3f(1., .97, .92))
        # Directional light rays travel along local -Z.
        direction = Gf.Vec3d(-4., 5., -7.).GetNormalized()
        rotation = Gf.Rotation(Gf.Vec3d(0., 0., -1.), direction).GetQuat()
        UsdGeom.Xformable(key).AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation)
    return dict(scope='Session-layer lighting only', dome_path=dome_path,
                previous_dome_intensity=previous, dome_intensity=dome_intensity,
                key_path=key_path, key_intensity=key_intensity,
                key_color=[1., .97, .92], key_ray_direction=list(direction), key_angle_degrees=2.)
