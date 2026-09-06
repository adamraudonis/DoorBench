"""Presentation-only material binding for the native Isaac camera.
No collision geometry, physical materials, joints or robot state are changed.
"""


def apply_hero_floor(stage):
    from pxr import Sdf, UsdShade, UsdLux, UsdGeom, UsdPhysics, Usd, Gf

    ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
    colliders = [p for p in Usd.PrimRange(ground) if p.HasAPI(UsdPhysics.CollisionAPI)]

    def physical_bindings():
        return [
            str(
                UsdShade.MaterialBindingAPI(p)
                .ComputeBoundMaterial("physics")[0]
                .GetPath()
            )
            for p in colliders
        ]

    before = physical_bindings()
    material = UsdShade.Material.Define(stage, "/World/HeroFloorMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/HeroFloorMaterial/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.07, 0.095, 0.12)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
    UsdShade.MaterialBindingAPI.Apply(ground).Bind(
        material, UsdShade.Tokens.strongerThanDescendants
    )

    if physical_bindings() != before:
        raise RuntimeError("Presentation material altered the floor physics binding")
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/defaultDomeLight"))
    dome.GetIntensityAttr().Set(2000)
    key = UsdLux.DistantLight.Define(stage, "/World/HeroKeyLight")
    key.CreateIntensityAttr(750)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -25))
