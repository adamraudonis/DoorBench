"""Shared passive Door55 latch conversion from the verified Isaac opening fixture."""
def add_native_latch_tendon(stage):
    """Passive one-sided length constraint; angular gearing uses metres/degree."""
    import numpy as np
    from pxr import UsdPhysics,PhysxSchema
    prefix='/World/Door/Articulation/Joints/'
    prims={n:stage.GetPrimAtPath(prefix+n) for n in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')}
    scale=float(prims['leaf_latch_bolt_slide'].GetAttribute('doorbench:latch_coupling_scale').Get())
    for name,c in [('leaf_hinge',0.),('leaf_handle_hinge',-scale),('leaf_latch_bolt_slide',1.)]:
        prim=prims[name]
        if name=='leaf_hinge':
            PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim,'latch')
            axis=PhysxSchema.PhysxTendonAxisAPI(prim,'latch')
        else:axis=PhysxSchema.PhysxTendonAxisAPI.Apply(prim,'latch')
        angular=prim.IsA(UsdPhysics.RevoluteJoint)
        axis.CreateGearingAttr([c*np.pi/180 if angular else c])
        axis.CreateForceCoefficientAttr([c])
    root=PhysxSchema.PhysxTendonAxisRootAPI.Apply(prims['leaf_hinge'],'latch')
    root.CreateStiffnessAttr(0.)
    root.CreateDampingAttr(0.)
    root.CreateLimitStiffnessAttr(100000.)
    root.CreateLowerLimitAttr(0.)
    root.CreateUpperLimitAttr(10.)
    root.CreateRestLengthAttr(0.)
    return scale
