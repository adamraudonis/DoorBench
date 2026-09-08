"""Versioned translation of native robot passive joint terms into PhysX.

Legacy explicit friction is retained for historical replay. backend-dry-v2 uses
the simulator's static/dynamic friction solver; it needs separate whole-robot
qualification and does not imply identical MuJoCo constraint compliance.
"""
import numpy as np

PROFILES = ('legacy-tanh-v1', 'backend-dry-v2')


def passive_profile(motors, joint_names, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown passive-joint profile')
    names = list(joint_names)
    if len(set(names)) != len(names) or set(names) != set(motors['passive']):
        raise ValueError('Exact original passive joint name coverage required')
    values = np.asarray([[motors['passive'][n][k] for k in
                         ('damping', 'friction', 'armature', 'stiffness', 'springref')]
                        for n in names], float)
    if not np.isfinite(values).all() or np.any(values[:, :4] < 0):
        raise ValueError('Invalid native passive joint coefficients')
    if np.any(values[:, 3] != 0):
        raise ValueError('Passive joint springs need a separately qualified adapter')
    damping, friction = values[:, 0].copy(), values[:, 1].copy()
    backend = np.zeros((len(names), 3))
    if profile == 'backend-dry-v2':
        backend[:] = np.column_stack([friction, friction, damping])
        damping[:] = 0.; friction[:] = 0.
    return dict(profile=profile, joint_names=names,
                explicit_damping=damping, explicit_friction=friction,
                backend_friction_properties=backend,
                native_armature=values[:, 2].copy())


def configure_backend(view, declaration):
    """Set SI [static effort, dynamic effort, viscous coefficient] and read back.

The caller supplies only plant configuration. Nothing here becomes actor input.
API absence or a changed property is a startup failure, never a silent fallback.
"""
    import torch
    target = torch.tensor(declaration['backend_friction_properties'][None], dtype=torch.float32)
    view.set_dof_friction_properties(target, torch.tensor([0], dtype=torch.int32))
    actual = view.get_dof_friction_properties().cpu().numpy().copy()
    if actual.shape != tuple(target.shape) or not np.array_equal(actual, target.numpy()):
        raise ValueError('PhysX passive joint property readback differs from the declared profile')
    return dict(profile=declaration['profile'], joint_names=declaration['joint_names'],
                backend_friction_properties=actual[0].tolist(),
                property_order=['static_friction_effort_Nm', 'dynamic_friction_effort_Nm',
                                'viscous_friction_Nm_s_per_rad'],
                explicit_damping=declaration['explicit_damping'].tolist(),
                explicit_friction=declaration['explicit_friction'].tolist(),
                scope='Passive adapter readback; whole-robot and grasp qualification required separately')
