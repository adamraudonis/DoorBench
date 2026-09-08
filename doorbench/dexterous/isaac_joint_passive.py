"""Versioned translation of native robot passive joint terms into PhysX.

Legacy explicit friction is retained for historical replay. backend-dry-v2 uses
the simulator's static/dynamic friction solver; it needs separate whole-robot
qualification and does not imply identical MuJoCo constraint compliance.
"""
import copy
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
    expected_armature = np.asarray(declaration['native_armature'], dtype=np.float32)[None]
    armature = view.get_dof_armatures().cpu().numpy().copy()
    if armature.shape != expected_armature.shape or not np.array_equal(armature, expected_armature):
        raise ValueError('PhysX armature readback differs from original native coefficients')
    return dict(profile=declaration['profile'], joint_names=declaration['joint_names'],
                backend_friction_properties=actual[0].tolist(),
                native_armature_readback_kg_m2=armature[0].tolist(),
                property_order=['static_friction_effort_Nm', 'dynamic_friction_effort_Nm',
                                'viscous_friction_Nm_s_per_rad'],
                explicit_damping=declaration['explicit_damping'].tolist(),
                explicit_friction=declaration['explicit_friction'].tolist(),
                scope='Passive adapter readback; whole-robot and grasp qualification required separately')


class PassivePropertyInvariant:
    """Read-only interval guard for the opt-in profile; no property repair.

    Call after every physical step using the episode-relative interval end.
    On any mismatch, save receipt() before propagating the error. A failed
    interval remains named even if the ordinary state recorder has not yet
    appended it. Historical profiles must not instantiate this guard.
    """
    def __init__(self, declaration, *, physics_dt_s=.002):
        if declaration['profile'] != 'backend-dry-v2':
            raise ValueError('Per-interval passive guard is opt-in backend-dry-v2 only')
        if type(physics_dt_s) not in (int, float) or physics_dt_s != .002:
            raise ValueError('Passive guard requires the declared 2ms interval')
        self.names = tuple(declaration['joint_names'])
        self.expected = np.asarray(declaration['backend_friction_properties'], dtype=np.float32)[None].copy()
        self.armature = np.asarray(declaration['native_armature'], dtype=np.float32)[None].copy()
        if (not self.names or len(set(self.names)) != len(self.names) or
                self.expected.shape != (1, len(self.names), 3) or self.armature.shape != (1, len(self.names)) or
                not np.isfinite(self.expected).all() or not np.isfinite(self.armature).all() or
                np.any(self.expected < 0) or np.any(self.armature < 0) or
                np.any(np.asarray(declaration['explicit_damping']) != 0) or
                np.any(np.asarray(declaration['explicit_friction']) != 0)):
            raise ValueError('Invalid exact passive configuration or duplicated explicit passive terms')
        self.dt = float(physics_dt_s)
        self.attempts = self.checked = self.valid = 0
        self.last_time = None
        self.max_difference = np.zeros(3)
        self.max_armature_difference = 0.
        self.first_failure = None

    def check(self, view, *, time_s):
        if self.first_failure is not None:
            raise ValueError('Passive property guard already failed; preserve the failed prefix')
        self.attempts += 1
        actual = armature = None
        expected_time = (self.checked + 1) * self.dt
        finite_time = type(time_s) in (int, float) and np.isfinite(time_s)
        try:
            if not finite_time or abs(time_s - expected_time) > 1e-8:
                raise ValueError('Missing, repeated or invalid passive-property interval clock')
            # Snapshot both getter results immediately: backend buffers may be reused.
            actual = view.get_dof_friction_properties().cpu().numpy().copy()
            armature = view.get_dof_armatures().cpu().numpy().copy()
            self.checked += 1
            self.last_time = float(time_s)
            if (actual.shape != self.expected.shape or armature.shape != self.armature.shape or
                    not np.isfinite(actual).all() or not np.isfinite(armature).all()):
                raise ValueError('Missing, malformed or nonfinite passive property readback')
            self.max_difference = np.maximum(self.max_difference,
                np.max(abs(actual.astype(float)-self.expected.astype(float)), axis=(0, 1)))
            self.max_armature_difference = max(self.max_armature_difference,
                float(np.max(abs(armature.astype(float)-self.armature.astype(float)))))
            if not np.array_equal(actual, self.expected) or not np.array_equal(armature, self.armature):
                raise ValueError('Configured passive properties changed during the physical interval')
            self.valid += 1
        except Exception as exc:
            def safe_values(value):
                if value is None: return None
                if not np.isfinite(value).all(): return dict(shape=list(value.shape), nonfinite=True)
                return value.tolist()
            self.first_failure = dict(interval_index=self.attempts,
                expected_interval_end_s=expected_time, observed_interval_end_s=float(time_s) if finite_time else None,
                reason=type(exc).__name__ + ': ' + str(exc),
                backend_friction_properties=safe_values(actual), native_armature=safe_values(armature))
            raise

    def receipt(self):
        return dict(schema='doorbench.passive-property-invariant.v1', profile='backend-dry-v2',
            passed=self.first_failure is None and self.valid > 0,
            attempted_intervals=self.attempts, checked_intervals=self.checked, valid_intervals=self.valid,
            last_checked_interval_end_s=self.last_time, physics_dt_s=self.dt,
            joint_names=list(self.names), expected_float32_properties=self.expected[0].tolist(),
            property_order=['static_friction_effort_Nm', 'dynamic_friction_effort_Nm', 'viscous_friction_Nm_s_per_rad'],
            maximum_absolute_property_difference=self.max_difference.tolist(),
            expected_float32_armature_kg_m2=self.armature[0].tolist(),
            maximum_armature_difference_kg_m2=self.max_armature_difference,
            first_failure=copy.deepcopy(self.first_failure),
            scope='Read-only per-interval property persistence; no active plant repair, forces or controller inputs')
