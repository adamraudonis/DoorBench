"""Robot-only tactile thumb relief in the retained position-effort subspace.

No door, world root, contact identity, active plant or dynamics step is accepted.
The local taxel force resultant includes shear: it is not an oracle surface
normal. Physical contact/anatomy qualification remains an independent audit.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from .robot_thumb_flexion_force import RobotThumbFlexionForce

NAMES = tuple('rh_THJ' + str(k) for k in (5, 4, 3, 2, 1))
PROFILE = {
    'schema': 'doorbench.thumb-normal-admittance.v1',
    'start_s': 23., 'duration_s': 36., 'physics_dt_s': .002,
    'local_palmar_target_N': 3., 'filter_tau_s': .02,
    'force_admittance_m_per_N_s': .00015,
    'maximum_reference_speed_m_s': .0005,
    'maximum_reference_offset_m': .002,
    'position_error_gain_per_s': 2., 'orientation_error_gain_per_s': 1.,
    'maximum_joint_goal_offset_rad': .08,
    'maximum_joint_goal_speed_rad_s': .06,
    'maximum_joint_goal_acceleration_rad_s2': .2,
    'joint_interior_margin_rad': .01,
    'th2_interior_margin_rad': .035,
    'interior_velocity_gain_per_s': .5,
    'stop_before_actual_th2_margin_rad': .015,
    'maximum_thumb_frame_turn_rad': .12,
    'position_subspace': 'THJ5/4/3 plus K_inverse times the THJ2/1 effort-tangent vector',
    'scope': 'instance-specific sensor-balanced motor route plus local tactile thumb relief; no learned or vision claim',
}
PROFILE_V2=dict(PROFILE,schema='doorbench.thumb-normal-admittance.v2',
    interior_policy='Least-squares soft interior velocity desires when incompatible; original hard motion bounds and actual stop guard retained')


def validate_profile(value):
    expected=PROFILE_V2 if type(value) is dict and value.get('schema')==PROFILE_V2['schema'] else PROFILE
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError('Exact versioned thumb admittance profile required')
    for key, wanted in expected.items():
        actual = value[key]
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError('Unsupported thumb admittance setting: ' + key)
    return dict(value)


def intersect_scalar_bounds(column, low, high):
    """Pull independent joint bounds through a one-dimensional position span."""
    lower, upper = -np.inf, np.inf
    for scale, lo, hi in zip(column, low, high):
        if abs(scale) < 1e-12:
            if lo > 1e-12 or hi < -1e-12:
                raise ValueError('Unactuated thumb posture direction cannot satisfy its bound')
            continue
        a, b = sorted((lo / scale, hi / scale))
        lower, upper = max(lower, a), min(upper, b)
    if lower > upper + 1e-12:
        raise ValueError('Retained thumb posture span has incompatible bounds')
    return lower, upper


def soft_interior_coefficient(column,low,high,hard_lower,hard_upper):
    """Exact scalar convex hinge least squares inside unchanged hard bounds."""
    c=np.asarray(column,float);lo=np.asarray(low,float);hi=np.asarray(high,float)
    if (c.shape!=(2,) or lo.shape!=(2,) or hi.shape!=(2,) or
            not np.isfinite(np.r_[c,lo,hi,hard_lower,hard_upper]).all() or
            hard_lower>hard_upper or np.any(lo>hi)):
        raise ValueError('Finite feasible hard bounds and two soft desires required')
    knots=[hard_lower,hard_upper]
    for scale,l,h in zip(c,lo,hi):
        if abs(scale)>1e-12:knots.extend(np.clip([l/scale,h/scale],hard_lower,hard_upper))
    knots=np.unique(knots);candidates=list(knots)
    for a,b in zip(knots[:-1],knots[1:]):
        mid=(a+b)/2;v=c*mid;active=(v<lo)|(v>hi);target=np.clip(v,lo,hi)
        norm=float(c[active]@c[active])
        if norm>0:candidates.append(float(np.clip(c[active]@target[active]/norm,a,b)))
    def cost(x):
        v=c*x;return float(np.sum((v-np.clip(v,lo,hi))**2))
    return float(min(candidates,key=cost))


class RobotThumbNormalAdmittance:
    def __init__(self, model, joint_names, action_names, transmission, profile, original_motor_contract):
        self.profile = validate_profile(profile)
        self.mapper = RobotThumbFlexionForce(model, joint_names, action_names, transmission)
        self.m, self.names = model, tuple(joint_names)
        self.d = mujoco.MjData(model)
        self.qa = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.columns = np.array([self.names.index(n) for n in NAMES])
        self.va = np.array([model.joint(n).dofadr[0] for n in NAMES])
        self.limits = model.jnt_range[[model.joint(n).id for n in NAMES]].copy()
        self.thumb = model.site('rh_thdistal_touch').id
        self.palm = model.site('rh_palm_touch').id
        self.jp = np.zeros((3, model.nv)); self.jr = self.jp.copy()
        motors = original_motor_contract['actuators']
        if ([r['name'] for r in motors] != list(action_names)
                or not np.array_equal([r['force_range'] for r in motors], model.actuator_forcerange)):
            raise ValueError('Original motor contract order and force caps required')
        native_gain = np.array([r['kp'] for r in motors], float)
        if native_gain.shape != (61,) or not np.isfinite(native_gain).all() or np.any(native_gain <= 0):
            raise ValueError('Original positive motor gain calibration required')
        # SensorBalanceController's unstepped model uses force-normalized
        # affine motors. Its model gain array is not the actual policy gain.
        # Read the original contract admitted by that controller, then require
        # exact agreement with the live post-ramp policy before every use.
        self.gain = 16. * native_gain[self.mapper.groups['th'][1]]
        self.reset()

    def reset(self):
        self.last_time = None; self.nominal = None; self.target = None
        self.previous_velocity = np.zeros(5); self.reference_offset = np.zeros(3)
        self.filtered_force = None; self.filtered_vector = None
        self.anchor_position = None; self.anchor_rotation = None
        self.failed_reason = None

    def geometry(self, q):
        q = np.asarray(q, float)
        if q.shape != (69,) or not np.isfinite(q).all():
            raise ValueError('Exactly69 finite own encoders required')
        self.d.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        rp = self.d.site_xmat[self.palm].reshape(3, 3)
        rt = self.d.site_xmat[self.thumb].reshape(3, 3)
        position = rp.T @ (self.d.site_xpos[self.thumb] - self.d.site_xpos[self.palm])
        rotation = rp.T @ rt
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.thumb)
        # The five controlled joints are descendants of the fixed palm, so
        # the palm's Jacobian columns are zero in this thumb-only solve.
        return position.copy(), rotation.copy(), rp.T @ self.jp[:, self.va], rp.T @ self.jr[:, self.va]

    def basis(self, q):
        unit, _ = self.mapper.motor_bias(q, np.ones(5))
        pressure = unit[self.mapper.groups['th'][1]]
        if np.linalg.norm(pressure) < 1e-5:
            raise ValueError('Degenerate thumb pressure direction')
        tangent = np.array([pressure[1], -pressure[0]]) / self.gain
        if tangent[0] < 0: tangent *= -1
        tangent /= np.max(abs(tangent))
        basis = np.zeros((5, 4)); basis[:3, :3] = np.eye(3); basis[3:, 3] = tangent
        return basis, pressure

    def update(self, goals, q, force_sensor_xyz, local_palmar_force_N, *, now_s):
        if self.failed_reason is not None:
            raise RuntimeError('Rejected thumb admittance requires episode reset: ' + self.failed_reason)
        try:
            return self._update(goals, q, force_sensor_xyz, local_palmar_force_N, now_s=now_s)
        except Exception as exc:
            self.failed_reason = type(exc).__name__ + ': ' + str(exc)
            raise

    def _update(self, goals, q, force_sensor_xyz, load, *, now_s):
        if isinstance(now_s, (bool, np.bool_)) or not np.isscalar(now_s) or not np.isfinite(now_s):
            raise ValueError('Finite scalar local clock required')
        t = float(now_s); dt = .002
        if t < 23.-1e-9 or (self.last_time is None and abs(t-23.) > 1e-8) or (self.last_time is not None and abs(t-self.last_time-dt) > 1e-8):
            raise ValueError('Admittance begins at23s then consumes every2ms decision')
        vector = np.asarray(force_sensor_xyz, float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or isinstance(load, (bool,np.bool_)) or not np.isscalar(load) or not np.isfinite(load) or load < 0:
            raise ValueError('Finite own taxel resultant and nonnegative palmar projection required')
        if type(goals) is not dict or not set(NAMES) <= set(goals):
            raise ValueError('Named original thumb goals required')
        nominal = np.array([goals[n] for n in NAMES], float)
        if not np.isfinite(nominal).all(): raise ValueError('Finite thumb goals required')
        q = np.asarray(q, float); position, rotation, jp, jr = self.geometry(q)
        measured_margin = q[self.columns]-self.limits[:,0]
        if measured_margin[3] < self.profile['stop_before_actual_th2_margin_rad']:
            raise ValueError('Thumb admittance could not preserve15mrad actual THJ2 interior margin')
        if self.nominal is None:
            self.nominal = nominal.copy(); self.target = nominal.copy()
            self.anchor_position = position.copy(); self.anchor_rotation = rotation.copy()
            self.filtered_force = float(load); self.filtered_vector = vector.copy()
            self.last_time = t
            return dict(goals), dict(thumb_admittance_started=True, thumb_admittance_reference_offset_m=[0.,0.,0.],
                                    thumb_admittance_goal_velocity_rad_s=[0.]*5, thumb_admittance_scope=PROFILE['scope'],
                                    thumb_admittance_pressure_policy_gain_Nm_rad=self.gain.tolist())
        alpha = -np.expm1(-dt / self.profile['filter_tau_s'])
        self.filtered_force += alpha * (load-self.filtered_force)
        self.filtered_vector += alpha * (vector-self.filtered_vector)
        frame_error = Rotation.from_matrix(self.anchor_rotation @ rotation.T).as_rotvec()
        if np.linalg.norm(frame_error) > self.profile['maximum_thumb_frame_turn_rad']:
            raise ValueError('Thumb orientation left its declared local envelope')
        # Positive local taxel force points into the thumb, away from the loaded
        # pad surface. No contact counterpart or object normal is available.
        force_length = np.linalg.norm(self.filtered_vector)
        direction = rotation[:,2].copy()
        if force_length > .2:
            direction = rotation @ (self.filtered_vector/force_length)
            if direction @ rotation[:,2] < .2:
                raise ValueError('Measured resultant does not support a palmar relief direction')
        reference_velocity = np.zeros(3)
        if load >= .2:
            speed = np.clip(self.profile['force_admittance_m_per_N_s']*(self.filtered_force-3.),
                            -self.profile['maximum_reference_speed_m_s'], self.profile['maximum_reference_speed_m_s'])
            proposed = self.reference_offset + dt*speed*direction
            radius = self.profile['maximum_reference_offset_m']
            if np.linalg.norm(proposed) > radius: proposed *= radius/np.linalg.norm(proposed)
            reference_velocity = (proposed-self.reference_offset)/dt
            self.reference_offset = proposed
        desired_velocity = reference_velocity + 2.*(self.anchor_position+self.reference_offset-position)
        basis, pressure = self.basis(q)
        max_velocity = self.profile['maximum_joint_goal_speed_rad_s']
        max_acceleration = self.profile['maximum_joint_goal_acceleration_rad_s2']
        low = np.maximum(-max_velocity, self.previous_velocity-max_acceleration*dt)
        high = np.minimum(max_velocity, self.previous_velocity+max_acceleration*dt)
        goal_lo = np.maximum(self.limits[:,0]+.005, self.nominal-.08)
        goal_hi = np.minimum(self.limits[:,1]-.005, self.nominal+.08)
        low = np.maximum(low, (goal_lo-self.target)/dt)
        high = np.minimum(high, (goal_hi-self.target)/dt)
        # Interior recovery has priority over the least-squares pad target.
        # At a newly active barrier the required velocity is limited by the
        # already-declared acceleration envelope; the remaining deficit is
        # reported rather than causing an instantaneous target jump.
        interior = np.full(5, .01); interior[3] = .035
        need_lo = .5*(self.limits[:,0]+interior-q[self.columns])
        need_hi = .5*(self.limits[:,1]-interior-q[self.columns])
        low[:3] = np.maximum(low[:3], np.minimum(need_lo[:3], high[:3]))
        high[:3] = np.minimum(high[:3], np.maximum(need_hi[:3], low[:3]))
        lower = list(low[:3]); upper = list(high[:3])
        a,b = intersect_scalar_bounds(basis[3:,3], low[3:], high[3:])
        soft_conflict=False
        try:
            required_a,required_b = intersect_scalar_bounds(basis[3:,3], need_lo[3:], need_hi[3:])
            a,b = max(a,min(required_a,b)),min(b,max(required_b,a))
        except ValueError:
            if self.profile['schema']!=PROFILE_V2['schema']:raise
            # Desired inward velocities are soft objectives, not joint limits.
            # They can conflict in the single actuated flexion tangent. Admit
            # only the best compromise within every original hard bound.
            a=b=soft_interior_coefficient(basis[3:,3],need_lo[3:],need_hi[3:],a,b)
            soft_conflict=True
        lower.append(a); upper.append(b)
        lower,upper = np.asarray(lower),np.asarray(upper)
        if np.any(lower > upper+1e-12): raise ValueError('Infeasible bounded thumb velocity')
        # Position is primary; preserving the initial pad frame discourages
        # rolling onto an unqualified surface without inventing an object pose.
        matrix = np.vstack((1000.*jp@basis, .2*jr@basis, .001*np.eye(4)))
        rhs = np.r_[1000.*desired_velocity, .2*frame_error, np.zeros(4)]
        # scipy requires strictly separated bounds. Fixed coordinates remain
        # fixed by elimination, not by artificially widening an envelope.
        fixed = upper-lower <= 1e-12; coefficient = (lower+upper)/2
        if np.any(~fixed):
            fit = lsq_linear(matrix[:,~fixed], rhs-matrix[:,fixed]@coefficient[fixed],
                             bounds=(lower[~fixed],upper[~fixed]), method='bvls',tol=1e-11,max_iter=30)
            if not fit.success: raise ValueError('Bounded thumb velocity solve failed')
            coefficient[~fixed] = fit.x
        velocity = basis@coefficient
        self.target += dt*velocity; self.previous_velocity = velocity.copy(); self.last_time = t
        result = dict(goals); result.update(dict(zip(NAMES,self.target.tolist())))
        pressure_tangent_error = float(abs(pressure @ (self.gain*velocity[3:])))
        return result, dict(thumb_admittance_started=True, thumb_admittance_reference_offset_m=self.reference_offset.tolist(),
            thumb_admittance_soft_interior_conflict=soft_conflict,
            thumb_admittance_reference_velocity_m_s=reference_velocity.tolist(), thumb_admittance_goal_velocity_rad_s=velocity.tolist(),
            thumb_admittance_filtered_local_load_N=self.filtered_force, thumb_admittance_resultant_sensor_N=self.filtered_vector.tolist(),
            thumb_admittance_direction_palm=direction.tolist(), thumb_admittance_position_error_m=(self.anchor_position+self.reference_offset-position).tolist(),
            thumb_admittance_actual_margins_rad=measured_margin.tolist(), thumb_admittance_interior_velocity_deficit_rad_s=np.maximum(0.,need_lo-velocity).tolist(),
            thumb_admittance_effort_tangent_residual=pressure_tangent_error, thumb_admittance_frozen_base_goals_rad=self.nominal.tolist(),
            thumb_admittance_pressure_policy_gain_Nm_rad=self.gain.tolist(),
            thumb_admittance_calculator_time_s=self.d.time, thumb_admittance_scope=PROFILE['scope'])
