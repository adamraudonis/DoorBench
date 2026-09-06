"""Force-only native service gate for the original marine hook holdback.

The external 80 Nm closing load is a test load, not a person. Both repeated
opening/release/closing cycles use forces on real operator surfaces. This gate
does not certify an embodied task, marine pressure, strength or durability.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict

import numpy as np

_CACHE = OrderedDict()


def run_ship_holdback_qa(model, meta):
    """Undog, capture, release both hands, unload/lift the hook, and close twice."""
    import mujoco
    from .native_warnings import capture_native_warnings

    if meta.get('family') != 'ship_watertight':
        return {'applicable': False, 'ok': True, 'scope': 'No marine holdback'}
    hb = meta.get('ship_holdback')
    if not isinstance(hb, dict) or hb.get('schema_version') != 1:
        raise ValueError('A physical marine holdback is required')
    mounts = meta.get('marine_dog_mounts', [])
    if len(mounts) < 4:
        raise ValueError('Marine service test requires every real dog')
    leaf = model.joint(hb['leaf_joint']).id
    hook = model.joint(hb['hook_joint']).id
    hook_body = model.body(hb['hook_body']).id
    station = model.body(hb['station_body']).id
    hook_site = model.site(hb['release_site']).id
    jaw = model.geom(hb['load_face_geom']).id
    shoulder = model.geom(hb['load_shoulder_geom']).id
    shoulder_arm = model.geom(hb['load_shoulder_moving_geom']).id
    striker = model.geom(hb['striker_geom']).id
    opening_stops = [model.geom(name).id for name in hb['opening_stop_geoms']]
    if (len(opening_stops) != 2 or len(set(opening_stops)) != 2
            or any(model.geom_bodyid[g] != station for g in opening_stops)):
        raise ValueError('Opening stops must be distinct fixed station contact faces')
    carrier = int(model.geom_bodyid[striker])
    while carrier and carrier != model.jnt_bodyid[leaf]:
        carrier = int(model.body_parentid[carrier])
    if carrier != model.jnt_bodyid[leaf]:
        raise ValueError('Holdback striker must be carried by the actual leaf')
    if model.tendon_stiffness[model.tendon(hb['spring']).id] <= 0:
        raise ValueError('The declared return spring must have native stiffness')
    if (model.jnt_bodyid[hook] != hook_body or model.site_bodyid[hook_site] != hook_body
            or model.geom_bodyid[jaw] != hook_body
            or model.geom_bodyid[shoulder_arm] != hook_body
            or model.geom_bodyid[shoulder] != station
            or model.body_parentid[hook_body] != station):
        raise ValueError('Holdback joint, grip or load-bearing geometry binding differs')
    parent = station
    while parent:
        if model.body_jntnum[parent]:
            raise ValueError('Holdback station requires a fixed structural support')
        parent = int(model.body_parentid[parent])
    if model.jnt_type[hook] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError('Holdback release requires its physical pivot')
    linkage = meta.get('marine_dog_linkage')
    inputs = ([(linkage['input_joint'], 'wheel_grip_n', 6.)] if linkage else
              [(row['joint'], row['body'] + '_grip', 2.) for row in mounts])
    leaf_site = model.site(inputs[0][1]).id
    la, lv = int(model.jnt_qposadr[leaf]), int(model.jnt_dofadr[leaf])
    ha, hv = int(model.jnt_qposadr[hook]), int(model.jnt_dofadr[hook])
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    digest = hashlib.sha256(binary).hexdigest()
    key = hashlib.sha256(digest.encode() + json.dumps(
        {'version': 4, 'holdback': hb, 'mounts': mounts, 'linkage': linkage}, sort_keys=True).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        result = copy.deepcopy(_CACHE[key]); result['cache_hit'] = True
        return result
    d = mujoco.MjData(model)
    depth = peak = loop = gear = 0.
    depth_event = None
    trace, cycles, failures = [], [], []
    phase = 'initial'; next_sample = 0.; forces = {}

    def angle(name):
        return float(d.qpos[model.jnt_qposadr[model.joint(name).id]])

    def contact_load(geom_pair=None, participant=None):
        total = 0.
        for k, contact in enumerate(d.contact):
            ids = {int(contact.geom1), int(contact.geom2)}
            if (geom_pair is not None and ids == set(geom_pair)) or participant in ids:
                force = np.zeros(6); mujoco.mj_contactForce(model, d, k, force)
                total += max(0., float(force[0]))
        return total

    def apply(joint, site, effort):
        nonlocal peak
        tangent = np.cross(d.xaxis[joint], d.site_xpos[site] - d.xanchor[joint])
        radius = float(np.linalg.norm(tangent))
        if radius < .02:
            raise ValueError('Physical hand surface has insufficient turning radius')
        force = tangent * np.clip(effort, -120 * radius, 120 * radius) / (radius * radius)
        peak = max(peak, float(np.linalg.norm(force)))
        forces[model.site(site).name] = force.tolist()
        mujoco.mj_applyFT(model, d, force, np.zeros(3), d.site_xpos[site],
                         model.site_bodyid[site], d.qfrc_applied)

    def measure():
        nonlocal depth, depth_event, loop, gear
        for contact in d.contact:
            if -contact.dist > depth:
                depth = -float(contact.dist)
                depth_event = {'time_s': float(d.time), 'phase': phase, 'depth_m': depth,
                               'geoms': [model.geom(contact.geom1).name, model.geom(contact.geom2).name]}
        if linkage:
            for name in linkage['connect_equalities']:
                e = model.equality(name).id
                a, b = model.eq_obj1id[e], model.eq_obj2id[e]
                delta = (d.xpos[a] + d.xmat[a].reshape(3, 3) @ model.eq_data[e, :3]
                         - d.xpos[b] - d.xmat[b].reshape(3, 3) @ model.eq_data[e, 3:6])
                loop = max(loop, float(np.linalg.norm(delta)))
            gear = max(gear, abs(angle(linkage['output_joint']) - angle(linkage['input_joint']) / 6))
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('nonfinite_native_state')
        if messages or any(w.number for w in d.warning):
            raise RuntimeError('native_solver_warning')
        if depth > .003:
            raise RuntimeError('native_contact_exceeds_3mm_abort')

    def tick(action, load=0.):
        nonlocal next_sample, forces
        # Preserve the previous physical force while refreshing contact loads.
        mujoco.mj_forward(model, d)
        d.qfrc_applied[:] = 0.; forces = {}; action()
        d.qfrc_applied[lv] += load
        mujoco.mj_step(model, d); measure()
        if d.time >= next_sample:
            trace.append({'time_s': float(d.time), 'phase': phase,
                          'leaf_q': float(d.qpos[la]), 'hook_q': float(d.qpos[ha]),
                          'jaw_striker_load_N': contact_load((jaw, striker)),
                          'shoulder_load_N': contact_load((shoulder, shoulder_arm)),
                          'site_forces': copy.deepcopy(forces), 'external_closing_load_Nm': load})
            next_sample += .1

    def quintic(value):
        u = float(np.clip(value, 0., 1.))
        return u ** 3 * (10 - 15 * u + 6 * u * u)

    def leaf_hand(target, closing=False):
        apply(leaf, leaf_site, 70 * (target - d.qpos[la]) - 16 * d.qvel[lv]
              + model.dof_frictionloss[lv] * (-1 if closing else 1))

    with capture_native_warnings() as messages:
        try:
            for name, site_name, duration in inputs:
                phase = 'undog'
                j = model.joint(name).id; a, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
                site = model.site(site_name).id; start = float(d.time); initial = angle(name)
                for _ in range(round(duration / model.opt.timestep)):
                    target = initial + (model.jnt_range[j, 1] - initial) * quintic((d.time - start) / (4 if linkage else 1.4))
                    kp, kd = (12, 1.5) if linkage else (50, 3)
                    tick(lambda: apply(j, site, kp * (target - d.qpos[a]) - kd * d.qvel[v]
                                       + (0 if linkage else model.dof_frictionloss[v])))
            if min(angle(row['joint']) for row in mounts) < 1.45:
                raise RuntimeError('not_all_dogs_released')
            for cycle in range(2):
                record = {'cycle': cycle + 1}
                phase = 'open'; start = float(d.time); initial = float(d.qpos[la])
                target_open = hb['full_open_angle_rad'] - .010
                for _ in range(round(10 / model.opt.timestep)):
                    target = initial + (target_open - initial) * quintic((d.time - start) / 8)
                    tick(lambda: leaf_hand(target))
                record['opened_rad'] = float(d.qpos[la])
                if abs(d.qpos[la] - target_open) > .08:
                    failures.append('leaf_did_not_reach_capture_station')
                phase = 'hold'; start = float(d.time)
                held, jaw_load_s, shoulder_load_s = [], 0., 0.
                for _ in range(round(2 / model.opt.timestep)):
                    tick(lambda: None, -80.)
                    if d.time - start >= 1.5:
                        held.append(float(d.qpos[la]))
                        jaw_load_s += model.opt.timestep * (contact_load((jaw, striker)) > 5)
                        shoulder_load_s += model.opt.timestep * (contact_load((shoulder, shoulder_arm)) > 5)
                record.update(held_rad=float(d.qpos[la]), hold_tail_drift_rad=float(np.ptp(held)),
                              hands_free_s=2., closing_test_load_Nm=80., jaw_load_observed_s=float(jaw_load_s),
                              shoulder_load_observed_s=float(shoulder_load_s), hook_held_rad=float(d.qpos[ha]))
                if not hb['nominal_capture_angle_rad'] - .09 < d.qpos[la] < hb['full_open_angle_rad']:
                    failures.append('holdback_did_not_retain_open_leaf')
                if np.ptp(held) > .001 or jaw_load_s < .1 or shoulder_load_s < .1:
                    failures.append('hands_free_hook_and_shoulder_load_path_not_stable')
                if d.qpos[ha] <= model.jnt_range[hook, 0] + .01:
                    failures.append('hook_rests_on_numerical_limit_instead_of_shoulder')
                # Remove the artificial closing test load before manual
                # release; the real leaf hand makes clearance for the hook.
                phase = 'unload_release'; start = float(d.time)
                held_leaf = float(d.qpos[la])
                unloaded_s = 0.; release_start = None; hook_peak = 0.; stop_peak = stop_load_s = 0.
                for _ in range(round(4.5 / model.opt.timestep)):
                    mujoco.mj_forward(model, d)
                    unloaded_s = unloaded_s + model.opt.timestep if contact_load((jaw, striker)) < 5 else 0.
                    if (release_start is None and d.time - start >= .3 and unloaded_s >= .1
                            and d.qpos[la] >= held_leaf + .035):
                        release_start = float(d.time)
                    target = 0. if release_start is None else .85 * quintic(d.time - release_start)
                    # Take up the pocket clearance gradually before lifting
                    # the unloaded hook. A step to the distant safety target
                    # gave unnecessary kilonewton bumper impacts in the first
                    # service schedule despite its capped actual hand force.
                    leaf_target = held_leaf + (target_open - held_leaf) * quintic((d.time - start) / 2.5)
                    def release():
                        leaf_hand(leaf_target)
                        if release_start is not None:
                            apply(hook, hook_site, 4 * (target - d.qpos[ha]) - .2 * d.qvel[hv] + model.dof_frictionloss[hv])
                    tick(release)
                    hook_peak = max(hook_peak, float(np.linalg.norm(forces.get(hb['release_site'], (0., 0., 0.)))))
                    stop_load = sum(contact_load((striker, stop)) for stop in opening_stops)
                    stop_peak = max(stop_peak, stop_load)
                    stop_load_s += model.opt.timestep * (stop_load > 1.)
                record.update(release_started_after_unloading=release_start is not None,
                              release_hook_rad=float(d.qpos[ha]), release_leaf_rad=float(d.qpos[la]),
                              release_peak_hook_force_N=hook_peak, opening_stop_peak_load_N=float(stop_peak),
                              opening_stop_load_observed_s=float(stop_load_s))
                if release_start is None or d.qpos[ha] < .7:
                    failures.append('actual_hook_release_failed')
                if stop_load_s < .1:
                    failures.append('physical_opening_stop_reaction_not_observed')
                phase = 'close'; start = float(d.time); initial = float(d.qpos[la])
                for _ in range(round(10 / model.opt.timestep)):
                    target = initial * (1 - quintic((d.time - start) / 8))
                    def close():
                        leaf_hand(target, True)
                        if d.qpos[la] > hb['nominal_capture_angle_rad'] - .25:
                            apply(hook, hook_site, 4 * (.85 - d.qpos[ha]) - .2 * d.qvel[hv])
                    tick(close)
                record.update(closed_rad=float(d.qpos[la]), returned_hook_rad=float(d.qpos[ha]))
                cycles.append(record)
                if abs(d.qpos[la]) > .01 or abs(d.qpos[ha]) > .08:
                    failures.append('leaf_or_hook_did_not_return')
            mujoco.mj_forward(model, d); measure()
        except RuntimeError as error:
            failures.append(str(error))
    if depth > .001: failures.append('native_penetration_exceeds_1mm')
    if loop > .001: failures.append('native_linkage_residual_exceeds_1mm')
    if gear > .001: failures.append('native_gear_residual_exceeds_0_001rad')
    if peak > 120 + 1e-9: failures.append('manual_force_limit_exceeded')
    if messages or any(w.number for w in d.warning): failures.append('native_solver_warning')
    result = {'schema_version': 1, 'applicable': True, 'ok': not failures,
              'cache_hit': False, 'compiled_model_sha256': digest, 'cycles': cycles,
              'scope': 'Two native service cycles: actual-surface dog release and leaf/hook hand forces, hands-free arrest under an external 80 Nm closing test load, unloaded manual hook release, and full closing. No embodied task, strength, pressure or durability certification.',
              'max_penetration_m': depth, 'depth_event': depth_event, 'peak_hand_force_N': peak,
              'max_loop_residual_m': loop, 'max_gear_residual_rad': gear,
              'elapsed_native_s': float(d.time), 'timestep_s': float(model.opt.timestep),
              'native_warning_messages': list(messages), 'warning_counters': [int(w.number) for w in d.warning],
              'failures': sorted(set(failures)), 'trace': trace}
    _CACHE[key] = copy.deepcopy(result)
    while len(_CACHE) > 32: _CACHE.popitem(last=False)
    return result
