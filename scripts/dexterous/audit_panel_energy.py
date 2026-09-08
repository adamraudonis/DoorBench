#!/usr/bin/env python3
"""Audit recorded panel work, reaction torque and cup effort without simulation.

Uses archived mj_step wrenches and motor forces. Only geometric/inertia pipeline
components run on a separate mirror. Work is a 2 ms rectangle quadrature of
actual force times pre-integration velocity, not an exact energy balance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-time', type=float, default=69.5)
    parser.add_argument('--bin-seconds', type=float, default=.1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from doorbench.dexterous.interval_mechanics import contact_generalized_force, motor_power
    # The exact run's environment builds the same combined plant.
    from doorbench.dexterous.environment import DexterousDoorEnv
    from doorbench.dexterous.native_transition_archive import NativeTransitionArchive
    run = args.run.resolve()
    configuration = json.loads((run / 'manifest.json').read_text())['configuration']
    robot = Path(configuration['robot'])
    sim = DexterousDoorEnv(configuration['door'], robot, json.loads(robot.with_suffix('.audit.json').read_text()))
    m, d = sim.m, sim.d
    motors_path = Path(configuration['motors'])
    motors = json.loads(motors_path.read_text())
    transmission = np.zeros((m.nu, m.nv))
    groups = {name: [] for name in ('left_arm', 'left_fingers', 'right_arm', 'right_fingers', 'legs', 'torso')}
    for entry in motors['actuators']:
        a = m.actuator('robot/' + entry['name']).id
        for name, coefficient in entry['terms'].items():
            transmission[a, m.jnt_dofadr[m.joint('robot/' + name).id]] = coefficient
        names = list(entry['terms'])
        if any(n.startswith('lh_') and 'WRJ' not in n for n in names): group = 'left_fingers'
        elif any(n.startswith('rh_') and 'WRJ' not in n for n in names): group = 'right_fingers'
        elif any(n.startswith('left_') and ('hip_' in n or n.endswith(('knee', 'ankle'))) for n in names) or any(n.startswith('right_') and ('hip_' in n or n.endswith(('knee', 'ankle'))) for n in names): group = 'legs'
        elif names == ['torso']: group = 'torso'
        elif any(n.startswith(('left_', 'lh_')) for n in names): group = 'left_arm'
        else: group = 'right_arm'
        groups[group].append(a)
    cup = m.joint('robot/lh_LFJ5').id
    cupq, cupv = m.jnt_qposadr[cup], m.jnt_dofadr[cup]
    cupa = next(i for i in range(m.nu) if transmission[i, cupv] != 0)
    leaf = m.joint('leaf_hinge').id
    leafq, leafv = m.jnt_qposadr[leaf], m.jnt_dofadr[leaf]
    scene_v = np.flatnonzero([not m.joint(int(j)).name.startswith('robot/') for j in m.dof_jntid])
    limited = np.flatnonzero(m.jnt_limited)
    limited_q = m.jnt_qposadr[limited]
    jp, jr, mass = np.zeros((3, m.nv)), np.zeros((3, m.nv)), np.zeros((m.nv, m.nv))
    rows = []
    frame_error = 0.
    transmission_error = None
    for raw in NativeTransitionArchive.read(run / 'raw-transitions'):
        t, end = raw['interval_start_s'], raw['interval_end_s']
        if t < args.from_time - 1e-8: continue
        q, v = np.asarray(raw['qpos_before']), np.asarray(raw['qvel_before'])
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        if transmission_error is None:
            mujoco.mj_tendon(m, d)
            mujoco.mj_transmission(m, d)
            native = np.zeros_like(transmission)
            for motor in range(m.nu):
                start = d.moment_rowadr[motor]
                end_index = start + d.moment_rownnz[motor]
                native[motor,d.moment_colind[start:end_index]] = d.actuator_moment[start:end_index]
            transmission_error = float(np.max(abs(native-transmission)))
            if transmission_error > 1e-12:
                raise ValueError('Recorded motor contract differs from compiled transmission')
        mujoco.mj_crb(m, d)
        mujoco.mj_fullM(m, d, mass)
        ids = raw['body_ids']
        frame_error = max(frame_error, float(np.max(abs(d.xpos[ids] - raw['body_positions_world_m']))), float(np.max(abs(d.xmat[ids].reshape(-1, 3, 3) - raw['body_rotations_world']))))
        powers = motor_power(raw['actuator_force'], transmission, v)
        tau = np.zeros(m.nv)
        cup_panel = cup_other = 0.
        panel_normal = palm_normal = panel_work_rate = 0.
        for contact in raw['contacts']:
            for side, body in enumerate(contact['body']):
                if not m.body(body).name.startswith('robot/lh_'): continue
                other = contact['body'][1-side]
                mujoco.mj_jac(m, d, jp, jr, np.asarray(contact['position_world_m']), body)
                generalized, world = contact_generalized_force(contact['frame_world'], contact['wrench_contact_frame'], side, jp, jr)
                tau += generalized
                if m.body(other).name.startswith('leaf'):
                    cup_panel += generalized[cupv]
                    panel_normal += max(0., contact['wrench_contact_frame'][0])
                    if m.body(body).name == 'robot/lh_palm': palm_normal += max(0., contact['wrench_contact_frame'][0])
                    panel_work_rate += float(generalized @ v)
                else: cup_other += generalized[cupv]
        actual_force = np.asarray(raw['actuator_force'])
        up = d.xmat[m.body('robot/torso_link').id].reshape(3, 3)[:, 2]
        joint_violation = np.maximum(m.jnt_range[limited,0]-q[limited_q], q[limited_q]-m.jnt_range[limited,1])
        worst_joint = int(np.argmax(joint_violation))
        rows.append(dict(time_s=float(t), dt_s=float(end-t), leaf_angle_rad=float(q[leafq]), leaf_speed_rad_s=float(v[leafv]),
            panel_normal_force_N=float(panel_normal), palm_normal_force_N=float(palm_normal),
            cup_angle_rad=float(q[cupq]), cup_velocity_rad_s=float(v[cupv]), cup_motor_torque_Nm=float(actual_force[cupa]*transmission[cupa,cupv]),
            cup_contact_torque_Nm=float(tau[cupv]), cup_panel_torque_Nm=float(cup_panel), cup_other_contact_torque_Nm=float(cup_other),
            left_panel_contact_power_on_hand_W=float(panel_work_rate),
            kinetic_energy_J=float(.5*v@mass@v), gravitational_potential_energy_J=float(-np.sum(m.body_mass*(d.xipos@m.opt.gravity))),
            leaf_subtree_kinetic_energy_J=float(.5*v[scene_v]@mass[np.ix_(scene_v,scene_v)]@v[scene_v]),
            maximum_joint_violation_rad=max(0.,float(joint_violation[worst_joint])), worst_joint=m.joint(int(limited[worst_joint])).name,
            root_height_m=float(q[sim.root_qadr+2]), torso_tilt_deg=float(np.rad2deg(np.arccos(np.clip(up[2],-1,1)))),
            motor_power_W={g:float(powers[index].sum()) for g,index in groups.items()},
            positive_motor_power_W={g:float(np.maximum(powers[index],0).sum()) for g,index in groups.items()}))
    if not rows: raise ValueError('No archive intervals in requested range')
    bins = []
    for index in range(int(np.ceil((rows[-1]['time_s'] - args.from_time) / args.bin_seconds))):
        start = args.from_time + index * args.bin_seconds
        part = [x for x in rows if start-1e-8 <= x['time_s'] < start+args.bin_seconds-1e-8]
        if not part: continue
        bins.append(dict(start_s=float(start), end_s=part[-1]['time_s']+part[-1]['dt_s'], samples=len(part),
            peak_palm_N=max(x['palm_normal_force_N'] for x in part), mean_palm_N=float(np.mean([x['palm_normal_force_N'] for x in part])),
            cup_minimum_rad=min(x['cup_angle_rad'] for x in part), cup_contact_torque_min_Nm=min(x['cup_contact_torque_Nm'] for x in part),
            cup_motor_torque_min_Nm=min(x['cup_motor_torque_Nm'] for x in part), cup_motor_torque_max_Nm=max(x['cup_motor_torque_Nm'] for x in part),
            kinetic_start_J=part[0]['kinetic_energy_J'], kinetic_end_J=part[-1]['kinetic_energy_J'],
            leaf_kinetic_end_J=part[-1]['leaf_subtree_kinetic_energy_J'],
            signed_motor_work_J={g:sum(x['motor_power_W'][g]*x['dt_s'] for x in part) for g in groups},
            positive_motor_work_J={g:sum(x['positive_motor_power_W'][g]*x['dt_s'] for x in part) for g in groups},
            work_on_left_hand_from_panel_J=sum(x['left_panel_contact_power_on_hand_W']*x['dt_s'] for x in part)))
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = {k:np.asarray([x[k] for x in rows]) for k in rows[0] if not isinstance(rows[0][k],dict)}
    arrays.update({'motor_power_'+g:np.asarray([x['motor_power_W'][g] for x in rows]) for g in groups})
    np.savez_compressed(args.output/'intervals.npz', **arrays)
    result = dict(scope='Actual archived interval force/power diagnostic, no rerun or changed qualification. Work uses pre-state rectangular quadrature, not a complete energy conservation audit.',
        run=str(run), raw_manifest_sha256=hashlib.file_digest((run/'raw-transitions/manifest.json').open('rb'),'sha256').hexdigest(),
        original_passed=json.loads((run/'report.json').read_text())['passed'],
        motor_contract_sha256=hashlib.file_digest(motors_path.open('rb'),'sha256').hexdigest(), maximum_body_frame_error=frame_error,
        maximum_compiled_transmission_error=transmission_error,
        source_links=['https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce','https://mujoco.readthedocs.io/en/stable/computation/index.html#transmission'],
        cup_motor_original_force_range=m.actuator_forcerange[cupa].tolist(), cup_original_joint_range=m.jnt_range[cup].tolist(),
        peak_palm_sample=max(rows,key=lambda x:x['palm_normal_force_N']), minimum_cup_sample=min(rows,key=lambda x:x['cup_angle_rad']), bins=bins)
    result['worst_joint_sample'] = max(rows,key=lambda x:x['maximum_joint_violation_rad'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bins'},indent=2))
    sim.close()


if __name__ == '__main__': main()
