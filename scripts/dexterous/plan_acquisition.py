#!/usr/bin/env python3
"""Screen open-hand approaches to the verified Door55 grasp, without stepping physics.

This is a privileged geometric development tool, never a success evaluator. The
output is a candidate for a subsequent motor-driven, free-base simulation trial.
"""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from doorbench.dexterous.environment import DexterousDoorEnv


def collision_failures(m, d):
    failures = []
    for c in d.contact[:d.ncon]:
        bodies = [m.body(m.geom_bodyid[g]).name for g in c.geom]
        geoms = [m.geom(int(g)).name for g in c.geom]
        if not any(n.startswith('robot/') for n in bodies):
            continue
        foot = 'floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)
        if c.dist < (-.003 if foot else -.001):
            failures.append(dict(depth_m=-float(c.dist), bodies=bodies, geoms=geoms))
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', type=Path, required=True)
    p.add_argument('--door', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('configs/dexterous/isaac-door55-reference.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--opening-exponent', type=float, default=1.)
    p.add_argument('--retreat-distance', type=float, default=.10)
    p.add_argument('--offset', type=float, nargs=3)
    p.add_argument('--optimize-collisions', action='store_true')
    p.add_argument('--samples', type=int, default=101)
    args = p.parse_args()
    if args.samples<2 or not np.isfinite(args.opening_exponent) or args.opening_exponent<=0 or not np.isfinite(args.retreat_distance) or args.retreat_distance<=0 or (args.offset and not np.isfinite(args.offset).all()):
        p.error("Use at least two samples, finite offsets, and positive finite opening/retreat parameters")
    if args.output.exists():
        raise SystemExit("Use a new output directory to preserve failed candidates")
    from doorbench.dexterous.provenance import capture
    capture(Path(__file__).resolve().parents[2], args.output, {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, timing="before_geometry_screen")
    (args.output/"grasp-input.json").write_bytes(args.reference.read_bytes())
    r = json.loads(args.reference.read_text())
    s = DexterousDoorEnv(args.door, args.robot, json.loads(args.robot.with_suffix('.audit.json').read_text()))
    m, d = s.m, s.d
    s.reset(images=False, randomize=False)
    d.qpos[s.root_qadr:s.root_qadr+7] = r['initial_root']
    for n, q in r['initial_joints'].items():
        d.qpos[m.jnt_qposadr[m.joint('robot/'+n).id]] = q
    mujoco.mj_forward(m, d)
    base = d.qpos.copy()
    palm = m.site('robot/rh_palm_touch').id
    target = d.site_xpos[palm].copy()
    rotation = d.site_xmat[palm].reshape(3, 3).copy()
    names = r['workspace_fit']['joint_names']
    joints = [m.joint('robot/'+n).id for n in names]
    qa, va = m.jnt_qposadr[joints], m.jnt_dofadr[joints]
    low, high = m.jnt_range[joints].T.copy()
    high[names.index('right_shoulder_roll')] = -.04
    jp = np.zeros((3, m.nv)); jr = jp.copy()
    open_pose = base.copy()
    # Preserve finger spread and thumb opposition; open flexion joints only.
    for digit in ('FF', 'MF', 'RF', 'LF'):
        for joint, value in (('J1', 0.), ('J2', 0.), ('J3', .25)):
            open_pose[m.jnt_qposadr[m.joint('robot/rh_'+digit+joint).id]] = value
    for name, value in [('THJ1', 0.), ('THJ2', -.1)]:
        open_pose[m.jnt_qposadr[m.joint('robot/rh_'+name).id]] = value
    fingers = [j for j in s.joints if m.joint(j).name.startswith('robot/rh_') and 'WRJ' not in m.joint(j).name]
    fa = m.jnt_qposadr[fingers]
    distal = np.array([m.jnt_qposadr[j] for j in fingers if any(m.joint(j).name.endswith(x) for x in ('J1', 'J2'))])
    proximal = np.array([m.jnt_qposadr[j] for j in fingers if m.joint(j).name.endswith('J3')])
    variable_joints = joints+fingers
    xq = m.jnt_qposadr[variable_joints]; xv = m.jnt_dofadr[variable_joints]
    bounds = list(zip(m.jnt_range[variable_joints,0], m.jnt_range[variable_joints,1]))
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    best = None
    for dx in ([args.offset[0]] if args.offset else (0., -.08, .08, -.16, .16)):
        for dz in ([args.offset[2]] if args.offset else (0., .08, -.08, .16)):
            offset = np.array(args.offset if args.offset else [dx, -args.retreat_distance, dz])
            # Fit backwards from the known grasp to stay on its IK branch.
            d.qpos[:] = base
            path = []
            failures = []
            errors = []
            for k, fraction in enumerate(np.linspace(0, 1, args.samples)):
                position = target + offset*fraction
                opening = min(1., fraction*1.2)**args.opening_exponent
                d.qpos[fa] = base[fa]*(1-opening)+open_pose[fa]*opening
                d.qpos[distal] = base[distal]*(1-opening**.5)+open_pose[distal]*opening**.5
                d.qpos[proximal] = base[proximal]*(1-opening**2)+open_pose[proximal]*opening**2
                for _ in range(100):
                    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                    error = np.r_[5*(position-d.site_xpos[palm]),
                        Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3, 3).T).as_rotvec()]
                    mujoco.mj_jacSite(m, d, jp, jr, palm)
                    jac = np.vstack([5*jp[:, va], jr[:, va]])
                    change = jac.T@np.linalg.solve(jac@jac.T+.0001*np.eye(6), error)
                    d.qpos[qa] = np.clip(d.qpos[qa]+np.clip(change, -.05, .05), low, high)
                    if np.linalg.norm(error) < 1e-5:
                        break
                if args.optimize_collisions and k:
                    nominal = d.qpos[xq].copy()
                    previous = path[-1][xq].copy()
                    def objective(x):
                        d.qpos[xq] = x
                        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
                        dp = d.site_xpos[palm]-position
                        angle = Rotation.from_matrix(d.site_xmat[palm].reshape(3,3)@rotation.T).as_rotvec()
                        mujoco.mj_jacSite(m,d,jp,jr,palm)
                        value = 100*dp@dp + .1*angle@angle + .003*np.sum((x-nominal)**2) + .02*np.sum((x-previous)**2)
                        gradient = 200*jp[:,xv].T@dp + .2*jr[:,xv].T@angle + .006*(x-nominal) + .04*(x-previous)
                        for contact in d.contact[:d.ncon]:
                            bodyids = [int(m.geom_bodyid[g]) for g in contact.geom]
                            if not any(m.body(b).name.startswith('robot/') for b in bodyids):
                                continue
                            geoms = [m.geom(int(g)).name for g in contact.geom]
                            if 'floor' in geoms and any(m.body(b).name.endswith('_ankle_link') for b in bodyids):
                                continue
                            violation = min(0., float(contact.dist)+.0001)
                            if violation == 0.:
                                continue
                            normal = contact.frame[:3]
                            jac = np.zeros(len(x))
                            for sign, bid in zip((-1,1),bodyids):
                                mujoco.mj_jac(m,d,jp,jr,contact.pos,bid)
                                jac += sign*(normal@jp[:,xv])
                            value += 20000*violation*violation
                            gradient += 40000*violation*jac
                        return value, gradient
                    fit = minimize(objective, previous, jac=True, bounds=bounds,
                        method='L-BFGS-B', options=dict(maxiter=160, ftol=1e-12, gtol=1e-7))
                    d.qpos[xq] = fit.x
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_collision(m, d)
                errors.append(float(np.linalg.norm(position-d.site_xpos[palm])))
                orientation_error = float(np.linalg.norm(Rotation.from_matrix(rotation@d.site_xmat[palm].reshape(3,3).T).as_rotvec()))
                # Interior poses may detour around the handle; the final grasp remains exact.
                position_limit, angle_limit = (.03, .4) if args.optimize_collisions and k else (.003, .03)
                if errors[-1] > position_limit or orientation_error > angle_limit:
                    failures.append(dict(sample=k, reason='unreachable', error_m=errors[-1], orientation_error_rad=orientation_error))
                failures.extend(dict(sample=k, **f) for f in collision_failures(m, d))
                path.append(d.qpos.copy())
            # Screen interpolated configurations too: valid endpoints can hide collisions.
            for segment in range(len(path)-1):
                for substep in range(1,5):
                    u = substep/5
                    d.qpos[:] = path[segment]*(1-u)+path[segment+1]*u
                    mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d); mujoco.mj_collision(m,d)
                    failures.extend(dict(segment=segment, fraction=u, **f) for f in collision_failures(m,d))
            d.qpos[:] = path[-1]
            mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d); mujoco.mj_collision(m,d)
            lever = m.geom('leaf_handle_lever_col_n').id
            hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith('robot/rh_')]
            start_gap = min(mujoco.mj_geomDistance(m,d,g,lever,1.,None) for g in hand_geoms)
            if start_gap < .02:
                failures.append(dict(reason='hand starts too close to lever', gap_m=float(start_gap)))
            summary = dict(start_hand_lever_gap_m=float(start_gap), offset_m=offset.tolist(), passed=not failures,
                max_position_error_m=max(errors), failure_count=len(failures), failures=failures)
            np.savez_compressed(args.output/f"candidate-{len(candidates):03d}.npz", qpos=np.asarray(list(reversed(path))))
            candidates.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='failures'}), flush=True)
            if not failures and best is None:
                best = (offset, list(reversed(path)))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope=__doc__, passed=best is not None, configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, candidates=candidates)
    (args.output/'geometry-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    if best is not None:
        offset, path = best
        result = copy.deepcopy(r)
        result['initial_joints'] = {m.joint(j).name.removeprefix('robot/'):float(path[0][m.jnt_qposadr[j]]) for j in s.joints}
        result['acquisition'] = dict(scope='Privileged standstill open-hand reach candidate; no locomotion or physics validation',
            offset_m=offset.tolist(), target_position=target.tolist(), target_rotation=rotation.tolist(),
            joint_names=[m.joint(j).name.removeprefix('robot/') for j in s.joints],
            path_qpos=[q[s.qadr].tolist() for q in path], reach_samples=args.samples,
            grasp_reference=str(args.reference), grasp_joints=r['initial_joints'])
        (args.output/'reference.json').write_text(json.dumps(result, indent=2)+'\n')
    s.close()
    raise SystemExit(0 if best is not None else 1)


if __name__ == '__main__':
    main()
