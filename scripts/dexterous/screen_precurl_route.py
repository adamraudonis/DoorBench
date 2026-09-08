#!/usr/bin/env python3
"""Screen every interpolated route pose and require an actual hand-free start.

This is a geometric screen, not evidence of motor-driven task success. In
particular it rejects shallow finger/panel contacts that penetration-only gates
allow. The final seating phase intentionally permits physical contact.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path, required=True)
    parser.add_argument('--door', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        default=root/'configs/dexterous/door55-precurl-v2/reference.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref = json.loads(args.reference.read_text())
    env = DexterousDoorEnv(args.door, args.robot,
                          json.loads(args.robot.with_suffix('.audit.json').read_text()))
    try:
        env.reset(images=False, randomize=False)
        m, d = env.m, env.d
        d.qpos[env.root_qadr:env.root_qadr+7] = ref['initial_root']
        ids = [m.joint('robot/'+n).id for n in ref['acquisition']['joint_names']]
        qa = m.jnt_qposadr[ids]
        path = np.asarray(ref['acquisition']['path_qpos'])
        if path.ndim != 2 or path.shape[1] != len(ids) or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('Invalid finite named-joint path')
        pairs = [(m.jnt_qposadr[m.joint(f'robot/{hand}_{finger}J1').id],
                  m.jnt_qposadr[m.joint(f'robot/{hand}_{finger}J2').id])
                 for hand in ('lh', 'rh') for finger in ('FF', 'MF', 'RF', 'LF')]
        failures = []
        count = (len(path)-1)*5+1
        for index, u in enumerate(np.linspace(0, 1, count)):
            z = u*(len(path)-1)
            i = min(int(z), len(path)-2)
            d.qpos[qa] = path[i]*(1-z+i)+path[i+1]*(z-i)
            mujoco.mj_forward(m, d)
            if np.any(d.qpos[qa] < m.jnt_range[ids, 0]-1e-9) or np.any(d.qpos[qa] > m.jnt_range[ids, 1]+1e-9):
                failures.append(dict(sample=index, reason='individual joint limit'))
            if any(d.qpos[a] > d.qpos[b]+1e-9 for a, b in pairs):
                failures.append(dict(sample=index, reason='loopback inequality'))
            for contact in d.contact[:d.ncon]:
                bodies = [m.body(m.geom_bodyid[g]).name for g in contact.geom]
                geoms = [m.geom(int(g)).name for g in contact.geom]
                hand = any(n.startswith('robot/rh_') for n in bodies)
                robot = any(n.startswith('robot/') for n in bodies)
                foot = 'floor' in geoms and any(n.endswith('_ankle_link') for n in bodies)
                if hand and u <= .85 and contact.dist <= 0:
                    failures.append(dict(sample=index, reason='early hand contact',
                                         bodies=bodies, distance_m=float(contact.dist)))
                elif robot and contact.dist < (-.003 if foot else -.001):
                    failures.append(dict(sample=index, reason='penetration',
                                         bodies=bodies, distance_m=float(contact.dist)))
        report = dict(scope=__doc__, passed=not failures, samples=count,
                      early_contact_free_fraction=.85, failures=failures)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'failures'}))
    finally:
        env.close()
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
