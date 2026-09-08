#!/usr/bin/env python3
"""Emit a versioned receipt only after independent live simulation checks pass."""
import argparse
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from doorbench.dexterous.isaac_readiness import mechanics_profile,file_sha256,validate_passive_audit


def build_receipt(trial,native_robot,motor_path,profile):
    profile=mechanics_profile(profile);trial=Path(trial);native_robot=Path(native_robot);motor_path=Path(motor_path)
    audit=json.loads((trial/'kinematics-audit.json').read_text());assert audit['passed']
    rows=json.loads((trial/'trace.json').read_text());config=json.loads((trial/'configuration.json').read_text())
    native_audit=json.loads(native_robot.with_suffix('.audit.json').read_text());motors=json.loads(motor_path.read_text())
    native_hash=file_sha256(native_robot)
    if native_audit.get('mechanics_profile','upstream-v1')!=profile:raise ValueError('Native robot mechanics profile differs')
    if native_audit['robot_xml_sha256']!=native_hash or motors.get('source_xml_sha256')!=native_hash:raise ValueError('Native robot and imported motor hashes differ')
    if abs(config['robot_mass_kg']-native_audit['mass_kg'])>1e-4:raise ValueError('Imported mass differs')
    if len(motors['actuators'])!=native_audit['actuators']:raise ValueError('Imported actuator count differs')
    for motor in motors['actuators']:
        limits=native_audit['limits'][motor['name']]
        if limits['force']!=motor['force_range'] or limits['control']!=motor['control_range']:raise ValueError('Original motor caps changed')
    tendon_path=trial/'passive-tendon-audit.json'
    tendon_audit=json.loads(tendon_path.read_text()) if tendon_path.exists() else {}
    passive=validate_passive_audit(tendon_audit,motors,profile)
    assert len(rows)>=45 and rows[-1]['time_s']>=.9
    assert config['runtime_pose_writes']==0 and not config['direct_door_commands']
    material=config['contact_material_audit']
    assert material['colliders_checked']>0 and material['contract']['combine_mode']=='max'
    assert material['backend_values_verified'] and material['backend_offsets_verified']
    assert not (trial/'error.txt').exists()
    assert min(r['root'][2] for r in rows)>.7 and max(r['torso_tilt_deg'] for r in rows)<20
    assert max(abs(r['sim_time_s']-r['time_s']) for r in rows)<.003
    assert len(config['simulator_effort_limits'])==69 and min(config['simulator_effort_limits'])>0
    wrist=config['robot_joint_names'].index('right_wrist_yaw')
    assert max(r['joints'][wrist] for r in rows)-rows[0]['joints'][wrist]>.05,'Commanded wrist motion did not occur'
    assert all(max(abs(x-y) for x,y in zip(r['joint_torque_command'],r['joint_torque_sent']))<1e-4 for r in rows),'Motor commands were dropped before reaching PhysX'
    if profile=='shadow-loopback-v2':
        names=config['robot_joint_names'];pairs=[(names.index(f'{side}_{digit}J1'),names.index(f'{side}_{digit}J2')) for side in ('lh','rh') for digit in ('FF','MF','RF','LF')]
        passive['max_sampled_loopback_violation_rad']=max(0.,max(r['joints'][distal]-r['joints'][middle] for r in rows for distal,middle in pairs))
        if passive['max_sampled_loopback_violation_rad']>.02:raise ValueError('Live readiness violates passive loopback limit')
    robot_usd=Path(config['args']['robot_usd']).resolve();door_usd=Path(config['args']['door_usd']).resolve()
    paths=[native_robot,native_robot.with_suffix('.audit.json'),motor_path,robot_usd,door_usd]
    return dict(ready=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),mechanics_profile=profile,
        scope='Runtime, generated door QA, free robot, original motor caps, live standing physics/rendering and declared passive mechanics. Not acquisition/opening success.',
        trial=str(trial.resolve()),kinematics=audit,native_robot=str(native_robot.resolve()),native_robot_sha256=native_hash,
        motor_contract=str(motor_path.resolve()),robot_usd=str(robot_usd),door_usd=str(door_usd),passive_tendons=passive,
        input_hashes={str(p.resolve()):file_sha256(p) for p in paths},
        checksums={p.name:file_sha256(p) for p in trial.iterdir() if p.suffix in ('.json','.mp4')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trial','output','native-robot','motors'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mechanics-profile',default='upstream-v1');a=p.parse_args()
    receipt=build_receipt(a.trial,a.native_robot,a.motors,a.mechanics_profile)
    receipt['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
