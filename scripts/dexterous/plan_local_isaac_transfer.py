#!/usr/bin/env python3
"""Rebuild LH transfer from a qualified local Isaac terminal measurement.

Native routes provide numerical LH preferences only. All root/joint/door state
comes from the source Isaac archive and passes the existing exact-epoch and
2 micrometre / 2 microradian destination kinematics admission. Original geometry
audits remain unchanged. A route is a candidate for a new motor trial, not proof
of physical transfer or cross-engine collision-model equivalence.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.dexterous.plan_local_standing_transfer import (
    sha, write_json, resample_preferences, generate_fixed_body_path, generate_path,
)
from doorbench.dexterous.qualified_isaac_grasp import (
    EXTRACTED_FILES, AUDITED_FILES, validate_reports, validate_transfer_evidence,
)
from doorbench.dexterous.isaac_attained_state import extract_attained_state
from doorbench.dexterous.standing_body_record import extract_standing_body_poses
from doorbench.dexterous.destination_planner_admission import admit_destination_planner
from doorbench.dexterous.landed_left_planner import LandedLeftScene, JOINT_NAMES
from doorbench.dexterous.robot_design_identity import robot_design_identity
from doorbench.dexterous.standing_transfer import validate_route_geometry


def numerical_left_preferences(config):
    """Discard source route qualification, mechanism values, and root/leg path."""
    if config.get('schema') == 'doorbench.standing-transfer.v1':
        config = dict(schema='doorbench.left-palm-targets.v1',
            joint_names=config['left_joint_names'],targets=config['left_targets'])
    # The common reader consumes only explicit numeric LH fields.
    return resample_preferences(config)


def local_summary(report,audit):
    """Same coordinator conjunction, explicitly derived for a direct local run.

    This is not a cloud coordinator receipt. The shared qualification validator
    below independently checks all original report, interval and contact gates.
    """
    physical = report.get('passed') is True and bool(report.get('checks')) and all(v is True for v in report['checks'].values())
    independent = (audit.get('accounting_passed') is True and audit.get('task_passed') is True
                   and audit.get('independent_raw_contact_audit_complete') is True)
    surfaces = audit.get('invalid_loaded_patches') == 0
    return dict(passed=bool(physical and independent and surfaces),isaac_runtime_passed=physical,
        independent_audit_passed=independent,all_loaded_handle_patches_qualified=surfaces,
        provenance='Derived local direct-run summary; no historical/cloud coordinator file exists or is synthesized')


def load_local_source(run,audit_path):
    run,audit_path = Path(run).resolve(),Path(audit_path).resolve()
    trial = run/'trial'
    report_path = trial/'operation-report.json'
    paths = [audit_path,report_path]+[trial/name for name in sorted(EXTRACTED_FILES|AUDITED_FILES)]
    initial_report = json.loads(report_path.read_text())
    if 'standing_transfer' in initial_report:
        paths += [run/'isaac-transfer-audit.json',trial/'standing-transfer-steps.json.gz']
    hashes = {str(p):sha(p) for p in paths}
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    configuration = json.loads((trial/'configuration.json').read_text())
    motors = json.loads((trial/'motor-contract.json').read_text())
    provenance = json.loads((trial/'provenance.json').read_text())
    with np.load(trial/'acquisition-physics.npz',allow_pickle=False) as physics:
        binding = extract_attained_state(configuration=configuration,motor_contract=motors,
            provenance=provenance,physics=physics,time_s=report['duration_s'])
        bodies = extract_standing_body_poses(configuration,physics,time_s=binding['time_s'])
    extracted = dict(binding=binding,measured_bodies=bodies,
        input_sha256={name:hashes[str(trial/name)] for name in EXTRACTED_FILES},
        scope='Exact original recorded endpoint; no state interpolation or native state substitution')
    summary = local_summary(report,audit)
    validate_reports(report,audit,summary,extracted)
    for record in (extracted,audit):
        for name,digest in record['input_sha256'].items():
            if hashes[str(trial/name)] != digest:
                raise ValueError('Source evidence hash mismatch: '+name)
    if 'standing_transfer' in report:
        transfer = json.loads((run/'isaac-transfer-audit.json').read_text())
        validate_transfer_evidence(report,dict(independent_transfer_passed=transfer.get('passed') is True),
            transfer,{str(p):hashes[str(p)] for p in (report_path,trial/'standing-transfer-steps.json.gz')})
    if hashes != {str(p):sha(p) for p in paths}:
        raise ValueError('Source changed during exact endpoint admission')
    return extracted,motors,dict(passed=True,local_direct_summary=summary,
        time_s=binding['time_s'],state_sha256=binding['sha256'],input_sha256=hashes,
        qualification='Original completed Isaac operation plus independent raw-contact audit; no cloud coordination dependency')


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    output = args.output.resolve()
    output.mkdir(parents=True)
    robot,door,door_usd,preferences = [Path(p).resolve() for p in (args.robot,args.door,args.door_usd,args.preferences)]
    if door.is_dir():door = door/'door.xml'
    audit_path = args.independent_audit or args.isaac_source/'independent-contact-audit.json'
    try:
        extracted,motors,qualification = load_local_source(args.isaac_source,audit_path)
        if sha(robot) != extracted['binding']['robot_source_sha256'] or sha(door_usd) != extracted['binding']['door_source_sha256']:
            raise ValueError('Current robot XML and door USD differ from the recorded source')
        scene = LandedLeftScene(robot,door)
        measured,admission = admit_destination_planner(scene.m,extracted,
            motor_contract=motors,door_source_sha256=sha(door_usd))
        frozen = measured.qpos.copy()
        values = numerical_left_preferences(json.loads(preferences.read_text()))
        if not np.isfinite(args.receiving_normal_offset_m) or abs(args.receiving_normal_offset_m)>.005:
            raise ValueError('Bounded declared receiving plane preference required')
        values['position'][:,1] += args.receiving_normal_offset_m
        files = [Path(__file__).resolve(),Path(sys.modules['scripts.dexterous.plan_local_standing_transfer'].__file__).resolve(),
            ROOT/'scripts/dexterous/audit_standing_transfer_path.py',ROOT/'doorbench/dexterous/standing_transfer.py',
            ROOT/'doorbench/dexterous/landed_left_audit.py',ROOT/'doorbench/dexterous/transfer_contact_geometry.py',
            ROOT/'doorbench/dexterous/qualified_isaac_grasp.py',ROOT/'doorbench/dexterous/destination_planner_admission.py',
            ROOT/'doorbench/dexterous/destination_return_kinematics.py',ROOT/'doorbench/dexterous/destination_planning_coordinates.py']
        hashes = {**qualification['input_sha256'],**{str(p):sha(p) for p in [robot,door,door_usd,preferences,*files]}}
        for file in files:shutil.copy2(file,output/file.name)
        write_json(output/'extracted-state.json',extracted)
        write_json(output/'source-admission.json',dict(source=qualification,coordinates=admission,
            consumed_state=frozen,normalization_scope='Only existing float32 quaternion normalization; no other measured coordinate changed'))
        endpoint = None
        if args.path_profile == 'fixed-body-joint-v1':
            path,rows,endpoint = generate_fixed_body_path(scene,frozen,values,pitch_bump=args.reach_pitch_bump)
        else:
            if args.reach_pitch_bump:raise ValueError('Pitch arc requires fixed-body joint profile')
            path,rows = generate_path(scene,frozen,values,pose_weight=args.pose_weight)
        sampled = len(path)==101 and all(r['passed'] for r in rows)
        report_path = output/'report.json'
        write_json(report_path,dict(schema='doorbench.local-isaac-transfer-planning.v1',passed=sampled,
            source_qualification=qualification,destination_admission=admission,
            preferences_are_numerical_only=True,physics_steps=0,path_qpos=path,samples=rows,
            input_sha256=hashes,parameters=dict(path_profile=args.path_profile,
                reach_pitch_bump=args.reach_pitch_bump,receiving_normal_offset_m=args.receiving_normal_offset_m,
                pose_weight=args.pose_weight),receiving_endpoint_fit=endpoint,scope=__doc__))
        proof = output/'dense-audit.json'
        subprocess.run([sys.executable,str(ROOT/'scripts/dexterous/audit_standing_transfer_path.py'),
            '--robot',str(robot),'--door',str(door),'--path',str(report_path),'--output',str(proof)],check=True,cwd=ROOT)
        dense = json.loads(proof.read_text())
        if any(sha(p)!=digest for p,digest in hashes.items()):raise ValueError('Source changed while planning')
        m,d=scene.m,scene.d
        joint_names = motors['joint_names']
        jointqa = [m.joint('robot/'+n).qposadr[0] for n in joint_names]
        targets=[]
        for q in path:
            d.qpos[:]=q;mujoco.mj_kinematics(m,d);rotation=d.xmat[scene.leaf].reshape(3,3)
            targets.append(dict(phase='left_reach',leaf_rad=float(q[m.joint('leaf_hinge').qposadr[0]]),
                position=(rotation.T@(d.site_xpos[scene.palm]-d.xpos[scene.leaf])).tolist(),
                normal=(rotation.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]).tolist(),
                nominal=[float(q[m.joint('robot/'+n).qposadr[0]]) for n in JOINT_NAMES]))
        candidate=dict(schema='doorbench.local-isaac-transfer-candidate.v1',
            sampled_geometry_passed=sampled,dense_geometry_passed=dense['passed'],
            attained_source_qualification=dict(source=qualification,coordinates=admission),
            robot_source_design_identity=robot_design_identity(robot),robot_path=str(robot),door_path=str(door),
            robot_xml_sha256=sha(robot),door_xml_sha256=sha(door),scene_path_source=str(report_path),
            scene_path_sha256=sha(report_path),dense_audit_path=str(proof),dense_audit_sha256=sha(proof),
            joint_names=joint_names,root_path=path[:,scene.root:scene.root+7],joint_path=path[:,jointqa],
            left_joint_names=JOINT_NAMES,left_targets=targets,attained_time_s=qualification['time_s'],
            attained_trial=str(args.isaac_source.resolve()),physics_steps=0,
            scope='Fresh path from qualified actual PhysX endpoint; original-model sampled geometry only. New live PhysX contact and motion qualification still required.')
        if sampled and dense['passed']:
            route={**candidate,'schema':'doorbench.standing-transfer.v1','geometric_screen_passed':True}
            validate_route_geometry(route)
            write_json(output/'transfer.json',route)
        write_json(output/'candidate.json',candidate)
        print(json.dumps(dict(source_passed=True,sampled_passed=sampled,dense_passed=dense['passed'],
            route_exported=(output/'transfer.json').exists(),source_time_s=qualification['time_s'],output=str(output))))
        return 0 if sampled and dense['passed'] else 1
    except Exception as error:
        result=dict(passed=False,physics_steps=0,error=repr(error),scope='Retained source/admission/planning failure; no runtime route qualified')
        if hasattr(error,'receipt'):result['admission_receipt']=error.receipt
        write_json(output/'failure.json',result)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('isaac-source','robot','door','door-usd','preferences','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--independent-audit',type=Path)
    p.add_argument('--path-profile',choices=('fixed-body-joint-v1','rebase-cartesian-v1'),default='fixed-body-joint-v1')
    p.add_argument('--reach-pitch-bump',type=float,default=.4)
    p.add_argument('--receiving-normal-offset-m',type=float,default=0.)
    p.add_argument('--pose-weight',type=float,default=200.)
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
