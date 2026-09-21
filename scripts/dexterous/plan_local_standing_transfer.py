#!/usr/bin/env python3
"""Generate a fresh standing transfer from recorded state and LH preferences.

Old target positions and joint values are numerical preferences only. No old
route, target pass flag, model identity, or qualification is inherited. The
unchanged standing-transfer auditor decides geometry on the current robot.
Development inputs may be failed physical trials, but never produce transfer.json.
This program owns an unstepped model and cannot execute a physical trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.dexterous.landed_left_planner import LandedLeftScene, JOINT_NAMES
from doorbench.dexterous.landed_left_audit import static_pose_check
from doorbench.dexterous.robot_design_identity import robot_design_identity
from doorbench.dexterous.standing_transfer import validate_route_geometry


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    def numeric(item):
        return item.item() if isinstance(item, np.generic) else item.tolist()
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=numeric)+'\n')


def resample_preferences(config):
    """Read only explicit coordinates, ignoring every historical pass claim."""
    if config.get('schema') != 'doorbench.left-palm-targets.v1' or config.get('joint_names') != JOINT_NAMES:
        raise ValueError('Expected named left-palm target preferences')
    rows = config.get('targets', [])
    if len(rows) < 2 or any(row.get('phase') != 'left_reach' for row in rows):
        raise ValueError('A complete numeric left_reach preference path is required')
    result = {}
    for key, width in [('position', 3), ('normal', 3), ('nominal', 8)]:
        values = np.asarray([row[key] for row in rows], float)
        if values.shape != (len(rows), width) or not np.isfinite(values).all():
            raise ValueError('Invalid finite preference '+key)
        if key == 'normal' and not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError('Preference normals must be unit vectors')
        result[key] = np.column_stack([np.interp(np.linspace(0, 1, 101),
            np.linspace(0, 1, len(rows)), values[:, column]) for column in range(width)])
    norms = np.linalg.norm(result['normal'], axis=1)
    if np.any(norms < 1e-6):
        raise ValueError('Ambiguous interpolated palm normal')
    result['normal'] /= norms[:, None]
    return result


def source_qualification(trial):
    """Retain the actual existing reports; a missing/false result stays false."""
    trial = Path(trial).resolve()
    pad = trial/'independent-pad-audit.json'
    if not pad.exists():
        pad = trial/'independent-contact-audit.json'
    paths = [trial/'report.json', pad, trial/'independent-whole-handle-audit.json']
    rows = []
    for index,path in enumerate(paths):
        data = json.loads(path.read_text()) if path.exists() else {}
        checks = data.get('checks', {})
        passed = data.get('passed') is True and isinstance(checks, dict) and all(v is True for v in checks.values())
        bindings = {}
        binding_passed = index == 0
        if index:
            declared = data.get('input_sha256', {})
            binding_passed = isinstance(declared,dict) and bool(declared)
            if binding_passed:
                for name,digest in declared.items():
                    raw = Path(name)
                    candidates = [raw] if raw.is_absolute() else [trial/raw,ROOT/raw]
                    matches = {p.resolve() for p in candidates if p.is_file() and sha(p)==digest}
                    if len(matches) != 1:
                        binding_passed = False
                        break
                    bindings[str(matches.pop())] = digest
                required = {str(trial/'manifest.json'),str(trial/'raw-transitions/manifest.json')}
                if index == 1:
                    required.add(str(trial/'report.json'))
                binding_passed = binding_passed and required <= set(bindings)
            passed = passed and binding_passed
        rows.append(dict(path=str(path), exists=path.exists(), passed=passed,
                         sha256=sha(path) if path.exists() else None,
                         input_binding_passed=binding_passed,input_sha256=bindings))
    return dict(passed=all(r['passed'] for r in rows), reports=rows,
                scope='Existing source results retained; this planner does not requalify contact or physics')


def load_snapshot(trial, robot, door, scene, *, at_time=None):
    """Read an exact pre-step state and its synchronized, hash-bound FK witness."""
    trial, robot, door = map(Path, (trial, robot, door))
    manifest_path = trial/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if sha(robot) != manifest['inputs']['robot']['sha256'] or sha(door) != manifest['inputs']['door']['door.xml']:
        raise ValueError('Robot/door bytes differ from the measured trial')
    archive_path = trial/'raw-transitions/manifest.json'
    archive = json.loads(archive_path.read_text())
    if archive.get('schema') != 'doorbench.native-transitions.v1' or archive.get('complete') is not True or not archive.get('chunks'):
        raise ValueError('Complete native actual-state archive required')
    if at_time is not None and (not np.isfinite(at_time) or at_time < 0):
        raise ValueError('Finite nonnegative exact snapshot time required')
    if at_time is None:
        chunk = archive['chunks'][-1]
    else:
        selected = [c for c in archive['chunks'] if c['interval_start_s']-1e-8 <= at_time < c['interval_end_s']-1e-8]
        if len(selected) != 1:
            raise ValueError('Requested snapshot is outside the recorded intervals')
        chunk = selected[0]
    chunk_path = (trial/'raw-transitions'/chunk['file']).resolve()
    if chunk_path.parent != (trial/'raw-transitions').resolve() or sha(chunk_path) != chunk['sha256']:
        raise ValueError('Actual snapshot chunk path/hash changed')
    with np.load(chunk_path, allow_pickle=False) as raw:
        times = raw['interval_start_s']
        index = len(times)-1 if at_time is None else int(np.argmin(abs(times-at_time)))
        time = float(times[index])
        if len(times) != chunk['rows'] or (at_time is not None and abs(time-at_time) > 1e-8):
            raise ValueError('Requested time must match an actual stored state; no nearest-time substitution')
        if abs(float(raw['geometry_time_s'][index])-time) > 1e-9:
            raise ValueError('Recorded geometry and source state have different epochs')
        qpos = raw['qpos_before'][index].copy()
        qvel = raw['qvel_before'][index].copy()
        if qvel.shape != (scene.m.nv,) or not np.isfinite(qvel).all():
            raise ValueError('Complete finite recorded velocity required')
        state = scene.state_from_qpos(qpos, pose_time_s=time)
        lo, hi = raw['body_offsets'][index:index+2]
        ids = raw['body_ids'][lo:hi]
        positions = raw['body_positions_world_m'][lo:hi]
        rotations = raw['body_rotations_world'][lo:hi]
    m, d = scene.m, scene.d
    if not len(ids) or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= m.nbody):
        raise ValueError('Valid synchronized body-frame witness required')
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    error = max(float(np.max(abs(d.xpos[ids]-positions))),
                float(np.max(abs(d.xmat[ids].reshape(-1, 3, 3)-rotations))))
    if not np.isfinite(error) or error > 1e-9:
        raise ValueError('Current model FK differs from the actual recorded body poses')
    receipt = dict(trial=str(trial.resolve()), pose_time_s=time, chunk_index=index,
        state_epoch='qpos_before at geometry_time_s; no state reset or integration',
        maximum_recorded_fk_error=error, witness_bodies=len(ids), state=state,
        qpos=qpos.tolist(), qvel=qvel.tolist(),
        input_sha256={str(p.resolve()): sha(p) for p in (manifest_path, archive_path, chunk_path, robot, door)})
    return qpos, receipt


def generate_path(scene, frozen, preferences, *, pose_weight=200., margin_nodes=20):
    """Fit original joint/root coordinates; keep fingers and mechanism exact."""
    if not np.isfinite(pose_weight) or not 100 <= pose_weight <= 1000 or not 10 <= margin_nodes <= 40:
        raise ValueError('Use original rebase pose-weight and margin-transition bounds')
    m, d = scene.m, scene.d
    names = ['torso']+['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1']+JOINT_NAMES[1:]
    legs = [n for n in scene.robot_names if any(k in n for k in ('hip', 'knee', 'ankle'))]
    ids = np.array([m.joint('robot/'+n).id for n in names])
    legids = np.array([m.joint('robot/'+n).id for n in legs])
    qa = np.r_[m.jnt_qposadr[ids], np.arange(scene.root, scene.root+3), m.jnt_qposadr[legids]]
    lower = np.r_[m.jnt_range[ids, 0]+.025, frozen[scene.root:scene.root+3]+[-.08,-.08,-.047], m.jnt_range[legids, 0]+.025, [-.04,-.04,-.15]]
    upper = np.r_[m.jnt_range[ids, 1]-.025, frozen[scene.root:scene.root+3]+[.08,.08,.012], m.jnt_range[legids, 1]-.025, [.04,.04,.15]]
    d.qpos[:] = frozen
    mujoco.mj_kinematics(m, d)
    hand = m.site('robot/rh_palm_touch').id
    hp, hr = d.site_xpos[hand].copy(), d.site_xmat[hand].reshape(3,3).copy()
    feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    fp, fr = d.xpos[feet].copy(), d.xmat[feet].reshape(2,3,3).copy()
    root = Rotation.from_quat(frozen[scene.root+3:scene.root+7][[1,2,3,0]])
    leaf_rotation = d.xmat[scene.leaf].reshape(3,3).copy()
    leaf_position = d.xpos[scene.leaf].copy()
    start_position = leaf_rotation.T@(d.site_xpos[scene.palm]-leaf_position)
    start_normal = leaf_rotation.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]
    initial = np.r_[frozen[qa], np.zeros(3)]
    previous = initial.copy()
    nominal_indices = [names.index(name) for name in JOINT_NAMES]
    path, rows = [], []
    for i in range(101):
        u = i/100
        blend = u**3*(10+u*(-15+6*u))
        phase = min(1., i/margin_nodes)
        interior_blend = phase**3*(10+phase*(-15+6*phase))
        node_lower = (1-interior_blend)*np.minimum(lower,initial-1e-12)+interior_blend*lower
        node_upper = (1-interior_blend)*np.maximum(upper,initial+1e-12)+interior_blend*upper
        nominal = initial.copy()
        nominal[nominal_indices] += preferences['nominal'][i]-preferences['nominal'][0]
        receive_position = preferences['position'][i]+(1-blend)*(start_position-preferences['position'][0])
        receive_normal = preferences['normal'][i]+(1-blend)*(start_normal-preferences['normal'][0])
        if np.linalg.norm(receive_normal) < 1e-6:
            raise ValueError('Ambiguous rebased palm normal')
        receive_normal /= np.linalg.norm(receive_normal)

        def residual(x):
            q = frozen.copy()
            q[qa] = x[:-3]
            quat = (Rotation.from_rotvec(x[-3:])*root).as_quat()
            q[scene.root+3:scene.root+7] = quat[[3,0,1,2]]
            d.qpos[:] = q
            mujoco.mj_kinematics(m,d)
            return np.r_[pose_weight*(leaf_rotation.T@(d.site_xpos[scene.palm]-leaf_position)-receive_position),
                (pose_weight/10)*(leaf_rotation.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]-receive_normal),
                pose_weight*(d.site_xpos[hand]-hp), (pose_weight/10)*Rotation.from_matrix(d.site_xmat[hand].reshape(3,3)@hr.T).as_rotvec(),
                100*(x[15:18]-initial[15:18]), 10*x[-3:],
                *[np.r_[pose_weight*(d.xpos[b]-pos),(pose_weight/10)*Rotation.from_matrix(d.xmat[b].reshape(3,3)@rot.T).as_rotvec()] for b,pos,rot in zip(feet,fp,fr)],
                .001*(x-nominal), .01*(x-previous)]

        if i == 0:
            x, converged, nfev = initial.copy(), True, 0
        else:
            fit = least_squares(residual, np.clip(previous,node_lower,node_upper),
                bounds=(node_lower,node_upper), max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            x, converged, nfev = fit.x, bool(fit.success), int(fit.nfev)
        residual(x)
        previous = x.copy()
        q = d.qpos.copy()
        check = static_pose_check(m,d,coordinate=u)
        pe = float(max(np.linalg.norm(d.site_xpos[hand]-hp), max(np.linalg.norm(d.xpos[b]-pos) for b,pos in zip(feet,fp))))
        re = float(max(np.linalg.norm(Rotation.from_matrix(d.site_xmat[hand].reshape(3,3)@hr.T).as_rotvec()), max(np.linalg.norm(Rotation.from_matrix(d.xmat[b].reshape(3,3)@rot.T).as_rotvec()) for b,rot in zip(feet,fr))))
        receive_error = float(np.linalg.norm(leaf_rotation.T@(d.site_xpos[scene.palm]-leaf_position)-receive_position))
        normal_error = float(np.linalg.norm(leaf_rotation.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]-receive_normal))
        tilt = float(np.degrees(np.arccos(np.clip((Rotation.from_quat(q[scene.root+3:scene.root+7][[1,2,3,0]]).as_matrix())[2,2],-1,1))))
        row = dict(index=i, coordinate=u, max_position_error_m=pe, max_rotation_error_rad=re,
            receiving_position_error_m=receive_error, receiving_normal_error=normal_error,
            root_tilt_deg=tilt, solver_converged=converged, solver_nfev=nfev, static_check=check)
        row['passed'] = bool(check['passed'] and pe <= .001 and re <= .01 and receive_error <= .001 and normal_error <= .01 and tilt <= 4)
        rows.append(row)
        path.append(q)
        if i % 10 == 0 or not row['passed']:
            print(json.dumps({k:row[k] for k in ('index','passed','max_position_error_m','receiving_position_error_m','root_tilt_deg')}),flush=True)
    return np.asarray(path), rows


def generate_fixed_body_path(scene, frozen, preferences, *, pitch_bump=0.):
    """Fit a fresh receiving endpoint, then move only the seven LH arm joints.

    The old endpoint supplies the panel plane and palm normal. As in the existing
    landed-left planner, contact may shift by at most 15 cm along the panel. This
    is a declared new target, not acceptance of tracking error to the old target.
    Every path target is recomputed using this exact recorded root/torso state.
    """
    if not np.isfinite(pitch_bump) or not 0 <= pitch_bump <= 1.2:
        raise ValueError('Use a bounded 0..1.2 rad geometric shoulder-pitch arc')
    m,d = scene.m,scene.d
    ids = np.array([m.joint('robot/'+n).id for n in JOINT_NAMES[1:]])
    qa = m.jnt_qposadr[ids]
    lo,hi = m.jnt_range[ids,0]+.025,m.jnt_range[ids,1]-.025
    start = frozen[qa].copy()
    old_position,old_normal = preferences['position'][-1],preferences['normal'][-1]
    nominal = np.clip(preferences['nominal'][-1,1:],lo,hi)

    def pose(q):
        d.qpos[:] = frozen
        d.qpos[qa] = q
        mujoco.mj_kinematics(m,d)
        rot = d.xmat[scene.leaf].reshape(3,3)
        return rot.T@(d.site_xpos[scene.palm]-d.xpos[scene.leaf]),rot.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]

    def residual(q):
        position,normal = pose(q)
        # Reserve 1 mm inside the unchanged 15 cm target-displacement limit.
        tangent_excess = np.maximum(abs(position[[0,2]]-old_position[[0,2]])-.149,0.)
        return np.r_[200*(position[1]-old_position[1]),20*(normal-old_normal),
                     200*tangent_excess,.015*(q-nominal)]

    fit = least_squares(residual,nominal,bounds=(lo,hi),max_nfev=500,
        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    goal = fit.x
    position,normal = pose(goal)
    fit_checks = dict(panel_plane_error=abs(position[1]-old_position[1])<=.001,
        palm_normal_error=np.linalg.norm(normal-old_normal)<=.01,
        bounded_panel_tangent=np.all(abs(position[[0,2]]-old_position[[0,2]])<=.15),
        interior_arm=np.all(goal>=lo) and np.all(goal<=hi),solver_converged=bool(fit.success))
    endpoint = dict(passed=bool(all(fit_checks.values())),checks=fit_checks,
        position=position.tolist(),normal=normal.tolist(),old_position=old_position.tolist(),
        panel_tangent_shift_m=(position[[0,2]]-old_position[[0,2]]).tolist(),
        plane_error_m=float(abs(position[1]-old_position[1])),normal_error=float(np.linalg.norm(normal-old_normal)),
        scope='Fresh bounded endpoint fit; original target qualification not inherited',solver_nfev=int(fit.nfev))
    path,rows = [],[]
    for i in range(101):
        u = i/100
        blend = u**3*(10+u*(-15+6*u))
        q = frozen.copy()
        q[qa] = start+blend*(goal-start)
        # A declared geometric waypoint preference, independently audited below.
        q[qa[0]] += pitch_bump*np.sin(np.pi*u)**2
        d.qpos[:] = q
        check = static_pose_check(m,d,coordinate=u)
        row = dict(index=i,coordinate=u,passed=bool(check['passed'] and endpoint['passed']),
            static_check=check,exact_frozen_root_legs_torso_opposite_arm_and_fingers=True)
        path.append(q)
        rows.append(row)
    return np.asarray(path),rows,endpoint


def can_export_route(*, source_passed, sampled_passed, dense_passed, development):
    return all((source_passed is True, sampled_passed is True, dense_passed is True, development is False))


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    robot, door, trial, preferences = [Path(p).resolve() for p in (args.robot,args.door,args.source_run,args.preferences)]
    if door.is_dir():
        door = door/'door.xml'
    qualification = source_qualification(trial)
    if not qualification['passed'] and not args.development_source:
        raise ValueError('Source physics/pad/whole-handle audits must pass, or explicitly select --development-source (never exports a runtime route)')
    numeric_preferences = resample_preferences(json.loads(preferences.read_text()))
    if not np.isfinite(args.receiving_normal_offset_m) or abs(args.receiving_normal_offset_m) > .005:
        raise ValueError('Declare a receiving-plane preference offset within 5 mm; geometry gates are unchanged')
    numeric_preferences['position'][:,1] += args.receiving_normal_offset_m
    scene = LandedLeftScene(robot,door)
    frozen, snapshot = load_snapshot(trial,robot,door,scene,at_time=args.at_time)
    source_paths = [Path(__file__).resolve(), ROOT/'doorbench/dexterous/landed_left_planner.py',
        ROOT/'doorbench/dexterous/landed_left_audit.py', ROOT/'scripts/dexterous/audit_standing_transfer_path.py',
        ROOT/'doorbench/dexterous/standing_transfer.py', ROOT/'doorbench/dexterous/transfer_contact_geometry.py']
    hashes = {**snapshot['input_sha256'], str(preferences):sha(preferences),
              **{str(p):sha(p) for p in source_paths},
              **{r['path']:r['sha256'] for r in qualification['reports'] if r['exists']},
              **{p:digest for r in qualification['reports'] for p,digest in r['input_sha256'].items()}}
    args.output.mkdir(parents=True)
    for path in source_paths:
        shutil.copy2(path,args.output/path.name)
    write_json(args.output/'snapshot.json',snapshot)
    write_json(args.output/'preferences.json',dict(scope='Numerical preferences only; no old pass or source identity consumed',
        input=str(preferences), input_sha256=sha(preferences), values=numeric_preferences))
    identity = robot_design_identity(robot)
    endpoint = None
    if args.path_profile == 'fixed-body-joint-v1':
        path,rows,endpoint = generate_fixed_body_path(scene,frozen,numeric_preferences,pitch_bump=args.reach_pitch_bump)
    else:
        if args.reach_pitch_bump:
            raise ValueError('Pitch arc requires the fixed-body joint profile')
        path,rows = generate_path(scene,frozen,numeric_preferences,pose_weight=args.pose_weight,margin_nodes=args.initial_margin_transition_nodes)
    sampled_passed = len(path) == 101 and all(r['passed'] for r in rows)
    report = dict(schema='doorbench.local-standing-transfer-planning.v1',passed=sampled_passed,
        scope=__doc__,physics_steps=0,source_qualification=qualification,development_source=args.development_source,
        robot_source_design_identity=identity,source_snapshot_sha256=sha(args.output/'snapshot.json'),
        preferences_are_numerical_only=True,path_profile=args.path_profile,receiving_endpoint_fit=endpoint,
        receiving_normal_offset_m=args.receiving_normal_offset_m,
        reach_pitch_bump_rad=args.reach_pitch_bump,pose_weight=args.pose_weight,
        initial_margin_transition_nodes=args.initial_margin_transition_nodes,samples=rows,path_qpos=path,input_sha256=hashes)
    report_path = (args.output/'report.json').resolve()
    write_json(report_path,report)
    proof = (args.output/'dense-audit.json').resolve()
    subprocess.run([sys.executable,str(ROOT/'scripts/dexterous/audit_standing_transfer_path.py'),
        '--robot',str(robot),'--door',str(door),'--path',str(report_path),'--output',str(proof)],check=True,cwd=ROOT)
    dense = json.loads(proof.read_text())
    if any(sha(p) != digest for p,digest in hashes.items()):
        raise ValueError('Frozen source changed while planning')
    m,d = scene.m,scene.d
    joint_names = scene.robot_names
    jointqa = [m.joint('robot/'+n).qposadr[0] for n in joint_names]
    targets = []
    for q in path:
        d.qpos[:] = q
        mujoco.mj_kinematics(m,d)
        rotation = d.xmat[scene.leaf].reshape(3,3)
        targets.append(dict(phase='left_reach',leaf_rad=float(q[m.joint('leaf_hinge').qposadr[0]]),
            position=(rotation.T@(d.site_xpos[scene.palm]-d.xpos[scene.leaf])).tolist(),
            normal=(rotation.T@d.site_xmat[scene.palm].reshape(3,3)[:,2]).tolist(),
            nominal=[float(q[m.joint('robot/'+n).qposadr[0]]) for n in JOINT_NAMES]))
    candidate = dict(schema='doorbench.local-standing-transfer-candidate.v1',
        scope='Geometric candidate only. A failed source is not a qualified grasp or executable transfer.',
        source_qualification=qualification,development_source=args.development_source,
        sampled_geometry_passed=sampled_passed,dense_geometry_passed=dense['passed'],
        robot_path=str(robot),door_path=str(door),robot_xml_sha256=sha(robot),door_xml_sha256=sha(door),
        robot_source_design_identity=identity,scene_path_source=str(report_path),scene_path_sha256=sha(report_path),
        dense_audit_path=str(proof),dense_audit_sha256=sha(proof),joint_names=joint_names,
        root_path=path[:,scene.root:scene.root+7],joint_path=path[:,jointqa],left_joint_names=JOINT_NAMES,
        left_targets=targets,attained_trial=str(trial),attained_time_s=snapshot['pose_time_s'],
        attained_manifest_sha256=sha(trial/'manifest.json'),physics_steps=0)
    allowed = can_export_route(source_passed=qualification['passed'],sampled_passed=sampled_passed,
        dense_passed=dense['passed'],development=args.development_source)
    candidate['runtime_route_exported'] = allowed
    if allowed:
        route = {**candidate,'schema':'doorbench.standing-transfer.v1','geometric_screen_passed':True}
        validate_route_geometry(route)
        write_json(args.output/'transfer.json',route)
    write_json(args.output/'candidate.json',candidate)
    print(json.dumps(dict(sampled_geometry_passed=sampled_passed,dense_geometry_passed=dense['passed'],
        source_physics_passed=qualification['passed'],runtime_route_exported=allowed,output=str(args.output))),flush=True)
    return 0 if sampled_passed and dense['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','robot','door','preferences','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--at-time',type=float,help='Exact recorded pre-step time; default last completed interval start')
    parser.add_argument('--development-source',action='store_true',help='Explicit failed-source geometric development only; never writes transfer.json')
    parser.add_argument('--pose-weight',type=float,default=200.)
    parser.add_argument('--initial-margin-transition-nodes',type=int,default=20)
    parser.add_argument('--path-profile',choices=('rebase-cartesian-v1','fixed-body-joint-v1'),default='rebase-cartesian-v1')
    parser.add_argument('--reach-pitch-bump',type=float,default=0.,help='Declared shoulder-pitch arc for fixed-body-joint-v1, within0..1.2rad')
    parser.add_argument('--receiving-normal-offset-m',type=float,default=0.,help='Explicit panel-local Y target offset, within +/-5mm; all original geometry gates remain')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
