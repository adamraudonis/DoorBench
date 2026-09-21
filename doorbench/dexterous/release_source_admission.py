"""Bind a release plan to qualified recorded evidence without inventing return."""
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .grasp_verification import grasp_profile
from .json_record_stream import iter_json_object_array

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound_inputs(report, source):
    declared = report.get('input_sha256')
    if not isinstance(declared, dict) or not declared:
        raise ValueError('Independent source audit must bind its actual input bytes')
    result = {}
    for name, digest in declared.items():
        raw = Path(name)
        candidates = [raw] if raw.is_absolute() else [source/raw, ROOT/raw]
        matches = {p.resolve() for p in candidates if p.is_file() and sha(p) == digest}
        if len(matches) != 1:
            raise ValueError('Audited release source bytes changed: '+name)
        result[str(matches.pop())] = digest
    return result


def admit_release_source(source, *, profile='distal-pad-v1',
                         contact_audit_name='independent-pad-audit.json', measured_rest=False):
    """A stricter explicit bridge for native transfer; legacy return stays explicit."""
    source = Path(source).resolve()
    grasp_profile(profile)
    if Path(contact_audit_name).name != contact_audit_name:
        raise ValueError('Contact audit must be a named file within its source run')
    names = ['report.json', contact_audit_name, 'independent-whole-handle-audit.json']
    if measured_rest:
        names.append('independent-transfer-audit.json')
    inputs = {}
    reports = {}
    for index, name in enumerate(names):
        path = source/name
        row = json.loads(path.read_text())
        if row.get('passed') is not True or not all(v is True for v in row.get('checks', {}).values()):
            raise ValueError('Fully qualified physical release source required: '+name)
        inputs[str(path)] = sha(path)
        reports[name] = row
        if index:
            bindings = bound_inputs(row, source)
            inputs.update(bindings)
            if measured_rest:
                required = {str(source/'manifest.json'), str(source/'raw-transitions/manifest.json')}
                if index == 1:
                    required.add(str(source/'report.json'))
                if not required <= bindings.keys():
                    raise ValueError('Native source audit does not bind the actual episode archive')
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['configuration']
    def asset_path(name):
        p = Path(cfg[name])
        return p if p.is_absolute() else Path(manifest['working_directory'])/p
    robot = asset_path('robot')
    door = asset_path('door')/'door.xml'
    if sha(robot) != manifest['inputs']['robot']['sha256'] or sha(door) != manifest['inputs']['door']['door.xml']:
        raise ValueError('Original release-source model bytes changed')
    inputs.update({str(p.resolve()): sha(p) for p in (robot, door)})
    if cfg.get('grasp_profile', 'distal-pad-v1') != profile:
        raise ValueError('Release profile must match the prospectively declared source profile')
    if reports[contact_audit_name].get('grasp_profile', 'distal-pad-v1') != profile:
        raise ValueError('Independent contact source uses another grasp profile')
    inputs[str(manifest_path)] = sha(manifest_path)
    if profile != 'distal-pad-v1':
        declaration_path = source/'grasp-profile-definition.json'
        declaration = json.loads(declaration_path.read_text())
        original = Path(cfg['grasp_profile_definition'])
        if not original.is_absolute():
            original = Path(manifest['working_directory'])/original
        if (sha(original) != sha(declaration_path) or declaration.get('profile') != profile
                or declaration.get('robot_xml_sha256') != manifest['inputs']['robot']['sha256']
                or reports['report.json'].get('grasp_profile') != profile):
            raise ValueError('Source-bound prospective profile declaration changed')
        inputs[str(declaration_path)] = sha(declaration_path)
        inputs[str(original.resolve())] = sha(original)
    rest = None
    if measured_rest:
        archive_path = source/'raw-transitions/manifest.json'
        archive = json.loads(archive_path.read_text())
        if archive.get('complete') is not True or archive.get('schema') != 'doorbench.native-transitions.v1' or not archive.get('chunks'):
            raise ValueError('Complete actual native transfer archive required')
        if reports['report.json'].get('checks', {}).get('final_left_palm_support') is not True:
            raise ValueError('Palm-only physical transfer qualification required')
        trajectory = source/'trajectory.npz'
        with np.load(trajectory, allow_pickle=False) as data:
            terminal = float(data['terminal_time_s'])
            terminal_qpos = data['terminal_qpos'].copy()
        chunk = archive['chunks'][-1]
        chunk_path = (source/'raw-transitions'/chunk['file']).resolve()
        if chunk_path.parent != source/'raw-transitions' or sha(chunk_path) != chunk['sha256']:
            raise ValueError('Actual terminal transition chunk changed')
        with np.load(chunk_path, allow_pickle=False) as raw:
            if (not np.isfinite(terminal_qpos).all()
                    or not np.array_equal(raw['qpos_after'][-1], terminal_qpos)
                    or abs(float(raw['interval_end_s'][-1])-terminal) > 1e-8):
                raise ValueError('Release source trajectory differs from the actual terminal transition')
        physics = source/'physics-steps.json.gz'
        tail = deque(maxlen=251)
        with gzip.open(physics, 'rt') as stream:
            for row in iter_json_object_array(stream):
                tail.append(row)
        times = np.array([r['sim_time_s'] for r in tail])
        if (len(tail) != 251 or not np.isfinite(times).all()
                or abs(times[-1]-terminal) > 1e-8
                or not np.allclose(np.diff(times), .002, atol=1e-8, rtol=0)):
            raise ValueError('Continuous actual half-second endpoint evidence required')
        if not all(r['pad_grasp']['valid_pad_grasp']
                and r['left_surface']['palm_normal_load_N'] >= 2
                and abs(r['handle_angle_rad']) <= .05
                and abs(r['bolt_slide_m']) <= .001 and .075 <= r['door_q'] <= .10 for r in tail):
            raise ValueError('Actual endpoint lacks sustained opposed grip, palm support or mechanism rest')
        inputs.update({str(p): sha(p) for p in (archive_path, chunk_path, trajectory, physics)})
        rest = dict(terminal_time_s=terminal, window_samples=251,
            minimum_palm_load_N=min(r['left_surface']['palm_normal_load_N'] for r in tail),
            maximum_abs_operator_rad=max(abs(r['handle_angle_rad']) for r in tail),
            maximum_abs_latch_m=max(abs(r['bolt_slide_m']) for r in tail),
            endpoint_contacts=tail[-1]['pad_grasp']['contacts'],
            scope='Measured spring-rest transfer endpoint; no lever-return milestone')
    return dict(grasp_profile=profile, contact_audit_name=contact_audit_name,
                measured_rest=rest, input_sha256=inputs)
