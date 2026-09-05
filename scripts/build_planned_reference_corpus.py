#!/usr/bin/env python3
"""Parallel, resumable generation and independent validation of motion candidates.

Outputs are separate from the immutable native recordings and generated assets.
Acceptance means a complete proposal passes the independent sampled kinematic
and task-evidence checks; it never certifies dynamics or visual naturalness.
Each attempt gets a fresh directory. Existing attempts require exact provenance
and artifact checksums to resume; --force archives them before replacement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import faulthandler
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading
import traceback
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doorbench.benchmark_eligibility import is_benchmark_eligible, require_benchmark_eligible, collection_counts

SCHEMA = 'doorbench.planned-reference-corpus.v1'
RESULT_SCHEMA = 'doorbench.planned-reference-attempt.v1'
STATUSES = ('unresolved', 'rejected', 'accepted_kinematic')
SOURCE_FILES = ('door.xml', 'model.json', 'spec.json')
WORKER_THREADS = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
ISOLATION_MODE = 'fresh_python_process_per_door'
SCOPE = ('Complete proposal plus independent sampled kinematic/task-evidence checks. '
         'Source benchmark outcome is separate from new actor completion. '
         'No dynamics, balance, force-closure, continuous collision or personal visual approval.')


class TaskUnresolved(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def generator_provenance(root=ROOT):
    """Pure Python inventory; plan-only never imports the solver or MuJoCo."""
    root = Path(root)
    paths = sorted((root/'doorbench/reference').glob('*.py'))
    paths += [root/'doorbench/benchmark_eligibility.py', root/'scripts/validate_planned_reference.py',
              root/'scripts/build_planned_reference_corpus.py', root/'pyproject.toml']
    if not paths or not (root/'doorbench/reference/solve.py').is_file():
        raise ValueError('Generator source inventory is missing solve.py')
    versions = {}
    for package in ('numpy', 'mujoco', 'mink', 'daqp', 'qpsolvers', 'scipy', 'shapely'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    payload = {'files': {str(p.relative_to(root)): sha(p) for p in sorted(paths)},
               'runtime': {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                           'platform': platform.system(), 'machine': platform.machine(), 'packages': versions,
                           'worker_thread_environment': WORKER_THREADS, 'isolation_mode': ISOLATION_MODE}}
    return {**payload, 'sha256': identity(payload)}


def checked_output(out, assets, recordings):
    out = Path(out).resolve()
    # Both descendants and ancestors are unsafe: a corpus cannot be the source
    # directory, nor may a forced per-door rename encompass a source tree.
    for protected in (Path(assets).resolve(), Path(recordings).resolve()):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError(f'Output overlaps immutable input tree: {protected}')
    return out


def xml_dependencies(source, assets):
    """Include actual mesh/texture/include bytes used when compiling native XML.

    Merely hashing door.xml misses shared hardware OBJ edits. Generated inputs
    use ordinary compiler meshdir/texturedir paths; external resources outside
    the supplied asset tree fail closed rather than escaping the inventory.
    """
    assets = Path(assets).resolve(); files = set(); visited = set()

    def visit(path, inherited=None):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        root = ET.parse(path).getroot()
        compiler = root.find('compiler')
        attributes = dict(inherited or {})
        if compiler is not None:
            attributes.update(compiler.attrib)
        if attributes.get('strippath') == 'true':
            raise ValueError('Unsupported strippath compiler dependency semantics')
        for element in root.iter():
            name = element.get('file')
            if not name:
                continue
            base = path.parent
            if element.tag in ('mesh', 'skin', 'hfield'):
                base /= attributes.get('meshdir', attributes.get('assetdir', ''))
            elif element.tag == 'texture':
                base /= attributes.get('texturedir', attributes.get('assetdir', ''))
            dependency = (base/name).resolve()
            if dependency != assets and assets not in dependency.parents:
                raise ValueError(f'Native resource escapes assets: {dependency}')
            if not dependency.is_file():
                raise FileNotFoundError(f'Missing native resource: {dependency}')
            files.add(dependency)
            if element.tag == 'include':
                visit(dependency, attributes)
    visit(Path(source))
    return sorted(files)


def prepare_plan(assets, recordings, out, *, doors='all', fps=60, max_frames=None,
                 gait_profile='smooth', timeout_s=600., force=False, generator_root=ROOT):
    assets = Path(assets).resolve(); recordings = Path(recordings).resolve()
    out = checked_output(out, assets, recordings)
    if type(fps) is not int or not 10 <= fps <= 60:
        raise ValueError('fps must be an integer from 10 through 60')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 2):
        raise ValueError('max_frames must be at least two; truncated proposals cannot be accepted')
    if gait_profile not in ('smooth', 'controlled', 'wide_turns', 'compact'):
        raise ValueError('Unknown gait profile')
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout must be finite and positive')
    manifest_path = assets/'manifest.json'; manifest_hash = sha(manifest_path)
    manifest = read_json(manifest_path)
    inventory = manifest['doors']; ids = [row['id'] for row in inventory]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', x) for x in ids):
        raise ValueError('Dataset IDs must be unique safe directory names')
    eligible_ids = [row['id'] for row in inventory if is_benchmark_eligible(row)]
    selected = eligible_ids if doors == 'all' else [x.strip() for x in doors.split(',')]
    if not selected or len(selected) != len(set(selected)) or set(selected)-set(ids):
        raise ValueError('Requested doors must be unique IDs present in the asset manifest')
    by_id = {row['id']: row for row in inventory}
    for door_id in selected:
        require_benchmark_eligible(by_id[door_id], operation='planned corpus selection')
    recording_index_path = recordings/'index.json'; recording_index_hash = sha(recording_index_path)
    recording_index = read_json(recording_index_path)
    if recording_index.get('schema') != 'doorbench.reference-motion.v1' or recording_index.get('manifest_sha256') != manifest_hash:
        raise ValueError('Native recording index schema or asset-manifest binding mismatch')
    native_rows = recording_index['clips']
    native_by_id = {row['door_id']: row for row in native_rows}
    if len(native_by_id) != len(native_rows):
        raise ValueError('Native recording index has duplicate door IDs')
    generator = generator_provenance(generator_root)
    options = {'fps': fps, 'max_frames': max_frames, 'gait_profile': gait_profile}
    execution = {'isolation_mode': ISOLATION_MODE, 'timeout_s': float(timeout_s)}
    rows = []; cache = {}

    def cached_sha(path):
        path = Path(path); stat = path.stat(); key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in cache:
            cache[key] = sha(path)
        return cache[key]

    families = {row['id']: row.get('family', 'unknown') for row in inventory}
    for door_id in selected:
        row = {'door_id': door_id, 'family': families[door_id], 'status': 'unresolved', 'action': 'run'}
        try:
            directory = assets/'doors'/door_id
            require_benchmark_eligible(read_json(directory/'spec.json'), operation='planned corpus generation')
            sources = {name: cached_sha(directory/name) for name in SOURCE_FILES}
            native_clip = recordings/'clips'/f'{door_id}.json'
            native_trajectory = recordings/'trajectories'/f'{door_id}.npz'
            recording = read_json(native_clip)
            if recording.get('schema') != 'doorbench.reference-motion.v1' or recording.get('door_id') != door_id:
                raise ValueError('Native recording identity/schema mismatch')
            if recording.get('source_sha256') != sources:
                raise ValueError('Native recording is bound to different source bytes')
            outcome = recording.get('outcome')
            if not isinstance(outcome, dict) or outcome.get('door_id') != door_id or outcome.get('scenario') != recording.get('scenario'):
                raise ValueError('Native outcome identity/scenario mismatch')
            dependencies = {str(p.relative_to(assets)): cached_sha(p) for p in xml_dependencies(directory/'door.xml', assets)}
            recorded = {'clip.json': cached_sha(native_clip), 'trajectory.npz': cached_sha(native_trajectory)}
            published = native_by_id[door_id]
            if (published.get('clip') != f'clips/{door_id}.json' or published.get('trajectory') != f'trajectories/{door_id}.npz' or
                    published.get('clip_sha256') != recorded['clip.json'] or published.get('trajectory_sha256') != recorded['trajectory.npz'] or
                    published.get('source_sha256') != sources or published.get('scenario') != recording['scenario'] or
                    published.get('success') != outcome['success'] or published.get('outcome') != outcome['outcome']):
                raise ValueError('Native clip/trajectory/outcome differs from its frozen recording index')
            provenance = {'manifest_sha256': manifest_hash, 'source_sha256': sources, 'native_resources_sha256': dependencies,
                          'recording_index_sha256': recording_index_hash, 'recording_sha256': recorded,
                          'generator_sha256': generator['sha256'], 'options': options, 'execution': execution}
            inputs = {str(directory/name): digest for name, digest in sources.items()}
            inputs.update({str(assets/name): digest for name, digest in dependencies.items()})
            inputs.update({str(manifest_path): manifest_hash, str(native_clip): recorded['clip.json'], str(native_trajectory): recorded['trajectory.npz']})
            inputs[str(recording_index_path)] = recording_index_hash
            job = {'door_id': door_id, 'assets': str(assets), 'recordings': str(recordings), 'out': str(out),
                   'generator_root': str(Path(generator_root).resolve()), 'generator': generator, 'inputs': inputs,
                   'provenance': provenance, 'identity_sha256': identity(provenance), 'source_outcome': outcome,
                   'family': row['family'], 'force': bool(force)}
            row['job'] = job
            existing = out/door_id
            if existing.is_symlink():
                raise ValueError('Refuse a symlink at the per-door output path')
            if existing.exists() and not force:
                try:
                    resumed = resumable_result(job)
                    row.update(action='resume', status=resumed['status'], result=resumed)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(action='blocked', error=f'Existing output retained; use --force to archive and retry: {exc}')
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
            row.update(action='blocked', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    if (sha(manifest_path) != manifest_hash or sha(recording_index_path) != recording_index_hash or
            generator_provenance(generator_root) != generator):
        raise ValueError('Manifest, native index or generator changed during inventory; prepare again after source freeze')
    return {'schema': SCHEMA, 'created_at': timestamp(), 'scope': SCOPE, 'selected_ids': selected,
            **collection_counts(inventory),
            'manifest_sha256': manifest_hash, 'generator': generator, 'options': options,
            'execution': execution, 'rows': rows, 'out': str(out)}


def verify_job(job):
    require_benchmark_eligible(job, operation='planned corpus execution')
    require_benchmark_eligible(read_json(Path(job['assets'])/'doors'/job['door_id']/'spec.json'), operation='planned corpus execution')
    for path, expected in job['inputs'].items():
        if sha(path) != expected:
            raise ValueError(f'Input changed since preparation: {path}')
    if generator_provenance(job['generator_root']) != job['generator']:
        raise ValueError('Generator/runtime changed since preparation')


def acceptance(clip, validation, job, directory):
    """Fail closed even if an accidentally permissive validator says accepted."""
    bindings = (clip.get('schema') == 'doorbench.planned-reference.v1' and
                validation.get('schema') == 'doorbench.planned-reference-validation.v1' and
                clip.get('door_id') == job['door_id'] and
                clip.get('source_sha256') == job['provenance']['source_sha256'] and
                clip.get('proposal', {}).get('source_outcome') == job['source_outcome'] and
                validation.get('door_id') == job['door_id'] and
                validation.get('clip_sha256') == sha(directory/'clip.json') and
                validation.get('trajectory_sha256') == sha(directory/'trajectory.npz') and
                validation.get('source_sha256') == job['provenance']['source_sha256'])
    completion = validation.get('task_completion', {})
    complete = clip.get('complete_proposal') is True and completion.get('complete_proposal') is True
    source = job['source_outcome']
    source_success = source.get('success') is True and source.get('outcome') == 'success' and not source.get('error')
    accepted = (bindings and complete and source_success and validation.get('accepted') is True and
                validation.get('kinematic_accepted') is True and validation.get('status') == 'accepted_kinematic' and
                completion.get('evidence_pass') is True and not validation.get('failure_counts') and
                not (directory/'failure.json').exists())
    return bool(accepted), {'complete_proposal': complete, 'task_evidence_pass': completion.get('evidence_pass') is True,
                            'artifact_bindings_verified': bool(bindings), 'source_success_declared': source_success}


def resumable_result(job, directory=None):
    directory = Path(directory) if directory is not None else Path(job['out'])/job['door_id']
    result = read_json(directory/'result.json')
    if (result.get('schema') != RESULT_SCHEMA or result.get('identity_sha256') != job['identity_sha256'] or
            result.get('status') not in STATUSES or result.get('door_id') != job['door_id']):
        raise ValueError('Attempt source/generator/options identity mismatch')
    artifacts = result['artifacts']
    paths = list(directory.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Attempt contains a symlink')
    actual = {str(p.relative_to(directory)) for p in paths if p.is_file() and p != directory/'result.json'}
    if actual != set(artifacts):
        raise ValueError('Attempt artifact inventory changed (including stale failure files)')
    for name, digest in artifacts.items():
        path = directory/name
        if path.resolve().parent != directory.resolve() or path.is_symlink() or sha(path) != digest:
            raise ValueError(f'Attempt artifact changed: {name}')
    if result['status'] == 'accepted_kinematic':
        accepted, _ = acceptance(read_json(directory/'clip.json'), read_json(directory/'validation.json'), job, directory)
        if not accepted or 'failure.json' in artifacts:
            raise ValueError('Accepted attempt no longer satisfies the independent completion gate')
    return result


def publish_attempt(staged, destination, force):
    destination = Path(destination); staged = Path(staged)
    previous = None
    if destination.is_symlink():
        raise ValueError('Refuse a symlink at the per-door output path')
    if destination.exists():
        if not force:
            raise FileExistsError(f'Existing output retained: {destination}')
        history = destination.parent/'.history'; history.mkdir(exist_ok=True)
        previous = history/(destination.name+'.'+uuid.uuid4().hex)
        os.replace(destination, previous)
    try:
        os.replace(staged, destination)
    except BaseException:
        if previous is not None and not destination.exists():
            os.replace(previous, destination)
        raise
    return str(previous.relative_to(destination.parent)) if previous else None


def preflight_guide(job, guide_builder=None):
    """Cheap task-route proposal check before thousands of expensive IK frames.

    This is not an acceptance gate: guide existence and native source success
    never establish body feasibility or completion. The independent validator
    still checks every generated candidate. Route construction is repeated by
    solve_door so its existing interface and authored proposal remain intact.
    """
    if guide_builder is None:
        from doorbench.reference.guidance import make_guide
        guide_builder = make_guide
    options = job['provenance']['options']
    guide = guide_builder(Path(job['assets'])/'doors'/job['door_id'], job['recordings'],
                          fps=options['fps'], gait_profile=options['gait_profile'])
    meta = guide.metadata
    if meta.get('traversal') not in ('proposed', 'not_requested'):
        raise TaskUnresolved('guide_traversal_unresolved',
                             f"Full task route is unresolved: {meta.get('traversal_reason') or meta.get('traversal')}")
    return {'traversal': meta['traversal'], 'traversal_reason': meta.get('traversal_reason'),
            'scope': 'Route proposal only; independent kinematics and task completion have not passed.'}


def result_base(job, run_id):
    return {'schema': RESULT_SCHEMA, 'door_id': job['door_id'], 'family': job['family'], 'run_id': run_id,
            'created_at': timestamp(), 'status': 'unresolved', 'identity_sha256': job['identity_sha256'],
            'provenance': job['provenance'], 'generator': job['generator'], 'source_outcome': job['source_outcome'],
            'new_completion': None, 'scope': SCOPE}


def stage_marker(job, stage):
    # sys.__stderr__ remains the subprocess log even while solver stdout/stderr
    # are redirected to attempt.log. PID/stage evidence survives a native crash.
    print(json.dumps({'event': 'worker_stage', 'pid': os.getpid(), 'door_id': job['door_id'],
                      'stage': stage, 'time': timestamp()}), file=sys.__stderr__, flush=True)


def run_job(job, *, solver=None, validator=None, publish=True):
    """One process owns one fresh attempt. Injected functions support gate tests."""
    out = Path(job['out']); run_id = job.get('attempt_run_id', uuid.uuid4().hex)
    if not re.fullmatch(r'[a-f0-9]{32}', run_id):
        raise ValueError('Invalid attempt run ID')
    parent = out/'.attempts'/run_id; directory = parent/job['door_id']; directory.mkdir(parents=True)
    start = time.monotonic()
    result = result_base(job, run_id)
    try:
        with (directory/'attempt.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            stage_marker(job, 'verify_inputs_start')
            verify_job(job)
            stage_marker(job, 'verify_inputs_done')
            source = job['source_outcome']
            if source.get('success') is not True or source.get('outcome') != 'success' or source.get('error'):
                raise TaskUnresolved('native_source_unsuccessful',
                                     f"Source outcome is {source.get('outcome')}; a complete new task cannot be derived by this source-driven planner.")
            if solver is None:
                stage_marker(job, 'guide_preflight_start')
                result['guide_preflight'] = preflight_guide(job)
                stage_marker(job, 'guide_preflight_done')
                from doorbench.reference.solve import solve_door
                solver = solve_door
            if validator is None:
                from scripts.validate_planned_reference import validate
                validator = validate
            stage_marker(job, 'solve_door_start')
            solver(Path(job['assets'])/'doors'/job['door_id'], job['recordings'], parent, **job['provenance']['options'])
            stage_marker(job, 'solve_door_done')
            clip = read_json(directory/'clip.json')
            stage_marker(job, 'independent_validation_start')
            validation = validator(directory/'clip.json', directory/'trajectory.npz', job['assets'])
            stage_marker(job, 'independent_validation_done')
            atomic_json(directory/'validation.json', validation)
            passed, completion = acceptance(clip, validation, job, directory)
            stage_marker(job, 'final_input_verification_start')
            verify_job(job)
            stage_marker(job, 'final_input_verification_done')
            result.update(status='accepted_kinematic' if passed else 'rejected', new_completion=completion,
                          validation_status=validation.get('status'), failure_counts=validation.get('failure_counts', {}),
                          task_completion=validation.get('task_completion'), frames=clip.get('frames'), duration_s=clip.get('duration'))
            if not passed and not result['failure_counts']:
                result['failure_counts'] = {'corpus_acceptance_contract': 1}
    except Exception as exc:
        stage_marker(job, 'python_exception')
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        result.update(status='unresolved', error=error)
        if isinstance(exc, TaskUnresolved):
            result['reason_code'] = exc.code
            result['new_completion'] = {'complete_proposal': False, 'task_evidence_pass': False,
                                        'artifact_bindings_verified': False,
                                        'source_success_declared': job['source_outcome'].get('success') is True}
        atomic_json(directory/'failure.json', {'door_id': job['door_id'], 'run_id': run_id, 'status': 'unresolved', 'error': error})
    result['runtime_s'] = time.monotonic()-start
    result['artifacts'] = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    atomic_json(directory/'result.json', result)
    stage_marker(job, 'worker_result_written')
    if not publish:
        return result
    previous = publish_attempt(directory, out/job['door_id'], job['force'])
    if previous:
        result['previous_attempt'] = previous
        atomic_json(out/job['door_id']/'result.json', result)
    parent.rmdir()
    return result


def isolated_job(job, *, cancel=None, command_factory=None):
    """A native crash cannot poison another door's interpreter or future.

    The parent owns publication and keeps its process log open until the child
    has exited. This captures faulthandler/native stderr without invalidating an
    already-hashed worker log. command_factory is only a Python test seam; the
    public CLI always invokes this script's internal worker entry point.
    """
    run_id = uuid.uuid4().hex; out = Path(job['out'])
    parent = out/'.attempts'/run_id; parent.mkdir(parents=True)
    staged = parent/job['door_id']; request = {**job, 'attempt_run_id': run_id}
    job_path = parent/'job.json'; atomic_json(job_path, request)
    command = (command_factory(job_path) if command_factory else
               [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()), '--_worker-job', str(job_path)])
    started = time.monotonic(); process = None; failure = None
    execution = {**job['provenance']['execution'], 'returncode': None, 'pid': None, 'failure_kind': None}
    with (parent/'process.log').open('wb') as log:
        log.write((json.dumps({'event': 'parent_launch', 'parent_pid': os.getpid(), 'door_id': job['door_id'],
                              'run_id': run_id, 'time': timestamp()})+'\n').encode()); log.flush()
        try:
            if cancel is not None and cancel.is_set():
                failure = ('cancelled', 'Corpus stopped before this subprocess started')
            else:
                env = {**os.environ, **WORKER_THREADS, 'PYTHONFAULTHANDLER': '1', 'PYTHONUNBUFFERED': '1'}
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                execution['pid'] = process.pid
                while process.poll() is None:
                    elapsed = time.monotonic()-started
                    if cancel is not None and cancel.is_set():
                        failure = ('cancelled', 'Corpus stopped while the door subprocess was running')
                    elif elapsed >= execution['timeout_s']:
                        failure = ('timeout', f"Door subprocess exceeded {execution['timeout_s']:g} seconds")
                    if failure:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        break
                    try:
                        process.wait(timeout=min(.5, max(.01, execution['timeout_s']-elapsed)))
                    except subprocess.TimeoutExpired:
                        pass
                execution['returncode'] = process.returncode
                if failure is None and process.returncode != 0:
                    code = process.returncode
                    if code < 0:
                        try:
                            name = signal.Signals(-code).name
                        except ValueError:
                            name = f'signal {-code}'
                        failure = ('signal', f'Door subprocess terminated by {name}')
                        execution['signal'] = name
                    else:
                        failure = ('nonzero_exit', f'Door subprocess exited with code {code}')
        except Exception as exc:
            failure = ('launch_error', f'{type(exc).__name__}: {exc}')
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    execution['runtime_s'] = time.monotonic()-started
    if failure is None:
        try:
            result = resumable_result(job, staged)
            if result['run_id'] != run_id:
                raise ValueError('Worker returned a different attempt ID')
            if result['status'] != 'unresolved':
                verify_job(job)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failure = ('worker_contract', f'Worker output verification failed: {exc}')
    staged.mkdir(exist_ok=True)
    if failure:
        execution['failure_kind'] = failure[0]
        # Preserve any completed/partial child evidence. A failed execution is
        # unresolved, never a statement that the motion is physically infeasible.
        if (staged/'result.json').exists():
            os.replace(staged/'result.json', staged/'worker-result.json')
        result = result_base(job, run_id)
        result.update(reason_code='execution_failure', execution_failure_kind=failure[0],
                      error={'type': 'DoorExecutionFailure', 'message': failure[1], 'traceback': None})
        atomic_json(staged/'failure.json', {'door_id': job['door_id'], 'run_id': run_id,
                    'status': 'unresolved', 'reason_code': 'execution_failure', 'error': result['error'], 'execution': execution})
    shutil.copyfile(parent/'process.log', staged/'execution.log')
    result['execution'] = execution
    result.setdefault('runtime_s', execution['runtime_s'])
    result['artifacts'] = {p.name: sha(p) for p in sorted(staged.iterdir()) if p.is_file() and p.name != 'result.json'}
    atomic_json(staged/'result.json', result)
    previous = publish_attempt(staged, out/job['door_id'], job['force'])
    if previous:
        result['previous_attempt'] = previous
        atomic_json(out/job['door_id']/'result.json', result)
    atomic_json(parent/'execution.json', execution)
    return result


def entry_summary(row):
    result = row.get('result', {})
    source = result.get('source_outcome', row.get('job', {}).get('source_outcome', {}))
    return {'door_id': row['door_id'], 'family': row['family'], 'status': row['status'], 'action': row['action'],
            'error': row.get('error', result.get('error')),
            'source_outcome': {'success': source.get('success'), 'outcome': source.get('outcome'), 'scenario': source.get('scenario')},
            'new_completion': result.get('new_completion'), 'failure_counts': result.get('failure_counts'),
            'reason_code': row.get('reason_code', result.get('reason_code')),
            'execution_failure_kind': row.get('execution_failure_kind', result.get('execution_failure_kind')),
            'execution': result.get('execution'),
            'identity_sha256': row.get('job', {}).get('identity_sha256'),
            'result': row['door_id']+'/result.json' if result else None}


def write_snapshot(plan):
    rows = [entry_summary(row) for row in plan['rows']]; snapshot = uuid.uuid4().hex
    common = {'schema': SCHEMA, 'snapshot_id': snapshot, 'updated_at': timestamp(), 'scope': SCOPE,
              'selected_ids': plan['selected_ids'], 'manifest_sha256': plan['manifest_sha256'],
              'generator': plan['generator'], 'options': plan['options'], 'execution': plan['execution']}
    counts = Counter(row['status'] for row in rows)
    report = {**common, 'selected_count': len(rows), 'status_counts': {key: counts[key] for key in STATUSES},
              'action_counts': dict(Counter(row['action'] for row in rows)),
              'family_counts': dict(Counter(row['family'] for row in rows)),
              'accepted_ids': [row['door_id'] for row in rows if row['status'] == 'accepted_kinematic'],
              'source_outcome_counts': dict(Counter(str(row['source_outcome']['outcome']) for row in rows)),
              'execution_failure_counts': dict(Counter(row['execution_failure_kind'] or 'parent_error' for row in rows
                                                      if row['reason_code'] == 'execution_failure')),
              'personal_visual_review': 'not performed by this runner'}
    atomic_json(Path(plan['out'])/'index.json', {**common, 'doors': rows})
    atomic_json(Path(plan['out'])/'report.json', report)
    return report


@contextmanager
def corpus_lock(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with (Path(out)/'.corpus.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another corpus runner owns this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_plan(plan, workers, *, launcher=None, snapshot_interval_s=2.):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    refresh_results(plan)
    report = write_snapshot(plan)
    jobs = [row for row in plan['rows'] if row['action'] == 'run']
    if not jobs:
        return report
    # Threads only supervise OS subprocesses; no engine state enters the parent.
    # A child SIGSEGV cannot break a reusable Python worker pool or its queue.
    launcher = launcher or isolated_job
    cancel = threading.Event(); pool = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    last_snapshot = time.monotonic(); interrupted = False

    def collect(future, row):
        try:
            result = future.result()
            row.update(result=result, status=result['status'], action='completed')
        except Exception as exc:
            row.update(status='unresolved', action='worker_failed', reason_code='execution_failure',
                       execution_failure_kind='parent_error',
                       error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        print(f"{row['door_id']}: {row['status']} ({row['action']})", flush=True)

    try:
        for row in jobs:
            futures[pool.submit(launcher, row['job'], cancel=cancel)] = row
        for future in as_completed(futures):
            row = futures[future]
            collect(future, row)
            if time.monotonic()-last_snapshot >= snapshot_interval_s:
                report = write_snapshot(plan)
                last_snapshot = time.monotonic()
    except KeyboardInterrupt:
        interrupted = True; cancel.set()
        for future in futures:
            future.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=interrupted)
        if interrupted:
            for future, row in futures.items():
                if row['action'] != 'run':
                    continue
                if future.cancelled():
                    row.update(status='unresolved', action='not_started', reason_code='corpus_interrupted',
                               error='Corpus stopped before this queued door began')
                else:
                    collect(future, row)
            for row in jobs:
                if row['action'] == 'run':
                    row.update(status='unresolved', action='not_started', reason_code='corpus_interrupted',
                               error='Corpus stopped before this door was submitted')
    # Earlier accepted jobs must not remain current if source/generator bytes
    # changed while later jobs ran. Preserve their original attempt on disk,
    # but remove them from the current accepted corpus until explicitly rebuilt.
    refresh_results(plan)
    report = write_snapshot(plan)
    if interrupted:
        raise KeyboardInterrupt
    return report


def refresh_results(plan):
    for row in plan['rows']:
        if 'result' not in row:
            continue
        try:
            verify_job(row['job'])
            current = resumable_result(row['job'])
            if current['run_id'] != row['result']['run_id']:
                raise ValueError('Attempt was replaced during the corpus run')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='unresolved', action='blocked', error=f'Current input/output verification failed: {exc}')


def main(argv=None):
    raw_args = sys.argv[1:] if argv is None else list(argv)
    if raw_args[:1] == ['--_worker-job']:
        if len(raw_args) != 2:
            raise ValueError('Internal worker requires one structured job JSON')
        faulthandler.enable(all_threads=True)
        job = read_json(raw_args[1])
        stage_marker(job, 'worker_start')
        try:
            run_job(job, publish=False)
        finally:
            stage_marker(job, 'worker_end')
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default='assets'); parser.add_argument('--recordings', default='out/reference-motions')
    parser.add_argument('--out', default='out/reference-planned-corpus'); parser.add_argument('--doors', default='all')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1)); parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--gait-profile', choices=['smooth', 'controlled', 'wide_turns', 'compact'], default='smooth')
    parser.add_argument('--max-frames', type=int); parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--timeout', type=float, default=600., help='Maximum seconds per isolated door subprocess (default: 600)')
    parser.add_argument('--force', action='store_true', help='Archive existing selected door attempts and generate again')
    args = parser.parse_args(raw_args)
    if args.workers < 1:
        parser.error('workers must be positive')
    try:
        out = checked_output(args.out, args.assets, args.recordings)
        with corpus_lock(out):
            plan = prepare_plan(args.assets, args.recordings, out, doors=args.doors, fps=args.fps,
                                max_frames=args.max_frames, gait_profile=args.gait_profile,
                                timeout_s=args.timeout, force=args.force)
            if args.plan_only:
                atomic_json(out/'plan.json', plan)
                print(json.dumps({'selected': len(plan['rows']), 'actions': dict(Counter(r['action'] for r in plan['rows'])),
                                  'generator_sha256': plan['generator']['sha256'], 'plan': str(out/'plan.json')}))
                return 0
            previous_term = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, terminate)
            try:
                report = run_plan(plan, args.workers)
            finally:
                signal.signal(signal.SIGTERM, previous_term)
            print(json.dumps(report['status_counts']))
            return 2 if report['status_counts']['unresolved'] else 1 if report['status_counts']['rejected'] else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Corpus interrupted; active subprocesses stopped and completed attempts retained.\n')


if __name__ == '__main__':
    raise SystemExit(main())
