#!/usr/bin/env python3
"""Prepare and publish an immutable experimental planned-reference supplement.

Preparation is local and requires an idle, complete, integrity-checked corpus.
Only accepted motions enter research archives and browser playback. Native data
are referenced at their existing immutable Hub commit, never copied here.
Credentials are used only by the explicit publish command and never serialized.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from scripts import huggingface_release as common
except ModuleNotFoundError:  # Standalone helper shipped beside archive_helpers.py.
    import archive_helpers as common

SCHEMA = 'doorbench.planned-reference-release.v1'
BASE_REPO = 'adamraudonis/DoorBench'
BASE_TAG = 'v2026.09.05'
BASE_COMMIT = '6e17f0f588bf81fec0f04b2a329b471488164366'
BASE_RELEASE_SHA = 'b9c809bd405d72d1d5bc96a1611370926c3b245c02b6d8e7459199f7af329a73'
SCOPE = ('Experimental sampled kinematic replay on an original approximate adult rig. '
         'Native door motion is prescribed from scripted-hand recordings and retimed. '
         'No forces, balance, causal humanoid operation, grasp/lock semantics, '
         'original timed benchmark success or personal visual approval is certified.')
LICENSE = 'MIT'
BROWSER_LIMITS = {'packed_bytes': 64*1024**2, 'decoded_bytes': 256*1024**2,
                  'frames': 100000, 'pose_scalars_per_group': 16000000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix(tag):
    require(isinstance(tag, str) and re.fullmatch(r'planned-[A-Za-z0-9][A-Za-z0-9._-]*', tag),
            'Choose an experimental version beginning planned-; native release tags cannot be reused')
    return f'experimental/planned-reference/{tag}'


def native_dependency(path):
    """The base release metadata is small; no archive download is needed."""
    path = Path(path)
    require(common.sha256(path) == BASE_RELEASE_SHA, 'Native release metadata differs from pinned v2026.09.05')
    release = common.read(path)
    require(release['repo_id'] == BASE_REPO and release['release'] == BASE_TAG, 'Native release identity mismatch')
    inventory_path = path.parent/'inventory.json'
    require(common.sha256(inventory_path) == release['inventory_sha256'], 'Native release inventory checksum mismatch')
    inventory = common.read(inventory_path)
    base = f'https://huggingface.co/datasets/{BASE_REPO}/resolve/{BASE_COMMIT}/'
    return {'repo_id': BASE_REPO, 'release': BASE_TAG, 'commit': BASE_COMMIT, 'release_sha256': BASE_RELEASE_SHA,
            'manifest_sha256': release['summary']['dataset_manifest_sha256'],
            'recording_index_sha256': inventory['files']['reference-motions/index.json']['sha256'],
            'release_url': base+'release.json', 'inventory_sha256': release['inventory_sha256'],
            'components': {name: {**release['components'][name], 'url': base+release['components'][name]['path']}
                           for name in ('assets', 'reference-motions')},
            'browser_assets': 'Use the matching website asset bundle; MotionLab verifies source/hardware SHA-256. Native Hub assets are archive members, not individual URLs.'}


@contextmanager
def idle_corpus(corpus):
    """Read-only shared lock; never package while the runner owns its write lock."""
    import fcntl  # Preparation is local POSIX; the standalone downloader is portable.
    with (Path(corpus)/'.corpus.lock').open('rb') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Corpus is still running; packaging waits for completion') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def inspect_corpus(corpus, assets, recordings, native, *, expected_doors=1000, generator_root=ROOT):
    """Require every terminal result, exact runner provenance and current bytes."""
    from scripts.build_planned_reference_corpus import prepare_plan
    corpus = Path(corpus); index = common.read(corpus/'index.json'); report = common.read(corpus/'report.json')
    require(index.get('schema') == 'doorbench.planned-reference-corpus.v1' and index['snapshot_id'] == report.get('snapshot_id'),
            'Corpus index/report snapshots disagree')
    ids = [row['id'] for row in common.read(Path(assets)/'manifest.json')['doors']]
    rows = index['doors']; selected = index['selected_ids']
    require(len(ids) == expected_doors and len(set(ids)) == len(ids) and len(rows) == len(ids) and
            len(selected) == len(ids) and set(selected) == set(ids) and {r['door_id'] for r in rows} == set(ids),
            'Release requires every dataset door exactly once')
    require(index['manifest_sha256'] == native['manifest_sha256'] == common.sha256(Path(assets)/'manifest.json'),
            'Corpus dataset differs from the pinned native release')
    require(common.sha256(Path(recordings)/'index.json') == native['recording_index_sha256'],
            'Source recording index differs from the pinned native release')
    require(all(r.get('action') in ('completed', 'resume') and r.get('result') == f"{r['door_id']}/result.json" for r in rows),
            'Corpus has pending, blocked, unstarted or missing attempts; packaging waits for completion')
    options = index['options']; execution = index['execution']
    current = prepare_plan(assets, recordings, corpus, doors='all', fps=options['fps'],
                           max_frames=options['max_frames'], gait_profile=options['gait_profile'],
                           timeout_s=execution['timeout_s'], generator_root=generator_root)
    require(current['generator'] == index['generator'], 'Current generator/runtime differs from completed corpus')
    indexed = {row['door_id']: row for row in rows}; results = {}
    for row in current['rows']:
        prior = indexed[row['door_id']]
        require(row['action'] == 'resume', f"{row['door_id']}: result cannot resume with exact current provenance: {row.get('error')}")
        result = row['result']
        require(result['status'] == prior['status'] and result['identity_sha256'] == prior['identity_sha256'], 'Result/index status or identity mismatch')
        require(result['provenance']['recording_index_sha256'] == native['recording_index_sha256'], 'Attempt native-index binding mismatch')
        results[row['door_id']] = result
    counts = dict(Counter(r['status'] for r in rows))
    require(all(report['status_counts'].get(key, 0) == counts.get(key, 0) for key in ('accepted_kinematic', 'rejected', 'unresolved')),
            'Corpus status totals disagree')
    return {'index': index, 'results': results, 'counts': counts,
            'index_sha256': common.sha256(corpus/'index.json'), 'report_sha256': common.sha256(corpus/'report.json'),
            'accepted_bytes': sum((corpus/door/name).stat().st_size for door, result in results.items()
                                  if result['status'] == 'accepted_kinematic' for name in ('clip.json', 'trajectory.npz', 'validation.json'))}


def verify_commit(commit, generator, root=ROOT):
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit), 'Record a full committed source SHA before packaging')
    expected = {**generator['files']}
    for name in ('scripts/planned_reference_release.py', 'scripts/export_planned_reference_web.py',
                 'scripts/huggingface_release.py', 'LICENSE'):
        expected[name] = common.sha256(Path(root)/name)
    for name, digest in expected.items():
        common.safe_name(name)
        process = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=root, capture_output=True, check=False)
        require(process.returncode == 0 and hashlib.sha256(process.stdout).hexdigest() == digest,
                f'Source commit does not contain the frozen file: {name}')
    return expected


def redact(value):
    from scripts.export_planned_reference_web import public_text
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items() if key not in ('traceback', 'command', 'environment', 'pid')}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return public_text(value)


def project_rejection_reports(web, directory):
    """Preserve accepted audit bytes; label any redacted rejected report clearly."""
    from scripts.export_planned_reference_web import audit, encoded
    for row in web['doors']:
        if not row.get('reason'):
            failures = ', '.join(sorted(row.get('failure_counts') or {}))
            row['reason'] = ('Independent sampled kinematic and task-evidence checks passed.'
                             if row['status'] == 'accepted_kinematic' else
                             'Independent checks failed: '+failures if failures else
                             row.get('reason_code') or 'No accepted complete motion was produced.')
        if row['status'] == 'accepted_kinematic':
            continue
        descriptor = row.get('audits', {}).get('validation.json')
        if descriptor:
            original = common.regular_file(directory, descriptor['path']).read_bytes()
            require(hashlib.sha256(original).hexdigest() == descriptor['sha256'], 'Rejection report changed')
            projected = {'schema': 'doorbench.planned-reference-public-validation.v1',
                         'projection': 'Public report; local absolute paths and tracebacks removed.',
                         'original_validation_sha256': descriptor['sha256'], 'report': redact(json.loads(original))}
            row['audits']['validation.json'] = audit(directory, row['door_id'], 'validation', encoded(projected))
    common.write_json(directory/'index.json', web)


def reachable_web_files(web, directory):
    files = {'web/index.json': directory/'index.json'}
    for row in web['doors']:
        require(row['status'] == 'accepted_kinematic' or row.get('clip') is None, 'Nonaccepted motion must never be playable')
        for descriptor in [row.get('clip'), *row.get('audits', {}).values()]:
            if descriptor is None:
                continue
            path = common.regular_file(directory, descriptor['path'])
            require(path.stat().st_size == descriptor['bytes'] and common.sha256(path) == descriptor['sha256'], 'Web derivative checksum mismatch')
            files['web/'+descriptor['path']] = path
    return files


def browser_compatibility(web, directory, *, limits=None):
    """Measure exact derivatives, using bounded streaming decompression.

    Body counts come from the byte-exact accepted clip audit, already checked by
    the exporter. The decoded JSON hash binds the measured stream to its index;
    no giant Python pose object needs to be allocated just to count its bytes.
    """
    limits = limits or BROWSER_LIMITS
    maxima = dict.fromkeys(limits, 0); violations = []; checked = 0
    for row in web['doors']:
        if row['status'] != 'accepted_kinematic':
            require(row.get('clip') is None, 'Nonaccepted browser clip')
            continue
        descriptor = row['clip']; path = common.regular_file(directory, descriptor['path'])
        require(path.stat().st_size == descriptor['bytes'] and common.sha256(path) == descriptor['sha256'],
                f"{row['door_id']}: browser packed checksum mismatch")
        audit = row['audits']['clip.json']; source = common.regular_file(directory, audit['path'])
        require(source.stat().st_size == audit['bytes'] and common.sha256(source) == audit['sha256'], 'Browser clip audit changed')
        clip = common.read(source); frames = clip['frames']
        require(frames == descriptor['frames'], 'Browser/audit frame count mismatch')
        values = {'packed_bytes': descriptor['bytes'], 'frames': frames,
                  'pose_scalars_per_group': max(frames*len(clip[group]['body_names'])*7 for group in ('native', 'actor'))}
        digest = hashlib.sha256(); decoded = 0
        with gzip.open(path, 'rb') as stream:
            while block := stream.read(min(1024**2, limits['decoded_bytes']+1-decoded)):
                decoded += len(block); digest.update(block)
                if decoded > limits['decoded_bytes']:
                    break
        values['decoded_bytes'] = decoded
        if decoded <= limits['decoded_bytes']:
            require(digest.hexdigest() == descriptor['json_sha256'], 'Browser decoded checksum mismatch')
        exceeded = {key: {'observed': value, 'limit': limits[key],
                          **({'observed_is_lower_bound': True} if key == 'decoded_bytes' else {})}
                    for key, value in values.items() if value > limits[key]}
        for key, value in values.items():
            maxima[key] = max(maxima[key], value)
        if exceeded:
            violations.append({'door_id': row['door_id'], 'limits_exceeded': exceeded})
        checked += 1
    return {'compatible': not violations, 'accepted_clips_checked': checked, 'limits': limits,
            'maxima': maxima, 'violations': violations,
            'scope': 'Resource compatibility for exact full-rate accepted browser derivatives; no motion or acceptance changes.'}


def archive_accepted(audit, corpus, staging, shard_bytes):
    """Stream original accepted files directly; never copy or archive rejected NPZs."""
    require(isinstance(shard_bytes, int) and shard_bytes > 0, 'Archive shard size must be positive')
    groups = []; group = []; size = 0; inventory = {}; archives = {}; downloads = {}
    for door, result in sorted(audit['results'].items()):
        if result['status'] != 'accepted_kinematic':
            continue
        directory = Path(corpus)/door
        clip = common.read(directory/'clip.json'); xml = clip['actor']['mjcf_xml']
        require(isinstance(xml, str) and xml.startswith('<mujoco'), 'Accepted clip must contain its original actor MJCF')
        rig = staging/'rig-sources'/f'{door}.xml'; rig.parent.mkdir(exist_ok=True)
        rig.write_text(xml)
        files = {f'accepted/{door}/{name}': directory/name for name in ('clip.json', 'trajectory.npz', 'validation.json')}
        files[f'accepted/{door}/actor.xml'] = rig
        for name in ('clip.json', 'trajectory.npz', 'validation.json'):
            require(common.sha256(directory/name) == result['artifacts'][name], f'{door}: accepted artifact changed before archive')
        count = sum(path.stat().st_size for path in files.values())
        if group and size+count > shard_bytes:
            groups.append(group); group = []; size = 0
        group.append((door, files)); size += count
    if group:
        groups.append(group)
    for number, group in enumerate(groups):
        files = {name: path for _, mapping in group for name, path in mapping.items()}
        records = {name: {'sha256': common.sha256(path), 'bytes': path.stat().st_size, 'license': LICENSE} for name, path in sorted(files.items())}
        digest = hashlib.sha256(common.canonical(records)).hexdigest()
        name = f'accepted-{number:03d}-{digest[:16]}.tar.gz'
        archive = common.write_archive(files, records, staging/'archives'/name)
        archive['door_ids'] = [door for door, _ in group]
        archives[name] = archive; inventory[name] = records
        for door in archive['door_ids']:
            downloads[door] = {'archive': archive['path'], 'member_prefix': f'accepted/{door}/'}
    common.write_json(staging/'research-inventory.json', inventory)
    return archives, downloads


def check_research_inventory(release, inventory, rows):
    """Check shard namespaces before even a subset can be extracted."""
    require(len(rows) == 1000 and len({row['door_id'] for row in rows}) == 1000, 'Research status coverage mismatch')
    require(dict(Counter(row['status'] for row in rows)) == release['counts'], 'Research status totals mismatch')
    require(release.get('accepted_scenarios')==accepted_scenario_counts(rows),'Accepted source scenario counts mismatch')
    for row in rows:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['door_id']) is not None, 'Unsafe research door ID')
        require(row['status'] in ('accepted_kinematic', 'rejected', 'unresolved'), 'Unsupported research status')
        require(isinstance(row.get('reason'), str) and bool(row['reason']), 'Missing research status reason')
        if row['status'] != 'accepted_kinematic':
            require(row.get('clip') is None and row.get('research_download') is None, 'Nonaccepted research download')
    accepted = {row['door_id']: row for row in rows if row['status'] == 'accepted_kinematic'}
    archived = [door for archive in release['archives'].values() for door in archive['door_ids']]
    require(len(archived) == len(set(archived)) and set(archived) == set(accepted), 'Research archives must contain exactly the accepted door set')
    require(set(inventory) == set(release['archives']), 'Research archive inventory mismatch')
    for name, archive in release['archives'].items():
        require(common.safe_name(name) == Path(name).name and archive['path'] == 'archives/'+name, 'Unsafe archive namespace')
        require(archive['path'] in release['files'] and all(release['files'][archive['path']][key] == archive[key] for key in ('sha256', 'bytes')),
                'Archive is not bound to the release inventory')
        records = inventory[name]
        require(hashlib.sha256(common.canonical(records)).hexdigest() == archive['inventory_sha256'], 'Research member inventory checksum mismatch')
        expected = {f'accepted/{door}/{member}' for door in archive['door_ids'] for member in ('clip.json', 'trajectory.npz', 'validation.json', 'actor.xml')}
        require(set(records) == expected, 'Research archive includes unexpected or rejected payload')
        require(archive['file_count'] == len(records) and archive['expanded_bytes'] == sum(r['bytes'] for r in records.values()), 'Research archive inventory sizes mismatch')
        for door in archive['door_ids']:
            row = accepted[door]
            require(row['research_download'] == {'archive': archive['path'], 'member_prefix': f'accepted/{door}/'}, 'Research download points at a different door or shard')
            for member in ('clip.json', 'validation.json'):
                require(all(row['audits'][member][key] == records[f'accepted/{door}/{member}'][key] for key in ('sha256', 'bytes')),
                        'Research archive differs from accepted browser audit')


def accepted_scenario_counts(rows):
    counts=Counter()
    for row in rows:
        if row['status']=='accepted_kinematic':
            scenario=row.get('source_scenario')
            require(scenario in ('open_and_traverse','unlock_and_traverse','locked_recognize'),'Accepted row lacks a supported bound source scenario')
            counts[scenario]+=1
    return dict(sorted(counts.items()))


def release_files(folder):
    folder = Path(folder); release = common.read(folder/'release.json')
    require(release.get('schema') == SCHEMA and release.get('experimental') is True, 'Unsupported experimental release')
    require(release['path_in_repo'] == prefix(release['release']), 'Release prefix mismatch')
    require(release.get('complete_corpus') is True and release['doors'] == 1000, 'Publication requires a completed all-1000 corpus')
    files = {'release.json': common.regular_file(folder, 'release.json')}
    require(hashlib.sha256(common.canonical(release['files'])).hexdigest() == release['files_sha256'], 'Release inventory identity mismatch')
    for name, record in release['files'].items():
        path = common.regular_file(folder, name)
        require(path.stat().st_size == record['bytes'] and common.sha256(path) == record['sha256'], f'Prepared release file changed: {name}')
        files[name] = path
    require({'README.md', 'LICENSE', 'LIMITATIONS.md', 'download.py', 'archive_helpers.py', 'web/index.json',
             'native-dependency.json', 'research-inventory.json', 'status.jsonl'} <= set(files), 'Missing required research/support files')
    web = common.read(files['web/index.json']); rows = web['doors']
    require(web.get('schema') == 'doorbench.planned-reference-web-index.v1' and len(rows) == release['doors'] and
            len({row['door_id'] for row in rows}) == len(rows), 'Published web index must cover all 1000 unique statuses')
    require(web.get('corpus_index_sha256') == release['corpus_index_sha256'] and
            web.get('generator_sha256') == release['generator']['sha256'], 'Web index provenance mismatch')
    require(dict(Counter(row['status'] for row in rows)) == release['counts'], 'Published status totals mismatch')
    require([json.loads(line) for line in files['status.jsonl'].read_text().splitlines()] == rows, 'Status download differs from web index')
    accepted = set()
    for row in rows:
        require(row['status'] in ('accepted_kinematic', 'rejected', 'unresolved') and isinstance(row.get('reason'), str) and row['reason'],
                'Every release row requires an explicit status and reason')
        if row['status'] == 'accepted_kinematic':
            accepted.add(row['door_id'])
            require(isinstance(row.get('clip'), dict) and isinstance(row.get('research_download'), dict), 'Accepted row lacks verified playback/download')
        else:
            require(row.get('clip') is None and row.get('research_download') is None, 'Nonaccepted motion must not be playable or downloadable as accepted research')
        for descriptor in [row.get('clip'), *row.get('audits', {}).values()]:
            if descriptor is None:
                continue
            name = 'web/'+common.safe_name(descriptor['path'])
            require(name in release['files'] and all(release['files'][name][key] == descriptor[key] for key in ('sha256', 'bytes')),
                    'Web descriptor is not bound to a release file')
    inventory = common.read(files['research-inventory.json'])
    check_research_inventory(release, inventory, rows)
    for row in rows:
        if row['door_id'] not in accepted:
            continue
        clip = common.read(files['web/'+row['audits']['clip.json']['path']])
        validation = common.read(files['web/'+row['audits']['validation.json']['path']])
        members = inventory[Path(row['research_download']['archive']).name]
        scenario=clip.get('proposal',{}).get('source_outcome',{}).get('scenario')
        require(scenario in ('open_and_traverse','unlock_and_traverse','locked_recognize') and
                row.get('source_scenario')==scenario==clip.get('proposal',{}).get('scenario'),
                'Accepted source scenario differs from the bound source outcome')
        require(validation.get('accepted') is True and validation.get('kinematic_accepted') is True and
                validation.get('status') == 'accepted_kinematic' and not validation.get('failure_counts') and
                validation.get('task_completion', {}).get('evidence_pass') is True and
                validation.get('task_completion', {}).get('complete_proposal') is True and clip.get('complete_proposal') is True,
                'Accepted research validation/task evidence failed')
        require(clip['door_id'] == validation['door_id'] == row['door_id'] and
                clip['source_sha256'] == validation['source_sha256'] and
                validation['clip_sha256'] == row['audits']['clip.json']['sha256'] and
                validation['trajectory_sha256'] == clip['trajectory_sha256'] == members[f"accepted/{row['door_id']}/trajectory.npz"]['sha256'],
                'Accepted research artifact/source bindings disagree')
        require(hashlib.sha256(clip['actor']['mjcf_xml'].encode()).hexdigest() == members[f"accepted/{row['door_id']}/actor.xml"]['sha256'],
                'Original actor rig differs from accepted clip')
    require(common.read(files['native-dependency.json']) == release['native_dependency'] and
            release['native_dependency']['commit'] == BASE_COMMIT and release['native_dependency']['release_sha256'] == BASE_RELEASE_SHA,
            'Native dependency is not the pinned published release')
    require(web['manifest_sha256'] == release['native_dependency']['manifest_sha256'], 'Browser/native dataset manifest binding mismatch')
    compatibility = browser_compatibility(web, folder/'web')
    require(compatibility['compatible'] and compatibility == release['browser_compatibility'], 'Browser compatibility failed or changed')
    return release, files


def prepare(args):
    from scripts.export_planned_reference_web import export_corpus
    tag_prefix = prefix(args.release)
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id), 'Invalid Hub dataset repository ID')
    corpus = Path(args.corpus).resolve(); destination = Path(args.out).resolve()
    for source in (corpus, Path(args.assets).resolve(), Path(args.recordings).resolve()):
        require(not (destination == source or destination.is_relative_to(source) or source.is_relative_to(destination)), 'Release output overlaps an immutable input tree')
    native = native_dependency(args.native_release)
    if args.dry_run:
        try:
            with idle_corpus(corpus):
                checked = inspect_corpus(corpus, args.assets, args.recordings, native)
                with tempfile.TemporaryDirectory(prefix='doorbench-browser-check-') as temporary:
                    web = export_corpus(corpus, Path(temporary)/'web', args.assets)
                    require(web['corpus_index_sha256'] == checked['index_sha256'], 'Corpus changed during browser check')
                    compatibility = browser_compatibility(web, Path(temporary)/'web')
            plan = {'ready': compatibility['compatible'], 'doors': len(checked['results']), 'counts': checked['counts'],
                    'accepted_source_bytes': checked['accepted_bytes'], 'browser_compatibility': compatibility}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            plan = {'ready': False, 'reason': str(exc)}
        plan.update(schema=SCHEMA, dry_run=True, scope=SCOPE, native_dependency=native)
        output = destination.with_name(destination.name+'.plan.json'); common.write_json(output, plan)
        print(json.dumps({'ready': plan['ready'], 'plan': str(output)})); return plan
    require(not destination.exists(), 'Prepared output already exists; use a fresh release directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with idle_corpus(corpus):
        checked = inspect_corpus(corpus, args.assets, args.recordings, native)
        source_files = verify_commit(args.source_commit, checked['index']['generator'])
        # Archives are streamed. This conservative bound leaves room for gzip
        # overhead, the web derivative and a safety margin without duplicating NPZs.
        require(shutil.disk_usage(destination.parent).free > checked['accepted_bytes']*1.2+256*1024**2, 'Insufficient free disk for accepted-only release archives')
        with tempfile.TemporaryDirectory(prefix='.planned-release-', dir=destination.parent) as temporary:
            staging = Path(temporary)/'bundle'; staging.mkdir()
            web = export_corpus(corpus, staging/'web', args.assets)
            require(web['corpus_index_sha256'] == checked['index_sha256'], 'Corpus changed while exporting MotionLab')
            compatibility = browser_compatibility(web, staging/'web')
            require(compatibility['compatible'], 'Browser resource limits exceeded: '+json.dumps(compatibility['violations']))
            project_rejection_reports(web, staging/'web')
            web_files = reachable_web_files(web, staging/'web')
            # Discard only unreferenced derivatives created inside this private
            # staging directory (e.g. original rejected reports before redaction).
            keep = set(web_files.values())
            for path in (staging/'web').rglob('*'):
                if path.is_file() and path not in keep:
                    path.unlink()
            # Recheck remaining space after exact web bytes have been written;
            # compressed NPZ files may not shrink further inside a tarball.
            require(shutil.disk_usage(staging).free > checked['accepted_bytes']*1.02+256*1024**2,
                    'Insufficient remaining disk for accepted research archives')
            archives, downloads = archive_accepted(checked, corpus, staging, args.shard_mib*1024**2)
            for row in web['doors']:
                row['research_download'] = downloads.get(row['door_id'])
            common.write_json(staging/'web/index.json', web)
            with (staging/'status.jsonl').open('wb') as stream:
                for row in web['doors']:
                    stream.write(common.canonical(row)+b'\n')
            common.write_json(staging/'native-dependency.json', native)
            shutil.copyfile(ROOT/'LICENSE', staging/'LICENSE')
            shutil.copyfile(__file__, staging/'download.py')
            shutil.copyfile(ROOT/'scripts/huggingface_release.py', staging/'archive_helpers.py')
            (staging/'LIMITATIONS.md').write_text('# Experimental scope\n\n'+SCOPE+'\n\n'
                'Acceptance uses independent sampled FK, collision, contact, joint/derivative and actor-route evidence checks. '
                'It is not continuous collision certification. Source outcomes remain separate from the new actor motion. '
                'Prescribed door poses and retiming do not reproduce source forces, powered controls, closer behavior or the original task clock. '
                'Hands are geometric proxies without articulated fingers; grasp orientation, force closure, unlocking and recognition semantics remain unverified. '
                'A local planner failure, engine crash or timeout is unresolved, not a proof of human impossibility. '
                'No pretrained human assets, SMPL models, third-party motion weights or textures are included.\n')
            counts = checked['counts']
            scenarios=accepted_scenario_counts(web['doors'])
            traversals=scenarios.get('open_and_traverse',0)+scenarios.get('unlock_and_traverse',0)
            locked=scenarios.get('locked_recognize',0)
            (staging/'README.md').write_text(f'# DoorBench experimental planned references — {args.release}\n\n'
                f'{SCOPE}\n\nAll 1,000 dataset doors retain a status and reason. '
                f"Counts: {counts.get('accepted_kinematic', 0)} accepted kinematic, {counts.get('rejected', 0)} rejected, "
                f"{counts.get('unresolved', 0)} unresolved. Only accepted rows have a playable clip or research trajectory download.\n\n"
                f'Accepted entries comprise **{traversals} traversal references** and **{locked} locked-door checks**. '
                'Locked-door checks do not traverse the doorway. Recognition is declared by the source benchmark, '
                'not independently demonstrated by the new actor. Open/unlock mechanism semantics are also not certified.\n\n'
                'Accepted archives contain exact clip JSON, NPZ, independent validation JSON, and actor MJCF derived verbatim from the clip. '
                'Rejected/unresolved entries retain public reports and reasons; their trajectories and transient execution logs are omitted. '
                'Public audit projections retain original hashes and redact local paths/tracebacks.\n\n'
                f'Native simulation assets and source recordings: [{BASE_TAG}]({native["release_url"]}), pinned to `{BASE_COMMIT}`. '
                'They are dependencies and are not duplicated here. MotionLab uses matching site assets with SHA-256 checks.\n\n'
                'Files in this supplement are MIT licensed. See LICENSE and LIMITATIONS.md. The release manifest binds every download. '
                'Use a release tag or full Hub commit; a moving branch is not an experimental data version.\n\n'
                f'```sh\npython download.py download --repo-id {args.repo_id} --release {args.release} '
                f'--revision {args.release} --out planned-data\n```\n')
            shutil.rmtree(staging/'rig-sources', ignore_errors=True)
            files = {p.relative_to(staging).as_posix(): {'sha256': common.sha256(p), 'bytes': p.stat().st_size, 'license': LICENSE}
                     for p in sorted(staging.rglob('*')) if p.is_file()}
            release = {'schema': SCHEMA, 'experimental': True, 'release': args.release, 'repo_id': args.repo_id,
                       'path_in_repo': tag_prefix, 'complete_corpus': True, 'doors': len(web['doors']), 'counts': counts,'accepted_scenarios':scenarios,
                       'scope': SCOPE, 'source_commit': args.source_commit, 'source_files_sha256': source_files,
                       'corpus_index_sha256': checked['index_sha256'], 'corpus_report_sha256': checked['report_sha256'],
                       'generator': checked['index']['generator'], 'native_dependency': native, 'archives': archives,
                       'browser_compatibility': compatibility,
                       'web_index': 'web/index.json', 'files': files,
                       'files_sha256': hashlib.sha256(common.canonical(files)).hexdigest()}
            require(common.sha256(corpus/'index.json') == checked['index_sha256'] and
                    common.sha256(corpus/'report.json') == checked['report_sha256'], 'Corpus snapshot changed while packaging')
            verify_commit(args.source_commit, checked['index']['generator'])
            common.write_json(staging/'release.json', release)
            release_files(staging)
            os.replace(staging, destination)
    print(json.dumps({'prepared': str(destination), 'counts': counts, 'published': False})); return release


def verify_remote(api, repo, commit, files):
    # Hub metadata requests are bounded; the native helper compares Git blobs or
    # LFS SHA-256. Only read-only verification is retried, never publication.
    # A global retry/sleep budget also bounds failures spread across many batches.
    names = sorted(files)
    retries = 0
    slept = 0.0
    for offset in range(0, len(names), 20):
        batch = {name: files[name] for name in names[offset:offset+20]}
        for attempt in range(6):
            try:
                common.verify_public_files(api, repo, commit, batch)
                break
            except Exception as exc:
                response = getattr(exc, 'response', None)
                if (getattr(response, 'status_code', None) not in (429, 503) or
                        attempt == 5 or retries >= 12):
                    raise
                delay = float(2 ** (attempt+1))
                value = response.headers.get('Retry-After')
                if value:
                    # Accept standard nonnegative delay-seconds or timezone-aware
                    # HTTP dates. Malformed headers fall back to exponential delay.
                    try:
                        if re.fullmatch(r'[0-9]+', value.strip()):
                            requested = float(value.strip())
                        else:
                            date = parsedate_to_datetime(value)
                            if date.tzinfo is None:
                                raise ValueError('Retry-After date needs a timezone')
                            requested = max(0.0, (date-datetime.now(timezone.utc)).total_seconds())
                        delay = max(delay, requested)
                    except (TypeError, ValueError, OverflowError):
                        pass
                # Do not retry earlier than a long server-requested wait. Fail
                # with the original HTTP error if it exceeds this bounded call.
                if delay > 60 or slept+delay > 180:
                    raise
                retries += 1
                slept += delay
                time.sleep(delay)


def publish(args):
    release, local = release_files(args.folder)
    release_sha256=common.sha256(Path(args.folder)/'release.json')
    def unchanged_staging():
        current,current_files=release_files(args.folder)
        require(current==release and set(current_files)==set(local) and
                common.sha256(Path(args.folder)/'release.json')==release_sha256,
                'Prepared release changed during publication; no release tag may be created')
    if args.dry_run:
        return {'published': False, 'files': len(local), 'path_in_repo': release['path_in_repo']}
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RevisionNotFoundError
    token = Path(args.token_file).read_text().strip() if args.token_file else os.environ.get('HF_TOKEN')
    require(bool(token), 'Supply a private token file or HF_TOKEN only when explicitly publishing')
    api = HfApi(token=token); public = HfApi(token=False); repo = release['repo_id']; base = release['path_in_repo']
    require(not public.repo_info(repo, repo_type='dataset').private, 'Target dataset must already exist and be public')
    remote = {base+'/'+name: path for name, path in local.items()}
    try:
        existing = api.repo_info(repo, repo_type='dataset', revision=release['release'])
    except RevisionNotFoundError:
        existing = None
    if existing:
        previous = common.read(hf_hub_download(repo, base+'/release.json', repo_type='dataset', revision=existing.sha, token=False))
        require(previous == release, 'Release tag already identifies different bytes; choose a new experimental version')
        commit = existing.sha
    else:
        commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(args.folder), path_in_repo=base,
                                   allow_patterns=list(local), commit_message=f'Experimental planned references {release["release"]}').oid
    unchanged_staging()
    verify_remote(public, repo, commit, remote)
    unchanged_staging()
    if not existing:
        api.create_tag(repo, repo_type='dataset', tag=release['release'], revision=commit,
                       tag_message=f'Experimental release inventory {release["files_sha256"]}')
    require(public.repo_info(repo, repo_type='dataset', revision=release['release']).sha == commit, 'Public release tag resolved to unexpected bytes')
    receipt = {'repo_id': repo, 'release': release['release'], 'commit': commit,
               'counts': release['counts'], 'accepted_scenarios':release['accepted_scenarios'],'doors': release['doors'],
               'native_manifest_sha256': release['native_dependency']['manifest_sha256'],
               'release_sha256': release_sha256,
               'web_index_sha256': release['files']['web/index.json']['sha256'],
               'web_index_url': f'https://huggingface.co/datasets/{repo}/resolve/{commit}/{base}/web/index.json'}
    common.write_json(Path(args.folder)/'publication.json', receipt)
    print(json.dumps(receipt)); return receipt


def download(args):
    from huggingface_hub import HfApi, hf_hub_download
    require(args.revision == args.release or re.fullmatch(r'[0-9a-f]{40}', args.revision), 'Pin the experimental tag or a full Hub commit')
    base = prefix(args.release); commit = HfApi(token=False).repo_info(args.repo_id, repo_type='dataset', revision=args.revision).sha
    fetch = lambda name: Path(hf_hub_download(args.repo_id, base+'/'+common.safe_name(name), repo_type='dataset', revision=commit, token=False))
    release_path = fetch('release.json'); release = common.read(release_path)
    require(release.get('schema') == SCHEMA and release['repo_id'] == args.repo_id and release['release'] == args.release,
            'Downloaded experimental release identity mismatch')
    require(release['path_in_repo'] == base and release['complete_corpus'] is True and release['doors'] == 1000, 'Incomplete experimental release')
    require(hashlib.sha256(common.canonical(release['files'])).hexdigest() == release['files_sha256'], 'Downloaded file inventory mismatch')
    require(release.get('experimental') is True and release.get('browser_compatibility', {}).get('compatible') is True,
            'Unsupported experimental compatibility contract')
    def checked(name):
        path = fetch(name); record = release['files'][name]
        require(path.stat().st_size == record['bytes'] and common.sha256(path) == record['sha256'], f'Downloaded checksum mismatch: {name}')
        return path
    inventories = common.read(checked('research-inventory.json'))
    status_path = checked('status.jsonl')
    rows = [json.loads(line) for line in status_path.read_text().splitlines()]
    check_research_inventory(release, inventories, rows)
    native_path = checked('native-dependency.json')
    require(common.read(native_path) == release['native_dependency'] and
            release['native_dependency']['commit'] == BASE_COMMIT and release['native_dependency']['release_sha256'] == BASE_RELEASE_SHA,
            'Downloaded native dependency differs from the pinned release')
    web = common.read(checked('web/index.json'))
    require(web['doors'] == rows and web['manifest_sha256'] == release['native_dependency']['manifest_sha256'],
            'Downloaded web/status/native manifest binding mismatch')
    wanted = sorted(release['archives']) if args.archives == 'all' else args.archives.split(',')
    require(len(wanted) == len(set(wanted)) and not set(wanted)-set(release['archives']), 'Choose available unique archive names')
    out = Path(args.out).resolve(); require(not out.exists(), 'Download output already exists; choose a fresh directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.planned-download-', dir=out.parent) as temporary:
        staging = Path(temporary)/'verified'; staging.mkdir()
        for name in wanted:
            archive = release['archives'][name]; records = inventories[name]
            require(hashlib.sha256(common.canonical(records)).hexdigest() == archive['inventory_sha256'], 'Archive member inventory mismatch')
            common.extract_component(checked(archive['path']), archive, records, staging)
        for name in ('status.jsonl', 'native-dependency.json', 'README.md', 'LICENSE', 'LIMITATIONS.md', 'research-inventory.json'):
            shutil.copyfile(checked(name), staging/name)
        shutil.copyfile(release_path, staging/'release.json')
        common.write_json(staging/'installed.json', {'schema': SCHEMA, 'revision': commit, 'release': args.release, 'archives': wanted})
        os.replace(staging, out)
    return {'out': str(out), 'revision': commit, 'archives': wanted}


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--corpus', required=True); p.add_argument('--assets', default='assets')
    p.add_argument('--recordings', default='out/reference-motions'); p.add_argument('--native-release', default='out/huggingface-release/v2026.09.05/release.json')
    p.add_argument('--release', required=True); p.add_argument('--repo-id', default=BASE_REPO); p.add_argument('--source-commit')
    p.add_argument('--out', required=True); p.add_argument('--dry-run', action='store_true'); p.add_argument('--shard-mib', type=int, default=512); p.set_defaults(func=prepare)
    p = commands.add_parser('publish'); p.add_argument('--folder', required=True); p.add_argument('--token-file'); p.add_argument('--dry-run', action='store_true'); p.set_defaults(func=publish)
    p = commands.add_parser('download'); p.add_argument('--repo-id', default=BASE_REPO); p.add_argument('--release', required=True)
    p.add_argument('--revision', required=True); p.add_argument('--archives', default='all'); p.add_argument('--out', required=True); p.set_defaults(func=download)
    args = parser.parse_args(); result = args.func(args)
    if args.command != 'prepare':
        print(json.dumps(result))


if __name__ == '__main__':
    main()
