"""Relocatable, hash-verified inputs for one frozen sensor training experiment.

Original reports remain byte-identical. Relative paths only relocate their
storage; every demonstration still passes its independent admission checks.
"""
import hashlib
import json
from pathlib import Path

from .sensor_demonstrations import SensorDemonstration
from .correction_demonstrations import CorrectionDemonstration

SCHEMA = 'doorbench.sensor-training-bundle.v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bundled_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Bundle paths must be explicit relative paths')
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError('Bundle path escapes its root')
    return candidate


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != SCHEMA or not manifest.get('files_sha256'):
        raise ValueError('Expected a complete frozen training bundle manifest')
    for name, expected in manifest['files_sha256'].items():
        p = bundled_path(path.parent, name)
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f'Frozen training bundle hash mismatch: {name}')
    return manifest


def load_bundle(path):
    path = Path(path)
    manifest = verify_bundle(path)
    resolve = lambda value: bundled_path(path.parent, value)
    episodes = []
    for row in manifest['datasets']:
        if row['kind'] == 'qualified_teacher':
            episode = SensorDemonstration(resolve(row['run']), qualification=row['qualification'],
                legacy_teacher_receipt=resolve(row['legacy_teacher_receipt']),
                reset_observation_run=resolve(row['reset_observation_run']))
        elif row['kind'] == 'counterfactual_correction':
            episode = CorrectionDemonstration(resolve(row['labels']), source_run=resolve(row['run']))
        else:
            raise ValueError('Unknown frozen dataset admission route')
        if len(episode) != row['examples']:
            raise ValueError('Relocated dataset has a different causal prefix')
        episodes.append(episode)
    if not episodes:
        raise ValueError('No frozen datasets')
    return episodes, manifest
