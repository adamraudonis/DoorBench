"""Immutable historical result identities, checked against their actual Git assets.

Historical registration selects a manifest, never skips schema, episode or
aggregate validation. New submissions always use the current asset manifest.
No generated asset is copied into the provenance registry.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/'results/provenance/historical-baselines.json'
SCHEMA = 'doorbench.historical-result-provenance.v1'
ARCHIVED_BASELINES = frozenset(('g1_locomotion.json', 'random.json', 'scripted_hand.json', 'scripted_hand_human.json'))
MANIFEST_PATH = 'assets/manifest.json'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _revision(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{40}', value):
        raise ValueError('Historical result needs a full immutable Git commit')
    return value


@lru_cache(maxsize=32)
def _source_bytes(repo, revision):
    revision = _revision(revision)
    try:
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', revision+'^{commit}'], stderr=subprocess.PIPE).decode().strip()
        raw = subprocess.check_output(['git', '-C', str(repo), 'show', revision+':'+MANIFEST_PATH], stderr=subprocess.PIPE)
        blob = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', revision+':'+MANIFEST_PATH], stderr=subprocess.PIPE).decode().strip()
    except subprocess.CalledProcessError as error:
        raise ValueError('Historical source commit/manifest is unavailable; fetch the recorded revision (CI uses fetch-depth: 0)') from error
    if actual != revision or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != blob:
        raise ValueError('Historical Git object does not match the recorded revision/blob')
    return raw, blob


def source_manifest_receipt(repo, revision):
    raw, blob = _source_bytes(str(Path(repo).resolve()), revision)
    manifest = json.loads(raw)
    if not isinstance(manifest.get('doors'), list) or not manifest['doors']:
        raise ValueError('Historical source has no door inventory')
    return manifest, dict(source_commit=revision, path=MANIFEST_PATH, git_blob_oid=blob,
        sha256=sha256(raw), dataset_version=manifest.get('version'),
        dataset_generated=manifest.get('generated'), n_doors=len(manifest['doors']))


def freeze_historical_results(paths, *, repo=ROOT):
    """Bind unmodified, clean source-revision runs; scores are never rewritten."""
    registry = dict(schema=SCHEMA, manifests={}, results={})
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        doc = json.loads(raw)
        bench = doc.get('benchmark', {})
        revision = _revision(bench.get('commit'))
        if bench.get('dirty') is not False:
            raise ValueError('A dirty/unknown working tree cannot be proved by a source commit')
        _, receipt = source_manifest_receipt(repo, revision)
        if any(bench.get(key) != receipt[key] for key in ('dataset_version', 'dataset_generated')):
            raise ValueError('Recorded dataset metadata differs from the source manifest')
        if bench.get('n_doors_total') != receipt['n_doors']:
            raise ValueError('Historical result denominator differs from its source inventory')
        if path.name in registry['results']:
            raise ValueError('Duplicate historical result filename')
        registry['manifests'][revision] = receipt
        registry['results'][path.name] = dict(sha256=sha256(raw), source_commit=revision)
    return registry


def historical_context(path, doc, *, repo=ROOT, registry_path=REGISTRY):
    """Return (source manifest, context) for an explicitly registered immutable run."""
    try:
        registry = json.loads(Path(registry_path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Historical result provenance registry is missing or invalid') from error
    if (not isinstance(registry, dict) or registry.get('schema') != SCHEMA or
            not isinstance(registry.get('results'), dict) or not isinstance(registry.get('manifests'), dict) or
            not ARCHIVED_BASELINES <= set(registry['results'])):
        raise ValueError('Historical provenance registry is missing an immutable shipped baseline')
    entries = registry['results']
    name = Path(path).name
    if name not in entries:
        if doc.get('benchmark', {}).get('commit') in registry.get('manifests', {}):
            raise ValueError('Historical source revision needs an explicit result-byte registration')
        return None, None
    entry = entries[name]
    if not isinstance(entry, dict):
        raise ValueError('Historical result byte/source receipt is malformed')
    if sha256(Path(path).read_bytes()) != entry.get('sha256'):
        raise ValueError('Immutable historical result bytes changed: '+name+'; publish a separately named new run')
    revision = _revision(entry.get('source_commit'))
    bench = doc.get('benchmark', {})
    if bench.get('commit') != revision or bench.get('dirty') is not False:
        raise ValueError('Historical result source revision/cleanliness differs from its receipt')
    manifest, actual = source_manifest_receipt(repo, revision)
    if registry.get('manifests', {}).get(revision) != actual:
        raise ValueError('Historical manifest receipt is missing or differs from its actual Git source')
    if any(bench.get(key) != actual[key] for key in ('dataset_version', 'dataset_generated')) or bench.get('n_doors_total') != actual['n_doors']:
        raise ValueError('Historical dataset metadata differs from the source manifest')
    return manifest, dict(mode='historical-source-revision', **actual,
        result_sha256=entry['sha256'], registry_sha256=sha256(Path(registry_path).read_bytes()),
        note='Validated against the exact run revision; not an evaluation of current door mechanics')
