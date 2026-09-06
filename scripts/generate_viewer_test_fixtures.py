#!/usr/bin/env python3
"""Generate current-source JSON for viewer regression tests, separate from release media."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doorbench.build import build_model, _json_default
from doorbench.spec import generate_all

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
rows = []
for spec in generate_all():
    model = build_model(spec)
    folder = args.out / 'doors' / spec['id']
    folder.mkdir(parents=True)
    (folder / 'model.json').write_text(json.dumps(model.to_dict('full'), default=_json_default))
    (folder / 'spec.json').write_text(json.dumps(spec, default=_json_default))
    rows.append({'id': spec['id'], 'family': spec['family']})
    if len(rows) % 100 == 0:
        print(f'Generated {len(rows)} viewer fixtures', flush=True)
(args.out / 'manifest.json').write_text(json.dumps({'n_doors': len(rows), 'doors': rows}))
assert len(rows) == 1000, 'Unexpected fixture coverage'
