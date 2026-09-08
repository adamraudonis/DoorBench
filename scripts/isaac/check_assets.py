#!/usr/bin/env python3
"""Fail closed on generation errors, missing requested doors, or incomplete QA."""
import argparse
import json
from pathlib import Path


def check_assets(root, ids=None):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    rows=manifest.get('doors',[])
    if not rows or manifest.get('n_doors')!=len(rows):
        raise ValueError('Dataset manifest is empty or inconsistent')
    by_id={row['id']:row for row in rows}
    wanted=list(ids or by_id)
    if len(by_id)!=len(rows) or set(wanted)-set(by_id):
        raise ValueError('Manifest is missing requested doors or contains duplicates')
    for id in wanted:
        row=by_id[id]
        if row.get('error'):raise ValueError('Generation failed: '+id)
        folder=root/'doors'/id
        for name in ('door.usda','door.xml','spec.json','qa.json'):
            if not (folder/name).is_file():raise ValueError(f'Missing {id}/{name}')
        qa=json.loads((folder/'qa.json').read_text())
        if qa.get('signed_off') is not True:raise ValueError('Door has not passed QA: '+id)
    return wanted


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--ids',default='')
    a=p.parse_args();ids=check_assets(a.root,a.ids.split(',') if a.ids else None)
    print('ASSET_CHECK_OK '+json.dumps(ids))

if __name__=='__main__':main()
