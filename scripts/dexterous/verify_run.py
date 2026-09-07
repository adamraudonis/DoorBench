#!/usr/bin/env python3
"""Verify a copied run's archived source and bundled inputs without extracting it."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def verify(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    problems=[]
    archive_path=directory/'source.tar.gz'
    digest=lambda data:hashlib.sha256(data).hexdigest()
    if digest(archive_path.read_bytes())!=manifest['source_archive_sha256']:
        problems.append('Source archive checksum differs')
    with tarfile.open(archive_path) as archive:
        members=archive.getmembers()
        if len({m.name for m in members})!=len(members):
            problems.append('Duplicate archive paths')
        if {m.name for m in members}!=set(manifest['source_hashes']):
            problems.append('Source inventory differs')
        for member in members:
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                problems.append('Unsafe archive member: '+member.name)
                continue
            data=archive.extractfile(member).read()
            if digest(data)!=manifest['source_hashes'].get(member.name):
                problems.append('Source checksum differs: '+member.name)
    for name,record in manifest.get('inputs',{}).items():
        if not isinstance(record,dict) or 'bundled_path' not in record:
            continue
        path=(directory/record['bundled_path']).resolve()
        if not path.is_relative_to(directory):
            problems.append('Input path escapes run: '+name)
        elif not path.is_file() or digest(path.read_bytes())!=record['sha256']:
            problems.append('Bundled input missing or changed: '+name)
    return {'valid':not problems,'problems':problems,
            'scope':'Archived source and bundled inputs only; external assets, physics and policy quality require separate checks'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=verify(args.directory)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['valid'] else 1)
