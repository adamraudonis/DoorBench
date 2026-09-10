#!/usr/bin/env python3
"""Compare source collider identities/types with imported USD; no physics claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def audit(mjcf, usd):
    from pxr import Usd, UsdPhysics
    root = ET.parse(mjcf).getroot()
    types = {'mesh': 'Mesh', 'cylinder': 'Cylinder', 'box': 'Cube',
             'sphere': 'Sphere', 'capsule': 'Capsule'}
    expected = Counter()
    for body in root.findall('./worldbody//body'):
        for geom in body.findall('geom'):
            # Prepared MJCF must explicitly resolve defaults before this audit.
            if not all(k in geom.attrib for k in ('type', 'contype', 'conaffinity', 'name')):
                raise ValueError('Prepared explicit geometry attributes required')
            if not (int(geom.get('contype')) or int(geom.get('conaffinity'))):
                continue
            kind = geom.get('type')
            key = geom.get('mesh') if kind == 'mesh' else geom.get('name')
            expected[(body.get('name'), key, types[kind])] += 1
    stage = Usd.Stage.Open(str(Path(usd).resolve()))
    if stage is None:
        raise ValueError('Cannot open imported USD')
    actual = Counter()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        parts = str(prim.GetPath()).strip('/').split('/')
        if len(parts) != 4 or parts[0] != 'collisions' or parts[2] != parts[3]:
            raise ValueError('Unexpected collider source hierarchy: ' + str(prim.GetPath()))
        actual[(parts[1], parts[3], prim.GetTypeName())] += 1
    def rows(values):
        return [dict(body=b, geometry=g, usd_type=t, count=n)
                for (b, g, t), n in sorted(values.items())]
    return dict(passed=bool(expected) and expected == actual,
                expected_count=sum(expected.values()), actual_count=sum(actual.values()),
                missing=rows(expected-actual), unexpected=rows(actual-expected),
                scope='Collider body/name/type inventory only; not mesh dimensions, cooked hulls, contact or task qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mjcf', type=Path, required=True)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing audit evidence')
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.mjcf, args.usd)}
    before = hashes()
    result = audit(args.mjcf, args.usd)
    if before != hashes():
        raise ValueError('Input changed during audit')
    result['input_sha256'] = before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
