#!/usr/bin/env python3
"""Preserve exact native model dependencies and run inputs without relocation.

Stores original bytes and a path mapping. Rebinding paths on a different host
still requires source-design, compiled-geometry and physical qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
from doorbench.dexterous.robot_design_identity import robot_design_identity, _expanded_xml


def dependencies(path):
    path = path.resolve(strict=True)
    identity = robot_design_identity(path)  # Reject opaque asset providers.
    tree = _expanded_xml(path)
    spec = mujoco.MjSpec.from_file(str(path))
    files = {path}
    def includes(xml):
        for node in ET.parse(xml).iter('include'):
            child = (path.parent/node.attrib['file']).resolve(strict=True)
            if child not in files:
                files.add(child)
                includes(child)
    includes(path)
    compiler = {}
    for node in tree.findall('compiler'):
        compiler.update(node.attrib)
    directories = dict(mesh=spec.meshdir, texture=spec.texturedir, hfield=spec.meshdir, skin=spec.meshdir)
    for node in tree.iter():
        for key in ('file','fileright','fileleft','fileup','filedown','filefront','fileback'):
            if not node.get(key):
                continue
            p = Path(node.get(key).replace('\\','/'))
            if compiler.get('strippath') == 'true':
                p = Path(p.name)
            if not p.is_absolute():
                p = path.parent/directories[node.tag]/p
            files.add(p.resolve(strict=True))
    return files, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    manifest = json.loads((a.run/'manifest.json').read_text())
    config = manifest['configuration']
    files = set()
    identities = {}
    def input_path(value):
        path = Path(value)
        return path if path.is_absolute() else Path(manifest['working_directory'])/path
    for name in ('robot','door'):
        model = input_path(config[name])
        if model.is_dir():model /= 'door.xml'
        if model.read_bytes() != (a.run/(name+'-input.xml')).read_bytes():
            raise ValueError('Current model differs from captured run: '+name)
        found, identity = dependencies(model)
        files.update(found)
        identities[str(model.resolve())] = identity
        audit = model.with_suffix('.audit.json')
        if audit.exists():files.add(audit.resolve())
    for key in ('reference','motors','plan','release_path','body_reset','preparation','checkpoint','runtime_screen'):
        if config.get(key):files.add(input_path(config[key]).resolve(strict=True))
    source = a.run.with_name(a.run.name+'-source')
    files.update(p.resolve() for p in source.glob('*.json'))
    a.output.mkdir(parents=True,exist_ok=False)
    records = {}
    for original in sorted(files):
        data = original.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = Path('files')/(digest+original.suffix)
        target = a.output/relative
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(original,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Input changed while archiving: '+str(original))
        records[str(original)] = dict(path=str(relative),sha256=digest,bytes=len(data))
    result = dict(schema='doorbench.native-input-closure.v1',scope=__doc__,files=records,
                  source_designs=identities,run_manifest=manifest,
                  archive_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records.values()),output=str(a.output))))


if __name__ == '__main__':main()
