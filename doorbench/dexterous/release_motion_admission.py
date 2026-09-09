"""Bind an optional moving-leaf clearance screen to the consumed release route."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_motion_screen(config, screen, source):
    path=config.get('leaf_motion_audit_path')
    if path is None:return
    path=Path(path)
    if digest(path)!=config.get('leaf_motion_audit_sha256'):
        raise ValueError('Moving-leaf audit bytes changed')
    audit=json.loads(path.read_text())
    if (audit.get('schema')!='doorbench.release-leaf-motion-screen.v1' or
            audit.get('passed') is not True or audit.get('physics_steps')!=0 or
            audit.get('sampled_configurations')!=3003 or audit.get('failed_configurations')!=0 or
            not 0<audit.get('leaf_envelope_rad',0)<=.03 or
            not audit.get('minimum_clearance_m',-1)>=0):
        raise ValueError('Passing explicit moving-leaf clearance screen required')
    inputs=audit.get('input_sha256',{})
    for required in (Path(screen),Path(source)/'trajectory.npz'):
        if inputs.get(str(required))!=digest(required):
            raise ValueError('Moving-leaf audit belongs to another route or source')
    for name,expected in inputs.items():
        if digest(name)!=expected:raise ValueError('Moving-leaf audit input changed: '+name)
