"""Portable, explicit robot-mechanics identities for Isaac readiness receipts."""
import hashlib
import json
import math
from pathlib import Path

PROFILES=('upstream-v1','shadow-loopback-v2')


def mechanics_profile(value):
    if value not in PROFILES:raise ValueError('Unknown robot mechanics profile: '+str(value))
    return value


def robot_filename(profile):
    return 'h1-shadow.xml' if mechanics_profile(profile)=='upstream-v1' else 'h1-shadow-loopback-v2.xml'


def ready_directory(root,profile):
    base=Path(root)/'out/isaac-ready'
    return base if mechanics_profile(profile)=='upstream-v1' else base/profile


def cache_identity(source_sha256,profile):
    value={'source_sha256':source_sha256,'mechanics_profile':mechanics_profile(profile)}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_passive_audit(audit,motors,profile):
    """Validate serialized live solver readbacks; no USD-authorship-only pass."""
    mechanics_profile(profile)
    tendons=motors.get('passive_tendons',[])
    if motors.get('hand_mechanics_profile','upstream-v1')!=profile:
        raise ValueError('Motor contract mechanics profile differs')
    if profile=='upstream-v1':
        if tendons:raise ValueError('Legacy profile unexpectedly contains passive loopbacks')
        return {'profile':profile,'count':0,'backend_verified':False}
    from doorbench.dexterous.isaac_tendons import validate_contract
    validate_contract(tendons)
    expected={f'{side}_{digit}_loopback' for side in ('lh','rh') for digit in ('FF','MF','RF','LF')}
    if len(tendons)!=8 or {t['name'] for t in tendons}!=expected:
        raise ValueError('Corrected hand requires all eight passive loopbacks')
    if audit.get('profile')!=profile or audit.get('count')!=8 or {t['name'] for t in audit.get('tendons',[])}!=expected:
        raise ValueError('Missing corrected-hand live tendon audit')
    backend=audit.get('backend_readback',{})
    def flat(value):
        return sum((flat(x) for x in value),[]) if isinstance(value,list) else [float(value)]
    zero=('get_fixed_tendon_stiffnesses','get_fixed_tendon_dampings','get_fixed_tendon_rest_lengths','get_fixed_tendon_offsets')
    for name in zero:
        values=flat(backend.get(name,[]))
        if len(values)!=8 or any(x!=0 for x in values):raise ValueError('Missing live count or unexpected bilateral tendon force: '+name)
    stiffness=sorted(flat(backend.get('get_fixed_tendon_limit_stiffnesses',[])))
    want=sorted(t['physx_limit_stiffness'] for t in tendons)
    if len(stiffness)!=8 or any(not math.isclose(x,y,rel_tol=0,abs_tol=1e-5) for x,y in zip(stiffness,want)):
        raise ValueError('Live passive tendon stiffness differs')
    values=flat(backend.get('get_fixed_tendon_limits',[]))
    limits=sorted(zip(values[::2],values[1::2]));want=sorted(tuple(t['range_rad']) for t in tendons)
    if len(values)!=16 or any(not math.isclose(x,y,rel_tol=0,abs_tol=1e-6) for row,wanted in zip(limits,want) for x,y in zip(row,wanted)):
        raise ValueError('Live passive tendon limits differ')
    return {'profile':profile,'count':8,'backend_verified':True,'audit':audit}


def load_ready_receipt(path,*,expected_profile=None):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('ready') is not True:raise ValueError('Environment has not passed readiness')
    profile=mechanics_profile(receipt.get('mechanics_profile','upstream-v1'))
    if expected_profile is not None and profile!=mechanics_profile(expected_profile):
        raise ValueError('This demo requires '+expected_profile+'; legacy v1 readiness cannot satisfy it')
    if profile=='shadow-loopback-v2':
        for name in ('native_robot','native_robot_sha256','motor_contract','robot_usd','door_usd','input_hashes'):
            if not receipt.get(name):raise ValueError('Readiness predates versioned input verification: '+name)
        if not receipt.get('passive_tendons',{}).get('backend_verified') or receipt['passive_tendons'].get('count')!=8:
            raise ValueError('Readiness lacks eight verified live loopbacks')
        required=[receipt[name] for name in ('native_robot','motor_contract','robot_usd','door_usd')]
        required.append(str(Path(receipt['native_robot']).with_suffix('.audit.json')))
        if any(str(Path(name).resolve()) not in receipt['input_hashes'] for name in required):
            raise ValueError('Readiness is missing required asset hashes')
        for name,digest in receipt['input_hashes'].items():
            if file_sha256(name)!=digest:raise ValueError('Ready input changed: '+name)
        if file_sha256(receipt['native_robot'])!=receipt['native_robot_sha256']:
            raise ValueError('Ready native robot changed')
        motors=json.loads(Path(receipt['motor_contract']).read_text())
        if motors.get('source_xml_sha256')!=receipt['native_robot_sha256']:
            raise ValueError('Ready imported motors refer to a different native robot')
        validate_passive_audit(receipt['passive_tendons'].get('audit',{}),motors,profile)
    return receipt
