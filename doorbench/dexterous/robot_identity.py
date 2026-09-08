"""Path-independent, version-pinned identity of a compiled teacher robot model.

This is a source-model contract, not an Isaac runtime-physics attestation. File
locations are excluded; compiled geometry, named bindings, native mechanisms,
motor arrays and solver settings are retained. Opaque plugins are rejected.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SCHEMA = 'doorbench.compiled-robot-identity.v1'
FLOAT_DECIMALS = 12
PREFIXES = ('body_', 'jnt_', 'dof_', 'geom_', 'site_', 'pair_', 'exclude_',
            'eq_', 'tendon_', 'wrap_', 'actuator_', 'sensor_', 'mesh_',
            'hfield_', 'flex_', 'bvh_', 'tree_', 'plugin_')


def array_receipt(value):
    array = np.asarray(value)
    if array.dtype.kind == 'f':
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite compiled physical model data')
        data = np.round(array.astype('<f8'), FLOAT_DECIMALS)
        data = np.where(data == 0, 0., data).astype('<f8')  # Canonical positive zero.
        kind = 'float64-rounded-12-decimal-places'
    elif array.dtype.kind in 'iub':
        data = array.astype('<i8')
        kind = 'int64'
    else:
        raise ValueError('Unsupported compiled physical array type')
    return dict(shape=list(array.shape), encoding=kind,
                sha256=hashlib.sha256(data.tobytes(order='C')).hexdigest())


def compiled_robot_identity(model, *, sensor_plugins=None):
    """Fingerprint actual compiled arrays, independent of asset/XML locations."""
    if model.nplugin:
        if not sensor_plugins or any(np.any(getattr(model,name)>=0) for name in ('body_plugin','geom_plugin','actuator_plugin')):
            raise ValueError('Opaque model plugins require their own physical identity contract')
        if set(model.sensor_plugin[model.sensor_plugin>=0])!=set(range(model.nplugin)):
            raise ValueError('Every allowed plugin must be exclusively a declared sensor')
    arrays = {}
    for name in sorted(dir(model)):
        if name.endswith('_pathadr'):
            continue
        if not (name.startswith(PREFIXES) or name in ('qpos0', 'qpos_spring')):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = array_receipt(value)
    options = {}
    for name in sorted(dir(model.opt)):
        if name.startswith('_'):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, (int, float, np.ndarray)):
            options[name] = array_receipt(np.asarray(value))
    bindings = {}
    for kind, count in (('BODY',model.nbody),('JOINT',model.njnt),('GEOM',model.ngeom),
                        ('SITE',model.nsite),('TENDON',model.ntendon),('ACTUATOR',model.nu),
                        ('SENSOR',model.nsensor),('MESH',model.nmesh),('HFIELD',model.nhfield),
                        ('EQUALITY',model.neq)):
        object_type = getattr(mujoco.mjtObj, 'mjOBJ_'+kind)
        bindings[kind.lower()] = [mujoco.mj_id2name(model,object_type,index) for index in range(count)]
    payload = dict(schema=SCHEMA,mujoco_version=mujoco.__version__,
                   canonicalization=dict(float_decimal_places=FLOAT_DECIMALS,
                       integer_encoding='signed little-endian 64-bit',
                       excluded='asset paths, render-only assets, memory layout and XML labels'),
                   state_sizes={name:int(getattr(model,name)) for name in ('nq','nv','nu','na','nsensordata','nmocap')},
                   bindings=bindings,physical_arrays=arrays,solver_options=options,sensor_plugins=sensor_plugins or [])
    encoded = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {**payload,'sha256':hashlib.sha256(encoded).hexdigest()}


def robot_file_identity(path):
    # A same-path asset edit inside the filesystem timestamp granularity can
    # otherwise reuse an earlier compiled mesh from MuJoCo's process cache.
    # This only clears compiler assets; existing models and plant state remain.
    mujoco.mj_clearCache(mujoco.mj_getCache())
    spec=mujoco.MjSpec.from_file(str(Path(path)))
    model=spec.compile()
    declarations=[]
    if model.nplugin:
        tree=ET.fromstring(spec.to_xml())
        for node in tree.iter('plugin'):
            if node.get('plugin')!='mujoco.sensor.touch_grid':
                raise ValueError('Only the version-pinned built-in touch-grid sensor plugin is supported')
            declarations.append(dict(attributes=dict(sorted(node.attrib.items())),
                config=sorted((child.get('key'),child.get('value')) for child in node.findall('config'))))
        if not declarations:raise ValueError('Missing source plugin declarations')
    return compiled_robot_identity(model,sensor_plugins=declarations)


def verify_robot_identity(path, expected):
    if expected.get('schema') != SCHEMA:
        raise ValueError('Unsupported compiled robot identity schema')
    actual = robot_file_identity(path)
    if expected.get('mujoco_version') != actual['mujoco_version']:
        raise ValueError('Compiled robot identity requires the pinned MuJoCo version')
    if expected.get('sha256') != actual['sha256']:
        raise ValueError('Compiled robot geometry, mechanics, motor or solver identity differs')
    return actual
