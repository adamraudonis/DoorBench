"""Read BVH capture without changing its time, proportions or joint motion.

Offsets and positions use the file's native length unit. Callers must supply the
documented scale and proper coordinate rotation; this reader does not infer units,
remove calibration poses, smooth data, or invent prop/contact measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Joint:
    name: str
    parent: int
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_start: int
    end_site: bool = False


@dataclass(frozen=True)
class Capture:
    joints: tuple[Joint, ...]
    values: np.ndarray
    frame_time: float

    @property
    def times(self):
        return np.arange(len(self.values), dtype=float) * self.frame_time


def read_bvh(path: str | Path) -> Capture:
    return parse_bvh(Path(path).read_text(encoding='utf-8-sig'))


def parse_bvh(text: str) -> Capture:
    parts = re.split(r'^\s*MOTION\s*$', text, flags=re.MULTILINE)
    if len(parts) != 2:
        raise ValueError('BVH must contain one MOTION section')
    tokens = iter(re.findall(r'\{|\}|[^\s{}]+', parts[0]))
    joints, names = [], set()
    channel_count = 0

    def expect(expected):
        value = next(tokens, None)
        if value != expected:
            raise ValueError(f'Expected {expected!r}, got {value!r}')

    def joint(parent, *, end_site=False, depth=0):
        nonlocal channel_count
        if depth > 128:
            raise ValueError('BVH hierarchy is too deep')
        name = joints[parent].name + '__end' if end_site else next(tokens, None)
        if not name or name in names:
            raise ValueError('Missing or duplicate joint name')
        names.add(name)
        expect('{'); expect('OFFSET')
        try:
            offset = tuple(float(next(tokens)) for _ in range(3))
        except (ValueError, StopIteration) as exc:
            raise ValueError('Invalid joint offset') from exc
        if not np.isfinite(offset).all():
            raise ValueError('Nonfinite joint offset')
        channels = ()
        if not end_site:
            expect('CHANNELS')
            try:
                count = int(next(tokens))
                if not 1 <= count <= 6:
                    raise ValueError('Unsupported channel count')
                channels = tuple(next(tokens) for _ in range(count))
            except (ValueError, StopIteration) as exc:
                raise ValueError('Invalid channel declaration') from exc
            allowed = {a + kind for a in 'XYZ' for kind in ('position', 'rotation')}
            if len(set(channels)) != len(channels) or set(channels) - allowed:
                raise ValueError('Invalid or repeated channel')
        index = len(joints)
        joints.append(Joint(name, parent, offset, channels, channel_count, end_site))
        channel_count += len(channels)
        while True:
            value = next(tokens, None)
            if value == '}':
                break
            if end_site:
                raise ValueError('End Site must not contain children')
            if value == 'JOINT':
                joint(index, depth=depth + 1)
            elif value == 'End':
                expect('Site'); joint(index, end_site=True, depth=depth + 1)
            else:
                raise ValueError(f'Unexpected hierarchy token {value!r}')

    expect('HIERARCHY'); expect('ROOT'); joint(-1)
    if next(tokens, None) is not None:
        raise ValueError('Unexpected tokens after root hierarchy')
    header = re.match(r'\s*Frames:\s*(\d+)\s*\n\s*Frame Time:\s*([^\s]+)\s*\n', parts[1])
    if not header:
        raise ValueError('Missing frame count or frame time')
    count, dt = int(header[1]), float(header[2])
    if count < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Frame count and time must be positive and finite')
    rows = [line.split() for line in parts[1][header.end():].splitlines() if line.strip()]
    if len(rows) != count or any(len(row) != channel_count for row in rows):
        raise ValueError('Motion rows disagree with declared frames or channels')
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite motion channel')
    values.setflags(write=False)
    return Capture(tuple(joints), values, dt)


def forward_kinematics(capture: Capture, *, length_scale: float, basis=None):
    """Return world joint positions and rotations in documented target coordinates.

    BVH rotations compose in declared intrinsic channel order. Translation
    channels apply in the joint's parent coordinates, alongside its static offset.
    ``basis`` rotates the source coordinate axes into the target world and must
    be a proper rotation; a reflection would silently exchange handedness.
    """
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('length_scale must be positive and finite')
    basis = np.eye(3) if basis is None else np.asarray(basis, dtype=float)
    if (basis.shape != (3, 3) or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(basis), 1, atol=1e-9, rtol=0)):
        raise ValueError('basis must be a proper coordinate rotation')
    frames, bones = len(capture.values), len(capture.joints)
    positions = np.zeros((frames, bones, 3))
    rotations = np.zeros((frames, bones, 3, 3))
    for j, bone in enumerate(capture.joints):
        local_pos = np.tile(np.asarray(bone.offset), (frames, 1))
        local_rot = np.tile(np.eye(3), (frames, 1, 1))
        for i, channel in enumerate(bone.channels):
            axis = 'XYZ'.index(channel[0])
            value = capture.values[:, bone.channel_start + i]
            if channel.endswith('position'):
                local_pos[:, axis] += value
            else:
                angle = np.deg2rad(value)
                c, s = np.cos(angle), np.sin(angle)
                a, b = (axis + 1) % 3, (axis + 2) % 3
                elementary = np.tile(np.eye(3), (frames, 1, 1))
                elementary[:, a, a] = elementary[:, b, b] = c
                elementary[:, a, b], elementary[:, b, a] = -s, s
                local_rot = local_rot @ elementary
        local_pos *= length_scale
        if bone.parent == -1:
            positions[:, j] = local_pos
            rotations[:, j] = local_rot
        else:
            parent_rot = rotations[:, bone.parent]
            positions[:, j] = positions[:, bone.parent] + np.einsum('tij,tj->ti', parent_rot, local_pos)
            rotations[:, j] = parent_rot @ local_rot
    return (np.einsum('ij,tkj->tki', basis, positions),
            np.einsum('ij,tkjl,ml->tkim', basis, rotations, basis))
