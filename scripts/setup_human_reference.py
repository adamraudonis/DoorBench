#!/usr/bin/env python3
"""Fetch pinned public human assets/capture, then build an isolated Blender character.

Generated assets and external tool code stay in ignored out/. No credentials or
Blender user preferences are needed. Capture-derived outputs retain CC BY 4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import urllib.request
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
COMMIT = '80919fa4682335c41847f761a4d79dcad4124732'
PACKAGES = [
    ('mpfb-80919fa.zip', f'https://github.com/makehumancommunity/mpfb2/archive/{COMMIT}.zip',
     '038e9f01ae3900ad11f24af887e184c0bfa20a00abe4411f750d09b21efaaadc'),
    ('makehuman_system_assets_cc0.zip',
     'https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip',
     'b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107'),
]
CAPTURE_URL = 'https://ndownloader.figshare.com/files/49174882'
BVH = 'sub-d02_ses-02_task-o03_tracksys-rokokosmartsuit1_run-01_motion.bvh'
BVH_SHA256 = '7104b8e750d5d8d35d19f52aa2d9cc721b36aa7b20022d7820952d98995a5a02'
# Pinned public ZIP local header offsets, compressed/uncompressed sizes and SHA.
# Fetch only these members of the 3 GB archive, including its original license.
MEMBERS = [
    ('ceti-age-kinematics/LICENSE', 538897464, 5927, 18665,
     '90e11e96f7704c1e46e3fcab857455da5fd7f06a509e96824011159351ff701f'),
    ('ceti-age-kinematics/dataset_description.json', 210185125, 768, 1379,
     '6db8f160b43117fbe229843e9853ebe912797b79eb4211fb11645384a41e810e'),
    ('ceti-age-kinematics/sub-d02/ses-02/bvh/' + BVH, 577337493, 212300, 710742,
     BVH_SHA256),
]


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def stream_sha(stream):
    # Keep the setup CLI compatible with the project's Python >=3.10 floor.
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def fetch(path, url, expected):
    if path.is_file() and sha(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + '.download')
    request = urllib.request.Request(url, headers={'User-Agent': 'DoorBench human-reference builder'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output, 1024 * 1024)
    if sha(temporary) != expected:
        temporary.unlink()
        raise ValueError(f'Pinned download hash mismatch: {path.name}')
    temporary.replace(path)


def byte_range(start, count):
    request = urllib.request.Request(CAPTURE_URL, headers={
        'Range': f'bytes={start}-{start + count - 1}', 'Accept-Encoding': 'identity',
        'User-Agent': 'DoorBench human-reference builder'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(
                f'bytes {start}-{start + count - 1}/'):
            raise ValueError('Capture host did not honor the exact byte range; refusing whole-archive fallback')
        data = response.read(count + 1)
    if len(data) != count:
        raise ValueError('Truncated or oversized capture byte range')
    return data


def fetch_member(folder, member):
    name, offset, compressed, size, expected = member
    target = folder / Path(name).name
    if target.is_file() and sha(target) == expected:
        return target
    header = byte_range(offset, 30)
    signature, version, flags, method, mtime, mdate, crc, csize, usize, nlen, elen = struct.unpack('<I5H3I2H', header)
    if signature != 0x04034B50 or flags & 1 or method != 8:
        raise ValueError('Unsupported capture ZIP member')
    # General-purpose bit 3 permits sizes in the later data descriptor. The
    # pinned member sizes and full uncompressed SHA remain authoritative.
    if not flags & 8 and (csize, usize) != (compressed, size):
        raise ValueError('Capture ZIP member sizes changed')
    metadata = byte_range(offset + 30, nlen + elen)
    if metadata[:nlen].decode('utf-8') != name:
        raise ValueError('Capture ZIP member name changed')
    packed = byte_range(offset + 30 + nlen + elen, compressed)
    inflater = zlib.decompressobj(-15)
    data = inflater.decompress(packed, size + 1)
    if len(data) != size or not inflater.eof or inflater.unused_data:
        raise ValueError('Capture ZIP payload length or stream mismatch')
    if hashlib.sha256(data).hexdigest() != expected or (not flags & 8 and zlib.crc32(data) != crc):
        raise ValueError('Capture ZIP payload checksum mismatch')
    target.write_bytes(data)
    return target


def extract(archive, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe external archive path')
        package.extractall(folder)


def verify_extracted(archive, folder):
    """Bind reused tool/data trees to archive bytes, before external imports.

    Existing Python bytecode is ignored here and must not be executed: the
    Blender builder selects a fresh per-process pycache prefix before import.
    """
    folder = Path(folder).resolve()
    expected = set()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            relative = Path(member.filename)
            if (relative.is_absolute() or '..' in relative.parts
                    or '\\' in member.filename
                    or relative.as_posix() != member.filename.rstrip('/')
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unsafe pinned archive member: ' + member.filename)
            if member.is_dir():
                continue
            if relative.as_posix() in expected:
                raise ValueError('Duplicate pinned archive member: ' + member.filename)
            expected.add(relative.as_posix())
            target = folder / relative
            if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(folder):
                raise ValueError(f'Pinned cache file missing or redirected: {target}; remove its cache tree and rerun setup')
            with package.open(member) as stream:
                digest = stream_sha(stream)
            if sha(target) != digest:
                raise ValueError(f'Pinned cache file changed: {target}; remove its cache tree and rerun setup')
    for target in folder.rglob('*'):
        if target.is_symlink():
            raise ValueError('Unexpected symlink in pinned cache: ' + str(target))
        if target.is_file() and target.relative_to(folder).as_posix() not in expected:
            if target.suffix == '.pyc' and target.parent.name == '__pycache__':
                continue
            raise ValueError('Unexpected file in pinned cache: ' + str(target))


def verify_toolcache(tool):
    for (name, _url, digest), folder in zip(PACKAGES, ['source', 'system-assets']):
        archive = Path(tool) / name
        if sha(archive) != digest:
            raise ValueError('Pinned external package checksum changed: ' + str(archive))
        verify_extracted(archive, Path(tool) / folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'build'])
    parser.add_argument('--toolcache', type=Path, default=REPO / 'out/toolcache/mpfb')
    parser.add_argument('--out', type=Path, default=REPO / 'out/human-reference')
    parser.add_argument('--blender', default='/Applications/Blender.app/Contents/MacOS/Blender')
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    tool = args.toolcache.resolve(); tool.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve(); source = out / 'source'; source.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name, url, digest in PACKAGES:
        path = tool / name
        fetch(path, url, digest)
        downloads.append(dict(file=str(path), url=url, bytes=path.stat().st_size, sha256=digest))
    (tool / 'download-manifest.json').write_text(json.dumps(downloads, indent=2) + '\n')
    capture_files = [fetch_member(source, member) for member in MEMBERS]
    (source / 'receipt.json').write_text(json.dumps({
        'dataset': 'CeTI-Age-Kinematics v2', 'doi': '10.6084/m9.figshare.26983645.v2',
        'license': 'CC-BY-4.0', 'archive_url': CAPTURE_URL,
        'files': [dict(path=p.name, sha256=sha(p)) for p in capture_files],
    }, indent=2) + '\n')
    if args.command == 'fetch':
        print(f'Pinned external tool, CC0 assets and CC BY capture ready: {out}')
        return
    addon = tool / 'source' / ('mpfb2-' + COMMIT) / 'src/mpfb'
    if not addon.is_dir():
        extract(tool / PACKAGES[0][0], tool / 'source')
    if not (tool / 'system-assets/skins/young_caucasian_male2').is_dir():
        extract(tool / PACKAGES[1][0], tool / 'system-assets')
    extension = tool / 'extension-repo'; extension.mkdir(exist_ok=True)
    link = extension / 'mpfb'
    if not link.exists():
        link.symlink_to(addon, target_is_directory=True)
    if link.resolve() != addon.resolve():
        raise ValueError('External extension points to a different MPFB revision')
    environment = dict(os.environ, BLENDER_USER_RESOURCES=str(tool / 'blender-user'))
    blender = [args.blender, '--background', '--factory-startup', '--python-exit-code', '1']
    render_args = ['--no-render'] if args.no_render else []
    subprocess.run(blender + ['--python', str(REPO / 'scripts/blender_reference_human.py'), '--',
        '--toolcache', str(tool), '--out', str(out / 'assets'), *render_args], env=environment, check=True)
    subprocess.run(blender + [str(out / 'assets/human-preview.blend'), '--python',
        str(REPO / 'scripts/blender_human_calibration.py'), '--', '--source', str(source / BVH),
        '--out', str(out / 'assets'), *render_args], env=environment, check=True)
    print(f'Blender human and separate capture-informed calibration ready: {out / "assets"}')


if __name__ == '__main__':
    main()
