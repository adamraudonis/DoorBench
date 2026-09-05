#!/usr/bin/env python
"""Build a complete, labelled visual screening atlas from the generated renders.

This records image coverage, never creates visual approval verdicts. Legacy open
thumbnails set only the primary joint and are explicitly labelled raw pose.
Usage: python scripts/inspection_atlas.py [--assets assets] [--out docs/review/takeover/atlas]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

VIEWS = [('iso', 'closed / front'), ('far_view', 'closed / reverse'),
         ('detail_handle', 'hardware'), ('iso_open', 'open / raw pose')]


def validate_render_record(source: Path, record: dict, door_id: str) -> dict:
    """Require current model provenance; additionally verify every newer source hash."""
    if record.get('door_id') != door_id:
        raise ValueError(f'{door_id}: render record belongs to another door')
    if not record.get('source_model_sha256'):
        raise ValueError(f'{door_id}: render record has no model provenance')
    verified = {}
    for kind, filename in [('model', 'model.json'), ('spec', 'spec.json'), ('xml', 'door.xml')]:
        key = f'source_{kind}_sha256'
        if key not in record:  # Earlier renders recorded only model.json.
            continue
        actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        if actual != record[key]:
            raise ValueError(f'{door_id}: stale render ({filename} changed); rerender before building the atlas')
        verified[key] = actual
    return verified


def pose_label(view: str, title: str, pose: dict) -> tuple[list[str], dict]:
    """Prescribed motion and failed loop solves remain explicit in both image and index."""
    if not isinstance(pose.get('forced_locked_pose'), bool):
        raise ValueError(f'{view}: missing forced_locked_pose in render record')
    residual = float(pose['loop_residual_m'])
    if not math.isfinite(residual) or residual < 0:
        raise ValueError(f'{view}: invalid loop residual')
    forced = pose['forced_locked_pose']
    unsolved = residual > 0.001
    lines = [title]
    if forced:
        lines.append('FORCED LOCKED POSE')
    if unsolved:
        lines.append(f'UNSOLVED LOOP: {residual * 1000:.2f} mm')
    return lines, {'label': ' | '.join(lines), 'forced_locked_pose': forced,
                   'loop_residual_m': residual, 'loop_unsolved': unsolved}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, default=Path('assets'))
    ap.add_argument('--out', type=Path, default=Path('docs/review/takeover/atlas'))
    ap.add_argument('--renders', type=Path, help='Fully framed output from render_inspection.py')
    a = ap.parse_args()
    views = [('front', 'closed / front'), ('reverse', 'closed / reverse'), ('hardware', 'hardware'), ('open', 'open / prescribed pose')] if a.renders else VIEWS
    raw = (a.assets / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    doors = sorted(manifest['doors'], key=lambda d: (d['family'], d['index']))
    # Validate every record before writing pages so stale images cannot enter a partial atlas.
    records = {}
    for d in doors if a.renders else []:
        record = json.loads((a.renders / d['id'] / 'render.json').read_text())
        provenance = validate_render_record(a.assets / 'doors' / d['id'], record, d['id'])
        poses = {v['view']: v for v in record['views']}
        labels = {view: pose_label(view, title, poses[view]) for view, title in views}
        records[d['id']] = (provenance, labels)
    a.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=10)
    cw, ch, header = 180, 135, 39
    tw, th = cw * len(views), ch + header
    pages = []
    for start in range(0, len(doors), 20):
        rows = doors[start:start + 20]
        sheet = Image.new('RGB', (2 * tw, 10 * th + 28), '#161a20')
        draw = ImageDraw.Draw(sheet)
        name = f'page_{start // 20 + 1:02d}.jpg'
        draw.text((8, 5), f'DoorBench / {name} / visual screening only; prescribed poses, inspect render residuals before accepting mechanisms', fill='#ffffff', font=font)
        entries = []
        for n, d in enumerate(rows):
            x, y = (n % 2) * tw, (n // 2) * th + 28
            draw.text((x + 4, y + 2), f"{d['id']} | {d['leaf']['slab']} | {d['operator']}", fill='#ffd786', font=font)
            draw.text((x + 4, y + 17), f"{d['closer']} / {d['latch']} / {d['lock']} / {d['mass_kg']} kg", fill='#bfc9d8', font=small)
            files = []
            for col, (view, title) in enumerate(views):
                lines, pose_info = records[d['id']][1][view] if a.renders else ([title], {})
                path = a.renders / d['id'] / f'{view}.jpg' if a.renders else a.assets / 'doors' / d['id'] / f'thumb_{view}.jpg'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing required image for {d["id"]}: {path}')
                with Image.open(path) as im:
                    thumb = ImageOps.contain(im.convert('RGB'), (cw, ch), Image.Resampling.LANCZOS)
                px, py = x + col * cw, y + header
                sheet.paste(thumb, (px, py))
                draw.rectangle((px, py, px + cw, py + 13 * len(lines)), fill='#20252c')
                for line_index, line in enumerate(lines):
                    draw.text((px + 3, py + 1 + 13 * line_index), line,
                              fill='#ffba92' if line_index else '#dddddd', font=small)
                files.append({'view': view, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **pose_info})
            entry = {'door_id': d['id'], 'cell': n, 'images': files}
            if a.renders:
                entry['verified_render_provenance'] = records[d['id']][0]
            entries.append(entry)
        sheet.save(a.out / name, quality=68, optimize=True)
        pages.append({'file': name, 'doors': entries})
    index = {'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'generated_dataset': manifest['generated'],
             'door_count': len(doors), 'image_count': len(doors) * len(views),
             'scope': 'Whole-dataset visual screening; no automatic visual sign-off. Rendered poses are prescribed, not evidence of successful dynamic actuation.' if a.renders else 'Legacy raw primary-joint poses; no visual sign-off.',
             'pages': pages}
    (a.out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(f'{len(pages)} pages, {len(doors)} doors, {index["image_count"]} images -> {a.out}')


if __name__ == '__main__':
    main()
