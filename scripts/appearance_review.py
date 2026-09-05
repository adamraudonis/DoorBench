#!/usr/bin/env python3
"""Build labelled contact sheets from actual Blender output for personal review.

This does not synthesize or retouch the rendered images. Every thumbnail links
back to a checksummed full-size image and its render metadata in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--renders', default='out/appearance')
    ap.add_argument('--assets', default='assets')
    ap.add_argument('--out', default='docs/review/blender/atlas')
    a = ap.parse_args()
    source, out = Path(a.renders).resolve(), Path(a.out).resolve()
    index = json.loads((source/'index.json').read_text())
    manifest = json.loads((Path(a.assets)/'manifest.json').read_text())
    doors = {d['id']:d for d in manifest['doors']}
    entries = [r for r in index['renders'] if r['image']]
    out.mkdir(parents=True, exist_ok=True)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = ImageFont.truetype(font_path, 13) if Path(font_path).is_file() else ImageFont.load_default()
    title = ImageFont.truetype(font_path, 20) if Path(font_path).is_file() else font
    records = []
    for start in range(0, len(entries), 20):
        group = entries[start:start+20]
        sheet = Image.new('RGB', (1000, 1460), '#111820')
        draw = ImageDraw.Draw(sheet)
        draw.text((14,12), f'DoorBench / Blender Cycles / {start+1}–{start+len(group)} of {len(entries)}', font=title, fill='#eef0e9')
        for i, entry in enumerate(group):
            path = source/entry['image']
            metadata = json.loads((source/entry['metadata']).read_text())
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if checksum != metadata['artifact_sha256'][path.name]:
                raise ValueError(f'Image checksum mismatch: {path}')
            x, y = 12+(i%4)*247, 50+(i//4)*278
            im = ImageOps.contain(Image.open(path).convert('RGB'), (235,235), Image.Resampling.LANCZOS)
            sheet.paste(im, (x+(235-im.width)//2, y+(235-im.height)//2))
            label = f"{entry['door_id']} / v{entry['variant']}"
            draw.text((x,y+238), label, font=font, fill='#e8e7df')
            recipe = entry['recipe']
            context = recipe['wall'].removeprefix('wall_')+' / '+recipe['floor'].removeprefix('floor_')
            draw.text((x,y+255), context, font=font, fill='#adb8c2')
            records.append({**entry, 'family':doors[entry['door_id']]['family'], 'image_sha256':checksum,
                            'sheet':f'page_{start//20+1:03d}.jpg', 'sheet_position':i})
        draw.text((14,1440), 'Actual saved renders; preview sample settings are not a photographic or physical sign-off.', font=font, fill='#adb8c2')
        sheet.save(out/f'page_{start//20+1:03d}.jpg', quality=83, optimize=True)
    summary = {'source':str(source), 'count':len(records), 'doors':len({r['door_id'] for r in records}),
               'families':len({r['family'] for r in records}), 'sheets':(len(records)+19)//20,
               'renders':records}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='renders'},indent=2))


if __name__ == '__main__':
    main()
