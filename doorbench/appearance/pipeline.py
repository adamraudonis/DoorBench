"""Prepare reproducible appearance jobs outside Blender and run them in batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def find_blender(explicit=None):
    candidates = [explicit, os.environ.get('DOORBENCH_BLENDER'), shutil.which('blender'),
                  '/Applications/Blender.app/Contents/MacOS/Blender']
    for value in candidates:
        if value and Path(value).is_file():
            return str(Path(value).resolve())
    raise FileNotFoundError('Blender is required. Set DOORBENCH_BLENDER or pass --blender /path/to/blender.')


def select_doors(manifest, selection):
    rows = manifest['doors']
    if selection == 'all':
        return rows
    if selection == 'families':
        # Stable representative of every family; full coverage is still available with all.
        seen = set()
        return [d for d in rows if d['family'] not in seen and not seen.add(d['family'])]
    ids = set(selection.removeprefix('ids:').split(','))
    known = {d['id'] for d in rows}
    if ids-known:
        raise ValueError('Unknown door IDs: '+', '.join(sorted(ids-known)))
    return [d for d in rows if d['id'] in ids]


def prepare_job(assets, door_id, out, *, seed=0, variant=0, quality='photo', width=960, height=960,
                wall=None, floor=None, door_finish=None, lighting=None, view='front', state=None,
                validate_only=False, save_blend=False, device='auto', texture_library=None):
    from .catalog import resolve_recipe
    from .state import capture_initial_state, validate_snapshot
    source = Path(assets).resolve()/'doors'/door_id
    spec = json.loads((source/'spec.json').read_text())
    model = json.loads((source/'model.json').read_text())
    if spec['id'] != door_id or model['name'] != door_id:
        raise ValueError('Door ID does not match source files')
    mesh_names = {g['mesh_name'] for b in model['bodies'] for g in b['geoms'] if g['type']=='mesh'}
    hardware = source.parent.parent/'hardware'
    for name in mesh_names:
        if not (hardware/f'{name}.obj').is_file():
            raise FileNotFoundError(hardware/f'{name}.obj')
    recipe = resolve_recipe(spec, seed=seed+variant, wall=wall, floor=floor, door_finish=door_finish, lighting=lighting)
    recipe['render_device'] = device
    reference_state = capture_initial_state(source)
    state = reference_state if state is None else validate_snapshot(state, expected_door_id=door_id)
    if state.get('door_id') != door_id:
        raise ValueError('Snapshot door_id does not match requested door')
    unnamed = state.get('unnamed_source_objects', {})
    if unnamed.get('body_ids') or unnamed.get('geom_ids'):
        raise ValueError('Snapshot contains unnamed bodies or geometry without a render-source mapping')
    for filename in ('spec.json','model.json','door.xml'):
        expected = state.get('source', {}).get(filename+'_sha256')
        if expected and hashlib.sha256((source/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Stale simulation snapshot: {filename} changed since capture')
    source_bodies = {b['name'] for b in model['bodies']} | {'world', 'world_env'}
    extra_bodies = set(state['body_world'])-source_bodies
    if extra_bodies:
        raise ValueError('Snapshot contains scene bodies without render geometry: '+', '.join(sorted(extra_bodies))+'. Export their geometry before rendering a complete robot observation.')
    missing_bodies = {b['name'] for b in model['bodies']}-set(state['body_world'])
    if missing_bodies:
        raise ValueError('Snapshot lacks full-tier source bodies: '+', '.join(sorted(missing_bodies)))
    extra_geoms = set(state.get('geom_world',{}))-{g['name'] for b in model['bodies'] for g in b['geoms']}
    if extra_geoms:
        raise ValueError('Snapshot contains geometry without a render source: '+', '.join(sorted(extra_geoms)))
    for body in model['bodies']:
        for geom in body['geoms']:
            native = state.get('geom_world', {}).get(geom['name'])
            if native is None or geom['type']=='mesh':
                continue
            if native['geom_type'] != geom['type'] or any(abs(x-y)>max(1e-6,abs(x)*2e-6) for x,y in zip(geom['size'],native['size'])):
                raise ValueError(f"Snapshot geometry differs from render source: {geom['name']}")
    job = {'schema_version':1, 'door_id':door_id,'variant':variant,'door_dir':str(source),'hardware_dir':str(hardware),
           'out_dir':str(Path(out).resolve()/door_id/f'variant_{variant:03d}'), 'recipe':recipe,'state':state,
           'reference_state':reference_state, 'texture_library':texture_library,
           'seed':seed+variant,'quality':quality,'width':width,'height':height,'view':view,
           'validate_only':validate_only,'save_blend':save_blend,
           'renderer_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
           'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('spec.json','model.json','door.xml')},
           'mesh_sha256':{name:hashlib.sha256((hardware/f'{name}.obj').read_bytes()).hexdigest() for name in sorted(mesh_names)}}
    job['job_sha256'] = digest(job)
    return job


def run_jobs(jobs, out, *, blender=None, resume=False):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((out/'index.json').read_text()).get('renders',[])
    except (OSError,ValueError,AttributeError):
        previous = []
    pending, cached = [], []
    for job in jobs:
        path = Path(job['out_dir'])/'render.json'
        if resume and path.is_file():
            try:
                old = json.loads(path.read_text())
                if not isinstance(old,dict): old = {}
            except (OSError,ValueError):
                old = {}
            def artifact_ok(filename):
                artifact = Path(job['out_dir'])/filename
                return artifact.is_file() and hashlib.sha256(artifact.read_bytes()).hexdigest() == old.get('artifact_sha256',{}).get(filename)
            image_ok = job['validate_only'] or artifact_ok('rgb.png')
            blend_ok = not job['save_blend'] or artifact_ok('scene.blend')
            if old.get('job_sha256') == job['job_sha256'] and image_ok and blend_ok:
                cached.append(old); continue
        pending.append(job)
    result_path = out/'worker_results.json'
    config = out/'jobs.json'
    config.write_text(json.dumps({'jobs':pending,'result_path':str(result_path)},allow_nan=False)+'\n')
    failed = []
    results = cached
    if pending:
        result_path.unlink(missing_ok=True)
        worker = Path(__file__).with_name('blender_worker.py')
        log = out/'blender.log'
        command = [find_blender(blender),'--background','--factory-startup','--python-exit-code','1','--python',str(worker),'--','--jobs',str(config)]
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result_path.is_file():
            try:
                data = json.loads(result_path.read_text()); results = cached+data['results']; failed = data['failures']
            except (OSError,ValueError,KeyError,TypeError):
                failed = [{'door_id':None,'error':'Blender result file is incomplete or invalid'}]
        if proc.returncode and not failed:
            failed = [{'door_id':None,'error':f'Blender exited {proc.returncode}; see {log}'}]
    by_key = {(r['door_id'],r['job_sha256']):r for r in results}
    entries = []
    for job in jobs:
        result = by_key.get((job['door_id'],job['job_sha256']))
        if result:
            relative = Path(job['out_dir']).relative_to(out)
            entries.append({'door_id':job['door_id'],'variant':job['variant'],'recipe':job['recipe'],
                            'image':str(relative/'rgb.png') if result['rendered'] else None,
                            'metadata':str(relative/'render.json'),
                            'blend':str(relative/'scene.blend') if job['save_blend'] else None,
                            'quality':job['quality'],'source_sha256':job['source_sha256']})
    batch_completed = len(entries)
    requested = {(j['door_id'],j['variant']) for j in jobs}
    # Rendering another door/variant should not erase the rest of the catalogue.
    # Replace every requested slot, including failed ones, to exclude stale images.
    for entry in previous if isinstance(previous,list) else []:
        try:
            if (entry['door_id'],entry['variant']) in requested:
                continue
            metadata_path = (out/entry['metadata']).resolve()
            if not metadata_path.is_relative_to(out):
                continue
            metadata = json.loads(metadata_path.read_text())
            for key in ('image','blend'):
                if entry.get(key):
                    artifact = (out/entry[key]).resolve()
                    if not artifact.is_relative_to(out) or hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata['artifact_sha256'][artifact.name]:
                        raise ValueError('Stale appearance artifact')
            entries.append(entry)
        except (OSError,ValueError,KeyError,TypeError):
            continue
    entries.sort(key=lambda e:(e['door_id'],e['variant']))
    index = {'schema_version':1,'renderer':'Blender Cycles','door_count':len({e['door_id'] for e in entries}),
             'requested':len(jobs),'batch_completed':batch_completed,
             'completed':len(entries),'rendered':sum(bool(e['image']) for e in entries),'failed':failed,'renders':entries}
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in index.items() if k!='renders'},indent=2), flush=True)
    if failed or batch_completed!=len(jobs):
        raise RuntimeError(f'Incomplete appearance batch; see {out}/index.json and blender.log')
    return index


def render_trajectory(assets, door_id, snapshots, out, *, seed=0, blender=None, **appearance):
    """Offline vision observations with one fixed appearance recipe across the trajectory.

    Capture snapshots from live simulation with state.capture_mujoco_state. Rendering
    is deliberately outside the physics step: Cycles is not a real-time RL backend.
    """
    jobs = []
    for frame, state in enumerate(snapshots):
        job = prepare_job(assets, door_id, out, seed=seed, state=state, **appearance)
        job['out_dir'] = str(Path(out).resolve()/door_id/f'frame_{frame:06d}')
        job['variant'] = frame
        job.pop('job_sha256')
        job['job_sha256'] = digest(job)
        jobs.append(job)
    if not jobs:
        raise ValueError('Trajectory contains no snapshots')
    return run_jobs(jobs,out,blender=blender)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command',required=True)
    commands.add_parser('catalog',help='List interchangeable wall, floor, finish and lighting presets')
    textures = commands.add_parser('fetch-textures',help='Download the curated CC0 PBR texture library from Poly Haven')
    textures.add_argument('--out',default='out/appearance-textures')
    textures.add_argument('--resolution',choices=['1k','2k'],default='2k')
    p = commands.add_parser('render',help='Render arbitrary doors with Blender Cycles; use --validate-only to build-check')
    p.add_argument('--assets',default='assets'); p.add_argument('--doors',default='families',help='all | families | comma-separated IDs')
    p.add_argument('--out',default='out/appearance'); p.add_argument('--blender')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--variants',type=int,default=1)
    p.add_argument('--quality',choices=['preview','photo'],default='photo')
    p.add_argument('--device',choices=['auto','CPU','METAL'],default='auto')
    texture_options = p.add_mutually_exclusive_group()
    texture_options.add_argument('--textures',type=Path,help='Scanned PBR manifest; automatically uses out/appearance-textures/manifest.json when present')
    texture_options.add_argument('--procedural-only',action='store_true',help='Use built-in procedural materials without the scan library')
    p.add_argument('--width',type=int,default=960); p.add_argument('--height',type=int,default=960)
    for name in ('wall','floor','door-finish','lighting'):
        p.add_argument('--'+name)
    p.add_argument('--view',default='front',choices=['front','reverse'])
    p.add_argument('--state',type=Path,help='Authoritative simulation snapshot; exactly one door and one variant')
    p.add_argument('--validate-only',action='store_true'); p.add_argument('--save-blend',action='store_true'); p.add_argument('--resume',action='store_true')
    a = ap.parse_args(argv)
    if a.command == 'fetch-textures':
        from .textures import fetch_library
        print(f'Poly Haven CC0 texture manifest: {fetch_library(a.out,resolution=a.resolution)}',flush=True)
        return
    if a.command == 'catalog':
        from .catalog import WALLS,FLOORS,DOOR_FINISHES,LIGHTING
        print(json.dumps({'walls':WALLS,'floors':FLOORS,'door_finishes':DOOR_FINISHES,'lighting':LIGHTING},indent=2)); return
    if a.variants<1 or a.width<32 or a.height<32 or a.width>8192 or a.height>8192:
        ap.error('variants must be positive; width/height must be between 32 and 8192')
    manifest = json.loads((Path(a.assets)/'manifest.json').read_text())
    doors = select_doors(manifest,a.doors)
    if a.state and (len(doors)!=1 or a.variants!=1):
        ap.error('--state requires exactly one door and one variant')
    snapshot = json.loads(a.state.read_text()) if a.state else None
    texture_library = None
    texture_path = a.textures or Path('out/appearance-textures/manifest.json')
    if not a.procedural_only and (a.textures or texture_path.is_file()):
        from .textures import load_texture_library
        texture_library = load_texture_library(texture_path)
        print(f'Using scanned PBR materials: {texture_path}',flush=True)
    elif not a.procedural_only:
        print('Using procedural PBR materials. Run appearance fetch-textures to enable the Poly Haven scan library.',flush=True)
    jobs=[]
    for d in doors:
        for variant in range(a.variants):
            jobs.append(prepare_job(a.assets,d['id'],a.out,seed=a.seed,variant=variant,quality=a.quality,width=a.width,height=a.height,
                                    wall=a.wall,floor=a.floor,door_finish=a.door_finish,lighting=a.lighting,view=a.view,state=snapshot,
                                    validate_only=a.validate_only,save_blend=a.save_blend,device=a.device,texture_library=texture_library))
        if len(jobs)%50==0:
            print(f'Prepared {len(jobs)} Blender jobs',flush=True)
    run_jobs(jobs,a.out,blender=a.blender,resume=a.resume)


if __name__ == '__main__':
    main()
