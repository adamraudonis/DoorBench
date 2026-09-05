#!/usr/bin/env python3
"""Blender phase storyboards for independently accepted motion, without saving scenes.

These diagnostic images hide contextual walls for visibility. They are sampled
visual-review aids, not photorealistic observations or full-motion approvals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_samples(phases, times):
    """Always include start/end and samples from every contiguous phase."""
    if len(phases) != len(times) or len(times) < 2:
        raise ValueError('Aligned phases and times are required')
    if any(not isinstance(p,str) or not p for p in phases) or any(not math.isfinite(float(t)) for t in times) or float(times[0])!=0 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Phases must be nonempty labels on a finite increasing zero-based timeline')
    result = {0, len(times)-1}
    begin = 0
    for end in range(1, len(phases)+1):
        if end < len(phases) and phases[end] == phases[begin]: continue
        fractions = (0., .25, .5, .75, 1.) if phases[begin] in ('operate', 'traverse') else (.5,)
        for f in fractions:
            result.add(begin+round((end-1-begin)*f))
        begin = end
    return [{'index': i, 'time_s': float(times[i]), 'phase': phases[i]} for i in sorted(result)]


def provenance(job, paths, appearance_job):
    """Read current dependency bytes; a fresh prepared job is required on resume."""
    dependencies={key:sha(ROOT/path) for key,path in {
        'script_sha256':'scripts/blender_motion_storyboards.py',
        'renderer_sha256':'scripts/blender_planned_motion.py',
        'replay_helper_sha256':'scripts/blender_reference_motion.py',
        'inventory_helper_sha256':'scripts/export_planned_reference_web.py'}.items()}
    for name,expected in job['source_sha256'].items():
        if sha(Path(job['door_dir'])/name)!=expected:raise ValueError(f'Storyboard source changed: {name}')
    for name,expected in job['mesh_sha256'].items():
        if sha(Path(job['hardware_dir'])/(name+'.obj'))!=expected:raise ValueError(f'Storyboard hardware changed: {name}')
    for name,expected in job['renderer_sha256'].items():
        if sha(ROOT/'doorbench/appearance'/name)!=expected:raise ValueError(f'Storyboard appearance renderer changed: {name}')
    return {'door_id':job['door_id'],'input_sha256':{name:sha(path) for name,path in paths.items()},
            'appearance_job_sha256':sha(appearance_job),'job_sha256':job['job_sha256'],
            'source_sha256':job['source_sha256'],'mesh_sha256':job['mesh_sha256'],
            'appearance_renderer_sha256':job['renderer_sha256'],**dependencies}


def load_storyboard_inputs(config):
    from scripts.blender_planned_motion import load_inputs, load_validation_report
    from scripts.export_planned_reference_web import verified_artifacts
    config=Path(config);job=json.loads(config.read_text())
    paths={name:Path(job[name]) for name in ('clip','trajectory','validation','result')}
    before={name:sha(path) for name,path in paths.items()}
    result=json.loads(paths['result'].read_text())
    if result.get('status')!='accepted_kinematic' or 'failure.json' in result.get('artifacts',{}):raise ValueError('Attempt is not accepted for storyboard review')
    files=verified_artifacts(paths['result'].parent,result)
    if 'failure.json' in files:raise ValueError('Accepted attempt contains a failure file')
    for key,name in [('clip','clip.json'),('trajectory','trajectory.npz'),('validation','validation.json')]:
        if paths[key].resolve()!=paths['result'].parent.resolve()/name or hashlib.sha256(files[name]).hexdigest()!=before[key]:raise ValueError('Storyboard artifact/result binding mismatch')
    inputs=load_inputs(job['appearance_job'],paths['clip'],paths['trajectory'])
    verification=load_validation_report(paths['validation'],paths['clip'],paths['trajectory'],inputs[0])
    if result.get('door_id')!=inputs[1]['door_id'] or result.get('provenance',{}).get('source_sha256')!=inputs[0]['source_sha256']:raise ValueError('Storyboard result source/door mismatch')
    source=result.get('source_outcome',{})
    if not all(result.get('new_completion',{}).get(k) is True for k in ['complete_proposal','artifact_bindings_verified','task_evidence_pass','source_success_declared']) or source.get('success') is not True or source.get('outcome')!='success' or source.get('error') or inputs[1].get('proposal',{}).get('source_outcome')!=source:raise ValueError('Storyboard result lacks complete source/task-evidence bindings')
    expected=provenance(inputs[0],paths,job['appearance_job'])
    if before!=expected['input_sha256']:raise ValueError('Storyboard motion inputs changed while loading')
    samples=phase_samples(inputs[1]['phases'],inputs[2]['actor_time'])
    return inputs,verification,expected,samples,paths,job['appearance_job']


def checked_image(directory,name):
    if not isinstance(name,str) or re.fullmatch(r'frame-[0-9]{2,}\.png',name) is None:raise ValueError('Unsafe storyboard image path')
    path=Path(directory)/name
    if path.is_symlink() or path.resolve().parent!=Path(directory).resolve():raise ValueError('Storyboard image is a symlink or escapes output')
    return path


def checked_report(directory,expected,samples,*,require_sheet=False):
    from PIL import Image
    directory=Path(directory);report=json.loads((directory/'storyboard.json').read_text())
    if report.get('schema')!='doorbench.motion-storyboard.v1' or any(report.get(k)!=v for k,v in expected.items()):raise ValueError('Stale storyboard input or renderer provenance')
    actual=report.get('samples',[])
    if len(actual)!=len(samples):raise ValueError('Storyboard phase coverage changed')
    for ordinal,(sample,wanted) in enumerate(zip(actual,samples)):
        if any(sample.get(k)!=v for k,v in wanted.items()) or sample.get('image')!=f'frame-{ordinal:02d}.png':raise ValueError('Storyboard phase sample mismatch')
        check=sample.get('pose_check',{})
        if any(not isinstance(check.get(k),(int,float)) or not math.isfinite(check[k]) or not 0<=check[k]<=2e-5 for k in ['max_position_error_m','max_rotation_error_rad']):raise ValueError('Storyboard lacks passing source pose checks')
        path=checked_image(directory,sample['image'])
        if sha(path)!=sample.get('sha256'):raise ValueError('Storyboard image changed')
        with Image.open(path) as im:
            if im.size!=(480,360) or im.format!='PNG':raise ValueError('Storyboard image format or dimensions changed')
            im.verify()
    if require_sheet:
        picture=directory/'contact-sheet.jpg'
        if picture.is_symlink() or not picture.is_file() or report.get('contact_sheet_sha256')!=sha(picture):raise ValueError('Storyboard contact sheet changed')
    return report


def sampled_arrays(arrays,samples):
    """Exact slices after full verification; generic full-motion replay is unchanged."""
    import numpy as np
    indices=np.asarray([s['index'] for s in samples],dtype=int);n=len(arrays['actor_time'])
    if len(indices)<2 or indices[0]!=0 or indices[-1]!=n-1 or np.any(np.diff(indices)<=0):raise ValueError('Storyboard samples must include ordered original endpoints')
    return {key:value[indices].copy() if value.ndim and value.shape[0]==n else value for key,value in arrays.items()}


def verify_rendered_pose(scene,inputs,index):
    """Check evaluated Blender world body poses against the original full NPZ."""
    import numpy as np
    job,clip,arrays,model,spec,mapping=inputs
    native={o.get('doorbench_body'):o for o in scene.objects if o.type=='EMPTY' and o.get('doorbench_body')}
    actor={o.get('doorbench_actor_body'):o for o in scene.objects if o.type=='EMPTY' and o.get('doorbench_actor_body')}
    if set(native)!=set(mapping) or set(actor)!=set(clip['actor']['body_names']):raise ValueError('Rendered source/actor body coverage mismatch')
    position_error=0.;angle_error=0.
    for objects,names,prefix in [(native,mapping,'body_'),(actor,{name:i for i,name in enumerate(clip['actor']['body_names'])},'actor_body_')]:
        for name,body_index in names.items():
            matrix=objects[name].matrix_world;pos=np.asarray(tuple(matrix.translation));quat=np.asarray(tuple(matrix.to_quaternion()))
            target_pos=arrays[prefix+'pos'][index,body_index];target_quat=arrays[prefix+'quat'][index,body_index]
            position_error=max(position_error,float(np.linalg.norm(pos-target_pos)))
            # Robust sign-independent rotation distance, including Blender float precision.
            quat=quat/np.linalg.norm(quat);target_quat=target_quat/np.linalg.norm(target_quat)
            delta=min(np.linalg.norm(quat-target_quat),np.linalg.norm(quat+target_quat))
            angle_error=max(angle_error,float(4*math.asin(min(1.,delta/2))))
    if position_error>2e-5 or angle_error>2e-5:raise ValueError(f'Storyboard pose mismatch at original frame {index}: {position_error:g} m, {angle_error:g} rad')
    return {'max_position_error_m':position_error,'max_rotation_error_rad':angle_error,'position_tolerance_m':2e-5,'rotation_tolerance_rad':2e-5}


def worker(config):
    import bpy
    from scripts.blender_planned_motion import build
    config = Path(config);out=config.parent
    inputs,verification,expected,samples,paths,appearance_job=load_storyboard_inputs(config)
    selected=list(inputs);selected[2]=sampled_arrays(inputs[2],samples)
    metadata = build(*selected, out/'unsaved.blend', verification=verification, save_scene=False)
    clip, arrays = inputs[1:3]
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 480; scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'; scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.; scene.view_settings.gamma = 1.
    scene.display.shading.light = 'STUDIO'; scene.display.shading.color_type = 'OBJECT'
    scene.display.shading.show_shadows = True; scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'; scene.display.shading.show_specular_highlight = True
    scene.display.shading.background_type = 'WORLD'; scene.world.color = (.72,.74,.75)
    scene.camera.data.dof.use_dof = False
    hidden = []
    for obj in scene.objects:
        semantic = obj.get('doorbench_semantic', obj.get('semantic', ''))
        if semantic in ('appearance_context', 'wall', 'ceiling', 'motion_status_annotation'):
            obj.hide_render = True; hidden.append(obj.name)
        if semantic == 'planned_rig_collision_surface': obj.color = (.035,.38,.42,1)
        elif semantic in ('operator','handle','hinge','lock'): obj.color = (.7,.46,.1,1)
        elif semantic == 'floor': obj.color = (.73,.73,.70,1)
        else: obj.color = (.49,.34,.21,1)
    for ordinal, sample in enumerate(samples):
        frame = 1+sample['time_s']*scene.render.fps
        scene.frame_set(math.floor(frame), subframe=frame-math.floor(frame))
        bpy.context.view_layer.update()
        sample['pose_check']=verify_rendered_pose(scene,inputs,sample['index'])
        path = out/f'frame-{ordinal:02d}.png'
        scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
        sample.update(image=path.name, sha256=sha(path))
    if expected!=provenance(inputs[0],paths,appearance_job):
        raise ValueError('Storyboard inputs or renderer changed during rendering')
    report = {'schema':'doorbench.motion-storyboard.v1', 'door_id':clip['door_id'],
              'scope':'Sampled visual inspection only; contextual walls hidden for visibility. Not a full-motion style approval.',
              **expected,
              'blender_version':bpy.app.version_string, 'engine':'BLENDER_WORKBENCH',
              'duration_s':metadata['duration'], 'samples':samples, 'hidden_render_only':hidden}
    report['animation_scope']='Only selected original frames are keyed and rendered. Full input/report verification precedes slicing; no between-sample interpolation is approved.'
    report['original_frames']=len(arrays['actor_time']);report['keyed_frames']=len(samples)
    (out/'storyboard.json').write_text(json.dumps(report, indent=2)+'\n')


def assemble(directory,expected=None,samples=None):
    from PIL import Image, ImageDraw, ImageFont
    directory = Path(directory)
    if expected is None or samples is None:
        _,_,expected,samples,_,_=load_storyboard_inputs(directory/'worker.json')
    report=checked_report(directory,expected,samples)
    samples = report['samples']; columns = 4; width, height = 480, 390
    canvas = Image.new('RGB', (columns*width, 100+math.ceil(len(samples)/columns)*height), '#edf0eb')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
        title = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 27)
    except OSError: font = title = ImageFont.load_default()
    draw.text((18,15), f"{report['door_id']} | {report['duration_s']:.1f} s | sampled kinematic acceptance", fill='#18332b', font=title)
    draw.text((18,55), 'Phase samples only. Context walls hidden. No full-motion style, forces or balance certification.', fill='#435249', font=font)
    for i, sample in enumerate(samples):
        path = checked_image(directory,sample['image'])
        if sha(path) != sample['sha256']: raise ValueError('Storyboard image changed')
        x, y = (i%columns)*width, 100+(i//columns)*height
        with Image.open(path) as im: canvas.paste(im.convert('RGB'), (x,y))
        draw.text((x+12,y+364), f"{sample['time_s']:.1f}s | {sample['phase'].replace('_',' ')}", fill='#18332b', font=font)
    target = directory/'contact-sheet.jpg'; canvas.save(target, quality=90)
    report['contact_sheet_sha256'] = sha(target)
    (directory/'storyboard.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='out/reference-planned-corpus-v1')
    parser.add_argument('--assets', default='assets')
    parser.add_argument('--out', default='out/planned-motion-storyboards')
    parser.add_argument('--doors', default='all')
    parser.add_argument('--worker-config')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    if args.worker_config: return worker(args.worker_config)
    from doorbench.appearance.pipeline import prepare_job, find_blender
    corpus, out, assets = (Path(v).resolve() for v in (args.corpus,args.out,args.assets))
    if any(out == p or out in p.parents or p in out.parents for p in (corpus,assets)):
        raise ValueError('Storyboard output must not overlap source data')
    index = json.loads((corpus/'index.json').read_text())
    if index.get('schema')!='doorbench.planned-reference-corpus.v1' or index.get('manifest_sha256')!=sha(assets/'manifest.json'):raise ValueError('Storyboard corpus schema or source manifest mismatch')
    selected = None if args.doors == 'all' else set(args.doors.split(','))
    failures=[]; complete=[]
    for row in index['doors']:
        door = row['door_id']
        if row.get('status') != 'accepted_kinematic' or (selected is not None and door not in selected): continue
        if Path(door).name != door or door in ('.','..'): raise ValueError('Unsafe door ID')
        source = corpus/door; directory = out/door
        if source.is_symlink() or directory.is_symlink():raise ValueError('Storyboard source/output directories must not be symlinks')
        directory.mkdir(parents=True,exist_ok=True)
        if any(p.is_symlink() for p in directory.iterdir()):raise ValueError('Storyboard output contains a symlink')
        if row.get('result')!=f'{door}/result.json':raise ValueError('Unsafe or mismatched corpus result path')
        paths={name:source/filename for name,filename in [('clip','clip.json'),('trajectory','trajectory.npz'),('validation','validation.json'),('result','result.json')]}
        attempt=json.loads(paths['result'].read_text())
        if attempt.get('identity_sha256')!=row.get('identity_sha256') or attempt.get('provenance',{}).get('generator_sha256')!=index['generator']['sha256']:raise ValueError('Storyboard corpus result identity/generator mismatch')
        report_path=directory/'storyboard.json'
        appearance=prepare_job(assets,door,directory,quality='preview',width=480,height=360,view='iso')
        job_path=directory/'appearance-job.json';job_path.write_text(json.dumps(appearance)+'\n')
        config={k:str(v) for k,v in paths.items()};config['appearance_job']=str(job_path)
        config_path=directory/'worker.json';config_path.write_text(json.dumps(config)+'\n')
        command=[find_blender(),'--background','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).resolve()),'--','--worker-config',str(config_path)]
        try:
            _,_,expected,samples,_,_=load_storyboard_inputs(config_path)
            if report_path.is_file():
                try:
                    checked_report(directory,expected,samples,require_sheet=True)
                    complete.append(door);continue
                except (ValueError,OSError,KeyError,TypeError):pass
            with (directory/'blender.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode: raise RuntimeError(f'Blender exited {result.returncode}')
            assemble(directory);complete.append(door);print(door,'rendered',flush=True)
        except Exception as exc:
            failures.append({'door_id':door,'error':str(exc)});print(door,str(exc),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'index.json').write_text(json.dumps({'completed':complete,'failures':failures,'scope':'Phase samples, not full-motion visual approval'},indent=2)+'\n')
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
