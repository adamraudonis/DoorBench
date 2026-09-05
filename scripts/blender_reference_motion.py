"""Build an optional, packed Blender replay of recorded doors and a kinematic figure.

Run with Blender --background --python this_file -- --job job.json --clip clip.json
--trajectory trajectory.npz --out scene.blend [--render-time 3 --image frame.png].
The figure is a visual reference, not a collision-validated controlled humanoid.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOINTS = ['pelvis','chest','neck','head','shoulder_l','elbow_l','wrist_l',
          'shoulder_r','elbow_r','wrist_r','hip_l','knee_l','ankle_l','hip_r','knee_r','ankle_r']
BONES = [[0,1],[1,2],[2,3],[1,4],[4,5],[5,6],[1,7],[7,8],[8,9],
         [0,10],[10,11],[11,12],[0,13],[13,14],[14,15]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(job_path, clip_path, trajectory_path):
    """Validate authoritative native body poses before touching the Blender scene."""
    import numpy as np
    from doorbench.appearance.pipeline import digest
    job, clip = [json.loads(Path(p).read_text()) for p in (job_path, clip_path)]
    if job.get('job_sha256') != digest({k:v for k,v in job.items() if k != 'job_sha256'}):
        raise ValueError('Prepared appearance job checksum mismatch; prepare the job again')
    if clip.get('schema') != 'doorbench.reference-motion.v1' or clip.get('door_id') != job['door_id']:
        raise ValueError('Clip schema or door_id does not match prepared job')
    if clip.get('up_axis') != 'Z' or clip.get('units') != 'metres/radians/seconds':
        raise ValueError('Clip must use metres/radians/seconds and Z up')
    if clip.get('avatar_joint_names') != JOINTS or clip.get('avatar_bones') != BONES:
        raise ValueError('Unsupported reference figure joint order or bone graph')
    if set(job['source_sha256']) != {'spec.json','model.json','door.xml'} or clip.get('source_sha256') != job['source_sha256']:
        raise ValueError('Clip source hashes do not match prepared job')
    source = Path(job['door_dir'])
    for name, expected in job['source_sha256'].items():
        if sha256(source/name) != expected:
            raise ValueError(f'Source changed after preparation: {name}')
    for name, expected in job['renderer_sha256'].items():
        if sha256(ROOT/'doorbench/appearance'/name) != expected:
            raise ValueError(f'Appearance source changed after preparation: {name}')
    for name, expected in job['mesh_sha256'].items():
        if sha256(Path(job['hardware_dir'])/(name+'.obj')) != expected:
            raise ValueError(f'Source mesh changed after preparation: {name}')
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {name:archive[name].copy() for name in archive.files}
    for name, array in arrays.items():
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError(f'Trajectory {name} must contain finite numeric values')
    names = clip.get('native', {}).get('body_names', [])
    if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Native body_names must be unique nonempty names')
    for key in ('time','actor_time'):
        value = arrays.get(key)
        if value is None or value.ndim != 1 or len(value) < 2 or abs(float(value[0])) > 1e-6 or np.any(np.diff(value) <= 0):
            raise ValueError(f'{key} must start at zero and strictly increase')
    n, a, b = len(arrays['time']), len(arrays['actor_time']), len(names)
    for key, shape in {'body_pos':(n,b,3), 'body_quat':(n,b,4), 'actor_joints':(a,16,3)}.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f'Trajectory {key} must have shape {shape}')
    if not np.allclose(np.linalg.norm(arrays['body_quat'], axis=-1), 1, atol=2e-5, rtol=0):
        raise ValueError('Native body_quat must contain unit WXYZ quaternions')
    if any(np.any(np.linalg.norm(arrays['actor_joints'][:,b]-arrays['actor_joints'][:,a], axis=-1) < 1e-6) for a,b in BONES):
        raise ValueError('Actor bones must have nonzero length in every sample')
    lead, fps = clip.get('lead_in_s'), clip.get('fps')
    if isinstance(lead,bool) or not isinstance(lead,(int,float)) or not math.isfinite(lead) or lead < 0:
        raise ValueError('lead_in_s must be finite and nonnegative')
    if isinstance(fps,bool) or not isinstance(fps,(int,float)) or not math.isfinite(fps) or not 1 <= fps <= 120 or fps != int(fps):
        raise ValueError('fps must be an integer from 1 to 120')
    if abs(float(arrays['actor_time'][-1])-lead-float(arrays['time'][-1])) > 1e-4:
        raise ValueError('Native and actor durations disagree with lead_in_s')
    # The NPZ does not carry string metadata: bind it to the clip through their
    # shared actor coordinates and scalar native qpos, not filenames alone.
    clip_times = np.asarray(clip.get('times',[]))
    if clip_times.shape != (a,) or not np.allclose(clip_times, arrays['actor_time'], atol=6e-5, rtol=0):
        raise ValueError('Clip times disagree with trajectory actor_time')
    actor = np.asarray(clip.get('avatar',[]))
    if actor.shape != (a,48) or not np.allclose(actor.reshape(a,16,3), arrays['actor_joints'], atol=6e-5, rtol=0):
        raise ValueError('Clip avatar disagrees with trajectory actor_joints')
    qpos = arrays.get('qpos')
    addresses = clip.get('native', {}).get('qpos_addresses', [])
    if qpos is None or qpos.ndim != 2 or qpos.shape[0] != n or any(type(i) is not int or not 0 <= i < qpos.shape[1] for i in addresses):
        raise ValueError('Trajectory qpos shape or native qpos_addresses are invalid')
    door_q = np.asarray(clip.get('door_q', []))
    expected_q = np.stack([np.interp(np.maximum(arrays['actor_time']-lead,0),arrays['time'],qpos[:,i]) for i in addresses],axis=1)
    if door_q.shape != expected_q.shape or not np.allclose(door_q, expected_q, atol=2e-4, rtol=0):
        raise ValueError('Clip door_q disagrees with native trajectory qpos')
    model, spec = [json.loads((source/name).read_text()) for name in ('model.json','spec.json')]
    aliases = job['reference_state'].get('body_aliases', {})
    mapping = {}
    for body in model['bodies']:
        name = body['name']
        native_name = name if name in names else aliases.get(name)
        if native_name not in names:
            raise ValueError(f'No native body pose for source body {name}')
        mapping[name] = names.index(native_name)
    if set(names)-set(mapping.keys())-set(aliases.values())-{'world'}:
        raise ValueError('Native trajectory contains bodies without source geometry')
    return job, clip, arrays, model, spec, mapping


def _linear(obj):
    action = obj.animation_data.action
    if hasattr(action, 'fcurves'):
        curves = action.fcurves
    else:
        curves = [curve for layer in action.layers for strip in layer.strips
                  for curve in strip.channelbag(obj.animation_data.action_slot).fcurves]
    for curve in curves:
        curve.extrapolation = 'CONSTANT'
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'


def build(job, clip, arrays, model, spec, mapping, out, render_time=None, image=None, input_hashes=None):
    import bpy
    import numpy as np
    from mathutils import Vector, Quaternion
    from doorbench.appearance.blender_worker import reset_scene, build_door, primitive, explicit_camera
    from doorbench.appearance.blender_materials import configure_texture_library
    from doorbench.appearance.blender_environment import configure_scene, build_environment, frame_camera, _context
    from doorbench.appearance.blender_details import build_details
    reset_scene()
    configure_texture_library(job.get('texture_library'))
    camera_state = job['state'].get('camera')
    width,height = camera_state['resolution'] if camera_state else (job['width'],job['height'])
    scene = configure_scene(job['recipe'],quality=job['quality'],width=width,height=height,seed=job['seed'])
    objects = build_door(model,spec,job['reference_state'],job['recipe'],Path(job['hardware_dir']))
    build_environment(model,spec,job['recipe'],objects)
    build_details(objects,spec,job['seed'])
    bodies = {o['doorbench_body']:o for o in scene.objects if o.type == 'EMPTY' and o.get('doorbench_body')}
    fps, lead = int(clip['fps']), clip['lead_in_s']
    native_times = np.r_[0, arrays['time']+lead] if lead else arrays['time']
    positions = np.concatenate([arrays['body_pos'][:1],arrays['body_pos']]) if lead else arrays['body_pos']
    quats = np.concatenate([arrays['body_quat'][:1],arrays['body_quat']]) if lead else arrays['body_quat']
    for name,obj in bodies.items():
        obj.rotation_mode = 'QUATERNION'
        previous = None
        for sec,pos,quat in zip(native_times,positions[:,mapping[name]],quats[:,mapping[name]]):
            q = Quaternion(tuple(float(v) for v in quat))
            if previous is not None and q.dot(previous) < 0: q.negate()
            obj.location, obj.rotation_quaternion = pos,q
            obj.keyframe_insert('location',frame=1+float(sec)*fps)
            obj.keyframe_insert('rotation_quaternion',frame=1+float(sec)*fps)
            previous = q.copy()
        _linear(obj)
    material = bpy.data.materials.new('Original DoorBench reference figure')
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.065,.27,.32,1)
    shader.inputs['Roughness'].default_value = .37
    shader.inputs['Metallic'].default_value = .1
    actor = []
    radii = [.10,.12,.045,.10,.05,.045,.035,.05,.045,.035,.05,.05,.045,.05,.05,.045]
    def tagged(obj):
        obj['doorbench_visual_only'] = True
        obj['doorbench_physics_export'] = False
        obj['doorbench_semantic'] = 'kinematic_reference_figure'
        obj.data.materials.append(material)
        actor.append(obj)
        return obj
    for i,name in enumerate(JOINTS):
        obj = tagged(primitive({'type':'sphere','size':[radii[i]],'name':'actor::'+name},{},Path(job['hardware_dir'])))
        obj['reference_joint_index'] = i
        for sec,pose in zip(arrays['actor_time'], arrays['actor_joints'][:,i]):
            obj.location = pose
            obj.keyframe_insert('location',frame=1+float(sec)*fps)
        _linear(obj)
    for i,(a,b) in enumerate(BONES):
        delta = arrays['actor_joints'][0,b]-arrays['actor_joints'][0,a]
        length = float(np.linalg.norm(delta))
        radius = min(.10 if i==0 else .05,length*.2)
        obj = tagged(primitive({'type':'capsule','size':[radius,length/2-radius],'name':f'actor::bone_{a}_{b}'},{},Path(job['hardware_dir'])))
        obj['reference_bone'] = [a,b]
        obj.rotation_mode = 'QUATERNION'
        previous = None
        for sec,pose in zip(arrays['actor_time'],arrays['actor_joints']):
            delta = Vector(pose[b]-pose[a])
            q = delta.to_track_quat('Z','Y')
            if previous is not None and q.dot(previous) < 0: q.negate()
            obj.location = (pose[a]+pose[b])/2
            obj.rotation_quaternion = q
            obj.scale = (1,1,delta.length/length)
            for field in ('location','rotation_quaternion','scale'):
                obj.keyframe_insert(field,frame=1+float(sec)*fps)
            previous=q.copy()
        _linear(obj)
    scene.render.fps=fps
    scene.frame_start=1
    scene.frame_end=math.ceil(1+float(arrays['actor_time'][-1])*fps)
    # Fix default framing across the complete source motion and actor path.
    # Authored local geometry stays parented to native body transforms.
    bounds = [Vector(arrays['actor_joints'].min(axis=(0,1))-.16),Vector(arrays['actor_joints'].max(axis=(0,1))+.16)]
    for obj in objects.values():
        if _context(obj): continue
        local = [obj.matrix_basis @ Vector(c) for c in obj.bound_box]
        idx=mapping[obj['doorbench_body']]
        for pos,quat in zip(arrays['body_pos'][:,idx],arrays['body_quat'][:,idx]):
            q=Quaternion(tuple(float(v) for v in quat)); p=Vector(pos)
            bounds.extend(p+q@v for v in local)
    low=[min(v[i] for v in bounds) for i in range(3)]
    high=[max(v[i] for v in bounds) for i in range(3)]
    mesh=bpy.data.meshes.new('Temporary trajectory bounds')
    mesh.from_pydata(list(itertools.product(*zip(low,high))),[],[])
    proxy=bpy.data.objects.new('Temporary trajectory bounds',mesh)
    bpy.context.collection.objects.link(proxy)
    bpy.context.view_layer.update()
    scene.camera=explicit_camera(camera_state) if camera_state else frame_camera({'trajectory':proxy},spec,view=job['view'],width=width,height=height)
    bpy.data.objects.remove(proxy,do_unlink=True)
    bpy.data.meshes.remove(mesh)
    sec=lead+float(arrays['time'][-1])*.5 if render_time is None else render_time
    if not math.isfinite(sec) or not 0 <= sec <= float(arrays['actor_time'][-1])+1e-5:
        raise ValueError('render-time must lie within the actor timeline')
    frame=1+sec*fps
    scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame))
    scene['doorbench_motion_scope']='Recorded door physics; kinematic visual figure, not a controlled or collision-validated humanoid.'
    metadata={'schema':'doorbench.blender-reference-motion.v1','door_id':job['door_id'],
              'job_sha256':job['job_sha256'],'source_sha256':job['source_sha256'],
              'renderer_sha256':job['renderer_sha256'],'mesh_sha256':job['mesh_sha256'],
              'texture_library':job.get('texture_library'), **(input_hashes or {}),
              'source_visual_geoms':len(objects),'body_mapping':mapping,'fps':fps,'lead_in_s':lead,
              'native_samples':len(arrays['time']),'actor_samples':len(arrays['actor_time']),
              'duration':float(arrays['actor_time'][-1]),'render_time':sec,'figure_objects':len(actor),
              'camera':'explicit calibrated snapshot' if camera_state else 'fixed complete trajectory framing',
              'limitations':clip.get('limitations',[]),
              'interpolation':'Linear position/quaternion components between samples; sample poses are authoritative.',
              'blender_version':bpy.app.version_string,'script_sha256':sha256(__file__)}
    text=bpy.data.texts.new('DoorBench motion provenance.json')
    text.write(json.dumps(metadata,indent=2))
    out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if image:
        image=Path(image).resolve();image.parent.mkdir(parents=True,exist_ok=True)
        scene.render.filepath=str(image)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    if image: bpy.ops.render.render(write_still=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('job','clip','trajectory','out'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--render-time',type=float)
    parser.add_argument('--image')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    input_hashes={'clip_sha256':sha256(args.clip),'trajectory_sha256':sha256(args.trajectory)}
    inputs=load_inputs(args.job,args.clip,args.trajectory)
    if input_hashes != {'clip_sha256':sha256(args.clip),'trajectory_sha256':sha256(args.trajectory)}:
        raise ValueError('Motion inputs changed during validation; retry after recording completes')
    metadata=build(*inputs,args.out,args.render_time,args.image,input_hashes)
    metadata.update(**input_hashes,blend_sha256=sha256(args.out))
    if args.image: metadata['image_sha256']=sha256(args.image)
    Path(args.out).with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__': main()
