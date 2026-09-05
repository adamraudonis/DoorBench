import {describe,it,expect} from 'bun:test';
import * as THREE from 'three';
import {artifactURL,buildPlannedPlayer,fetchPlannedClip,motionTaskDetail,motionTaskLabel,sha256,validateMotionIndex,validatePlannedClip,type PlannedClip,type MotionEntry} from './plannedReferenceMotion';
import {buildScene} from './scene';
import type {ModelJ} from './types';

function fixture(){
  const names=Array.from({length:16},(_,i)=>`actor_${i}`),pose=(x:number,y:number,z:number)=>[x,y,z,1,0,0,0];
  const native=[pose(0,0,0).concat(pose(1,2,0),pose(1,3,1)),pose(0,0,0).concat([2,1,0,Math.SQRT1_2,0,0,Math.SQRT1_2],pose(1,1,1))];
  const clip={schema:'doorbench.planned-reference-web.v1',door_id:'fixture',status:'accepted_kinematic',source_scenario:'open_and_traverse',scope:'sampled',identity_sha256:'a'.repeat(64),duration:1,times:[0,1],native_time:[0,.2],phases:['approach','operate'],foot_contact:[[1,1],[1,0]],hand_contact:[[0,0],[0,1]],native_resources_sha256:{},
    source_sha256:{'model.json':'b'.repeat(64),'spec.json':'c'.repeat(64),'door.xml':'d'.repeat(64)},
    native:{body_names:['world','leaf','handle'],poses:native},actor:{body_names:names,poses:[names.flatMap((_,i)=>pose(i*.1,-1,1)),names.flatMap((_,i)=>pose(i*.1,-.5,1))],geometries:names.map(name=>({name:`${name}_geom`,body_name:name,type:'sphere',size:[.03,0,0],pos:[0,0,0],quat_wxyz:[1,0,0,0]}))}} as PlannedClip;
  const entry={door_id:'fixture',family:'swing_single',status:'accepted_kinematic',source_scenario:clip.source_scenario,identity_sha256:clip.identity_sha256,clip:{path:'clips/fixture.json.gz',sha256:'e'.repeat(64),json_sha256:'f'.repeat(64),bytes:10,duration:1,frames:2},audits:{}} as MotionEntry;
  return {clip,entry};
}
const model={name:'fixture',tier:'full',materials:{wood:{rgba:[.5,.3,.1,1]}},equalities:[],tendons:[],meta:{},bodies:[
  {name:'world_env',pos:[0,0,0],quat:[1,0,0,0],geoms:[]},
  {name:'leaf',parent:'world_env',pos:[.4,.5,0],quat:[1,0,0,0],joint:{name:'hinge',type:'hinge',axis:[0,0,1],pos:[.2,.1,0],range:[0,2],modeled_at:.3},geoms:[{name:'leaf_geom',type:'box',size:[.3,.02,.5],pos:[.5,0,.5],quat:[1,0,0,0],visual:true,material:'wood'}]},
  {name:'handle',parent:'leaf',pos:[.5,-.1,1],quat:[1,0,0,0],joint:{name:'operator',type:'hinge',axis:[1,0,0],pos:[.01,.02,.03],range:[0,2]},geoms:[]}
]} as unknown as ModelJ;

async function downloadFixture(){
  const {clip,entry}=fixture(),encode=(s:string)=>new TextEncoder().encode(s).buffer;
  const sources=new Map([['model.json',encode(JSON.stringify(model))],['spec.json',encode('{}')],['door.xml',encode('<mujoco/>')]]);
  for(const [name,bytes]of sources)clip.source_sha256[name]=await sha256(bytes);
  const raw=encode(JSON.stringify(clip)),packed=new Uint8Array(Bun.gzipSync(raw)).buffer;
  entry.clip!.sha256=await sha256(packed);entry.clip!.json_sha256=await sha256(raw);entry.clip!.bytes=packed.byteLength;
  return {clip,entry,sources,packed};
}

function trackAbortListener(signal:AbortSignal){
  const add=signal.addEventListener.bind(signal),remove=signal.removeEventListener.bind(signal);const listeners=new Set<EventListenerOrEventListenerObject>();
  signal.addEventListener=((type:string,listener:EventListenerOrEventListenerObject,options:any)=>{if(type==='abort')listeners.add(listener);add(type,listener,options);}) as typeof signal.addEventListener;
  signal.removeEventListener=((type:string,listener:EventListenerOrEventListenerObject,options:any)=>{if(type==='abort')listeners.delete(listener);remove(type,listener,options);}) as typeof signal.removeEventListener;
  return listeners;
}

describe('planned motion source and articulation',()=>{
  it('applies native world poses to nested body containers, including pivots and world alias',async()=>{
    const {clip}=fixture(),built=await buildScene(model,{showEnv:true});built.root.position.set(.7,-.4,.2);built.root.rotation.z=.4;
    const player=buildPlannedPlayer(clip,built);player.setTime(1);
    for(const [name,index] of [['world_env',0],['leaf',1],['handle',2]] as const){
      const container=built.bodies.get(name)!.userData.container as THREE.Object3D;
      const actual=container.getWorldPosition(new THREE.Vector3());expect(actual.distanceTo(new THREE.Vector3().fromArray(clip.native.poses[1],index*7))).toBeLessThan(1e-12);
      const expected=clip.native.poses[1].slice(index*7+3,index*7+7);expect(Math.abs(container.getWorldQuaternion(new THREE.Quaternion()).dot(new THREE.Quaternion(expected[1],expected[2],expected[3],expected[0])))).toBeCloseTo(1,12);
    }
    const mesh=built.root.getObjectByName('leaf_geom')!;expect(mesh.getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(2,1.5,.5))).toBeLessThan(1e-12);
    expect(player.group.children.length).toBe(16);expect(player.group.children.every(b=>b.children.length===1)).toBeTrue();
    player.setTime(.5);expect(player.group.children[0].position.y).toBeCloseTo(-.75,12);player.dispose();built.dispose();
  });
  it('rejects wrong identity, malformed poses, bad primitives and nonaccepted playback',()=>{
    const {clip,entry}=fixture();expect(validatePlannedClip(clip,entry)).toBe(clip);
    expect(()=>validatePlannedClip({...clip,door_id:'other'},entry)).toThrow('identity');
    expect(()=>validatePlannedClip({...clip,status:'rejected'},entry)).toThrow('accepted');
    const bad=structuredClone(clip);bad.actor.poses[0][3]=0;expect(()=>validatePlannedClip(bad,entry)).toThrow('quaternion');
    const badSize=structuredClone(clip);badSize.actor.geometries[0].size[0]=-.1;expect(()=>validatePlannedClip(badSize,entry)).toThrow('size');
    expect(()=>validateMotionIndex({schema:'doorbench.planned-reference-web-index.v1',manifest_sha256:'a'.repeat(64),doors:[{...entry,status:'rejected'}]} as any)).toThrow('accepted');
    expect(()=>artifactURL('../secret','https://example.test/index.json')).toThrow('Unsafe');
    expect(()=>artifactURL('https://other.test/clip','https://example.test/index.json')).toThrow('Unsafe');
    const index={schema:'doorbench.planned-reference-web-index.v1',manifest_sha256:'a'.repeat(64),doors:[entry]} as any;
    expect(validateMotionIndex(index)).toBe(index);
    expect(()=>validateMotionIndex({...index,doors:[{...entry,identity_sha256:undefined}]})).toThrow('identity');
    expect(()=>validateMotionIndex({...index,doors:[{...entry,clip:{...entry.clip,bytes:64*1024*1024+1}}]})).toThrow('size');
  });
  it('restores original door materials after brown/gold mode',async()=>{
    const built=await buildScene(model);const mesh=built.root.getObjectByName('leaf_geom') as THREE.Mesh;const original=mesh.material;
    built.setDiagnostic(true);expect(mesh.material).not.toBe(original);built.setDiagnostic(false);expect(mesh.material).toBe(original);built.dispose();
  });
  it('labels locked checks separately using the bound source scenario, never phase names',()=>{
    const {clip,entry}=fixture();clip.source_scenario='locked_recognize';entry.source_scenario='locked_recognize';clip.phases=['traverse','traverse'];
    expect(validatePlannedClip(clip,entry).source_scenario).toBe('locked_recognize');
    expect(motionTaskLabel(clip.source_scenario)).toBe('Locked-door check');expect(motionTaskDetail(clip.source_scenario)).toContain('does not traverse');
    expect(motionTaskLabel('unlock_and_traverse')).toBe('Traversal reference');expect(motionTaskDetail('unlock_and_traverse')).toContain('not independently certified');
    expect(()=>validatePlannedClip(clip,{...entry,source_scenario:'open_and_traverse'})).toThrow('scenario');
    expect(()=>validateMotionIndex({schema:'doorbench.planned-reference-web-index.v1',manifest_sha256:'a'.repeat(64),doors:[{...entry,source_scenario:undefined}]} as any)).toThrow('scenario');
  });
  it('verifies compressed bytes and served source hashes before returning playable data',async()=>{
    const {clip,entry}=fixture();const encode=(s:string)=>new TextEncoder().encode(s).buffer;
    const sources=new Map([['model.json',encode(JSON.stringify(model))],['spec.json',encode('{}')],['door.xml',encode('<mujoco/>')]]);
    for(const [name,bytes] of sources)clip.source_sha256[name]=await sha256(bytes);
    const raw=encode(JSON.stringify(clip)),packed=new Uint8Array(Bun.gzipSync(raw)).buffer;entry.clip!.sha256=await sha256(packed);entry.clip!.json_sha256=await sha256(raw);entry.clip!.bytes=packed.byteLength;
    const original=globalThis.fetch;let corrupt=false,stale=false;const requests:string[]=[];
    globalThis.fetch=(async(url:any)=>{
      const path=String(url);requests.push(path);if(path.includes('/clips/'))return new Response(corrupt?encode('broken'):packed);
      const name=path.split('/').pop()!;return new Response(stale&&name==='model.json'?encode('{}'):sources.get(name));
    }) as typeof fetch;
    try{
      expect((await fetchPlannedClip(entry,'https://example.test/index.json')).clip.door_id).toBe('fixture');
      corrupt=true;requests.length=0;await expect(fetchPlannedClip(entry,'https://example.test/index.json')).rejects.toThrow('checksum');expect(requests.length).toBe(1);
      corrupt=false;stale=true;await expect(fetchPlannedClip(entry,'https://example.test/index.json')).rejects.toThrow('does not match');
      await expect(fetchPlannedClip({...entry,status:'rejected'},'https://example.test/index.json')).rejects.toThrow('not accepted');
    }finally{globalThis.fetch=original;}
  });
  it('recovers transient clip/source HTTP errors while retaining checksum and schema verification',async()=>{
    const {clip,entry}=fixture(),encode=(s:string)=>new TextEncoder().encode(s).buffer;
    const sources=new Map([['model.json',encode(JSON.stringify(model))],['spec.json',encode('{}')],['door.xml',encode('<mujoco/>')]]);
    for(const [name,bytes]of sources)clip.source_sha256[name]=await sha256(bytes);
    const raw=encode(JSON.stringify(clip)),packed=new Uint8Array(Bun.gzipSync(raw)).buffer;
    entry.clip!.sha256=await sha256(packed);entry.clip!.json_sha256=await sha256(raw);entry.clip!.bytes=packed.byteLength;
    const original=globalThis.fetch,timer=globalThis.setTimeout;const calls=new Map<string,number>();
    globalThis.setTimeout=((callback:any)=>timer(callback,0)) as typeof setTimeout;
    globalThis.fetch=(async(url:any)=>{const path=String(url),count=(calls.get(path)??0)+1;calls.set(path,count);
      if(path.includes('/clips/'))return count===1?new Response('throttled',{status:429,headers:{'Retry-After':'0'}}):new Response(packed);
      const name=path.split('/').pop()!;return name==='model.json'&&count===1?new Response('busy',{status:503}):new Response(sources.get(name));
    }) as typeof fetch;
    try{
      expect((await fetchPlannedClip(entry,'https://example.test/index.json')).clip.door_id).toBe('fixture');
      expect(calls.get('https://example.test/clips/fixture.json.gz')).toBe(2);expect(calls.get('./assets/doors/fixture/model.json')).toBe(2);
      const invalid=encode(JSON.stringify({...clip,schema:'unsupported'})),gz=new Uint8Array(Bun.gzipSync(invalid)).buffer;
      const badEntry={...entry,clip:{...entry.clip!,sha256:await sha256(gz),json_sha256:await sha256(invalid),bytes:gz.byteLength}};
      let schemaRequests=0;globalThis.fetch=(async()=>{schemaRequests++;return new Response(gz);}) as typeof fetch;
      await expect(fetchPlannedClip(badEntry,'https://example.test/index.json')).rejects.toThrow('identity');expect(schemaRequests).toBe(1);
    }finally{globalThis.fetch=original;globalThis.setTimeout=timer;}
  });
  it.each(['permanent','checksum'])('cancels sibling backoff after a terminal %s source failure',async(kind)=>{
    const {entry,sources,packed}=await downloadFixture();const outer=new AbortController(),listeners=trackAbortListener(outer.signal);
    const original=globalThis.fetch,timer=globalThis.setTimeout,clear=globalThis.clearTimeout;
    const pending=new Map<number,()=>void>();let next=0,cancelled=0;const requests:string[]=[],signals:AbortSignal[]=[];
    globalThis.setTimeout=((callback:()=>void)=>{pending.set(++next,callback);return next;}) as unknown as typeof setTimeout;
    globalThis.clearTimeout=((id:number)=>{if(pending.delete(id))cancelled++;}) as unknown as typeof clearTimeout;
    globalThis.fetch=(async(url:any,options:any)=>{const path=String(url);requests.push(path);signals.push(options.signal);
      if(path.includes('/clips/'))return new Response(packed);
      if(path.endsWith('model.json'))return kind==='permanent'?new Response('missing',{status:404}):new Response('changed bytes');
      if(path.endsWith('spec.json'))return new Response('busy',{status:429,headers:{'Retry-After':'2'}});
      return new Response(sources.get('door.xml'));
    }) as typeof fetch;
    try{
      const expected=kind==='permanent'?'Cannot verify doors/fixture/model.json (404)':'Motion does not match served doors/fixture/model.json';
      await expect(fetchPlannedClip(entry,'https://example.test/index.json',outer.signal)).rejects.toThrow(expected);
      // Advance any remaining retry callbacks after the returned rejection.
      // Without group cancellation this sends another spec request.
      expect(cancelled).toBe(1);expect(pending.size).toBe(0);
      for(const callback of pending.values())callback();
      await new Promise(resolve=>timer(resolve,0));
      expect(requests.filter(p=>p.endsWith('spec.json')).length).toBe(1);
      expect(new Set(signals).size).toBe(1);expect(signals[0]).not.toBe(outer.signal);expect(signals[0].aborted).toBeTrue();
      expect(outer.signal.aborted).toBeFalse();expect(listeners.size).toBe(0);
    }finally{globalThis.fetch=original;globalThis.setTimeout=timer;globalThis.clearTimeout=clear;}
  });
  it('forwards caller cancellation to the internal request and removes its listener',async()=>{
    const {entry}=fixture(),outer=new AbortController(),listeners=trackAbortListener(outer.signal);const original=globalThis.fetch;
    let requested:AbortSignal|undefined;
    globalThis.fetch=((_url,options)=>{requested=options!.signal!;return new Promise((_resolve,reject)=>requested!.addEventListener('abort',()=>reject(requested!.reason),{once:true}));}) as typeof fetch;
    try{
      const error=new Error('Changed selected door'),promise=fetchPlannedClip(entry,'https://example.test/index.json',outer.signal);
      expect(requested).not.toBe(outer.signal);expect(listeners.size).toBe(1);outer.abort(error);
      await expect(promise).rejects.toBe(error);expect(requested!.aborted).toBeTrue();expect(listeners.size).toBe(0);
    }finally{globalThis.fetch=original;}
  });
  it('unlinks the caller after a complete successful verified download',async()=>{
    const {entry,sources,packed}=await downloadFixture(),outer=new AbortController(),listeners=trackAbortListener(outer.signal),original=globalThis.fetch;
    const signals=new Set<AbortSignal>();
    globalThis.fetch=(async(url,options)=>{signals.add(options!.signal!);const path=String(url);return new Response(path.includes('/clips/')?packed:sources.get(path.split('/').pop()!));}) as typeof fetch;
    try{
      const loaded=await fetchPlannedClip(entry,'https://example.test/index.json',outer.signal);
      expect(loaded.clip.door_id).toBe('fixture');expect(loaded.files.size).toBe(3);expect(signals.size).toBe(1);expect(listeners.size).toBe(0);
      outer.abort();expect([...signals][0].aborted).toBeFalse();
    }finally{globalThis.fetch=original;}
  });
});
