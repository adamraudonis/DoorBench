import {describe,it,expect} from 'bun:test';
import * as THREE from 'three';
import {artifactURL,buildPlannedPlayer,fetchPlannedClip,sha256,validateMotionIndex,validatePlannedClip,type PlannedClip,type MotionEntry} from './plannedReferenceMotion';
import {buildScene} from './scene';
import type {ModelJ} from './types';

function fixture(){
  const names=Array.from({length:16},(_,i)=>`actor_${i}`),pose=(x:number,y:number,z:number)=>[x,y,z,1,0,0,0];
  const native=[pose(0,0,0).concat(pose(1,2,0),pose(1,3,1)),pose(0,0,0).concat([2,1,0,Math.SQRT1_2,0,0,Math.SQRT1_2],pose(1,1,1))];
  const clip={schema:'doorbench.planned-reference-web.v1',door_id:'fixture',status:'accepted_kinematic',scope:'sampled',identity_sha256:'a'.repeat(64),duration:1,times:[0,1],native_time:[0,.2],phases:['approach','operate'],foot_contact:[[1,1],[1,0]],hand_contact:[[0,0],[0,1]],native_resources_sha256:{},
    source_sha256:{'model.json':'b'.repeat(64),'spec.json':'c'.repeat(64),'door.xml':'d'.repeat(64)},
    native:{body_names:['world','leaf','handle'],poses:native},actor:{body_names:names,poses:[names.flatMap((_,i)=>pose(i*.1,-1,1)),names.flatMap((_,i)=>pose(i*.1,-.5,1))],geometries:names.map(name=>({name:`${name}_geom`,body_name:name,type:'sphere',size:[.03,0,0],pos:[0,0,0],quat_wxyz:[1,0,0,0]}))}} as PlannedClip;
  const entry={door_id:'fixture',family:'swing_single',status:'accepted_kinematic',identity_sha256:clip.identity_sha256,clip:{path:'clips/fixture.json.gz',sha256:'e'.repeat(64),json_sha256:'f'.repeat(64),bytes:10,duration:1,frames:2},audits:{}} as MotionEntry;
  return {clip,entry};
}
const model={name:'fixture',tier:'full',materials:{wood:{rgba:[.5,.3,.1,1]}},equalities:[],tendons:[],meta:{},bodies:[
  {name:'world_env',pos:[0,0,0],quat:[1,0,0,0],geoms:[]},
  {name:'leaf',parent:'world_env',pos:[.4,.5,0],quat:[1,0,0,0],joint:{name:'hinge',type:'hinge',axis:[0,0,1],pos:[.2,.1,0],range:[0,2],modeled_at:.3},geoms:[{name:'leaf_geom',type:'box',size:[.3,.02,.5],pos:[.5,0,.5],quat:[1,0,0,0],visual:true,material:'wood'}]},
  {name:'handle',parent:'leaf',pos:[.5,-.1,1],quat:[1,0,0,0],joint:{name:'operator',type:'hinge',axis:[1,0,0],pos:[.01,.02,.03],range:[0,2]},geoms:[]}
]} as unknown as ModelJ;

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
  it('verifies compressed bytes and served source hashes before returning playable data',async()=>{
    const {clip,entry}=fixture();const encode=(s:string)=>new TextEncoder().encode(s).buffer;
    const sources=new Map([['model.json',encode(JSON.stringify(model))],['spec.json',encode('{}')],['door.xml',encode('<mujoco/>')]]);
    for(const [name,bytes] of sources)clip.source_sha256[name]=await sha256(bytes);
    const raw=encode(JSON.stringify(clip)),packed=new Uint8Array(Bun.gzipSync(raw)).buffer;entry.clip!.sha256=await sha256(packed);entry.clip!.json_sha256=await sha256(raw);entry.clip!.bytes=packed.byteLength;
    const original=globalThis.fetch;let corrupt=false,stale=false;
    globalThis.fetch=(async(url:any)=>{
      const path=String(url);if(path.includes('/clips/'))return new Response(corrupt?encode('broken'):packed);
      const name=path.split('/').pop()!;return new Response(stale&&name==='model.json'?encode('{}'):sources.get(name));
    }) as typeof fetch;
    try{
      expect((await fetchPlannedClip(entry,'https://example.test/index.json')).clip.door_id).toBe('fixture');
      corrupt=true;await expect(fetchPlannedClip(entry,'https://example.test/index.json')).rejects.toThrow('checksum');
      corrupt=false;stale=true;await expect(fetchPlannedClip(entry,'https://example.test/index.json')).rejects.toThrow('does not match');
      await expect(fetchPlannedClip({...entry,status:'rejected'},'https://example.test/index.json')).rejects.toThrow('not accepted');
    }finally{globalThis.fetch=original;}
  });
});
