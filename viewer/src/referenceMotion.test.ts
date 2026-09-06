import {expect,test,describe,it} from 'bun:test';
import {frameAt,validateClip,buildReferencePlayer,type ReferenceClip} from './referenceMotion';
import type {BuiltScene} from './scene';
import * as THREE from 'three';

function nativeClip(){return {
  schema:'doorbench.native-motion.v1',door_id:'db0148_garage_sectional',scenario:'open_and_traverse',
  duration:1,times:[0,1],joint_names:['lift','arm'],door_q:[[0,0],[1,.5]],
  targets:[[0,0,1],[0,1,1]],phases:['wait','open'],
  oracle_contacts:[[],[{site:'lift_handle_grip',joint:null,position:[0,1,1],force_N:[0,0,100]}]],
  outcome:{success:true,outcome:'success',sim_time:1},source_sha256:{},
} as unknown as ReferenceClip;}

test('native playback retains recorded constraint poses without creating a human',()=>{
  const clip=validateClip(nativeClip(),'db0148_garage_sectional');
  expect(clip.avatar_bones).toEqual([]);
  const player=buildReferencePlayer(clip);let q:number[]=[];
  const built={setRecordedJoints:(_names:string[],values:number[])=>{q=values;}} as BuiltScene;
  player.setTime(.8,built);expect(q).toEqual([0,0]);
  player.setTime(1,built);expect(q).toEqual([1,.5]);
  expect(player.group.name).toBe('native_mechanism_contacts');
  expect(player.group.getObjectByName('pelvis')).toBeUndefined();
  player.dispose();
});

test('native contact timeline rejects corrupted positions and missing frames',()=>{
  const bad=nativeClip();bad.oracle_contacts![1][0].position=[NaN,0,0];
  expect(()=>validateClip(bad,bad.door_id)).toThrow();
  const missing=nativeClip();missing.oracle_contacts!.pop();
  expect(()=>validateClip(missing,missing.door_id)).toThrow();
});

test('native free-root playback uses observed world poses without interpolation',()=>{
  const source=nativeClip();
  source.native={body_names:['free_link','child'],joint_types:['mjJNT_FREE','mjJNT_HINGE'],
    poses:[[1,2,3,1,0,0,0, 1,2,4,1,0,0,0],[5,6,7,1,0,0,0, 5,6,9,1,0,0,0]]};
  const clip=validateClip(source,source.door_id);const player=buildReferencePlayer(clip);
  const root=new THREE.Group(),free=new THREE.Group(),child=new THREE.Group();
  root.add(free);free.add(child);free.userData.container=free;child.userData.container=child;
  const built={setRecordedJoints:()=>{},bodies:new Map([['child',child],['free_link',free]])} as unknown as BuiltScene;
  player.setTime(.8,built);
  expect(free.getWorldPosition(new THREE.Vector3()).toArray()).toEqual([1,2,3]);
  expect(child.getWorldPosition(new THREE.Vector3()).toArray()).toEqual([1,2,4]);
  player.setTime(1,built);
  expect(child.getWorldPosition(new THREE.Vector3()).toArray()).toEqual([5,6,9]);
  const bad=nativeClip();bad.native={body_names:['free_link'],joint_types:['mjJNT_FREE']};
  expect(()=>validateClip(bad,bad.door_id)).toThrow('Free roots');
  player.dispose();
});

describe('reference playback',()=>{
 it('handles irregular terminal samples and clamps outside the timeline',()=>{
  expect(frameAt([0,.05,.073],.06)).toEqual([1,2,(.06-.05)/(.073-.05)]);
  expect(frameAt([0,.05,.073],-1)).toEqual([0,1,0]);
  expect(frameAt([0,.05,.073],9)).toEqual([2,2,0]);
 });
 it('rejects wrong doors and malformed recordings before constructing a scene',()=>{
  expect(()=>validateClip({schema:'other'} as ReferenceClip,'door')).toThrow();
  const c={schema:'doorbench.reference-motion.v1',door_id:'door',duration:1,times:[0,1],joint_names:['hinge'],avatar_joint_names:['pelvis','chest','neck','head','shoulder_l','elbow_l','wrist_l','shoulder_r','elbow_r','wrist_r','hip_l','knee_l','ankle_l','hip_r','knee_r','ankle_r'],avatar_bones:[],door_q:[[0],[1]],avatar:[Array(48).fill(0),Array(48).fill(0)],targets:[[0,0,0],[0,0,0]],phases:['wait','wait'],hand_error_m:[0,0],hand_active:[0,0]} as unknown as ReferenceClip;
  expect(validateClip(c,'door')).toBe(c);
  expect(()=>validateClip({...c,avatar_joint_names:['head']},'door')).toThrow();
  expect(()=>validateClip({...c,hand_error_m:[0,NaN]},'door')).toThrow();
  expect(()=>validateClip({...c,times:[0,0]},'door')).toThrow();
  expect(()=>validateClip({...c,door_q:[[0],[NaN]]},'door')).toThrow();
  expect(()=>validateClip({...c,avatar_bones:[[0,16]]},'door')).toThrow();
 });
});
