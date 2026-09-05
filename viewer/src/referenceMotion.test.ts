import { describe,it,expect } from 'bun:test';
import { frameAt,validateClip,type ReferenceClip } from './referenceMotion';
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
