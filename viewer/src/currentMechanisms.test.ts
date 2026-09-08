/** Current generated assemblies and historical snapshots are distinct contracts.
 * A physical catch or pinned rod must never inherit an old proxy animation.
 */
import {describe, expect, test} from 'bun:test';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import type {ModelJ} from './types';
import {openClosePhases, operatorJoints, operatorsAreIndividual, requiresRecordedPhysics, sliderReaction, type JointLike} from './doorLogic';
import {LoopSolver, hasLoops} from './kinematics';
import {openingProcedure} from './mechanismInspection';
import {CodeLock, keypadOf, keypadRows, type KeypadJ} from './keypad';

const root=process.env.DOORBENCH_TEST_ASSETS;
function load(id:string) {
  const folder=path.join(root!,'doors',id);
  const model:ModelJ=JSON.parse(readFileSync(path.join(folder,'model.json'),'utf8'));
  const spec=JSON.parse(readFileSync(path.join(folder,'spec.json'),'utf8'));
  const joints=new Map<string,JointLike>();
  for(const b of model.bodies) if(b.joint) joints.set(b.joint.name,{name:b.joint.name,q:b.joint.modeled_at??0,modeledAt:b.joint.modeled_at??0,range:b.joint.range});
  return {model,spec,joints};
}

describe.skipIf(!root)('current generated mechanisms',()=>{
  test('all 1000 doors have bounded previews and procedures; contact assemblies require recorded physics',()=>{
    const manifest=JSON.parse(readFileSync(path.join(root!,'manifest.json'),'utf8'));
    expect(manifest.doors).toHaveLength(1000);
    let guarded=0,animated=0;
    for(const row of manifest.doors){
      const {model,spec,joints}=load(row.id);
      const preview=openClosePhases(model,joints);
      for(const name of operatorJoints(model))expect(joints.has(name)).toBe(true);
      expect(openingProcedure(model,spec).steps.length).toBeGreaterThan(0);
      if(requiresRecordedPhysics(model)){
        guarded++;
        expect(preview.phases).toEqual([]);
        expect(preview.note).toContain('recorded physics');
        for(const name of [model.meta.primary_joint,model.meta.secondary_joint].filter(Boolean)){
          expect(sliderReaction(model,joints,name,.4).blocked).toBe(true);
        }
      }else{
        animated+=Number(preview.phases.length>0);
        for(const move of preview.phases.flatMap(p=>[p,...p.followers??[]])){
          const j=joints.get(move.joint)!;
          expect(j).toBeDefined();expect(Number.isFinite(move.to)).toBe(true);
          if(j.range){expect(move.to).toBeGreaterThanOrEqual(j.range[0]);expect(move.to).toBeLessThanOrEqual(j.range[1]);}
        }
      }
    }
    expect(guarded).toBeGreaterThan(100);expect(animated).toBeGreaterThan(100);
  });

  test('rebuilt Dutch, paired, marine, hatch and vault fixtures use their actual contact assemblies',()=>{
    for(const [id,key] of [
      ['db0118_dutch','dutch_joining_bolt'],['db0149_swing_double','paired_leaf_holds'],
      ['db0168_ship_watertight','marine_dog_mounts'],['db0744_ship_watertight','marine_dog_mounts'],
      ['db0380_hatch_floor','hatch_support'],['db0179_vault','vault_boltwork']]){
      const {model}=load(id);expect(model.meta[key]).toBeTruthy();expect(requiresRecordedPhysics(model)).toBe(true);
    }
    const individual=load('db0168_ship_watertight').model;
    expect(operatorsAreIndividual(individual)).toBe(true);
    expect(operatorJoints(individual)).toEqual(individual.meta.marine_dog_mounts.map((r:any)=>r.joint));
    const linked=load('db0744_ship_watertight').model;
    expect(operatorJoints(linked)).toEqual(['wheel_hinge']);
    expect(linked.meta.marine_dog_linkage!.connect_equalities.length).toBeGreaterThanOrEqual(4);
  });

  test('valid keypad codes request real catches without changing joint limits',()=>{
    for(const id of ['db0526_swing_single','db0086_swing_single']){
      const {model,joints}=load(id);const kp=keypadOf(model) as KeypadJ;
      expect(kp.release).toBe('physical_catch');
      expect(joints.has(model.meta.keypad.physical_catch_joint)).toBe(true);
      expect(keypadRows(kp)).toHaveLength(5);expect(kp.buttons).toHaveLength(10);
      const before=JSON.stringify(model.bodies.map(b=>b.joint?.range));
      const lock=new CodeLock(kp);[...kp.code!].forEach((c,i)=>lock.press(c,i*.2));
      expect(lock.unlocked).toBe(true);
      expect(JSON.stringify(model.bodies.map(b=>b.joint?.range))).toBe(before);
      if(kp.bolt_joint){expect(joints.has(kp.bolt_joint)).toBe(true);expect(kp.motor_force_N!).toBeGreaterThan(0);}
    }
  });

  test('all current linkage loops stay closed; prescribed marine followers are never numerical unknowns',()=>{
    const manifest=JSON.parse(readFileSync(path.join(root!,'manifest.json'),'utf8'));const failures:string[]=[];let count=0;
    for(const row of manifest.doors){
      const {model}=load(row.id);if(!hasLoops(model))continue;count++;
      const s=new LoopSolver(model);const primary=s.art.joints.get(model.meta.primary_joint)!;
      const marine=model.meta.marine_dog_linkage;
      const prescribed=new Set(marine?[marine.output_joint,...marine.dog_joints,...marine.rod_joints]:[]);
      for(const name of s.coupled)expect(s.owned.has(name)).toBe(prescribed.has(name));
      for(const loop of s.loops)for(const joint of loop.joints)expect(s.coupled.has(joint.joint.name)).toBe(false);
      for(let i=0;i<=36;i++){
        const [lo,hi]=primary.range??[0,1.5],q=lo+(hi-lo)*i/36;
        s.setQ(primary.name,q);if(model.meta.secondary_joint)s.setQ(model.meta.secondary_joint,q);
        if(marine){const [a,b]=marine.input_range_rad;s.setQ(marine.input_joint,a+(b-a)*i/36);}
        for(const e of model.equalities)if(e.kind==='joint'&&[model.meta.primary_joint,model.meta.secondary_joint].includes(e.b))s.setQ(e.a,e.polycoeff[0]+e.polycoeff[1]*q);
        for(const loop of s.solve())if(!(loop.separation<1e-3))failures.push(`${row.id} ${loop.name} at ${i}: ${loop.separation}m`);
      }
      failures.push(...s.warnings.map(w=>`${row.id}: ${w}`));
    }
    expect(count).toBeGreaterThan(0);expect(failures).toEqual([]);
  });
});
