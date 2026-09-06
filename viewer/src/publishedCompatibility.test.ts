/** Compatibility with the immutable release is separate from current-generator regression fixtures. */
import {describe, expect, test} from 'bun:test';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import type {ModelJ} from './types';
import {openClosePhases, operatorJoints, operatorReturnPhase, type JointLike} from './doorLogic';
import {deadboltControls, openingProcedure} from './mechanismInspection';

const root=process.env.DOORBENCH_PUBLISHED_ASSETS;
describe.skipIf(!root)('published dataset compatibility',()=>{
  test('every released door has finite, existing mechanism targets and an inspection guide',()=>{
    const manifest=JSON.parse(readFileSync(path.join(root!,'manifest.json'),'utf8'));
    expect(manifest.doors).toHaveLength(1000);
    const errors:string[]=[];
    for(const row of manifest.doors){
      const folder=path.join(root!,'doors',row.id);
      const model:ModelJ=JSON.parse(readFileSync(path.join(folder,'model.json'),'utf8'));
      const spec=JSON.parse(readFileSync(path.join(folder,'spec.json'),'utf8'));
      const joints=new Map<string,JointLike>();
      for(const b of model.bodies)if(b.joint)joints.set(b.joint.name,{name:b.joint.name,q:b.joint.modeled_at??0,modeledAt:b.joint.modeled_at??0,range:b.joint.range});
      try{
        const phases=openClosePhases(model,joints).phases.flatMap(p=>[p,...p.followers??[]]);
        for(const p of phases){
          const j=joints.get(p.joint);
          if(!j||!Number.isFinite(p.to))errors.push(`${row.id}: invalid phase ${p.joint}`);
          else if(j.range&&(p.to<j.range[0]-1e-6||p.to>j.range[1]+1e-6))errors.push(`${row.id}: phase outside native range ${p.joint}`);
        }
        for(const name of operatorJoints(model)){
          if(!joints.has(name))errors.push(`${row.id}: missing operator ${name}`);
          const phase=operatorReturnPhase(model,joints,name);
          if(phase&&!Number.isFinite(phase.to))errors.push(`${row.id}: invalid return ${name}`);
        }
        for(const turn of deadboltControls(model))if(!joints.has(turn.joint)||!joints.has(turn.bolt))errors.push(`${row.id}: missing deadbolt coupling`);
        const guide=openingProcedure(model,spec);
        if(!guide.steps.length||guide.steps.some(s=>!s.trim()))errors.push(`${row.id}: empty guide`);
      }catch(e){errors.push(`${row.id}: ${String(e)}`);}
    }
    expect(errors).toEqual([]);
  });
});
