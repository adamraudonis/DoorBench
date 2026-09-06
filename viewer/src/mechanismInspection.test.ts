import { test, expect } from 'bun:test';
import { buildScene } from './scene';
import { deadboltControls, openingProcedure } from './mechanismInspection';
import type { ModelJ } from './types';
const geom = (name: string, semantic: string) => ({name,semantic,type:'box',size:[.1,.02,.1],pos:[0,0,0],quat:[1,0,0,0],visual:true,material:'steel'});
function fixture(): ModelJ {
  const joint=(name:string,role:string,range=[0,1])=>({name,role,range,type:'hinge',axis:[0,0,1],pos:[0,0,0],robot_interactive:true});
  return {name:'test',materials:{steel:{rgba:[.4,.4,.4,1]}},meta:{operator_joint:'handle'},tendons:[],equalities:[{kind:'joint',a:'leaf_deadbolt_slide',b:'leaf_deadbolt_thumbturn_hinge',polycoeff:[0,.02,0,0,0]}],bodies:[
    {name:'leaf',pos:[0,0,0],quat:[1,0,0,0],joint:joint('hinge','primary'),geoms:[geom('slab','leaf'),geom('glass','glass')]},
    {name:'handle',parent:'leaf',pos:[1,0,0],quat:[1,0,0,0],joint:joint('handle','operator'),geoms:[geom('handle_mesh','operator')]},
    {name:'turn',parent:'leaf',pos:[0,0,0],quat:[1,0,0,0],joint:joint('leaf_deadbolt_thumbturn_hinge','lock'),geoms:[geom('turn_mesh','lock')]},
    {name:'bolt',parent:'leaf',pos:[0,0,0],quat:[1,0,0,0],joint:{...joint('leaf_deadbolt_slide','lock',[0,.02]),type:'slide',robot_interactive:false},geoms:[geom('bolt_mesh','lock')]},
    {name:'wall',pos:[0,0,0],quat:[1,0,0,0],geoms:[geom('wall_mesh','wall'),geom('floor_mesh','floor')]},
  ]} as unknown as ModelJ;
}
test('hidden leaf preserves visible child hardware and coupled deadbolt movement', async()=>{
  const m=fixture(),b=await buildScene(m,{showEnv:true});b.setMechanismsOnly(true);
  for(const name of ['slab','glass','wall_mesh','floor_mesh']) expect(b.root.getObjectByName(name)!.visible).toBe(false);
  expect(b.bodies.get('leaf')!.visible).toBe(true);expect(b.root.getObjectByName('handle_mesh')!.visible).toBe(true);
  b.setJoint('handle',1);expect(b.joints.get('leaf_deadbolt_slide')!.q).toBe(0);
  b.setJoint(deadboltControls(m)[0].joint,1);expect(b.joints.get('leaf_deadbolt_slide')!.q).toBeCloseTo(.02);
  b.setJoint('hinge',1);b.root.updateMatrixWorld(true);expect(b.bodies.get('handle')!.matrixWorld.elements[12]).toBeCloseTo(Math.cos(1));
  b.setMechanismsOnly(false);expect(b.root.getObjectByName('slab')!.visible).toBe(true);b.dispose();
});
test('a thumbturn without a bolt coupling is not advertised as a deadbolt control',()=>{
  const m=fixture();m.equalities=[];expect(deadboltControls(m)).toEqual([]);
});
test('procedure distinguishes legacy multipoint from rebuilt operation and inaccessible locks',()=>{
  const m=fixture(),s={lock:{model:'multipoint',engaged:true,robot_side_release:true}};
  expect(openingProcedure(m,s).steps.join(' ')).toContain('unavailable');
  m.meta.multipoint_locks=[{}];const steps=openingProcedure(m,s).steps;
  expect(steps[0]).toContain('thumbturn');expect(steps[1]).toContain('Depress');
  s.lock.robot_side_release=false;expect(openingProcedure(m,s).steps.join(' ')).toContain('no declared release');
});
