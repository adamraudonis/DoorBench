import {expect,test} from 'bun:test';
import * as THREE from 'three';
import {routePulley,nativeCablePoints,type CablePulley} from './cableRouting';

test('two-to-one cable wraps around the sheave surface with continuous entry and exit',()=>{
  const p:CablePulley={kind:'cylinder',position:[1,0,0],axis:[0,0,1],radius:.1,side_point:[1.2,0,0]};
  const before=new THREE.Vector3(0,-.1,0),after=new THREE.Vector3(0,.1,0);
  const arc=routePulley(before,after,p);
  expect(arc[0].distanceTo(new THREE.Vector3(1,-.1,0))).toBeLessThan(1e-8);
  expect(arc.at(-1)!.distanceTo(new THREE.Vector3(1,.1,0))).toBeLessThan(1e-8);
  expect(Math.max(...arc.map(v=>v.x))).toBeCloseTo(1.1,3);
  const length=before.distanceTo(arc[0])+arc.reduce((s,v,i)=>s+(i?v.distanceTo(arc[i-1]):0),0)+after.distanceTo(arc.at(-1)!);
  expect(length).toBeCloseTo(2+Math.PI*.1,4);
  for(const point of arc) expect(point.distanceTo(new THREE.Vector3(1,0,0))).toBeCloseTo(.1,8);
});

test('native tangent pairs never become a chord through the pulley bore',()=>{
  const points=nativeCablePoints({name:'cable',length_m:2+Math.PI*.1,max_length_m:2+Math.PI*.1,
    nodes:[{point:[0,-.1,0],geom_name:null},{point:[1,-.1,0],geom_name:'sheave'},
      {point:[1,.1,0],geom_name:'sheave'},{point:[0,.1,0],geom_name:null}],
    wrap_geometries:{sheave:{kind:'cylinder',position:[1,0,0],axis:[0,0,1],radius:.1,side_point:[1.2,0,0]}}});
  expect(points.length).toBeGreaterThan(60);
  for(const point of points.slice(1,-1)) expect(point.distanceTo(new THREE.Vector3(1,0,0))).toBeCloseTo(.1,8);
});
