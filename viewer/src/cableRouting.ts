import * as THREE from 'three';

export interface CablePulley {
  kind: 'cylinder'|'sphere'; position: number[]; axis: number[];
  radius: number; side_point: number[]|null;
}
export interface NativeCableFrame {
  schema_version: number; units: string; frame: string;
  cables: {name:string; length_m:number; max_length_m:number;
    nodes:{point:number[];geom_name:string|null}[];
    wrap_geometries:Record<string,CablePulley>}[];
}
const vec=(p:number[])=>new THREE.Vector3(p[0],p[1],p[2]);
const twoPi=2*Math.PI;
const positive=(a:number)=>(a%twoPi+twoPi)%twoPi;

/** Circular surface arc joining native tangent points, with continuous tangents. */
export function pulleyArc(a:THREE.Vector3,b:THREE.Vector3,before:THREE.Vector3,after:THREE.Vector3,p:CablePulley) {
  const center=vec(p.position), axis=vec(p.axis).normalize();
  const radial=a.clone().sub(center); const axial=radial.dot(axis);
  const u=radial.addScaledVector(axis,-axial).normalize(),v=axis.clone().cross(u);
  const rb=b.clone().sub(center), end=Math.atan2(rb.dot(v),rb.dot(u));
  const entering=a.clone().sub(before).normalize();
  const sign=v.dot(entering)>=0?1:-1;
  const angle=sign>0?positive(end):-positive(-end);
  const exitTangent=u.clone().multiplyScalar(-Math.sin(angle)).addScaledVector(v,Math.cos(angle)).multiplyScalar(sign);
  if(exitTangent.dot(after.clone().sub(b).normalize())<.99) throw Error('Discontinuous cable tangent');
  const n=Math.max(2,Math.ceil(Math.abs(angle)/(.05)));
  return Array.from({length:n+1},(_,i)=>center.clone().addScaledVector(axis,axial+(rb.dot(axis)-axial)*i/n)
    .addScaledVector(u,p.radius*Math.cos(angle*i/n)).addScaledVector(v,p.radius*Math.sin(angle*i/n)));
}

/** Static inspection uses exact external tangencies, not lines through pulley centers. */
export function routePulley(before:THREE.Vector3,after:THREE.Vector3,p:CablePulley) {
  if(p.kind!=='cylinder') throw Error('Static spherical cable routing is not supported');
  const c=vec(p.position),axis=vec(p.axis).normalize(),r1=before.clone().sub(c),r2=after.clone().sub(c);
  const z1=r1.dot(axis),z2=r2.dot(axis);
  if(Math.abs(z1-z2)>1e-4) throw Error('Inspection cable endpoints must share a pulley plane');
  r1.addScaledVector(axis,-z1);r2.addScaledVector(axis,-z2);
  const d1=r1.length(),d2=r2.length();
  if(d1<p.radius-1e-6||d2<p.radius-1e-6) throw Error('Cable endpoint lies inside its pulley');
  const u=r1.normalize(),v=axis.clone().cross(u),b=Math.atan2(r2.dot(v),r2.dot(u));
  const alpha=Math.acos(Math.min(1,p.radius/d1)),beta=Math.acos(Math.min(1,p.radius/d2));
  const point=(q:number)=>c.clone().addScaledVector(axis,z1).addScaledVector(u,p.radius*Math.cos(q)).addScaledVector(v,p.radius*Math.sin(q));
  const candidates:THREE.Vector3[][]=[];
  for(const a of [alpha,-alpha]) for(const e of [b+beta,b-beta]) {
    try { candidates.push(pulleyArc(point(a),point(e),before,after,p)); } catch { /* opposite tangent branch */ }
  }
  if(!candidates.length) throw Error('No continuous pulley route');
  const side=p.side_point?vec(p.side_point).sub(c).normalize():null;
  const missesSide=(arc:THREE.Vector3[])=>side&&Math.max(...arc.map(point=>point.clone().sub(c).normalize().dot(side)))<.995?1:0;
  // Two continuous tangent branches can both pass the side point. Choose
  // the shorter route there; the long branch crosses the two free spans.
  candidates.sort((a,b)=>missesSide(a)-missesSide(b)||a.length-b.length);
  return candidates[0];
}

export function nativeCablePoints(c:NativeCableFrame['cables'][number]) {
  const points:THREE.Vector3[]=[];
  for(let i=0;i<c.nodes.length;i++) {
    const node=c.nodes[i],next=c.nodes[i+1];
    if(node.geom_name && node.geom_name===next?.geom_name) {
      const pulley=c.wrap_geometries[node.geom_name];
      if(!pulley||!c.nodes[i-1]||!c.nodes[i+2]) throw Error('Incomplete native pulley route');
      points.push(...pulleyArc(vec(node.point),vec(next.point),vec(c.nodes[i-1].point),vec(c.nodes[i+2].point),pulley));i++;
    } else points.push(vec(node.point));
  }
  return points;
}
