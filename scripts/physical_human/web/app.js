import * as THREE from 'three';
import { OrbitControls } from './lib/OrbitControls.js';
const $=id=>document.getElementById(id);
const [data,checks]=await Promise.all(['replay.json','checks.json'].map(async u=>{const r=await fetch(u);if(!r.ok)throw Error(`Cannot load ${u}`);return r.json()}));
$('loading').remove();
const viewport=$('viewport'), scene=new THREE.Scene();scene.background=new THREE.Color('#e9ebe4');scene.fog=new THREE.Fog('#e9ebe4',7,17);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.35;viewport.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(37,1,.015,40);camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,renderer.domElement);orbit.enableDamping=true;orbit.dampingFactor=.1;orbit.minDistance=.15;orbit.maxDistance=8;orbit.maxPolarAngle=Math.PI*.49;
scene.add(new THREE.HemisphereLight('#fffdf4','#657265',2.3));
const sun=new THREE.DirectionalLight('#fff9e9',4);sun.position.set(-2,-3,5);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-3,right:3,top:3,bottom:-3,near:.1,far:14});sun.shadow.bias=-.00015;sun.shadow.normalBias=.009;scene.add(sun);
const fill=new THREE.DirectionalLight('#c2dbed',1.5);fill.position.set(2,2,3);scene.add(fill);
const meshes=data.geoms.map(g=>{
 let geo;const [a,b,c]=g.size;
 switch(g.type){case 0:geo=new THREE.PlaneGeometry(100,100);break;case 2:geo=new THREE.SphereGeometry(a,24,16);break;case 3:geo=new THREE.CapsuleGeometry(a,2*b,6,16);geo.rotateX(Math.PI/2);break;case 4:geo=new THREE.SphereGeometry(1,24,16);geo.scale(a,b,c);break;case 5:geo=new THREE.CylinderGeometry(a,a,2*b,24);geo.rotateX(Math.PI/2);break;case 6:geo=new THREE.BoxGeometry(2*a,2*b,2*c);break;default:throw Error('Unsupported primitive '+g.type)}
 const hardware=/lever|latch|rose|hinge_\d|strike_near|strike_far/.test(g.name),human=/^actor_|^hand_/.test(g.body),hand=g.name.startsWith('hand_');
 let color=hardware?'#bb8639':g.name==='door_leaf'?'#855339':human?(hand?'#deded0':'#3d7775'):'#70796f';if(g.name==='floor')color='#d9dfd0';
 const material=new THREE.MeshStandardMaterial({color,metalness:hardware?.62:.04,roughness:hardware?.32:.67});
 const mesh=new THREE.Mesh(geo,material);mesh.castShadow=g.type!==0;mesh.receiveShadow=true;mesh.userData.base=color;mesh.userData.hardware=hardware;scene.add(mesh);return mesh;
});
const contactGeo=new THREE.SphereGeometry(.0035,10,8),contactMat=new THREE.MeshBasicMaterial({color:'#e7a037',depthTest:false});const dots=Array.from({length:60},()=>{const mesh=new THREE.Mesh(contactGeo,contactMat);mesh.renderOrder=10;mesh.visible=false;scene.add(mesh);return mesh});
const groups=[['Palm','palm'],['Index','finger3'],['Middle','finger2'],['Ring','finger1'],['Little','finger0'],['Thumb','thumb']];
$('fingerbars').innerHTML=groups.map(([label,id])=>`<div class="finger"><label>${label}</label><i><em id="bar-${id}"></em></i><span id="n-${id}">0.0 N</span></div>`).join('');
const rows=data.report.rows,end=data.time.at(-1),maxForce=Math.max(...rows.map(r=>r.touch_n),1);
const points=rows.map((r,i)=>`${i/(rows.length-1)*280},${70-r.touch_n/maxForce*62}`).join(' ');
$('chart').innerHTML=`<polyline points="${points}" fill="none" stroke="#b18a49" stroke-width="1.4"/><line id="cursor" x1="0" x2="0" y1="0" y2="74" stroke="#466855" stroke-width="1.5"/>`;
$('withtouch').textContent=checks.baseline.max_door_deg.toFixed(1)+'° opened';$('withouttouch').textContent=checks['no-touch']?checks['no-touch'].max_door_deg.toFixed(2)+'°':'Not run';$('blocked').textContent=checks.blocked?checks.blocked.max_door_deg.toFixed(2)+'°':'Not run';
let playing=true,clock=0,speed=1,view='scene',last=performance.now(),manualOrbit=false;
const poses=new THREE.Vector3(),quat=new THREE.Quaternion(),poseB=new THREE.Vector3(),quatB=new THREE.Quaternion();
const gripIndex=data.geoms.findIndex(g=>g.name==='lever_grip'),boltIndex=data.geoms.findIndex(g=>g.name==='latch_bolt');
function setView(v){view=v;manualOrbit=false;meshes.forEach((mesh,i)=>{const g=data.geoms[i];const actor=/^actor_|^hand_/.test(g.body);mesh.visible=v==='scene'||!actor||g.name.startsWith('hand_l_')});orbit.maxPolarAngle=v==='scene'?Math.PI*.49:Math.PI*.85;$('hint').textContent=v==='scene'?'Drag to orbit · Scroll to zoom':'Hand isolated · Drag to inspect contact';document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===v));if(v==='scene'){camera.position.set(-2.55,-3.7,2.20);orbit.target.set(.48,-.18,1.05)}else{const at=meshes[v==='hand'?gripIndex:boltIndex].position;orbit.target.copy(at);camera.position.copy(at).add(new THREE.Vector3(v==='hand'?-.27:.33,v==='hand'?-.32:-.49,v==='hand'?-.08:.27))}orbit.update()}
function seek(t){clock=THREE.MathUtils.clamp(t,0,end);last=performance.now()}
function update(t){
 let k=Math.max(0,Math.min(rows.length-2,Math.floor((t-data.time[0])/(data.time[1]-data.time[0])))),u=THREE.MathUtils.clamp((t-data.time[k])/(data.time[k+1]-data.time[k]),0,1);
 const a=data.frames[k],b=data.frames[k+1],r=rows[k];
 meshes.forEach((mesh,i)=>{poses.fromArray(a[i]);poseB.fromArray(b[i]);mesh.position.lerpVectors(poses,poseB,u);quat.fromArray(a[i],3);quatB.fromArray(b[i],3);mesh.quaternion.slerpQuaternions(quat,quatB,u);mesh.material.emissive.set('#000000');});
 const values=Object.fromEntries(groups.map(([,id])=>[id,0]));
 const contacts=data.report.contacts[k];dots.forEach((dot,i)=>{dot.visible=i<contacts.length;if(dot.visible){dot.position.fromArray(contacts[i]);dot.scale.setScalar(.7+Math.min(2,contacts[i][3]/12))}});
 for(const c of contacts){const n=c[4].startsWith('hand_')?c[4]:c[5];for(const [,id]of groups)if(n.includes(id))values[id]+=c[3];const i=data.geoms.findIndex(g=>g.name===n);if(i>=0)meshes[i].material.emissive.setRGB(Math.min(.5,c[3]/35),Math.min(.23,c[3]/80),0)}
 for(const [,id]of groups){$('bar-'+id).style.width=Math.min(100,values[id]*5)+'%';$('n-'+id).textContent=values[id].toFixed(1)+' N'}
 for(const [id,key,unit]of [['angle','door_deg','°'],['lever','lever_deg','°'],['latch','latch_mm','mm'],['force','touch_n','N']])$(id).innerHTML=`${Math.max(0,r[key]).toFixed(1)}<small>${unit}</small>`;
 const phase=r.phase==='settle'?'Ready':r.phase;const pidx=['settle','reach','grasp','press lever','pull','release'].indexOf(r.phase);$('phase').textContent=phase;$('step').textContent=String(Math.max(1,pidx)).padStart(2,'0');
 $('time').value=t/end*1000;$('clock').textContent=`${t.toFixed(2)} / ${end.toFixed(2)} s`;const x=t/end*280;$('cursor').setAttribute('x1',x);$('cursor').setAttribute('x2',x);
 document.querySelectorAll('.chapters button').forEach((button,i)=>button.classList.toggle('current',i===pidx-1));
 if(view!=='scene'&&!manualOrbit){const at=meshes[view==='hand'?gripIndex:boltIndex].position;const delta=at.clone().sub(orbit.target);camera.position.add(delta);orbit.target.copy(at)}
}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';$('play').setAttribute('aria-label',playing?'Pause playback':'Play playback')};$('restart').onclick=()=>seek(0);$('time').oninput=e=>seek(+e.target.value/1000*end);$('speed').onchange=e=>speed=+e.target.value;
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>seek(+b.dataset.time));
$('xray').onchange=e=>meshes.forEach((mesh,i)=>{if(data.geoms[i].name==='door_leaf'){mesh.material.transparent=e.target.checked;mesh.material.opacity=e.target.checked?.12:1;mesh.material.depthWrite=!e.target.checked;mesh.castShadow=!e.target.checked}});
orbit.addEventListener('start',()=>{manualOrbit=true});window.addEventListener('keydown',e=>{if(e.code==='Space'&&!['INPUT','BUTTON','SELECT'].includes(document.activeElement.tagName)){e.preventDefault();$('play').click()}});
new ResizeObserver(()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}).observe(viewport);
update(0);setView('scene');
function frame(now){const dt=Math.max(0,Math.min(.05,(now-last)/1000));last=now;if(playing){clock+=dt*speed;if(clock>end+.7)clock=0}update(Math.min(clock,end));orbit.update();renderer.render(scene,camera);requestAnimationFrame(frame)}requestAnimationFrame(frame);
window.__prototype={data,checks,seek,pause:()=>{playing=false;$('play').textContent='Play'},setView};
