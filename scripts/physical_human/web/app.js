import * as THREE from 'three';
import { OrbitControls } from './lib/OrbitControls.js';

const $ = id => document.getElementById(id);
const [data, checks] = await Promise.all(['replay.json', 'checks.json'].map(async url => {
  const response = await fetch(url);
  if (!response.ok) throw Error(`Cannot load ${url}`);
  return response.json();
}));
$('loading').remove();
const viewport = $('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#536166');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
viewport.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(40, 1, .008, 40);
camera.up.set(0, 0, 1);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = .12;
orbit.minDistance = .10;
orbit.maxDistance = 8;
scene.add(new THREE.HemisphereLight('#fffdf4', '#657265', 2.3));
const sun = new THREE.DirectionalLight('#fff9e9', 3);
sun.position.set(-2, -3, 5);
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: .1, far: 14 });
sun.shadow.bias = -.00015;
sun.shadow.normalBias = .001;
scene.add(sun);
const fill = new THREE.DirectionalLight('#c2dbed', 1.5);
fill.position.set(2, 2, 3);
scene.add(fill);

const meshes = data.geoms.map(g => {
  let geometry;
  const [a, b, c] = g.size;
  switch (g.type) {
    case 0: geometry = new THREE.PlaneGeometry(100, 100); break;
    case 2: geometry = new THREE.SphereGeometry(a, 24, 16); break;
    case 3: geometry = new THREE.CapsuleGeometry(a, 2 * b, 6, 16); geometry.rotateX(Math.PI / 2); break;
    case 4: geometry = new THREE.SphereGeometry(1, 24, 16); geometry.scale(a, b, c); break;
    case 5: geometry = new THREE.CylinderGeometry(a, a, 2 * b, 24); geometry.rotateX(Math.PI / 2); break;
    case 6: geometry = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c); break;
    default: throw Error('Unsupported primitive ' + g.type);
  }
  const hardware = /lever|latch|rose|hinge_\d|strike_near|strike_far/.test(g.name);
  const human = /^actor_|^hand_/.test(g.body);
  const hand = g.name.startsWith('hand_');
  let color = hardware ? '#c79543' : g.name === 'door_leaf' ? '#855339' : human ? '#3d7775' : '#70796f';
  if (g.name === 'floor') color = '#bec6be';
  if (hand) color = new THREE.Color().setRGB(...g.rgba.slice(0, 3));
  const material = new THREE.MeshStandardMaterial({ color, metalness: hardware ? .45 : .04, roughness: hardware ? .37 : .67 });
  if (g.group === 4) { material.transparent = true; material.opacity = .20; material.depthWrite = false; }
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = g.type !== 0;
  mesh.receiveShadow = true;
  scene.add(mesh);
  return mesh;
});
const contactGeometry = new THREE.SphereGeometry(.0018, 10, 8);
const contactMaterial = new THREE.MeshBasicMaterial({ color: '#ef946b', depthTest: false });
const dots = Array.from({ length: 60 }, () => {
  const mesh = new THREE.Mesh(contactGeometry, contactMaterial);
  mesh.renderOrder = 10;
  mesh.visible = false;
  scene.add(mesh);
  return mesh;
});
const groups = [['Index', 'index'], ['Middle', 'middle'], ['Ring', 'ring'], ['Little', 'little'], ['Thumb pads', 'thumb'], ['Palm / base', 'palm']];
function digitOf(name) {
  if (/proximal_thumb|distal_thumb/.test(name)) return 'thumb';
  for (const [number, digit] of [[2, 'index'], [3, 'middle'], [4, 'ring'], [5, 'little']]) {
    if (name.includes(number + 'proxph') || name.includes('midph' + number) || name.includes('distph' + number)) return digit;
  }
  return 'palm';
}
$('fingerbars').innerHTML = groups.map(([label, id]) => `<div class="finger"><label>${label}</label><i><em id="bar-${id}"></em></i><span id="n-${id}">0.0 N</span></div>`).join('');
const qualityCount = Object.keys(checks.baseline.quality_checks).length;
$('quality-status').textContent = checks.baseline.quality_passed ? `${qualityCount} / ${qualityCount} passed` : 'FAILED';
$('kinematic-status').textContent = checks.baseline.kinematics.passed ? 'Passed' : 'FAILED';
$('contact-coverage').textContent = (checks.baseline.grasp.phases.pull.opposed_fraction * 100).toFixed(2) + '%';
const angleLabels = ['Thumb CMC flex', 'Thumb CMC spread', 'Thumb MCP', 'Thumb IP', 'Wrist flex', 'Wrist deviation'];
$('angles').innerHTML = angleLabels.map((label, i) => `<div class="test"><span>${label}</span><b id="joint-angle-${i}">0°</b></div>`).join('');
const rows = data.report.rows;
const end = data.time.at(-1);
const maxForce = Math.max(...rows.map(r => r.touch_n), 1);
$('duration').textContent = end.toFixed(2) + ' s';
const points = rows.map((r, i) => `${i / (rows.length - 1) * 280},${70 - r.touch_n / maxForce * 62}`).join(' ');
$('chart').innerHTML = `<polyline points="${points}" fill="none" stroke="#b18a49" stroke-width="1.4"/><line id="cursor" x1="0" x2="0" y1="0" y2="74" stroke="#466855" stroke-width="1.5"/>`;
$('withtouch').textContent = checks.baseline.max_door_deg.toFixed(1) + '° opened';
$('withouttouch').textContent = checks['no-touch'].max_door_deg.toFixed(2) + '°';
$('blocked').textContent = checks.blocked.max_door_deg.toFixed(2) + '°';
let playing = true, clock = 0, speed = 1, view = 'thumb', last = performance.now(), manualOrbit = false;
const poses = new THREE.Vector3(), quat = new THREE.Quaternion(), poseB = new THREE.Vector3(), quatB = new THREE.Quaternion();
const gripIndex = data.geoms.findIndex(g => g.name === 'lever_grip');
const phaseOrder = ['settle', 'reach', 'place around lever', 'grasp', 'settle grip', 'press lever', 'pull', 'hold open'];
function doorVisibility() {
  meshes.forEach((mesh, i) => {
    if (data.geoms[i].name !== 'door_leaf') return;
    mesh.material.transparent = $('xray').checked;
    mesh.material.opacity = $('xray').checked ? .08 : 1;
    mesh.material.depthWrite = !$('xray').checked;
    mesh.material.needsUpdate = true;
    mesh.castShadow = !$('xray').checked;
  });
}
function setView(value) {
  view = value;
  manualOrbit = false;
  sun.castShadow = view === 'scene';
  scene.background.set(view === 'scene' ? '#e9ebe4' : '#536166');
  meshes.forEach((mesh, i) => {
    const g = data.geoms[i];
    const actor = /^actor_|^hand_/.test(g.body);
    mesh.visible = (view === 'scene' || !actor || g.name.startsWith('hand_l_')) && (g.group !== 4 || $('envelopes').checked);
  });
  orbit.maxPolarAngle = view === 'scene' ? Math.PI * .49 : Math.PI * .85;
  $('hint').textContent = view === 'scene' ? 'Drag to orbit · Scroll to zoom' : 'Actual hand bones · Drag to inspect · Scroll to zoom';
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  if (view === 'scene') {
    camera.position.set(-2.55, -3.7, 2.20);
    orbit.target.set(.48, -.35, 1.05);
  }
  update(Math.min(clock, end));
  orbit.update();
}
function seek(time) { clock = THREE.MathUtils.clamp(time, 0, end); last = performance.now(); }
function update(time) {
  const k = Math.max(0, Math.min(rows.length - 2, Math.floor((time - data.time[0]) / (data.time[1] - data.time[0]))));
  const u = THREE.MathUtils.clamp((time - data.time[k]) / (data.time[k + 1] - data.time[k]), 0, 1);
  const a = data.frames[k], b = data.frames[k + 1], row = rows[k];
  meshes.forEach((mesh, i) => {
    poses.fromArray(a[i]); poseB.fromArray(b[i]); mesh.position.lerpVectors(poses, poseB, u);
    quat.fromArray(a[i], 3); quatB.fromArray(b[i], 3); mesh.quaternion.slerpQuaternions(quat, quatB, u);
  });
  data.angle_names.forEach((name, i) => { $('joint-angle-' + i).textContent = data.angles_deg[k][i].toFixed(1) + '°'; });
  const values = Object.fromEntries(groups.map(([, id]) => [id, 0]));
  const contacts = data.report.contacts[k];
  dots.forEach((dot, i) => {
    dot.visible = $('contacts').checked && i < contacts.length;
    if (dot.visible) { dot.position.fromArray(contacts[i]); dot.scale.setScalar(.7 + Math.min(1, contacts[i][3] / 12)); }
  });
  for (const contact of contacts) {
    const name = contact[4].startsWith('hand_') ? contact[4] : contact[5];
    values[digitOf(name)] += contact[3];
  }
  for (const [, id] of groups) {
    $('bar-' + id).style.width = Math.min(100, values[id] * 5) + '%';
    $('n-' + id).textContent = values[id].toFixed(1) + ' N';
  }
  for (const [id, key, unit] of [['angle', 'door_deg', '°'], ['lever', 'lever_deg', '°'], ['latch', 'latch_mm', 'mm'], ['force', 'touch_n', 'N']]) {
    $(id).innerHTML = `${Math.max(0, row[key]).toFixed(1)}<small>${unit}</small>`;
  }
  const grasp = row.grasp;
  $('thumb-force').innerHTML = `${grasp.thumb_normal_force_n.toFixed(1)}<small>N</small>`;
  const working = ['press lever', 'pull', 'hold open'].includes(row.phase);
  const sidesCorrect = grasp.four_fingers_together && grasp.thumb_on_opposite_side && grasp.thumb_below_grip;
  $('side-status').textContent = working ? (sidesCorrect ? 'Four fingers together · Thumb below and opposite' : 'CHECK GRASP PLACEMENT') : 'The open hand approaches before the fingers close';
  $('opposition-status').textContent = working ? `${grasp.opposed_loaded_fingers} / 4 fingers opposed to a loaded thumb` : 'Preparing the grasp';
  $('phase').textContent = row.phase === 'settle' ? 'Ready' : row.phase;
  $('step').textContent = String(Math.max(1, phaseOrder.indexOf(row.phase))).padStart(2, '0');
  $('time').value = time / end * 1000;
  $('clock').textContent = `${time.toFixed(2)} / ${end.toFixed(2)} s`;
  const x = time / end * 280;
  $('cursor').setAttribute('x1', x); $('cursor').setAttribute('x2', x);
  document.querySelectorAll('.chapters button').forEach(button => button.classList.toggle('current', button.dataset.phase === row.phase));
  if (view !== 'scene' && !manualOrbit) {
    const at = meshes[gripIndex].position.clone().add(new THREE.Vector3(0, 0, .015));
    const door = THREE.MathUtils.lerp(row.door_deg, rows[k + 1].door_deg, u);
    const azimuth = THREE.MathUtils.degToRad((view === 'thumb' ? 140 : 310) - door);
    const elevation = THREE.MathUtils.degToRad(view === 'thumb' ? -12 : -18);
    const distance = .28;
    camera.position.copy(at).add(new THREE.Vector3(-Math.cos(azimuth) * Math.cos(elevation), -Math.sin(azimuth) * Math.cos(elevation), -Math.sin(elevation)).multiplyScalar(distance));
    orbit.target.copy(at);
  }
}
$('play').onclick = () => {
  playing = !playing;
  $('play').textContent = playing ? 'Pause' : 'Play';
  $('play').setAttribute('aria-label', playing ? 'Pause playback' : 'Play playback');
};
$('restart').onclick = () => seek(0);
$('time').oninput = event => seek(+event.target.value / 1000 * end);
$('speed').onchange = event => { speed = +event.target.value; };
document.querySelectorAll('[data-view]').forEach(button => { button.onclick = () => setView(button.dataset.view); });
document.querySelectorAll('[data-time]').forEach(button => { button.onclick = () => seek(+button.dataset.time); });
$('envelopes').onchange = () => setView(view);
$('xray').onchange = doorVisibility;
orbit.addEventListener('start', () => { manualOrbit = true; });
window.addEventListener('keydown', event => {
  if (event.code === 'Space' && !['INPUT', 'BUTTON', 'SELECT'].includes(document.activeElement.tagName)) { event.preventDefault(); $('play').click(); }
});
new ResizeObserver(() => {
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  camera.updateProjectionMatrix();
}).observe(viewport);
doorVisibility();
setView('thumb');
function frame(now) {
  const dt = Math.max(0, Math.min(.05, (now - last) / 1000));
  last = now;
  if (playing) { clock += dt * speed; if (clock > end + .7) clock = 0; }
  update(Math.min(clock, end));
  orbit.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
