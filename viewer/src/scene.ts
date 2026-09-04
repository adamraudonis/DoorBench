import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import type { BodyJ, GeomJ, ModelJ } from "./types";

const ASSETS = "./assets";
const objCache = new Map<string, Promise<THREE.BufferGeometry>>();
const loader = new OBJLoader();

function loadObj(key: string): Promise<THREE.BufferGeometry> {
  let p = objCache.get(key);
  if (!p) {
    p = new Promise((resolve, reject) => {
      loader.load(`${ASSETS}/hardware/${key}.obj`, (grp) => {
        const geos: THREE.BufferGeometry[] = [];
        grp.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) geos.push(m.geometry);
        });
        if (!geos.length) return reject(new Error("empty obj " + key));
        const g = geos.length === 1 ? geos[0] : mergeGeometries(geos);
        g.computeVertexNormals();
        resolve(g);
      }, undefined, reject);
    });
    objCache.set(key, p);
  }
  return p;
}

function mergeGeometries(geos: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const positions: number[] = [];
  for (const g of geos) {
    const p = g.getAttribute("position");
    const idx = g.getIndex();
    if (idx) for (let i = 0; i < idx.count; i++) { const k = idx.getX(i); positions.push(p.getX(k), p.getY(k), p.getZ(k)); }
    else for (let i = 0; i < p.count; i++) positions.push(p.getX(i), p.getY(i), p.getZ(i));
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return out;
}

export interface JointHandle {
  name: string;
  body: string;
  type: "hinge" | "slide";
  axis: THREE.Vector3;
  pos: THREE.Vector3;
  range: [number, number] | null;
  node: THREE.Object3D;      // the node whose transform we drive
  q: number;
  modeledAt: number;
  label: string;
  role: string;
  interactive: boolean;
}

export interface BuiltScene {
  root: THREE.Group;
  joints: Map<string, JointHandle>;
  bodies: Map<string, THREE.Object3D>;
  bounds: THREE.Box3;
  setJoint: (name: string, q: number, propagate?: boolean) => void;
  dispose: () => void;
}

const Z_TO_Y = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2); // three cylinders are Y-aligned; our cylinders are Z-aligned

function makeMaterial(model: ModelJ, name: string, cache: Map<string, THREE.Material>): THREE.Material {
  let m = cache.get(name);
  if (m) return m;
  const mj = model.materials[name];
  const rgba = mj?.rgba ?? [0.7, 0.7, 0.7, 1];
  const transparent = (mj?.transparent ?? false) || rgba[3] < 0.99;
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    roughness: mj?.roughness ?? 0.6,
    metalness: mj?.metallic ?? 0,
    transparent,
    opacity: transparent ? Math.min(rgba[3], 0.55) : 1,
    side: transparent ? THREE.DoubleSide : THREE.FrontSide,
    depthWrite: !transparent,
  });
  if (mj?.emissive && (mj.emissive[0] || mj.emissive[1] || mj.emissive[2])) mat.emissive = new THREE.Color(mj.emissive[0], mj.emissive[1], mj.emissive[2]);
  cache.set(name, mat);
  return mat;
}

export function geomMesh(g: GeomJ, mat: THREE.Material): THREE.Object3D | null {
  let geo: THREE.BufferGeometry | null = null;
  const q = new THREE.Quaternion(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
  if (g.type === "box") geo = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
  else if (g.type === "cylinder") { geo = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 28); q.multiply(Z_TO_Y); }
  else if (g.type === "capsule") { geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 6, 16); q.multiply(Z_TO_Y); }
  else if (g.type === "sphere") geo = new THREE.SphereGeometry(g.size[0], 20, 14);
  else return null;
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
  mesh.quaternion.copy(q);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = g.name;
  mesh.userData = { semantic: g.semantic, label: g.part_label, collision: g.collision };
  return mesh;
}

export async function buildScene(model: ModelJ, opts: { showCollision?: boolean; showEnv?: boolean } = {}): Promise<BuiltScene> {
  const root = new THREE.Group();
  root.name = model.name;
  const matCache = new Map<string, THREE.Material>();
  const bodies = new Map<string, THREE.Object3D>();
  const joints = new Map<string, JointHandle>();
  const pending: Promise<void>[] = [];
  const bounds = new THREE.Box3();

  // body nodes: body -> [jointPivot -> [inner -> geoms + children]]
  for (const b of model.bodies) {
    const node = new THREE.Group();
    node.name = b.name;
    node.position.set(b.pos[0], b.pos[1], b.pos[2]);
    node.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    let container: THREE.Object3D = node;
    if (b.joint) {
      const j = b.joint;
      const pivot = new THREE.Group();
      pivot.name = j.name + "_pivot";
      pivot.position.set(j.pos[0], j.pos[1], j.pos[2]);
      const inner = new THREE.Group();
      inner.position.set(-j.pos[0], -j.pos[1], -j.pos[2]);
      pivot.add(inner);
      node.add(pivot);
      container = inner;
      joints.set(j.name, {
        name: j.name, body: b.name, type: j.type, axis: new THREE.Vector3(...j.axis).normalize(), pos: new THREE.Vector3(...j.pos),
        range: j.range, node: pivot, q: j.modeled_at ?? 0, modeledAt: j.modeled_at ?? 0, label: j.label, role: j.role, interactive: j.robot_interactive,
      });
    }
    (node as any).userData = { container, semantic: b.semantic, label: b.label, static: b.static, mass: b.mass };
    for (const g of b.geoms) {
      if (!g.visual && !opts.showCollision) continue;
      if (!opts.showEnv && (g.semantic === "wall" || g.semantic === "floor")) continue;
      const mat = g.visual ? makeMaterial(model, g.material, matCache) : new THREE.MeshBasicMaterial({ color: 0xff3366, wireframe: true, transparent: true, opacity: 0.4 });
      if (g.type === "mesh" && g.mesh_name) {
        const key = g.mesh_name;
        const gpos = g.pos, gquat = g.quat;
        pending.push(loadObj(key).then((geo) => {
          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.set(gpos[0], gpos[1], gpos[2]);
          mesh.quaternion.set(gquat[1], gquat[2], gquat[3], gquat[0]);
          mesh.castShadow = true; mesh.receiveShadow = true;
          mesh.name = g.name;
          mesh.userData = { semantic: g.semantic, label: g.part_label };
          container.add(mesh);
        }).catch(() => {}));
      } else {
        const mesh = geomMesh(g, mat);
        if (mesh) container.add(mesh);
      }
    }
    bodies.set(b.name, node);
    if (b.parent && bodies.has(b.parent)) ((bodies.get(b.parent) as any).userData.container as THREE.Object3D).add(node);
    else root.add(node);
  }
  await Promise.all(pending);
  root.updateMatrixWorld(true);
  // bounds of door-ish parts (exclude env)
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh && m.userData?.semantic !== "wall" && m.userData?.semantic !== "floor") {
      m.geometry.computeBoundingBox();
      const bb = m.geometry.boundingBox!.clone().applyMatrix4(m.matrixWorld);
      bounds.union(bb);
    }
  });

  // couplings: driven joint = c0 + c1 * driver
  const couplings = model.equalities.filter((e) => e.kind === "joint" && e.b);
  const tendons = model.tendons ?? [];

  function applyJoint(h: JointHandle) {
    const dq = h.q - h.modeledAt;
    if (h.type === "hinge") {
      h.node.quaternion.setFromAxisAngle(h.axis, dq);
      h.node.position.copy(h.pos);
    } else {
      h.node.quaternion.identity();
      h.node.position.copy(h.pos).addScaledVector(h.axis, dq);
    }
  }

  function setJoint(name: string, q: number, propagate = true) {
    const h = joints.get(name);
    if (!h) return;
    if (h.range) q = Math.min(Math.max(q, h.range[0]), h.range[1]);
    h.q = q;
    applyJoint(h);
    if (!propagate) return;
    for (const c of couplings) {
      if (c.b === name) {
        const d = joints.get(c.a);
        if (d) { const [c0, c1] = c.polycoeff; setJoint(c.a, c0 + c1 * q, false); }
      }
    }
    // one-sided tendon: bolt_q >= scale * handle_q  -> visualise bolt following the handle
    for (const t of tendons) {
      const [[bolt, cb], [driver, cd]] = t.sites as [string, number][];
      if (driver === name) {
        const d = joints.get(bolt);
        if (d) { const target = Math.max(0, (-cd / cb) * q); setJoint(bolt, Math.max(target, 0), false); }
      }
    }
  }
  for (const h of joints.values()) applyJoint(h);

  return {
    root, joints, bodies, bounds, setJoint,
    dispose: () => { root.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); } }); for (const m of matCache.values()) m.dispose(); },
  };
}
