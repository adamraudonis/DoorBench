// three.js view of a compiled MuJoCo model: one mesh per mjGeom, transforms copied from geom_xpos / geom_xmat every
// frame.  Rendering straight from the physics state (instead of re-driving the model.json scene graph) means what
// you see is exactly what MuJoCo simulates, including closer arms closed by `connect` equalities and the latch bolt
// riding over the strike.  Mesh geometry comes from the compiled model (mesh_vert / mesh_face), so no second
// download of the OBJ files; materials reuse the model.json roughness / metallic / transparency like scene.ts.
import * as THREE from "three";
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";
import type { GeomJ, ModelJ } from "../types";

const Z_TO_Y = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
const MJ_SPHERE = 2, MJ_CAPSULE = 3, MJ_ELLIPSOID = 4, MJ_CYLINDER = 5, MJ_BOX = 6, MJ_MESH = 7;

interface Item { gid: number; mesh: THREE.Mesh; env: boolean; collisionOnly: boolean; zAligned: boolean }

export interface Pick { gid: number; body: number; point: THREE.Vector3; name: string }

export class MjRenderer {
  readonly root = new THREE.Group();
  readonly items: Item[] = [];
  readonly bounds = new THREE.Box3();
  private geoms = new Map<number, THREE.BufferGeometry>();
  private mats = new Map<string, THREE.Material>();
  private colMat = new THREE.MeshBasicMaterial({ color: 0xff3366, wireframe: true, transparent: true, opacity: 0.4 });
  private geomNames: string[] = [];
  private byGeomName = new Map<string, GeomJ>();
  showEnv = true;
  showCollision = false;
  private readonly m4 = new THREE.Matrix4();

  constructor(private mj: MainModule, private model: MjModel, private modelJ: ModelJ) {
    this.root.name = modelJ.name + "_mujoco";
    for (const b of modelJ.bodies) for (const g of b.geoms) this.byGeomName.set(g.name, g);
    this.build();
  }

  private build() {
    const { mj, model } = this;
    const G = mj.mjtObj.mjOBJ_GEOM.value, MAT = mj.mjtObj.mjOBJ_MATERIAL.value;
    const type = model.geom_type as ArrayLike<number>, size = model.geom_size as ArrayLike<number>, dataid = model.geom_dataid as ArrayLike<number>;
    const group = model.geom_group as ArrayLike<number>, matid = model.geom_matid as ArrayLike<number>, grgba = model.geom_rgba as ArrayLike<number>, mrgba = model.mat_rgba as ArrayLike<number>;
    for (let g = 0; g < model.ngeom; g++) {
      const name = mj.mj_id2name(model, G, g);
      this.geomNames[g] = name;
      const gj = this.byGeomName.get(name);
      const semantic = gj?.semantic ?? "";
      if (semantic === "floor" || name === "floor") continue;          // the viewport draws its own ground
      const collisionOnly = group[g] === 3 || (gj ? gj.collision && !gj.visual : false);
      const env = semantic === "wall";
      let geo: THREE.BufferGeometry | null = null;
      let zAligned = false;
      const s0 = size[3 * g], s1 = size[3 * g + 1], s2 = size[3 * g + 2];
      switch (type[g]) {
        case MJ_BOX: geo = this.cached(`box:${s0},${s1},${s2}`, () => new THREE.BoxGeometry(2 * s0, 2 * s1, 2 * s2)); break;
        case MJ_SPHERE: geo = this.cached(`sph:${s0}`, () => new THREE.SphereGeometry(s0, 20, 14)); break;
        case MJ_CAPSULE: geo = this.cached(`cap:${s0},${s1}`, () => new THREE.CapsuleGeometry(s0, 2 * s1, 6, 16)); zAligned = true; break;
        case MJ_CYLINDER: geo = this.cached(`cyl:${s0},${s1}`, () => new THREE.CylinderGeometry(s0, s0, 2 * s1, 28)); zAligned = true; break;
        case MJ_ELLIPSOID: geo = this.cached(`ell:${s0},${s1},${s2}`, () => { const e = new THREE.SphereGeometry(1, 20, 14); e.scale(s0, s1, s2); return e; }); break;
        case MJ_MESH: geo = this.meshGeometry(dataid[g]); break;
        default: continue;
      }
      if (!geo) continue;
      let mat: THREE.Material;
      if (collisionOnly) mat = this.colMat;
      else {
        const mi = matid[g];
        const rgba: [number, number, number, number] = mi >= 0 ? [mrgba[4 * mi], mrgba[4 * mi + 1], mrgba[4 * mi + 2], mrgba[4 * mi + 3]] : [grgba[4 * g], grgba[4 * g + 1], grgba[4 * g + 2], grgba[4 * g + 3]];
        const mname = mi >= 0 ? mj.mj_id2name(model, MAT, mi) : `geom:${rgba.join(",")}`;
        mat = this.material(mname, rgba);
      }
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = name;
      mesh.castShadow = !env && !collisionOnly;
      mesh.receiveShadow = true;
      mesh.userData = { gid: g, semantic, label: gj?.part_label ?? name };
      this.items.push({ gid: g, mesh, env, collisionOnly, zAligned });
      this.root.add(mesh);
    }
    this.applyVisibility();
  }

  private cached(key: string, make: () => THREE.BufferGeometry): THREE.BufferGeometry {
    const k = key.length;   // keys are unique strings; map on a hash of the string
    let h = 0; for (let i = 0; i < k; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
    const id = -1 - (h >>> 0);   // negative ids: primitives; non-negative: mesh ids
    let g = this.geoms.get(id);
    if (!g) { g = make(); this.geoms.set(id, g); }
    return g;
  }

  private meshGeometry(meshId: number): THREE.BufferGeometry {
    let g = this.geoms.get(meshId);
    if (g) return g;
    const { model } = this;
    const va = (model.mesh_vertadr as ArrayLike<number>)[meshId], vn = (model.mesh_vertnum as ArrayLike<number>)[meshId];
    const fa = (model.mesh_faceadr as ArrayLike<number>)[meshId], fn = (model.mesh_facenum as ArrayLike<number>)[meshId];
    const verts = (model.mesh_vert as Float32Array).slice(3 * va, 3 * (va + vn));
    const faces = (model.mesh_face as Int32Array).slice(3 * fa, 3 * (fa + fn));
    g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    g.setIndex(new THREE.Uint32BufferAttribute(Uint32Array.from(faces), 1));
    g.computeVertexNormals();
    this.geoms.set(meshId, g);
    return g;
  }

  private material(name: string, rgba: [number, number, number, number]): THREE.Material {
    let m = this.mats.get(name);
    if (m) return m;
    const mj = this.modelJ.materials[name];
    const transparent = (mj?.transparent ?? false) || rgba[3] < 0.99;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: mj?.roughness ?? 0.6, metalness: mj?.metallic ?? 0,
      transparent, opacity: transparent ? Math.min(rgba[3], 0.55) : 1, side: transparent ? THREE.DoubleSide : THREE.FrontSide, depthWrite: !transparent,
    });
    if (mj?.emissive && (mj.emissive[0] || mj.emissive[1] || mj.emissive[2])) mat.emissive = new THREE.Color(mj.emissive[0], mj.emissive[1], mj.emissive[2]);
    this.mats.set(name, mat);
    return mat;
  }

  setVisibility(showEnv: boolean, showCollision: boolean) { this.showEnv = showEnv; this.showCollision = showCollision; this.applyVisibility(); }
  private applyVisibility() { for (const it of this.items) it.mesh.visible = (this.showEnv || !it.env) && (this.showCollision || !it.collisionOnly); }

  /** Copy the geom poses from mjData. */
  update(data: MjData) {
    const xpos = data.geom_xpos as ArrayLike<number>, xmat = data.geom_xmat as ArrayLike<number>;
    for (const it of this.items) {
      const g = it.gid, o = 9 * g;
      it.mesh.position.set(xpos[3 * g], xpos[3 * g + 1], xpos[3 * g + 2]);
      this.m4.set(xmat[o], xmat[o + 1], xmat[o + 2], 0, xmat[o + 3], xmat[o + 4], xmat[o + 5], 0, xmat[o + 6], xmat[o + 7], xmat[o + 8], 0, 0, 0, 0, 1);
      it.mesh.quaternion.setFromRotationMatrix(this.m4);
      if (it.zAligned) it.mesh.quaternion.multiply(Z_TO_Y);
    }
  }

  /** Bounds of the door parts (walls excluded) at the current pose. */
  computeBounds(): THREE.Box3 {
    this.bounds.makeEmpty();
    this.root.updateMatrixWorld(true);
    for (const it of this.items) {
      if (it.env || it.collisionOnly) continue;
      it.mesh.geometry.computeBoundingBox();
      this.bounds.union(it.mesh.geometry.boundingBox!.clone().applyMatrix4(it.mesh.matrixWorld));
    }
    return this.bounds;
  }

  /** Raycast the visible, non-environment geoms of moving bodies. */
  pick(raycaster: THREE.Raycaster): Pick | null {
    const bodyOf = this.model.geom_bodyid as ArrayLike<number>;
    const meshes = this.items.filter((it) => it.mesh.visible && !it.env && bodyOf[it.gid] > 0).map((it) => it.mesh);
    const hits = raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const h = hits[0];
    const gid = (h.object as THREE.Mesh).userData.gid as number;
    return { gid, body: bodyOf[gid], point: h.point.clone(), name: this.geomNames[gid] };
  }

  dispose() {
    for (const g of this.geoms.values()) g.dispose();
    for (const m of this.mats.values()) m.dispose();
    this.colMat.dispose();
    this.root.clear();
  }
}
