import * as THREE from 'three';

/** Place the geometry container at an observed world pose, after its parent. */
export function applyBodyWorld(container:THREE.Object3D,position:THREE.Vector3,quaternion:THREE.Quaternion) {
  container.parent?.updateWorldMatrix(true,false);
  const world=new THREE.Matrix4().compose(position,quaternion,new THREE.Vector3(1,1,1));
  if(container.parent)world.premultiply(container.parent.matrixWorld.clone().invert());
  world.decompose(container.position,container.quaternion,container.scale);container.updateMatrixWorld(true);
}
