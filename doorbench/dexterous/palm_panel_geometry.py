"""Geometric palm flattening that preserves the actual supporting plane.

The input cloud contains original collision vertices in palm-site coordinates.
No forces or physical contact qualification are inferred from this operation.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def flatten_palm_goal(position,rotation,vertices,fraction):
    p=np.asarray(position,float);r=np.asarray(rotation,float);v=np.asarray(vertices,float)
    if (p.shape!=(3,) or r.shape!=(3,3) or v.ndim!=2 or v.shape[1]!=3 or len(v)<4
            or not np.isfinite(np.r_[p,r.ravel(),v.ravel(),fraction]).all()
            or not np.allclose(r.T@r,np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(r),1.,atol=1e-8) or not 0<=fraction<=1):
        raise ValueError('Require finite original palm geometry and a proper rotation')
    z=r[:,2];target=np.array([0.,-1.,0.]);cross=np.cross(z,target)
    length=np.linalg.norm(cross);angle=np.arctan2(length,z@target)
    if length<1e-10:
        if z@target<0:raise ValueError('Antiparallel palm requires a declared rotation axis')
        result=r.copy()
    else:result=Rotation.from_rotvec(cross/length*angle*fraction).as_matrix()@r
    shift=float(np.max(v@r[1,:])-np.max(v@result[1,:]))
    result_p=p.copy();result_p[1]+=shift
    return result_p,result,shift


def original_palm_vertices(model,data,site):
    """Read original convex palm collision meshes in the measured site frame."""
    import mujoco
    cloud=[];body=model.site_bodyid[site];rotation=data.site_xmat[site].reshape(3,3)
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=body or not model.geom_contype[g]:continue
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('This screen requires the original mesh-only palm body')
        mesh=model.geom_dataid[g];begin=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        world=model.mesh_vert[begin:begin+count]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        cloud.append((world-data.site_xpos[site])@rotation)
    if not cloud:raise ValueError('Missing original palm collision vertices')
    return np.vstack(cloud)
