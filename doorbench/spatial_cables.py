"""Read-only native cable routing for recording and exact inspection rendering."""
from __future__ import annotations
import numpy as np


def native_cable_paths(model, data, cable_names):
    """Return actual MuJoCo tangent points and pulley geometry in world metres.

    MuJoCo exposes paired storage arrays: flatten wrap_xpos to (-1,3) and
    wrap_obj to (-1,) BEFORE slicing at ten_wrapadr/ten_wrapnum. Consecutive
    nodes naming one pulley are its entry/exit tangent points, not its centre.
    A renderer must connect that pair with the indicated circular surface arc;
    the side point selects the wrapping branch, including an ambiguous180°.
    Caller forwards the state first. This function never advances or edits it.
    """
    positions=np.asarray(data.wrap_xpos).reshape(-1,3)
    objects=np.asarray(data.wrap_obj).reshape(-1)
    cables=[]
    for name in cable_names:
        tid=model.tendon(name).id
        start=int(data.ten_wrapadr[tid]);count=int(data.ten_wrapnum[tid])
        nodes=[];geometries={}
        path_start=int(model.tendon_adr[tid]);path_count=int(model.tendon_num[tid])
        side_sites={int(model.wrap_objid[i]):int(model.wrap_prm[i])
                    for i in range(path_start,path_start+path_count) if int(model.wrap_type[i]) in (4,5)}
        for index in range(start,start+count):
            gid=int(objects[index]);geom_name=model.geom(gid).name if gid>=0 else None
            nodes.append({'point':positions[index].tolist(),'geom_name':geom_name})
            if gid>=0 and geom_name not in geometries:
                kind='cylinder' if int(model.geom_type[gid])==5 else 'sphere'
                side=side_sites.get(gid,-1)
                geometries[geom_name]={'geom_id':gid,'kind':kind,
                    'position':np.asarray(data.geom_xpos[gid]).tolist(),
                    'axis':np.asarray(data.geom_xmat[gid]).reshape(3,3)[:,2].tolist(),
                    'radius':float(model.geom_size[gid,0]),
                    'side_point':np.asarray(data.site_xpos[side]).tolist() if side>=0 else None}
        cables.append({'name':name,'length_m':float(data.ten_length[tid]),
            'max_length_m':float(model.tendon_range[tid,1]),'nodes':nodes,'wrap_geometries':geometries})
    return {'schema_version':1,'units':'metres','frame':'world','cables':cables}
