"""Conservative clear intervals through a door's actual native aperture.

The synthetic base has no body collider. Joint angle alone cannot establish
passability (notably a bypass panel opening away from the centre). Project
native collision bounds into the doorway band before admitting a standing
0.5 m wide, 1.8 m tall traveller. This is a clearance check, not humanoid gait.
"""
from __future__ import annotations
import numpy as np

SUPPORTED = {'swing_single','swing_double','automatic_swing','cold_storage','pivot',
             'baby_gate','gate_swing','dutch','saloon','sliding_single','sliding_bypass',
             'automatic_sliding','elevator','gate_sliding','bifold','accordion',
             'garage_tiltup','garage_sectional','rollup'}


def _hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) <= 2:
        return points
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def _clip_y(poly, value, above):
    result=[]
    if not poly: return result
    previous=poly[-1]; was=previous[1]>=value if above else previous[1]<=value
    for current in poly:
        inside=current[1]>=value if above else current[1]<=value
        if inside != was:
            a=(value-previous[1])/(current[1]-previous[1])
            result.append((previous[0]+a*(current[0]-previous[0]),value))
        if inside: result.append(current)
        previous,was=current,inside
    return result


class Passage:
    def __init__(self, model, spec, metadata, model_ir=None, width=.5, height=1.8, exclude_bodies=(), body_depth=.6):
        self.m=model; self.enabled=spec['family'] in SUPPORTED
        self.half=spec['opening']['width']/2; self.y=float(metadata.get('wall_y',0))
        self.width=width; self.height=height
        self._time=None;self._result=None;self._q=None
        semantics={g['name']:g.get('semantic') for b in (model_ir or {}).get('bodies',[]) for g in b.get('geoms',[])}
        excluded=set(exclude_bodies)
        self.ids=np.array([i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name not in excluded
                           and (model.geom_contype[i] or model.geom_conaffinity[i])
                           and int(model.geom_type[i]) != 0 and semantics.get(model.geom(i).name) not in ('floor','seal')],dtype=int)
        signs=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        self.local=model.geom_aabb[self.ids,:3,None]+model.geom_aabb[self.ids,3:,None]*signs.T[None,:,:]
        # A curtain or rear slider can sit behind the architectural wall
        # plane. Establish the crossing band from its authored closed stock,
        # including the traveller's finite front/back extent. A narrow wall
        # slice otherwise labels an entirely closed offset door as passable.
        import mujoco
        rest=mujoco.MjData(model)
        mujoco.mj_kinematics(model,rest)
        corners=self._corners(rest)
        half_wall=max(.05,spec['opening'].get('wall_thickness',.1)/2)
        near,far=self.y-half_wall,self.y+half_wall
        for gid,xyz in zip(self.ids,corners):
            if model.geom_bodyid[gid]==0:continue
            if xyz[2].max()<=.05 or xyz[2].min()>=height:continue
            if xyz[0].max()<=-self.half or xyz[0].min()>=self.half:continue
            near=min(near,float(xyz[1].min()));far=max(far,float(xyz[1].max()))
        self.near=near-.5*body_depth
        self.far=far+.5*body_depth

    def _corners(self,data):
        matrices=data.geom_xmat[self.ids].reshape(-1,3,3)
        return np.einsum('gij,gjk->gik',matrices,self.local)+data.geom_xpos[self.ids,:,None]

    def intervals(self,data):
        if not self.enabled: return None
        if self._time is not None:
            elapsed=float(data.time)-self._time
            if elapsed==0 and np.array_equal(data.qpos,self._q): return self._result
            # Delaying a newly clear declaration is conservative. Never cache
            # a positive opening across physics steps while a leaf can close.
            if 0<elapsed<.02 and not self._result: return self._result
        blocked=[]
        corners=self._corners(data)
        for xyz in corners:
            # Below-floor sills do not remove the whole human opening. Raised
            # obstacles and low overhead leaves remain conservatively blocked.
            if xyz[2].max()<=.05 or xyz[2].min()>=self.height: continue
            if xyz[1].max()<self.near or xyz[1].min()>self.far: continue
            poly=_clip_y(_clip_y(_hull(xyz[:2].T),self.near,True),self.far,False)
            if poly:
                lo=max(-self.half,min(p[0] for p in poly));hi=min(self.half,max(p[0] for p in poly))
                if hi>lo: blocked.append((lo,hi))
        free=[];edge=-self.half
        for lo,hi in sorted(blocked):
            if lo-edge>=self.width+.01: free.append((edge+self.width/2+.005,lo-self.width/2-.005))
            edge=max(edge,hi)
        if self.half-edge>=self.width+.01: free.append((edge+self.width/2+.005,self.half-self.width/2-.005))
        self._time=float(data.time);self._result=free;self._q=data.qpos.copy()
        return free
