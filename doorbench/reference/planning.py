"""Scene-aware contact and route proposals for the constrained reference solver.

These are proposals, not certificates: a disk swept through projected collision
geometry is deliberately conservative. The whole-body solver and independent
3-D audit decide whether an exported motion is acceptable.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Point, LineString, MultiPoint, box
from shapely.ops import unary_union
from shapely.prepared import prep


def heading(direction):
    """Z rotation from the rig's +Y-forward rest pose."""
    d = np.asarray(direction)
    return float(math.atan2(-d[0], d[1]))


def smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10+t*(-15+6*t))


@dataclass(frozen=True)
class Stance:
    xy: np.ndarray
    yaw: float
    pelvis_height: float
    hand: str
    clearance: float


class NoRoute(RuntimeError):
    """No proposal was found; this is not a mathematical infeasibility proof."""


class SceneNavigator:
    """Private native MuJoCo state and conservative standing-body route search."""

    def __init__(self, door_dir, radius=.23, height=1.72, resolution=.09):
        import mujoco
        self.mj = mujoco
        self.directory = Path(door_dir)
        self.model = mujoco.MjModel.from_xml_path(str(self.directory/'door.xml'))
        self.data = mujoco.MjData(self.model)
        self.radius = float(radius)
        self.height = float(height)
        self.resolution = float(resolution)
        self.local = {}
        self.floor_bounds = None
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                continue
            if name == 'floor':
                p=self.model.geom_pos[g]; s=self.model.geom_size[g]
                self.floor_bounds=(p[0]-s[0]+radius,p[1]-s[1]+radius,
                                   p[0]+s[0]-radius,p[1]+s[1]-radius)
                continue
            typ=int(self.model.geom_type[g]); s=self.model.geom_size[g]
            if typ == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=int(self.model.geom_dataid[g]); a=int(self.model.mesh_vertadr[mid]); n=int(self.model.mesh_vertnum[mid])
                vertices=self.model.mesh_vert[a:a+n]
                lo,hi=vertices.min(axis=0),vertices.max(axis=0)
            else:
                if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE): half=np.repeat(s[0],3)
                elif typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE): half=np.array([s[0],s[0],s[0]+s[1]])
                elif typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER): half=np.array([s[0],s[0],s[1]])
                else: half=s
                lo,hi=-half,half
            self.local[g]=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        self.update(self.data.qpos.copy())

    def update(self, native_qpos, *, height=None):
        q=np.asarray(native_qpos,float)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all(): raise ValueError('Invalid native qpos')
        self.data.qpos[:]=q
        self.mj.mj_kinematics(self.model,self.data)
        h=self.height if height is None else float(height)
        obstacles=[]
        for g,corners in self.local.items():
            xyz=corners @ self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
            if xyz[:,2].min()>h or xyz[:,2].max()<.10: continue
            hull=MultiPoint(xyz[:,:2]).convex_hull
            if not hull.is_empty: obstacles.append(hull)
        self.obstacles=unary_union(obstacles)
        self.blocked=self.obstacles.buffer(self.radius,quad_segs=6)
        self.prepared=prep(self.blocked)

    def clear(self, xy):
        xy=np.asarray(xy,float)
        if self.floor_bounds:
            l,b,r,t=self.floor_bounds
            if not (l<=xy[0]<=r and b<=xy[1]<=t): return False
        return not self.prepared.intersects(Point(xy[:2]))

    def segment_clear(self,a,b):
        return self.clear(a) and self.clear(b) and not self.blocked.intersects(LineString([a,b]))

    def stance(self, target, previous_xy, *, approach_side=-1, preferred_hand=None,
               previous_yaw=None,previous_height=.94):
        """Prefer upright reachable stances with room for the finite feet.

        The reach shell uses this rig's shoulder height and arm lengths. It is
        only a proposal filter; swept feet, body clearance and contact still
        require the whole-body solve and independent trajectory validation.
        """
        target=np.asarray(target,float); previous_xy=np.asarray(previous_xy,float)[:2]
        if (previous_yaw is not None and preferred_hand and self.clear(previous_xy)
                and self.obstacles.distance(Point(previous_xy)) >= .30):
            right=np.array([math.cos(previous_yaw),math.sin(previous_yaw)])
            shoulder=np.r_[previous_xy+right*(.18 if preferred_hand=='right_hand' else -.18),previous_height+.41]
            distance=np.linalg.norm(target-shoulder)
            if .20<distance<.535:
                return Stance(previous_xy.copy(),previous_yaw,previous_height,preferred_hand,
                              float(self.obstacles.distance(Point(previous_xy))))
        candidates=[]
        preferred=heading(target[:2]-previous_xy)
        # Both arms and several directions matter for opposite hinge handedness.
        for distance in [.28,.34,.40,.46]:
            for angle in np.linspace(-math.pi,math.pi,32,endpoint=False):
                facing=np.array([-math.sin(angle),math.cos(angle)])
                right=np.array([math.cos(angle),math.sin(angle)])
                for hand,sign in [('right_hand',1),('left_hand',-1)]:
                    xy=target[:2]-facing*distance-right*sign*.15
                    clearance=float(self.obstacles.distance(Point(xy)))
                    if not self.clear(xy) or clearance < .30: continue
                    # Avoid switching to the inaccessible other face through a slab.
                    facing_delta=abs(math.atan2(math.sin(angle-preferred),math.cos(angle-preferred)))
                    side_cost=.8 if approach_side*xy[1]<-.1 else 0.
                    switch_cost=.45 if preferred_hand and hand != preferred_hand else 0.
                    for pelvis in sorted({.94,float(np.clip(target[2]+.05,.43,.94))},reverse=True):
                        shoulder=np.r_[xy+right*sign*.18,pelvis+.41]
                        reach=np.linalg.norm(target-shoulder)
                        if not .20<reach<.545: continue
                        cost=(np.linalg.norm(xy-previous_xy)+.12*facing_delta+side_cost
                              +.05*(hand=='left_hand')+switch_cost+1.5*(.94-pelvis))
                        candidates.append((cost,Stance(xy,angle,pelvis,hand,clearance)))
        if not candidates: raise NoRoute('No upright reachable stance candidate at this hardware target')
        return min(candidates,key=lambda x:x[0])[1]

    def route(self, start, goal):
        """A* followed by collision-checked visibility simplification."""
        start=np.asarray(start,float)[:2]; goal=np.asarray(goal,float)[:2]
        if not self.clear(start): raise NoRoute('Start footprint intersects scene geometry')
        if not self.clear(goal): raise NoRoute('Goal footprint intersects scene geometry')
        if self.segment_clear(start,goal): return np.array([start,goal])
        step=self.resolution
        lo=np.minimum(start,goal)-1.5; hi=np.maximum(start,goal)+1.5
        if self.floor_bounds:
            lo=np.maximum(lo,self.floor_bounds[:2]);hi=np.minimum(hi,self.floor_bounds[2:])
        shape=np.ceil((hi-lo)/step).astype(int)+1
        def point(i): return lo+np.asarray(i)*step
        def nearby(x):
            center=np.rint((x-lo)/step).astype(int)
            candidates=[]
            for dx in range(-2,3):
                for dy in range(-2,3):
                    v=tuple(center+[dx,dy]);p=point(v)
                    if all(0<=v[k]<shape[k] for k in (0,1)) and self.segment_clear(x,p):
                        candidates.append((np.linalg.norm(p-x),v))
            if not candidates: raise NoRoute('No connected route grid node')
            return min(candidates)[1]
        source,sink=nearby(start),nearby(goal)
        queue=[(0.,source)]; costs={source:0.};parents={}; closed=set()
        shifts=[(x,y) for x in (-1,0,1) for y in (-1,0,1) if x or y]
        while queue:
            _,cur=heapq.heappop(queue)
            if cur in closed: continue
            if cur==sink: break
            closed.add(cur)
            for dx,dy in shifts:
                nxt=(cur[0]+dx,cur[1]+dy)
                if not all(0<=nxt[k]<shape[k] for k in (0,1)) or nxt in closed: continue
                if not self.segment_clear(point(cur),point(nxt)): continue
                cost=costs[cur]+math.hypot(dx,dy)*step
                if cost<costs.get(nxt,math.inf):
                    costs[nxt]=cost;parents[nxt]=cur
                    heapq.heappush(queue,(cost+np.linalg.norm(point(nxt)-point(sink)),nxt))
        if sink not in costs: raise NoRoute('No path within the bounded search region')
        route=[goal,point(sink)];cur=sink
        while cur!=source:
            cur=parents[cur];route.append(point(cur))
        route.append(start);route=np.array(route[::-1])
        clean=[route[0]];i=0
        while i<len(route)-1:
            j=len(route)-1
            while j>i+1 and not self.segment_clear(route[i],route[j]):j-=1
            clean.append(route[j]);i=j
        return np.array(clean)

    def site(self,name):
        sid=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_SITE,name)
        if sid<0: raise KeyError(name)
        return self.data.site_xpos[sid].copy()

    def passage_route(self,start,goal,pass_plane):
        """Require crossing inside the declared aperture, never around the wall."""
        normal=np.asarray(pass_plane['normal'],float)
        if abs(normal[1])<.99:raise NoRoute('Non-vertical passage needs a separate climbing/crawling planner')
        center=np.asarray(pass_plane['center'],float)
        half=float(pass_plane['width'])/2-self.radius
        if half<=0:raise NoRoute('Declared aperture is narrower than the standing-body route proxy')
        choices=[]
        for x in np.linspace(center[0]-half,center[0]+half,13):
            a=np.array([x,center[1]-.30]);b=np.array([x,center[1]+.30])
            if not self.segment_clear(a,b):continue
            try:
                before=self.route(start,a);after=self.route(b,goal)
            except NoRoute:continue
            route=np.concatenate([before,after])
            # Reject any auxiliary crossing outside this declared passage.
            valid=True
            for p,q in zip(route,route[1:]):
                if (p[1]-center[1])*(q[1]-center[1])<0:
                    s=(center[1]-p[1])/(q[1]-p[1]);cross=p[0]+s*(q[0]-p[0])
                    if abs(cross-center[0])>half:valid=False
            if valid:
                clean=[route[0]];i=0
                while i<len(route)-1:
                    j=len(route)-1
                    while j>i+1:
                        p,q=route[i],route[j];cross_ok=True
                        if (p[1]-center[1])*(q[1]-center[1])<0:
                            s=(center[1]-p[1])/(q[1]-p[1]);cross_ok=abs(p[0]+s*(q[0]-p[0])-center[0])<=half
                        if cross_ok and self.segment_clear(p,q):break
                        j-=1
                    clean.append(route[j]);i=j
                route=np.array(clean)
                choices.append((np.linalg.norm(np.diff(route,axis=0),axis=1).sum(),route))
        if not choices:raise NoRoute('No standing-body route through the declared aperture at this door pose')
        return min(choices,key=lambda v:v[0])[1]
