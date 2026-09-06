"""Original ordinary-door glazing sections shared by geometry and mass.

Dimensions are metres. Stock uses the existing slab's effective material
density; panes, retaining stops and glazing elastomer have distinct volumes.
This is a geometric construction, not a fire/impact/thermal certification.
"""
from __future__ import annotations

import math
from . import materials as M
from .panels import glazing_layout
from .construction_dimensions import MULTIPOINT_CASE_DEPTH_M, MULTIPOINT_GLAZING_STOCK_WEB_M

STYLES = frozenset(("glass_full", "glass_half", "glass_15_lite", "glass_10_lite",
    "glass_6_lite", "glass_9_lite", "glass_1_lite_top", "glass_oval", "glass_fan",
    "steel_half_glass", "glass_vision", "steel_vision", "porthole",
    "sectional_long_windows", "glass_sidelite_style"))


def uses_ordinary_glazing(leaf):
    slab = M.SLABS[leaf['slab']]
    return bool(leaf.get('glazing') and leaf['glazing'].get('panel_style') in STYLES
        and leaf['slab'] not in M.FRAMED_GLASS_SLABS
        and leaf.get('panel_style') != 'glass_frameless'
        and not (slab.monolithic and slab.core_material in ('glass_clear', 'mirror')))


def area(poly):
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly, poly[1:]+poly[:1])))/2


def rectangle(x, z, w, h):
    return [(x,z),(x+w,z),(x+w,z+h),(x,z+h)]


def _clip(poly, a, b, inside):
    def distance(p):
        return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
    out=[]
    for p,q in zip(poly,poly[1:]+poly[:1]):
        dp,dq=distance(p),distance(q)
        keep_p=(dp>=0) if inside else (dp<=0)
        keep_q=(dq>=0) if inside else (dq<=0)
        if keep_p:out.append(p)
        if keep_p != keep_q:
            f=dp/(dp-dq);out.append((p[0]+f*(q[0]-p[0]),p[1]+f*(q[1]-p[1])))
    clean=[]
    for p in out:
        if not clean or math.dist(p,clean[-1])>1e-10:clean.append(p)
    if len(clean)>1 and math.dist(clean[0],clean[-1])<1e-10:clean.pop()
    return clean if len(clean)>=3 and area(clean)>1e-12 else []


def subtract(poly, cutter):
    """Nonoverlapping convex pieces of a convex polygon minus a convex hole."""
    remaining=poly;pieces=[]
    for a,b in zip(cutter,cutter[1:]+cutter[:1]):
        if not remaining:break
        outside=_clip(remaining,a,b,False)
        if outside:pieces.append(outside)
        remaining=_clip(remaining,a,b,True)
    return pieces


def _contour(style, bounds, inset=0.):
    x,z,w,h=bounds;x+=inset;z+=inset;w-=2*inset;h-=2*inset
    if min(w,h)<=0:raise ValueError('Glazing contour has no usable area')
    if style not in ('glass_oval','glass_fan','porthole'):return rectangle(x,z,w,h)
    if style=='glass_fan':
        # A true half-elliptical fan light, including its flat lower edge.
        return [(x+w/2+w/2*math.cos(i*math.pi/32),z+h*math.sin(i*math.pi/32)) for i in range(33)]
    return [(x+w/2+w/2*math.cos(i*math.tau/64),z+h/2+h/2*math.sin(i*math.tau/64)) for i in range(64)]


def construction(leaf, width=None, height=None, *, spec=None):
    """Return exact convex component prisms, mass and visible apertures.

    Existing layout rectangles specify cut-glass bounds. Ordinary rectangles
    remain rectangles; oval, fan and porthole profiles use convex faceted
    contours, never a convex hull across the surrounding stock's opening.
    """
    if not uses_ordinary_glazing(leaf):raise ValueError('Not an ordinary glazed leaf')
    W=leaf['width'] if width is None else width
    H=leaf['height'] if height is None else height
    t=leaf['thickness'];glazing=leaf['glazing'];gt=glazing.get('thickness',.006)
    if not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in (W,H,t,gt)):
        raise ValueError('Glazing dimensions must be finite positive numbers')
    slab=M.SLABS[leaf['slab']];style=glazing['panel_style']
    # Solid timber stops for wood products; formed 1.2 mm metal channels for
    # metal/composite shells. Separate actual-density BOM, not solid steel fill.
    timber=not slab.skin_material.startswith('steel') and slab.skin_material not in ('stainless','aluminum','fiberglass_frp','pvc')
    if timber and gt+.004>=t:raise ValueError('Timber glazing needs space for opposed retaining stops')
    stop_material=slab.core_material if slab.monolithic and timber else ('pine' if timber else 'aluminum')
    stop_depth=t if timber else max(t,gt+.0068)
    density=slab.area_density(t)/t
    parts=[];panes=[];roughs=[]
    layouts=glazing_layout(style,W,H)
    adjustments=[]
    multipoint=bool(spec and spec.get('lock',{}).get('model')=='multipoint')
    reserve=MULTIPOINT_CASE_DEPTH_M+MULTIPOINT_GLAZING_STOCK_WEB_M if multipoint else .127
    if (style not in ('glass_vision','steel_vision','porthole','sectional_long_windows') or multipoint) and layouts:
        left=min(r[0]for r in layouts);right=max(r[0]+r[2]for r in layouts)
        new_left=max(left,.129);new_right=min(right,W-reserve-.002)
        if new_left!=left or new_right!=right:
            if new_right-new_left<.10:raise ValueError('Glazed leaf is too narrow for supported 127 mm stiles')
            scale=(new_right-new_left)/(right-left)
            layouts=[(new_left+(x-left)*scale,z,w*scale,h)for x,z,w,h in layouts]
            adjustments.append('Minimum 127 mm supporting edge stiles outside the 2 mm glazing seat')
            if multipoint:adjustments.append('Latch-side stock reserves the multipoint mortise plus a 10 mm supporting web')
    if leaf.get('pet_flap') and layouts:
        bottom=min(r[1]for r in layouts);top=max(r[1]+r[3]for r in layouts)
        raised=max(bottom,leaf['pet_flap']['height']+.078)
        if raised>bottom:
            if top-raised<.10:raise ValueError('Pet opening leaves no supported glazed region')
            scale=(top-raised)/(top-bottom)
            layouts=[(x,raised+(z-bottom)*scale,w,h*scale)for x,z,w,h in layouts]
            adjustments.append('Raised glazing over a continuous solid pet-opening rail')
    def part(name,polygon,y0,y1,material,rho,semantic):
        if y1-y0<=1e-10 or area(polygon)<=1e-12:return
        parts.append({'name':name,'polygon_xz':[list(p) for p in polygon],'y_bounds':[y0,y1],
            'material':material,'density':rho,'semantic':semantic,
            'volume_m3':area(polygon)*(y1-y0),'mass_kg':area(polygon)*(y1-y0)*rho})
    def ring(name,outer,inner,y0,y1,material,rho,semantic):
        for i,p in enumerate(subtract(outer,inner)):part(f'{name}_{i}',p,y0,y1,material,rho,semantic)
    for k,bounds in enumerate(layouts):
        x,z,w,h=bounds
        if min(w,h)<=.02:raise ValueError('Glazing layout is too small for a supported pane')
        outer=_contour(style,bounds,-.002);pane=_contour(style,bounds);visible=_contour(style,bounds,.008)
        if min(p[0] for p in outer)<=0 or max(p[0] for p in outer)>=W or min(p[1] for p in outer)<=0 or max(p[1] for p in outer)>=H:
            raise ValueError('Glazing opening removes a perimeter stile or rail')
        # Rectangular stock outside the bounding opening remains box stock for
        # later real lock/spindle mortises. Curved corner infill is convex pieces.
        rough=rectangle(x-.002,z-.002,w+.004,h+.004)
        roughs.append((x-.002,z-.002,w+.004,h+.004))
        for j,p in enumerate(subtract(rough,outer)):
            part(f'slab_curve_{k}_{j}',p,-t/2,t/2,'stock',density,'leaf')
        part(f'glass_{k}',pane,-gt/2,gt/2,glazing['material'],M.MATERIALS[glazing['material']].density,'glass')
        inner_web=_contour(style,bounds,-.0008)
        edge_outer=outer if timber else inner_web
        ring(f'glazing_edge_{k}',edge_outer,pane,-gt/2,gt/2,'rubber',M.MATERIALS['rubber'].density,'glazing_retainer')
        if not timber:
            # Continuous formed casing is supported by the complete slab
            # thickness, even when its thicker pane sits proud of thin steel.
            ring(f'glazing_web_{k}',outer,inner_web,-stop_depth/2,stop_depth/2,
                stop_material,M.MATERIALS[stop_material].density,'glazing_retainer')
        for side in (-1,1):
            tape=sorted((side*gt/2,side*(gt/2+.001)))
            ring(f'glazing_tape_{k}_{side}',pane,visible,*tape,'rubber',M.MATERIALS['rubber'].density,'glazing_retainer')
            start,end=sorted((side*(gt/2+.001),side*stop_depth/2))
            rho=M.MATERIALS[stop_material].density
            if timber:
                ring(f'glazing_stop_{k}_{side}',outer,visible,start,end,stop_material,rho,'glazing_retainer')
            else:
                lip=sorted((side*(gt/2+.001),side*(gt/2+.0022)))
                ring(f'glazing_lip_{k}_{side}',inner_web,visible,*lip,stop_material,rho,'glazing_retainer')
        panes.append({'index':k,'glass_name':f'glass_{k}','glass_thickness_m':gt,
            'glass_polygon_xz':[list(p)for p in pane],'rough_polygon_xz':[list(p)for p in outer],
            'visible_polygon_xz':[list(p)for p in visible],'cut_glass_bounds_m':list(bounds)})
    # Full-height perimeter stiles remain single boxes, rather than being
    # fragmented by unrelated pane heights. Interior rails/muntins are exact
    # nonoverlapping remaining material, not decorative bars over glass.
    xs=sorted({0.,W,*[r[0]for r in roughs],*[r[0]+r[2]for r in roughs]})
    stock=[]
    for a,b in zip(xs,xs[1:]):
        mid=(a+b)/2;occupied=sorted((z,z+h)for x,z,w,h in roughs if x<mid<x+w)
        cursor=0.
        for c,d in occupied:
            if c>cursor+1e-10:stock.append(rectangle(a,cursor,b-a,c-cursor))
            cursor=max(cursor,d)
        if cursor<H-1e-10:stock.append(rectangle(a,cursor,b-a,H-cursor))
    for k,p in enumerate(stock):part(f'slab_{k}',p,-t/2,t/2,'stock',density,'leaf')
    masses={key:sum(p['mass_kg'] for p in parts if p['semantic']==semantic)
        for key,semantic in [('slab_kg','leaf'),('glass_kg','glass'),('retainer_kg','glazing_retainer')]}
    return {'schema':'doorbench.ordinary-glazing.v1','width_m':W,'height_m':H,'thickness_m':t,
        'style':style,'panes':panes,'parts':parts,**masses,'total_kg':sum(masses.values()),
        'glass_edge_clearance_m':.002,'face_tape_m':.001,'stop_bite_m':.008,
        'stop_material':stop_material,'stock_density_kg_m3':density,
        'retainer_depth_m':stop_depth,'layout_adjustments':adjustments,
        'hardware_edge_reserve_m':reserve if multipoint else None,
        'scope':'Real openings and fixed supported panes, original simplified stops. Stock retains the catalogue effective slab density; no core-cell, fastener-strength, glass-breakage, fire or impact certification.'}
