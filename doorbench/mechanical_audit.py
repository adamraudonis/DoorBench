"""Independent inventory audit of mass scope, physical panel area and operators.

This does not replace collision/dynamic QA or certify material constructions.
It never uses physics.leaf_mass as its expected-mass oracle. Collision envelopes
are projected once per physical panel body; overlapping proxies are unioned.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import materials as M
from . import hardware as H
from .ir import quat_to_mat

FRAMED_GLASSES = frozenset({'storefront_alu', 'storefront_alu_igu', 'patio_slider_glass'})


def rectangle_union_area(rectangles):
    """Exact area of axis-aligned rectangles, including duplicate collision layers."""
    rectangles = [tuple(map(float, r)) for r in rectangles if r[2] > r[0] and r[3] > r[1]]
    xs = sorted({v for r in rectangles for v in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rectangles if r[0] < right and r[2] > left)
        length, end = 0.0, -math.inf
        for lo, hi in intervals:
            length += max(0.0, hi-max(lo, end))
            end = max(end, hi)
        total += (right-left)*length
    return total


def _box_rect(geom, axes):
    p=np.asarray(geom['pos'], dtype=float)
    h=np.abs(quat_to_mat(geom['quat'])) @ np.asarray(geom['size'], dtype=float)
    a,b=axes
    return [p[a]-h[a],p[b]-h[b],p[a]+h[a],p[b]+h[b]]


def _stock_area_density(spec):
    """Same published material catalogue, independently applied to measured area."""
    leaf=spec['leaf'];slab=M.SLABS[leaf['slab']];t=float(leaf['thickness'])
    if leaf['slab']=='chain_link_gate' and leaf.get('infill_thickness') is not None:
        w,h=float(leaf['width']),float(leaf['height'])
        r,wall=t/2,.0016
        line=math.pi*(r*r-(r-wall)**2)*M.MATERIALS['steel_galvanized'].density
        return (2*(w+h)*line+2.4*max(0,w-2*t)*max(0,h-2*t))/(w*h)
    glazing=leaf.get('glazing') or {}
    f=float(glazing.get('area_fraction',0))
    glass=M.MATERIALS[glazing.get('material','glass_clear')].density*float(glazing.get('thickness',.006))
    return slab.area_density(t)*(1-f)+f*glass


def material_inventory(spec, model):
    """Return actual panel area and material estimate with explicit model limits."""
    leaf=spec['leaf'];fam=spec['family'];w,h,t=(float(leaf[k]) for k in ('width','height','thickness'))
    bodies=[b for b in model['bodies'] if b['semantic']=='leaf' and not b['static']]
    panels=[];limitations=[]
    if fam in ('turnstile_tripod','turnstile_fullheight'):
        # The authored arm tubes are represented by solid collision capsules.
        # Use the declared hollow tube wall, not the capsules' solid volume.
        if fam=='turnstile_tripod':
            arms=sum(g['name'].startswith('arm_') and g['name'].endswith('_col') for b in bodies for g in b['geoms'])
            lengths=[.5]*arms
            hub=3.0
            formula='sum(arm lengths)*pi*(0.019^2-0.0175^2)*7900 + 3 kg hub'
        else:
            lengths=[2*float(g['size'][1]) for b in bodies for g in b['geoms'] if re.fullmatch(r'wing_\d+_arm_\d+',g['name'])]
            wings=len({g['name'].split('_')[1] for b in bodies for g in b['geoms'] if re.fullmatch(r'wing_\d+_arm_\d+',g['name'])})
            hub=4.0*wings
            formula='sum(arm lengths)*pi*(0.019^2-0.0175^2)*7900 + 4 kg per wing column allocation'
        mass=sum(lengths)*math.pi*(.019**2-.0175**2)*7900+hub
        limitations.append('Hollow 1.5 mm tube wall and hub/column allocation are authored parameters, not verified from collision solids.')
        return {'mode':'hollow_tube_rotor','leaf_body_count':len(bodies),'material_panel_count':len(lengths),'area_m2':None,'nominal_unit_area_m2':w*h,'area_multiple':None,'stock_material_mass_kg':mass,'glass_volume_m3':0.0,'panels':[],'formula':formula,'limitations':limitations}
    if fam=='revolving':
        for b in bodies:
            for g in b['geoms']:
                if re.fullmatch(r'wing_\d+_glass',g['name']):
                    x,y,z=map(float,g['size'])
                    panels.append({'body':b['name'],'part':g['name'],'area_m2':4*x*z,'width_m':2*x,'height_m':2*z,'thickness_m':2*y})
        density=2500*t
        limitations.append('Glass-only lower bound; actual rotor shaft, wing rails, stiles and push bars add mass.')
        formula='sum(actual wing glass box volumes)*2500 kg/m^3; frame/hardware excluded'
    else:
        axes=(0,1) if fam in ('hatch_floor','hatch_ceiling') else (0,2)
        for b in bodies:
            geoms=[g for g in b['geoms'] if g['type']=='box' and g['collision'] and g['semantic'] in ('leaf','glass')]
            rectangles=[_box_rect(g,axes) for g in geoms]
            if not rectangles:
                limitations.append(f"{b['name']}: no planar box envelope; inspect non-box construction separately")
                continue
            area=rectangle_union_area(rectangles)
            low=np.min(np.asarray(rectangles)[:,:2],axis=0);high=np.max(np.asarray(rectangles)[:,2:],axis=0)
            panels.append({'body':b['name'],'area_m2':area,'width_m':float(high[0]-low[0]),'height_m':float(high[1]-low[1]),'thickness_m':t})
        density=1250*t if fam=='strip_curtain' else _stock_area_density(spec)
        formula='sum(per-body union of slab/glass collision-box projected areas) * catalogue construction kg/m^2'
        limitations.append('Stock-area estimate: decorative overlays, small mortise depth, cutout material changes and hardware masses require a separate bill of materials.')
    area=sum(p['area_m2'] for p in panels)
    glass_volume=sum(8*math.prod(g['size']) for b in bodies for g in b['geoms'] if g['type']=='box' and g['visual'] and g['semantic']=='glass')
    material_mass=area*density
    if leaf['slab'] in FRAMED_GLASSES:
        if model.get('meta',{}).get('framed_glass_constructions'):
            # Independently integrate actual visible primitive material. Ignore
            # mass_override and invisible collision proxies/calibrated budgets.
            material_mass=sum(8*math.prod(g['size'])*g['density'] for b in bodies for g in b['geoms']
                              if g['type']=='box' and g['visual'] and g['semantic'] in ('leaf','glass','seal') and g['density']>100)
            formula='sum(actual visible frame/pane/gasket primitive volumes * material density), ignoring mass overrides'
        else:
            material_mass=glass_volume*2500
            formula='legacy solid-glass geometry volume *2500 kg/m^3; frame/hardware excluded'
    return {'mode':'planar_stock_area','leaf_body_count':len(bodies),'material_panel_count':len(panels),'area_m2':area,'nominal_unit_area_m2':w*h,'area_multiple':area/(w*h),'stock_material_mass_kg':material_mass,'glass_volume_m3':glass_volume,'panels':panels,'formula':formula,'limitations':limitations}


def audit_model(spec, model):
    """JSON-safe findings. Every reported issue includes its scope and evidence."""
    inventory=material_inventory(spec,model)
    moving=[b for b in model['bodies'] if not b['static']]
    mass=sum(float(b['mass']) for b in moving)
    expected=inventory['stock_material_mass_kg']
    issues=[]
    if expected>0 and mass<.90*expected:
        issues.append({'code':'moving_mass_below_material_estimate','actual_kg':mass,'material_estimate_kg':expected,'ratio':mass/expected})
    dimensions=[v for p in inventory['panels'] for v in (p['width_m'],p['height_m'],p['thickness_m'])]
    if any(not math.isfinite(v) or v<=0 for v in dimensions):
        issues.append({'code':'invalid_panel_dimensions'})
    if spec['leaf']['slab'] in FRAMED_GLASSES:
        suspicious=[g['name'] for b in moving if b['semantic']=='leaf' for g in b['geoms'] if g['type']=='box' and g['semantic']=='glass' and g['collision'] and abs(2*g['size'][1]-spec['leaf']['thickness'])<1e-6 and spec['leaf']['thickness']>.025]
        if suspicious:
            issues.append({'code':'framed_glass_uses_frame_depth_as_solid_glass','slab':spec['leaf']['slab'],'frame_depth_m':spec['leaf']['thickness'],'glass_geoms':suspicious})
    op_geoms=[{'body':b['name'],'geom':g['name'],'collision':g['collision']} for b in model['bodies'] for g in b['geoms'] if g['semantic']=='operator']
    contacts=[{'body':b['name'],'site':s['name'],'role':s['role'],'static':b['static']} for b in model['bodies'] for s in b['sites'] if s['role'] in ('grip','push')]
    op_joints=[b['joint']['name'] for b in model['bodies'] if b.get('joint') and b['joint']['role']=='operator']
    op=spec['operator']['model']
    kind=H.OPERATORS[op].kind
    # A hasp is catalogued as an operator but realized by lock geometry; an
    # elevator_none entry deliberately has no manual hardware or grip sites.
    if kind=='hasp':
        op_geoms += [{'body':b['name'],'geom':g['name'],'collision':g['collision']} for b in model['bodies'] for g in b['geoms'] if g['semantic']=='lock']
    if kind!='none' and not op_geoms:
        issues.append({'code':'specified_operator_has_no_operator_geometry','operator':op})
    if kind!='none' and not contacts:
        issues.append({'code':'specified_operator_has_no_grip_or_push_site','operator':op})
    if kind!='none' and kind!='push_plate' and not any(g['collision'] for g in op_geoms):
        issues.append({'code':'specified_operator_has_no_operator_collider','operator':op})
    return {'door_id':spec['id'],'family':spec['family'],'slab':spec['leaf']['slab'],'operator':op,'mass_kg':mass,'leaf_body_masses':{b['name']:b['mass'] for b in moving if b['semantic']=='leaf'},'material_inventory':inventory,'operation_inventory':{'operator_geoms':op_geoms,'contact_sites':contacts,'operator_joints':op_joints,'declared_operator_joint':model['meta'].get('operator_joint')},'issues':issues}
