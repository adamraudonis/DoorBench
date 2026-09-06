"""Physical panel dimensions and hardware ownership before IR construction.

These conventions mirror authored leaf envelopes, not leaf.count blindly.
"""
from __future__ import annotations

from . import hardware as H
from . import materials as M
from .folding import FOLD_HINGE_GAP, fold_jamb_gap, fold_lead_gap, fold_meeting_gap, fold_groups


def mass_panels(spec):
    leaf=spec['leaf'];f=spec['family'];w,h,t=(leaf[k] for k in ('width','height','thickness'));n=leaf.get('count',1);wo=spec['opening']['width']
    result=[]
    def add(name,width=w,height=h,**kwargs):
        result.append({'body':name,'width':width,'height':height,**kwargs})
    if f=='swing_double':
        from .construction_dimensions import PAIRED_JAMB_GAP_M
        inset=.006 if H.HINGES[spec['hinge']['model']].kind in ('pivot_center','pivot_center_heavy') else PAIRED_JAMB_GAP_M
        mullion=leaf.get('astragal')=='removable_mullion'
        width=(wo-2*inset-(.056 if mullion else .003))/2
        add('leaf_a',width)
        add('leaf_b',width,inactive=not leaf.get('inactive_leaf',{}).get('active',False),secondary_pair=True)
    elif f=='saloon':
        add('leaf_a' if spec['kinematics'].get('pair',True) else 'leaf')
        if spec['kinematics'].get('pair',True):add('leaf_b')
    elif f=='dutch':
        split=leaf.get('dutch_split_height',h/2)
        add('leaf_lower',height=split-.012-.004,hardware_fraction=split/h)
        add('leaf_upper',height=h-split-.004,hardware_fraction=(h-split)/h)
    elif f in ('bifold','accordion'):
        accordion=bool(spec['kinematics'].get('accordion'));groups=fold_groups(n,accordion);per=n//groups
        for gi in range(groups):
            u=1 if gi==0 else -1;hx=-wo/2 if gi==0 else wo/2
            lead=u*(wo/2-fold_lead_gap(per,w,t)) if groups==1 else -u*fold_meeting_gap(per,w,t)
            for k in range(per):
                start=fold_jamb_gap(t) if k==0 else FOLD_HINGE_GAP
                end=abs(lead-(hx+u*k*w)) if k==per-1 else w-FOLD_HINGE_GAP
                add(f'panel_{gi}_{k}',end-start,hardware_fraction=1/n,operator_fraction=1.0 if k==per-1 else 0.0)
    elif f=='sliding_bypass':
        for i in range(n):add(f'leaf_{i}')
    elif f=='elevator' and spec['kinematics'].get('center_opening') or f=='automatic_sliding' and spec['kinematics'].get('bi_parting'):
        add('leaf_a');add('leaf_b')
    elif f=='strip_curtain':
        from .strips import segment_name, strip_layout
        layout=strip_layout(spec)
        for i in range(n):
            for k in range(layout['segments']):
                add(segment_name(i,k),leaf['strip_width'],height=layout['segment_length'],
                    strip_index=i,segment_index=k,hardware_fraction=0.)
    elif f in ('revolving','turnstile_fullheight','turnstile_tripod'):
        add('rotor',material_units=n if f=='revolving' else (spec['kinematics']['wings'] if f=='turnstile_fullheight' else 1),operator_fraction=n if f=='revolving' else 1)
    elif f in ('hatch_floor','hatch_ceiling'):
        add('hatch',height=spec['opening']['height']-.008)
    elif f=='garage_sectional':
        sections=int(spec['kinematics'].get('n_sections',4))
        operator_panel=sections-1
        if H.OPERATORS[spec['operator']['model']].kind=='t_handle':
            import math
            operator_panel=max(0,min(sections-1,math.floor((h+.05-min(spec['operator']['height'],h-.10))/(h/sections))))
        for i in range(sections):
            # Top-to-bottom independent material panels, with one installed
            # operator on its physical panel. Fractions preserve the complete
            # original material envelope and the single hardware allowance.
            add(f'section_{i}',height=h/sections,hardware_fraction=1/sections,
                operator_fraction=1. if i==operator_panel else 0.)
    elif f=='rollup':
        from .geometry.rollup import curtain_dimensions
        layout=curtain_dimensions(spec);count=layout['slat_count']
        for i in range(count):
            add(f'curtain_slat_{i}',height=layout['pitch_m'],hardware_fraction=1/count,
                operator_fraction=1. if i==count-1 else 0.)
    else:
        add({'rollup':'curtain','garage_sectional':'door','garage_tiltup':'door','pet_door':'flap'}.get(f,'leaf'))
    if leaf.get('pet_flap'):
        pf=leaf['pet_flap']
        for parent in list(result):
            prefix='' if parent['body']=='leaf' else parent['body']+'_'
            parent['cutout_area_m2']=(pf['width']+.002)*(pf['height']+.002)
            result.append({'body':prefix+'pet_flap','width':pf['width']-.008,'height':pf['height']-.008,
                           'embedded_slab':pf['slab'],'thickness':M.SLABS[pf['slab']].typical_thickness[0]})
    return result
