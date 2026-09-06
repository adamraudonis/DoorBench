"""Rear-face manual locking hardware for retractable garage panels.

Original generic fittings: real rod/keeper clearances and a slotted hasp.
Padlock key operation is not simulated; a closed padlock retains its hasp.
"""
from __future__ import annotations
from ..ir import ALL_TIERS
from . import common as C


def add_tiltup_lock(model, panel, world, spec, top_height, *, mount_height=None):
    lock=spec['lock'];kind=lock['model']
    if kind not in ('garage_slide_lock','padlock'):
        return
    width=spec['leaf']['width'];thickness=spec['leaf']['thickness']
    steel=C.mat_from_material(model,'steel_galvanized','mat_garage_lock')
    brass=C.mat_from_material(model,'brass','mat_garage_padlock')
    z=mount_height if mount_height is not None else (.65 if kind=='garage_slide_lock' else .18)
    surface=thickness/2
    # Rear-facing fittings are reachable from the garage interior. A steel
    # bracket runs from the side jamb to the keeper outside the panel edge.
    sign=-1 if kind=='garage_slide_lock' else 1
    world.geoms.append(C.box('garage_lock_jamb_bracket',(sign*(width/2+.135),surface-.007,z),
        (.10,.007,.04),steel,7850,True,True,ALL_TIERS,'lock','Keeper bracket bolted to side jamb'))
    if kind=='garage_slide_lock':
        bolt,info=C.add_barrel_bolt(model,panel,'garage_slide_lock',(-width/2,surface,z-top_height),
            (-1,0,0),(0,1,0),.18,.012,.065,bool(lock.get('engaged')),steel,
            protrusion=.052,standoff=.015,tiers=ALL_TIERS,role='lock',rod_semantic='lock',
            joint_name='garage_slide_lock_slide',grip_site='slide_lock_grip')
        keeper=(-width/2-.041,surface,z)
        C.add_keeper_loop(world.geoms,'garage_lock_keeper',keeper,
            (keeper[0],surface+.015,z),(-1,0,0),(0,1,0),.006,steel,ALL_TIERS,
            base=.030,bar=.005,bar_len=.014)
        if lock.get('engaged') and not lock.get('robot_side_release'):
            bolt.joint.range=(0.,.001)
        model.meta['garage_lock_hardware']={'kind':'rear_slide_bolt','joint':bolt.joint.name,
            'grip_site':'slide_lock_grip','engaged_q':0.,'released_q':.065,
            'keeper_prefix':'garage_lock_keeper','side':'garage_interior'}
        return
    locked=bool(lock.get('engaged'))
    hinge=(width/2-.14,surface,z-top_height)
    eye=(width/2+.06,surface+.022,z)
    hb=C.add_hasp_assembly(model,panel,world,'garage_lock',hinge,(0,1,0),(1,0,0),
        .23,.010,eye,(eye[0],surface,z),(0,1,0),locked,steel,brass,ALL_TIERS)
    # The hinge axis stands proud of its plate by a pin radius; the strap can
    # fold back without cutting through the fixed mounting spacer.
    hb.pos=(hb.pos[0],hb.pos[1]+.004,hb.pos[2])
    for g in hb.geoms:
        if g.name=='garage_lock_hasp_knuckle':
            g.pos=(0.,0.,0.)
    # Replace the generic solid strap by four bars around an actual 14x32 mm
    # slot. The staple passes through empty geometry, not an exclusion waiver.
    hb.geoms=[g for g in hb.geoms if g.name!='garage_lock_hasp_strap']
    for name,x0,x1,lat,hl in [('root',0.,.193,0.,.0175),('tip',.207,.23,0.,.0175),
                            ('slot_a',.193,.207,-.01675,.00075),('slot_b',.193,.207,.01675,.00075)]:
        hb.geoms.append(C.obox('garage_lock_hasp_'+name,(0,0,0),(1,0,0),(0,1,0),
            (x0+x1)/2,lat,.0015,(x1-x0)/2,hl,.0015,steel,True,ALL_TIERS,'lock','Slotted hasp strap'))
    if not locked:
        hb.joint.stiffness=.3;hb.joint.springref=2.8
    for g in world.geoms:
        if g.name.startswith('garage_lock_padlock'):
            g.collision=True
    # The remaining generic allowances describe surface-mounted bolt/hinge
    # joints, not an imaginary strap slot. Remove obsolete slot allowances.
    model.meta['clearance_allow']=[a for a in model.meta.get('clearance_allow',[])
        if a[0]!='garage_lock_hasp_strap' and a[1]!='garage_lock_hasp_strap']
    model.meta['garage_lock_hardware']={'kind':'rear_slotted_hasp_padlock','joint':hb.joint.name,
        'grip_site':'garage_lock_hasp_grip','engaged':locked,'released_q':2.8,
        'side':'garage_interior','key_operation':'not simulated; locked hasp retained by padlock',
        'slot_dimensions_m':[.014,.032]}
