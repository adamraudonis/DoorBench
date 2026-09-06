"""Bifold pivot/guide support geometry and independent-bank metadata.

Based on P C Henderson's Bifold fitting/CAD topology: each pair has fixed
jamb pivots plus one lead guide in an open top channel. Not a vendor load
rating: the authored material/weight remains a separate spec concern.
"""
from ..ir import Site, ALL_TIERS, QUAT_ID
from ..folding import FOLD_PIVOT_IN, FOLD_FLOOR_GAP, FOLD_TRACK_H
from . import common as C


def add_bifold_track(model, world, spec, material):
    width,height=spec['opening']['width'],spec['opening']['height']
    world.geoms.append(C.box('fold_track_web',(0,0,height-.002),(width/2,.013,.002),
        material,2700,True,True,ALL_TIERS,'track','Open guide channel upper web'))
    for sign in (-1,1):
        world.geoms.append(C.box(f'fold_track_flange_{sign}',(0,sign*.011,height-.017),
            (width/2,.002,.013),material,2700,True,True,ALL_TIERS,'track','Guide-channel side flange'))


def add_bifold_supports(model, world, spec, bodies, groups):
    width, height=spec['leaf']['width'],spec['leaf']['height']
    top=FOLD_FLOOR_GAP+height;ho=spec['opening']['height']
    steel=C.mat_from_material(model,'steel_galvanized','mat_fold_pivot')
    nylon=C.mat_from_material(model,'black_matte_metal','mat_fold_guide')
    banks=[]
    for index,(direction,origin) in enumerate(groups):
        pivot=model.body(f'panel_{index}_0');lead=model.body(f'panel_{index}_1')
        x=direction*FOLD_PIVOT_IN;world_x=origin+x
        world.geoms.append(C.box(f'fold_bottom_bracket_{index}',(origin+direction*.025,0,.004),
            (.025,.023,.004),steel,7850,True,True,ALL_TIERS,'hinge','Bottom pivot bearing plate on floor'))
        pivot.geoms.append(C.cyl(f'fold_bottom_pin_{index}',(x,0,(.008+.035)/2),.004,(.035-.008)/2,
            steel,(0,0,1),7850,True,True,ALL_TIERS,'hinge','Load-bearing bottom pivot pin'))
        pivot.geoms.append(C.cyl(f'fold_top_pin_{index}',(x,0,(top-.006+ho-.009)/2),.004,(ho-.009-top+.006)/2,
            steel,(0,0,1),7850,True,True,ALL_TIERS,'hinge','Top jamb pivot pin'))
        # Square socket surrounding the pin, with an actual 9 mm opening.
        for axis in (0,1):
            for sign in (-1,1):
                position=[world_x,0,ho-.012];position[axis]+=sign*.00825
                half=[.012,.012,.006];half[axis]=.00375
                world.geoms.append(C.box(f'fold_top_socket_{index}_{axis}_{sign}',position,half,
                    steel,7850,True,True,ALL_TIERS,'hinge','Open pivot-pin socket fixed in the top channel'))
        # Equal pivot-to-hinge and hinge-to-guide horizontal offsets. With the
        # existing face-hinge +/-t/2 offsets and q_fold=-2*q_pivot this point
        # stays exactly at world y=0, not merely approximately on the track.
        guide_x=direction*(width-FOLD_PIVOT_IN)
        lead.geoms.append(C.cyl(f'fold_guide_stem_{index}',(guide_x,0,(top-.006+ho-.014)/2),
            .004,(ho-.014-top+.006)/2,steel,(0,0,1),7850,True,True,ALL_TIERS,'hinge','Lead-panel guide stem'))
        lead.geoms.append(C.cyl(f'fold_guide_roller_{index}',(guide_x,0,ho-.014),.007,.0045,
            nylon,(0,0,1),1200,True,True,ALL_TIERS,'hinge','Nylon lead guide in channel'))
        lead.sites.append(Site(f'fold_guide_{index}',(guide_x,0,ho-.014),QUAT_ID,.005,'guide'))
        grips=[s.name for s in lead.sites if s.role=='grip']
        banks.append({'bank':index,'pivot_joint':pivot.joint.name,'fold_joint':lead.joint.name,
                      'grip_site':grips[0] if grips else None,'guide_site':f'fold_guide_{index}',
                      'guide_geom':f'fold_guide_roller_{index}','pivot_body':pivot.name,'lead_body':lead.name,
                      'closed_q':0.,'open_q':pivot.joint.range[1]})
    model.meta['folding_banks']=banks
    model.meta['folding_reference_sequence']='Operate each bank by its own grip; hold previously opened banks while switching. Banks have no mechanical coupling to one another.'
    model.meta['folding_support_reference']='https://www.pchenderson.com/media/documents/BIFOLD-FOLDING-WARDROBE-DOOR-GEAR-LVL-A.pdf'
    model.meta['folding_grip_reference']='https://www.johnsonhardware.com/content/Images/uploaded/Documentation/100FDCatalogPage.pdf'
