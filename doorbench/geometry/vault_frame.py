"""Frame-backed top and sill rebates for the original oversized vault opening."""
from ..ir import ALL_TIERS
from . import common as C


def add_vault_frame_rebates(model,spec):
    world=model.body('world_env');height=spec['leaf']['height'];width=spec['opening']['width']
    v=model.meta['v'];thickness=spec['leaf']['thickness']
    steel=C.mat_from_material(model,'steel','mat_vault_frame_rebate')
    # Use the same half-millimetre closing-face clearance as the supported
    # side rebates. A wider stop needs stock above/below the leaf, not a
    # cosmetic seal floating across its 20/10 mm running gaps.
    y=-v*(thickness/2+.016+.0005)
    world.geoms=[g for g in world.geoms if g.name not in ('stop_head','seal_head')]
    rows=[]
    for name,low,high,anchor in (
        ('stop_vault_head',.05+height-.020,spec['opening']['height']+.020,'jamb_head'),
        ('stop_vault_sill',.020,.05+.020,'sill_step'),
    ):
        if not any(g.name==anchor for g in world.geoms):raise ValueError('Vault rebate requires its authored frame support')
        world.geoms.append(C.box(name,(0,y,(low+high)/2),(width/2,.016,(high-low)/2),
            steel,7850,True,True,ALL_TIERS,'frame','Frame-backed vault closing rebate'))
        rows.append({'geom':name,'anchor_geom':anchor,'leaf_overlap_m':.020,'backing_overlap_m':.020,
                     'closed_face_clearance_m':.0005})
    model.meta['vault_frame_rebates']={'rows':rows,
        'scope':'Original frame-backed top and sill stops; no airtightness, blast, security or structural rating.'}
    model.meta['vault_closing_stops'].extend(row['geom'] for row in rows)
