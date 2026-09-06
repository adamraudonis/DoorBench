"""Original positive chain keeper for the manual material-chain hoist.

Manufacturers require a wall/jamb chain keeper on ordinary manual hoists. This
is an original spring-return sliding-pin implementation, not proprietary
integral-brake CAD. Native pin/roller contact, not a friction multiplier or
coordinate lock, carries the load. A second hand must withdraw the pin while
operating either chain strand; releasing the hands lets it re-engage.
"""
from __future__ import annotations
import numpy as np
from ..ir import Body, Joint, Site, SpatialSpring, QUAT_ID, ALL_TIERS
from . import common as C
from .marine_dogs import bearing_y


def add_chain_keeper(model, spec):
    if 'rollup_hoist' not in model.meta or spec.get('kinematics',{}).get('opener')!='chain_hoist':
        raise ValueError('Positive keeper requires the actual manual material-chain hoist')
    if model.meta['rollup_hoist'].get('keeper'):
        raise ValueError('A chain keeper is already installed')
    hoist=model.meta['rollup_hoist'];points=np.asarray(hoist['parameters']['points_yz'],float)
    mids=(points+np.roll(points,-1,axis=0))/2
    candidates=np.flatnonzero((mids[:,0]>.06)&(mids[:,1]>.8)&(mids[:,1]<1.6))
    if not len(candidates):raise ValueError('No straight, reachable chain strand for keeper')
    idx=int(min(candidates,key=lambda k:abs(mids[k,1]-1.2)))
    x,y,_=hoist['wheel_center_m'];y+=float(mids[idx,0]);z=float(mids[idx,1])
    steel=C.mat_from_material(model,'steel','mat_hoist_keeper')
    wall_back=spec['opening']['wall_thickness']/2
    mount=model.add_body(Body('hoist_keeper_housing',None,(x,y,z),static=True,
        semantic='frame',label='Wall-mounted bored chain-keeper housing'))
    # The real material strand bows away from its nominal vertical line while
    # loaded. Keep the fixed bearing behind that measured swept volume.
    bearing_y(mount,'hoist_keeper_front_bearing',(0,.040,0),steel,inner=.0058,outer=.016,half_length=.008,semantic='mechanism')
    bearing_y(mount,'hoist_keeper_spring_housing',(0,.122,0),steel,inner=.012,outer=.018,half_length=.085,semantic='mechanism')
    bearing_y(mount,'hoist_keeper_rear_bearing',(0,.215,0),steel,inner=.0058,outer=.018,half_length=.008,semantic='mechanism')
    # Capture the pin in a receiving bore across the complete chain channel.
    # A free cantilever tip lets a flexible strand bow around it under load.
    bearing_y(mount,'hoist_keeper_receiver',(0,-.025,0),steel,inner=.0058,outer=.020,half_length=.008,semantic='mechanism')
    for side in (-1,1):
        mount.geoms.append(C.box(f'hoist_keeper_chain_guide_{side}',(side*.018,.0075,0),
            (.003,.0405,.075),steel,7850,True,True,ALL_TIERS,'frame','Chain guide joins receiving bore to keeper housing'))
    mount.geoms.append(C.box('hoist_keeper_wall_arm',(.04,(wall_back-y+.215)/2,0),
        (.008,(y+.215-wall_back)/2,.024),steel,7850,True,True,ALL_TIERS,'frame','Steel keeper bracket bolted to wall'))
    mount.geoms.append(C.box('hoist_keeper_housing_bridge',(.026,.215,0),
        (.016,.008,.012),steel,7850,True,True,ALL_TIERS,'frame','Bearing housing joined to wall bracket'))
    keeper=model.add_body(Body('hoist_keeper_pin',None,(x,y,z),joint=Joint('hoist_keeper_release','slide',(0,1,0),
        range=(0.,.080),damping=.8,frictionloss=.2,role='mechanism',label='Pull +Y to withdraw chain keeper; release hand after controlled engagement'),
        semantic='mechanism',label='Positive steel chain-keeper pin and reachable pull handle'))
    keeper.geoms.append(C.cyl('hoist_keeper_shaft',(0,.115,0),.005,.155,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'lock','10 mm steel pin enters the open space between chain rollers'))
    keeper.geoms[-1].friction=(.12,.001,.0001)  # Same lubricated steel contact as the authored rollers.
    keeper.geoms[-1].solref=(.001,1.)
    keeper.geoms[-1].solimp=(.999,.99999,.0001)
    keeper.geoms.append(C.cyl('hoist_keeper_spring_shoulder',(0,.075,0),.010,.002,steel,(0,1,0),7850,
        True,True,ALL_TIERS,'mechanism','Compression-spring moving shoulder'))
    keeper.geoms.append(C.sphere('hoist_keeper_pull',(0,.285,0),.016,steel,7850,True,ALL_TIERS,'operator','Reachable keeper pull knob'))
    keeper.sites.append(Site('hoist_keeper_grip',(0,.301,0),QUAT_ID,.01,'grip'))
    keeper.sites.append(Site('hoist_keeper_spring_seat',(0,.075,0),QUAT_ID,.004,'spring_anchor'))
    mount.sites.append(Site('hoist_keeper_spring_cap',(0,.215,0),QUAT_ID,.004,'spring_anchor'))
    model.spatial_springs.append(SpatialSpring('hoist_keeper_return',('hoist_keeper_spring_seat','hoist_keeper_spring_cap'),
        200.,.150,damping=.4,width=.008,label='Guided compression return spring around keeper shaft; linear force model'))
    # Match the authored lubricated steel rollers at every new interface.
    # Positive pin capture, rather than extra guide friction, must hold load.
    for geom in (*mount.geoms,*keeper.geoms):
        geom.friction=(.12,.001,.0001)
        geom.solref=(.001,1.)
        geom.solimp=(.999,.99999,.0001)
    model.meta.setdefault('mechanism_mass_bodies',[]).append(keeper.name)
    model.meta.setdefault('physical_inertia_joints',[]).append(keeper.joint.name)
    result={'schema_version':1,'kind':'spring_return_positive_roller_chain_pin','body':keeper.name,
        'joint':keeper.joint.name,'grip_site':'hoist_keeper_grip','pin_geom':'hoist_keeper_shaft',
        'fixed_body':mount.name,'spring':'hoist_keeper_return','location_m':[x,y,z],
        'engaged_q_m':0.,'withdrawn_q_m':.080,'withdraw_axis_world':[0.,1.,0.],
        'hand_force_limit_N':120.,'pin_radius_m':.005,'front_bearing_bore_radius_m':.0058,
        'spring_stiffness_N_per_m':200.,'spring_free_length_m':.150,
        'excluded_chain_grip_z_m':[z-.12,z+.12],
        'initial_chain_gap_link_index':idx,
        'scope':'Original positive pin keeper. Operate with actual keeper withdrawal plus chain-strand force; engage while holding the chain, then release both hands. Not an automatic load brake. No rated strength, fatigue or embodied reach certification.',
        'sources':['https://www.janusintl.com/hubfs/janus_2019/pdf/model2000-install-Rev.-12.9.16.pdf',
                   'https://www.cornelliron.com/product/controlgard-chain-hoist']}
    hoist['keeper']=result
    return result
