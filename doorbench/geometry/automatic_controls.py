"""Explicit activation stations and electric exit-device retraction.

Generic assemblies following the activation/unlatching/opening sequence in
LCN automatic operator wiring diagrams; no claim of an OEM internal mechanism.
"""
from ..ir import Site, QUAT_ID, ALL_TIERS
from . import common as C
from .wall_buttons import add_wall_button


def add_automatic_controls(model, spec):
    actuator = spec.get('kinematics', {}).get('actuator') or {}
    sectional = model.meta.get('sectional_track')
    powered_lift = sectional and sectional['drive']['mode'] == 'powered'
    if spec['family'] not in ('automatic_swing', 'automatic_sliding') and not powered_lift:
        return
    sensor = 'push_button_wall' if powered_lift else actuator.get('sensor', 'motion')
    station = {'kind': sensor, 'buttons': [], 'wave_sites': []}
    world = model.body('world_env')
    mat = C.mat_from_material(model, 'stainless', 'mat_activation_plate')
    padmat = C.mat_from_material(model, 'aluminum_dark', 'mat_activation_pad')
    x = float(model.meta.get('u', 1.)) * (spec['opening']['width']/2 + .25)
    y = float(model.meta.get('wall_y', 0.))
    depth = spec['opening']['wall_thickness']/2
    if sensor in ('push_button', 'push_button_wall'):
        # Very wide apertures can leave less stock than the station width in
        # the generic environment. Extend its outer edge, preserving the
        # aperture edge, so the plate has continuous masonry behind it.
        side = 1 if x >= 0 else -1
        wall = next(g for g in world.geoms if g.name == ('wall_right' if side > 0 else 'wall_left'))
        required_outer = abs(x) + .065 + .02
        if side*wall.pos[0] + wall.size[0] < required_outer:
            inner = wall.pos[0] - side*wall.size[0]
            outer = side*required_outer
            wall.pos = ((inner+outer)/2, wall.pos[1], wall.pos[2])
            wall.size = (abs(outer-inner)/2, wall.size[1], wall.size[2])
        for face, tag in ((-1, 'n'), (1, 'p')):
            name = 'activation_button_' + tag
            body = add_wall_button(model,world,spec,name=name,x=x,height=1.05,
                face=face,radius=.052,travel=.004,colour=(.20,.21,.22,1.),
                joint_role='operator',site_name=name+'_push',plate_half=(.065,.065))
            body.label = 'Press to open'
            body.joint.robot_interactive = face < 0
            body.joint.label = 'Wall plate: press to activate automatic door'
            station['buttons'].append({'joint': body.joint.name, 'site': body.sites[0].name, 'face': face,
                                       'threshold_m': .002})
    elif sensor == 'wave_to_open':
        for face, tag in ((-1, 'n'), (1, 'p')):
            name = 'activation_wave_' + tag
            world.geoms.append(C.box(name, (x,y+face*(depth+.008),1.05), (.045,.008,.045),
                                     padmat,1000,True,True,ALL_TIERS,'sensor','Wave-to-open sensor'))
            world.sites.append(Site(name+'_zone', (x,y+face*(depth+.12),1.05),QUAT_ID,.08,'activation',ALL_TIERS))
            station['wave_sites'].append(world.sites[-1].name)
    else:
        # The presence head is mounted above the aperture; a wall button is
        # not falsely used as a stand-in for a microwave/infrared sensor.
        world.geoms.append(C.box('activation_presence_head',(0,y-depth-.015,spec['opening']['height']+.065),
                                 (.09,.015,.028),padmat,1000,True,True,ALL_TIERS,'sensor','Presence activation head'))
    model.meta['automatic_activation'] = station
    # Electrically retract the actual bar and its mechanically coupled latch.
    # Zero motor command adds no fictitious spring in unpowered manual mode.
    retractors = []
    for body in model.bodies:
        joint = body.joint
        if joint and '_exit_device_' in joint.name and joint.role == 'operator':
            name = joint.name + '_electric_retraction'
            model.meta.setdefault('actuators', []).append({'name':name,'joint':joint.name,'kind':'motor',
                                                          'gear':1.,'ctrlrange':[0.,110.], 'role':'latch_retraction'})
            retractors.append({'actuator':name,'joint':joint.name,'travel':joint.range[1], 'max_force':110.})
    model.meta['powered_latch_retraction'] = retractors
