"""Two-layer PVC curtain with native material flexures and retained contacts.

Each strip is a planar discrete beam, not a deformable-shell certification.
Its elastic bending response uses E*I/L and real segment material inertia.
"""
from __future__ import annotations

import math

from .. import materials as M
from ..ir import ALL_TIERS, Body, Joint, QUAT_ID, Site
from ..strips import segment_name, strip_layout
from . import common as C


def build_strip_curtain(spec,phys,model):
    layout=strip_layout(spec);w,h,t=(layout[k] for k in ('width','height','thickness'))
    count,n,length=layout['count'],layout['segments'],layout['segment_length']
    wo,ho=spec['opening']['width'],spec['opening']['height']
    world=C.add_floor_and_wall(model,spec,wall_half_width=max(2.5,wo/2+1),wall_height=ho+.6)
    steel=C.mat_from_material(model,'stainless','mat_strip_hanger')
    pvc=C.mat_from_material(model,'pvc_flexible','mat_strip')
    material=M.MATERIALS['pvc_flexible'];density=material.density
    # The catalog's 10 MPa is an authored flexible-PVC approximation; no
    # temperature-specific or strain-dependent measured curve is implied.
    modulus=material.youngs_modulus;second_moment=w*t**3/12;stiffness=modulus*second_moment/length
    segment_mass=w*t*length*density
    damping=2*.25*math.sqrt(stiffness*segment_mass*length**2/3)
    top=ho-.025;center_strip=count//2
    world.geoms.append(C.box('hanger_rail',(0,0,ho+.020),(wo/2+.035,.020,.020),steel,7900,
        semantic='frame',label='Wall-mounted strip hanging rail'))
    controls=[];flexures=[];output=[]
    for i in range(count):
        layer=(i-center_strip)%2
        x=-wo/2+.01+w/2+i*layout['pitch'];y=(layer-.5)*(t+.001)
        for side in (-1,1):
            world.geoms.append(C.box(f'strip_clamp_{i}_{side}',(x,y+side*(t/2+.001),top+.014),
                (w/2,.001,.013),steel,7900,semantic='frame',label='Fixed PVC clamping plate'))
        world.geoms.append(C.box(f'strip_hanger_{i}',(x,y,ho+.003),(.015,.005,.020),steel,7900,
            semantic='frame',label='Clamp hook fixed to hanging rail'))
        # The cut stock continues through the fixed jaws. Keep a named native
        # body so only its bonded interface to its own first moving segment can
        # omit self-contact, exactly like adjacent segments of one strip.
        tab_name=f'strip_{i}_clamp_tab'; tab_height=.028
        tab=Body(tab_name,None,(0,0,0),QUAT_ID,None,[],[],ALL_TIERS,'frame','Fixed PVC tab inside clamping jaws',static=True)
        tab.geoms.append(C.box(tab_name+'_pvc',(x,y,top+tab_height/2),(w/2,t/2,tab_height/2),pvc,density,
            True,True,ALL_TIERS,'leaf','Fixed PVC cut-stock reserve through clamp'))
        model.add_body(tab)
        model.meta.setdefault('native_fixed_body_names',[]).append(tab_name)
        model.contact_excludes.append((tab_name,segment_name(i,0)))
        parent=tab_name;push_sites=[];segment_bodies=[]
        target_index=max(0,min(n-1,int((top-1.)/length)))
        for k in range(n):
            name=segment_name(i,k);position=(x,y,top) if k==0 else (0,0,-length)
            body=Body(name,parent,position,QUAT_ID,None,[],[],ALL_TIERS,'leaf',f'PVC strip {i+1}, material segment {k+1}')
            joint=f'strip_{i}_hinge' if k==0 else name+'_bend'
            role=('primary' if i==center_strip else 'secondary') if k==0 else 'mechanism'
            body.joint=Joint(joint,'hinge',(1,0,0),(0,0,0),(-1.4,1.4),
                damping=damping,stiffness=stiffness*(2 if k==0 else 1),springref=0.,armature=0.,
                role=role,robot_interactive=k==0,label='PVC flexure (+ = toward far side)')
            body.geoms.append(C.box(name+'_pvc',(0,0,-length/2),(w/2,t/2,length/2),pvc,density,
                True,True,ALL_TIERS,'leaf','Flexible PVC material segment',friction=(.4,.003,.0001),
                solref=(.0002,1.)))
            # Constant impedance avoids a sharply varying contact spring at
            # crossing segment edges. The 0.2 ms contact time is resolved by
            # the explicit 0.1 ms native bound below; contacts are not masked.
            body.geoms[-1].solimp=(.95,.95,.0001)
            # Contacts on both real faces. The operated segment can bend while
            # the upper strip remains nearly vertical; site forces propagate
            # through the complete material chain in the native model.
            if k==target_index:
                local_z=max(-length+.01,min(-.01,1.-(top-k*length)))
                for face,tag in ((-1,'n'),(1,'p')):
                    site=f'strip_{i}_push_{tag}'
                    body.sites.append(Site(site,(0,face*t/2,local_z),QUAT_ID,.008,'push'))
                    push_sites.append(site)
            model.add_body(body);output.append(body);segment_bodies.append(name);flexures.append(joint);parent=name
        controls.append({'strip':i,'root_joint':f'strip_{i}_hinge','segment_bodies':segment_bodies,
            'push_sites':push_sites,'layer':layer,'fixed_tab_body':tab_name,'fixed_tab_geom':tab_name+'_pvc',
            'clamp_geoms':[f'strip_clamp_{i}_{side}' for side in (-1,1)],
            'fixed_stock_length_m':tab_height,'cut_stock_length_m':h+tab_height})
    world.sites.extend([Site('approach_point',(0,-1.5,0),QUAT_ID,.05,'approach'),
        Site('goal_point',(0,1.5,0),QUAT_ID,.05,'goal'),Site('door_plane_center',(0,0,ho/2),QUAT_ID,.02,'pass_plane')])
    model.meta.update({'primary_joint':f'strip_{center_strip}_hinge','operator_joint':None,'handle_height':1.,
        'both_ways':True,'n_strips':count,'material_flexure_joints':flexures,'native_timestep_s':.0001,'native_arena_memory_mib':16,
        'strip_curtain':{'schema_version':1,'model':'planar_discrete_elastic_strips','layout':layout,
            'controls':controls,'elastic_modulus_Pa':modulus,'density_kg_m3':density,
            'fixed_pvc_mass_kg':count*w*t*tab_height*density,
            'bending_stiffness_Nm_per_rad':stiffness,'bending_damping_Nms_per_rad':damping,
            'scope':'Planar flexural approximation with full interstrip contacts; torsion, lateral bending, temperature dependence and tear/fracture are not modeled.',
            'reference':'https://www.pvc-strip.co.uk/news/installing-pvc-strips-hook-type-pvc-curtain-kits/'}})
    return output
