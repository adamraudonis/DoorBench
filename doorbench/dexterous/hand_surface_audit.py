"""Privileged native contact audit; never an actor observation or routing API."""
import mujoco
import numpy as np


def native_hand_surface_loads(model, data, *, surface_body='leaf', side='lh'):
    """Separate actual palm loading from finger contacts against one surface.

    The surface +Y axis is the door's opening-side normal in this diagnostic.
    Loads use forces from the active physics step, including tangent components.
    Contact identifiers are evaluator-only; do not put this record in actor data.
    """
    if side not in ('lh','rh'):
        raise ValueError('Expected named left or right hand')
    surface=model.body(surface_body).id
    normal=data.xmat[surface].reshape(3,3)[:,1]
    prefix='robot/'+side+'_'
    loads={};contacts=[]
    for index,contact in enumerate(data.contact[:data.ncon]):
        bodies=[int(model.geom_bodyid[g]) for g in contact.geom]
        for hand_side in (0,1):
            name=model.body(bodies[hand_side]).name
            if not name.startswith(prefix) or bodies[1-hand_side]!=surface:
                continue
            wrench=np.zeros(6);mujoco.mj_contactForce(model,data,index,wrench)
            on_hand=contact.frame.reshape(3,3).T@wrench[:3]*(-1 if hand_side==0 else 1)
            load=max(0.,float(-normal@on_hand))
            if not np.isfinite(load):raise ValueError('Nonfinite hand-surface contact load')
            loads[name]=loads.get(name,0.)+load
            contacts.append(dict(hand_body=name,normal_load_N=load,distance_m=float(contact.dist)))
    return dict(total_normal_load_N=sum(loads.values()),
                palm_normal_load_N=loads.get(prefix+'palm',0.),
                body_normal_loads_N=loads,contacts=contacts,
                scope='privileged evaluator-only native contact forces')
