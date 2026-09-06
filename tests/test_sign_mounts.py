"""Adhesive door signs must seat on the actual substrate, not the frame plane."""
import pytest
from doorbench.geometry import common as C
from doorbench.ir import Body
from doorbench.build import build_model
from doorbench.spec import generate_all


def test_full_footprint_required_and_other_hardware_cannot_supply_substrate():
    body=Body('test',None)
    body.geoms.append(C.box('pane',(0,0,1),(.4,.003,.8),'glass',2500,semantic='glass'))
    body.geoms.append(C.box('floating_decoration',(0,.02,1),(.1,.002,.1),'metal',7800,semantic='decor'))
    assert C._label_mount_face(body,0,1,.06,.03,1,.025)==pytest.approx(.003)
    assert C._label_mount_face(body,0,1,.06,.03,-1,.025)==pytest.approx(.003)
    assert C._label_mount_face(body,.39,1,.06,.03,1,.025)==pytest.approx(.025)


def test_recessed_glass_pair_signs_seat_on_each_pane():
    spec=next(s for s in generate_all() if s['index']==10)
    model=build_model(spec)
    count=0
    for body in model.bodies:
        panes=[g for g in body.geoms if g.semantic=='glass' and g.type=='box']
        for sign in [g for g in body.geoms if g.name.startswith(('sign_push','sign_pull'))]:
            x,y,z=sign.pos;hx,hy,hz=sign.size;face=1 if y>0 else -1
            supports=[g for g in panes if abs(g.pos[0]-x)+hx<=g.size[0]+1e-9
                      and abs(g.pos[2]-z)+hz<=g.size[2]+1e-9]
            assert supports
            assert face*y-hy==pytest.approx(max(face*g.pos[1]+g.size[1] for g in supports),abs=1e-9)
            count+=1
    assert count==4
