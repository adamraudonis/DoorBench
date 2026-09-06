"""A named parent/child stop carries load without disabling parent filtering."""
from copy import deepcopy
from xml.etree import ElementTree as ET
import mujoco
import numpy as np
import pytest
from doorbench.ir import Model,Body,Joint,ALL_TIERS
from doorbench.geometry import common as C
from doorbench.export.mjcf import build_mjcf


def fixture():
    ir=Model('explicit_stop');mat=C.mat_rgba(ir,'steel',(.4,.4,.4,1))
    parent=Body('carrier',None,(0,0,1),joint=Joint('rotor','hinge',(0,0,1),range=(-1,1)))
    parent.geoms=[C.box('stop',(0,0,0),(.1,.1,.02),mat,7850,tiers=ALL_TIERS)]
    child=Body('arm',parent.name,(0,0,.12),joint=Joint('drop','slide',(0,0,1),range=(-.2,.2)))
    child.geoms=[C.box('toe',(0,0,0),(.02,.02,.02),mat,7850,tiers=ALL_TIERS)]
    ir.add_body(parent);ir.add_body(child)
    ir.meta['native_contact_pairs']=[{'geom1':'toe','geom2':'stop','solref':[.001,1.],
        'solimp':[.99,.999,.0001],'friction':[.04,.04,.0001,.00001,.00001]}]
    return ir


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_only_explicit_pair_stops_child_under_force(tier):
    for paired in (True,False):
        ir=fixture()
        if not paired:ir.meta['native_contact_pairs']=[]
        m=mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(ir,tier=tier,timestep=.0005),encoding='unicode'))
        d=mujoco.MjData(m);j=m.joint('drop').id;v=m.jnt_dofadr[j]
        for _ in range(1200):d.qfrc_applied[v]=-10;mujoco.mj_step(m,d)
        assert not (m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        if paired:
            assert abs(d.qpos[m.jnt_qposadr[j]]+.08)<.001
            assert any(set(c.geom)=={m.geom('toe').id,m.geom('stop').id} for c in d.contact)
        else:assert d.qpos[m.jnt_qposadr[j]]<-.19 and d.ncon==0
        assert not np.any(d.warning.number)


@pytest.mark.parametrize('defect',('unknown','visual','tier','duplicate','nan','unknown_field'))
def test_invalid_or_missing_pair_never_silently_drops_contact(defect):
    ir=fixture();pair=ir.meta['native_contact_pairs'][0]
    if defect=='unknown':pair['geom1']='missing'
    elif defect=='visual':ir.body('arm').geoms[0].collision=False
    elif defect=='tier':ir.body('arm').geoms[0].tiers=('simple','minimal')
    elif defect=='duplicate':ir.meta['native_contact_pairs'].append(deepcopy(pair))
    elif defect=='nan':pair['friction'][0]=float('nan')
    else:pair['margin']=1.
    with pytest.raises(ValueError):build_mjcf(ir)
