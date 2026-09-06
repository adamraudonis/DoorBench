"""Two independent trim cams must not add their throws into one latch case."""
from copy import deepcopy
from functools import lru_cache
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from doorbench.build import build_model
from doorbench.export.mjcf import build_mjcf
from doorbench.spec import generate_all
from doorbench.qa import _qa_step


@lru_cache(None)
def source():
    return build_model(next(s for s in generate_all() if s['index']==548))


def native(ir,tier):
    assets={g.mesh_name+'.obj':g.mesh.export(file_type='obj',include_normals=False,include_texture=False).encode()
            for b in ir.bodies for g in b.geoms if g.type=='mesh'}
    return mujoco.MjModel.from_xml_string(ET.tostring(build_mjcf(ir,tier=tier,mesh_dir_rel=''),encoding='unicode'),assets)


def cycle(m,row):
    d=mujoco.MjData(m);operators=[m.joint(n).id for n in row['operator_joints']]
    b=m.joint(row['bolt_joint']).id;states=[]
    # Bounded generalized input fixture isolates the concealed cam relation.
    # This is not an embodied two-hand task or proof of internal cam surfaces.
    for selected in ((0,),(),(1,),(),(0,1),()):
        for _ in range(round(1.5/m.opt.timestep)):
            d.qfrc_applied[:]=0.
            for i in selected:
                j=operators[i];d.qfrc_applied[m.jnt_dofadr[j]]=120. if int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_SLIDE) else 6.
            _qa_step(m,d)
        states.append({'inputs':selected,'operators':[float(d.qpos[m.jnt_qposadr[j]]) for j in operators],
                       'bolt':float(d.qpos[m.jnt_qposadr[b]]),'warnings':d.warning.number.copy(),
                       'penetration':max([-float(c.dist) for c in d.contact]+[0.])})
    return states


@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_either_or_both_inputs_fully_withdraw_once_and_return(tier):
    ir=source();row=ir.meta['independent_latch_inputs'][0];m=native(ir,tier)
    throw=float(m.jnt_range[m.joint(row['bolt_joint']).id,1])
    for state in cycle(m,row):
        assert not np.any(state['warnings']),state
        assert state['penetration']<.001,state
        assert abs(state['bolt']-(throw if state['inputs'] else 0.))<.001,state
        if len(state['inputs'])==2:
            assert all(q>=.97*m.jnt_range[m.joint(n).id,1] for n,q in zip(row['operator_joints'],state['operators'])),state


def test_old_additive_relation_cannot_complete_both_inputs_without_conflict():
    ir=deepcopy(source());row=ir.meta['independent_latch_inputs'][0]
    first=next(t for t in ir.tendons if t.name==row['tendons'][0]);second=next(t for t in ir.tendons if t.name==row['tendons'][1])
    first.sites += [(n,c) for n,c in second.sites if n!=row['bolt_joint']]
    ir.tendons.remove(second);m=native(ir,'full');state=cycle(m,row)[4]
    full=all(q>=.97*m.jnt_range[m.joint(n).id,1] for n,q in zip(row['operator_joints'],state['operators']))
    assert not full or state['penetration']>=.001 or np.any(state['warnings']),state
