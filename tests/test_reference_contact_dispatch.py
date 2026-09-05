"""Role selection must not broaden contact permissions or swallow integrity errors."""
import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from doorbench.reference import guidance, manual_contacts
from doorbench.reference.solve import ContactResolver
from tests.test_reference_manual_contacts import inputs


@pytest.fixture
def manual_input_tree(tmp_path, inputs):
    spec, ir, clip, source = inputs
    door=tmp_path/'assets/doors'/spec['id']; door.mkdir(parents=True)
    recordings=tmp_path/'recordings'; (recordings/'clips').mkdir(parents=True); (recordings/'trajectories').mkdir()
    for name,data in [('spec',spec),('model',ir)]: (door/f'{name}.json').write_text(json.dumps(data))
    (recordings/'clips'/f'{door.name}.json').write_text(json.dumps(clip))
    np.savez_compressed(recordings/'trajectories'/f'{door.name}.npz',**source)
    return door, recordings


def test_manual_dispatch_uses_selected_builder_and_preserves_options(manual_input_tree,monkeypatch):
    calls=[]; sentinel=object()
    def build(*args,**kwargs): calls.append((args,kwargs)); return SimpleNamespace(guide=sentinel)
    monkeypatch.setattr(manual_contacts,'build_manual_guide',build)
    monkeypatch.setattr(guidance,'SceneNavigator',lambda *a:pytest.fail('Selected schedule entered baseline'))
    assert guidance.make_guide(*manual_input_tree,fps=60,gait_profile='controlled') is sentinel
    assert calls==[(manual_input_tree,{'fps':60,'gait_profile':'controlled'})]


def test_selected_manual_integrity_error_propagates(manual_input_tree,monkeypatch):
    def corrupt(*args,**kwargs): raise ValueError('Source hash mismatch')
    monkeypatch.setattr(manual_contacts,'build_manual_guide',corrupt)
    monkeypatch.setattr(guidance,'SceneNavigator',lambda *a:pytest.fail('Integrity error fell back'))
    with pytest.raises(ValueError,match='Source hash mismatch'): guidance.make_guide(*manual_input_tree)


def test_unsupported_manual_shape_preserves_baseline_path(manual_input_tree,monkeypatch):
    door,recordings=manual_input_tree
    spec=json.loads((door/'spec.json').read_text()); spec['lock']['engaged']=True
    (door/'spec.json').write_text(json.dumps(spec))
    class BaselineReached(Exception): pass
    def baseline(*args): raise BaselineReached
    monkeypatch.setattr(guidance,'SceneNavigator',baseline)
    monkeypatch.setattr(manual_contacts,'build_manual_guide',lambda *a,**k:pytest.fail('Unsupported builder selected'))
    with pytest.raises(BaselineReached): guidance.make_guide(door,recordings)


@pytest.fixture
def resolver_fixture():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="panel" pos="0 0 1"><joint name="slide" type="slide" axis="1 0 0"/>
        <site name="grip"/><geom name="declared" type="box" size=".05 .02 .05"/>
        <geom name="unrelated" type="box" pos="0 -.08 0" size=".05 .02 .05"/>
      </body><body name="other" pos="1 0 1"><joint name="other_joint"/>
        <site name="other_grip"/><geom name="other_geom" type="box" size=".05 .02 .05"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    solver=SimpleNamespace(model=model,data=data,scene_geom_ids=list(range(model.ngeom)),floor_geom_ids=[])
    ir={'bodies':[{'geoms':[{'name':model.geom(i).name,'collision':True,'semantic':'operator'} for i in range(model.ngeom)]}]}
    source={'time':np.array([0.,1.]),'target':np.array([[0.,0.,1.],[0.,0.,1.]])}
    role={'id':'pull','joint_name':'slide','body_name':'panel','site_name':'grip','geom_name':'declared'}
    return solver,ir,source,role


def test_role_uses_fresh_site_and_only_declared_geometry(resolver_fixture):
    solver,ir,source,role=resolver_fixture
    resolver=ContactResolver(solver,ir,source,roles=[role])
    solver.data.qpos[0]=.4; mujoco.mj_forward(solver.model,solver.data)
    anchor,target,geom=resolver.resolve(.5,[.4,-.5,.94],role_id='pull')
    np.testing.assert_allclose(anchor,[.4,0,1],atol=1e-12)
    assert geom=='declared'  # The nearer unrelated geometry is never considered.
    np.testing.assert_allclose(target,[.4,-.0595,1],atol=1e-12)
    assert resolver.resolve(.5,[.4,-.5,.94],role_id=None)[2] is None


def test_missing_declared_hit_never_falls_back(resolver_fixture):
    solver,ir,source,role=resolver_fixture
    # Move only the declared primitive; another eligible collider remains near the site.
    solver.model.geom_pos[solver.model.geom('declared').id]=[1,0,0]
    solver.model.geom_sameframe[solver.model.geom('declared').id]=mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    mujoco.mj_forward(solver.model,solver.data)
    resolver=ContactResolver(solver,ir,source,roles=[role])
    assert resolver.resolve(.5,[0,-.5,.94],role_id='pull')[2] is None


@pytest.mark.parametrize('field,value',[('site_name','other_grip'),('geom_name','other_geom'),('joint_name','other_joint')])
def test_cross_body_contact_binding_rejected(resolver_fixture,field,value):
    solver,ir,source,role=resolver_fixture; role=copy.deepcopy(role); role[field]=value
    with pytest.raises(ValueError,match='one native joint body'): ContactResolver(solver,ir,source,roles=[role])


def test_unknown_or_unbound_contact_role_is_rejected(resolver_fixture):
    solver,ir,source,role=resolver_fixture
    with pytest.raises(ValueError,match='Unknown manual'):
        ContactResolver(solver,ir,source,roles=[role]).resolve(0,[0,-.5,.94],role_id='other')
    with pytest.raises(ValueError,match='without a bound schedule'):
        ContactResolver(solver,ir,source).resolve(0,[0,-.5,.94],role_id='pull')
