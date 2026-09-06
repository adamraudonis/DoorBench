"""Independent counterexamples for physical rotor supports and hand surfaces."""
import json
import math
import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.rotating_hardware_qa import run_rotating_hardware_qa


@pytest.fixture(scope='module')
def rotors(tmp_path_factory):
    root=tmp_path_factory.mktemp('rotating-hardware');rows=[]
    for spec in generate_all():
        if spec['family'] not in ('revolving','turnstile_tripod','turnstile_fullheight'):continue
        summary=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        path=root/'doors'/spec['id'];meta=json.loads((path/'model.json').read_text())['meta']
        rows.append((spec,meta,summary['files']['mjcf']))
    assert len(rows)==35
    return rows


def test_every_rotor_contact_has_a_surface_and_positive_push_moment_in_all_tiers(rotors):
    for spec,meta,files in rotors:
        for tier in ('full','simple','minimal'):
            model=mujoco.MjModel.from_xml_path(files[tier]);before=model.qpos0.copy()
            result=run_rotating_hardware_qa(model,meta)
            assert result['ok'],(spec['id'],tier,result)
            assert result['contacts']
            np.testing.assert_array_equal(model.qpos0,before)


def test_floating_bar_and_perpendicular_shaft_fail(rotors):
    for spec,meta,files in rotors:
        if spec['family']=='revolving' and spec['operator']['model']!='none':
            model=mujoco.MjModel.from_xml_path(files['full'])
            model.geom_pos[model.geom('wing_0_bar_mount_outer').id,2]+=.030
            result=run_rotating_hardware_qa(model,meta)
            assert not result['ok'];assert any('detached_mount' in x for x in result['failures'] if isinstance(x,dict));break
    spec,meta,files=next(x for x in rotors if x[0]['family']=='turnstile_tripod')
    model=mujoco.MjModel.from_xml_path(files['full']);g=model.geom('hub_boss').id
    model.geom_quat[g]=[math.sqrt(.5),0,math.sqrt(.5),0]
    assert not run_rotating_hardware_qa(model,meta)['ok']


def test_old_between_rung_point_and_vertical_normal_are_rejected(rotors):
    _,meta,files=next(x for x in rotors if x[0]['family']=='turnstile_fullheight')
    model=mujoco.MjModel.from_xml_path(files['full']);sid=model.site('wing_0_push').id
    model.site_pos[sid,2]=1.
    assert not run_rotating_hardware_qa(model,meta)['ok']
    model=mujoco.MjModel.from_xml_path(files['full']);model.site_quat[sid]=[1,0,0,0]
    result=run_rotating_hardware_qa(model,meta)
    assert not result['ok'];assert any('invalid_push_moment' in x for x in result['failures'] if isinstance(x,dict))


def test_native_credential_bolt_blocks_without_a_primary_range(rotors):
    for spec,meta,files in rotors:
        if not spec['kinematics'].get('locked_until_credential'):continue
        model=mujoco.MjModel.from_xml_path(files['full']);d=mujoco.MjData(model)
        j=model.joint(meta['primary_joint']).id
        assert not model.jnt_limited[j]
        row=meta['turnstile_locks'];contact_seen=False
        for _ in range(round(1./model.opt.timestep)):
            d.qfrc_applied[model.jnt_dofadr[j]]=20.
            mujoco.mj_step(model,d)
            contact_seen|=any(row['bolt_geom'] in [model.geom(g).name for g in c.geom]
                and any(model.geom(g).name in row['index_geoms'] for g in c.geom) for c in d.contact)
        assert abs(d.qpos[model.jnt_qposadr[j]])<.051
        assert contact_seen
        assert not np.any(d.warning.number)


def test_surface_push_rotates_every_unlocked_rotor_without_coordinate_resets(rotors):
    from doorbench.turnstile_locks import compile_turnstile_locks,apply_turnstile_locks
    from doorbench.turnstile_drop import compile_turnstile_drop,apply_turnstile_drop
    for spec,meta,files in rotors:
        if spec['kinematics'].get('locked_until_credential'):continue
        model=mujoco.MjModel.from_xml_path(files['full']);d=mujoco.MjData(model)
        c=meta['rotating_contacts'][0];sid=model.site(c['site']).id;jid=model.joint(c['joint']).id
        rules=compile_turnstile_locks(model,meta);drops=compile_turnstile_drop(model,meta);previous=mujoco.get_mjcb_passive()
        def callback(native,data):
            if native is model:
                apply_turnstile_locks(native,data,rules)
                apply_turnstile_drop(native,data,drops)
        try:
            mujoco.set_mjcb_passive(callback)
            for _ in range(round(3./model.opt.timestep)):
                mujoco.mj_kinematics(model,d)
                force=-40.*d.site_xmat[sid].reshape(3,3)[:,2]
                d.qfrc_applied[:]=0
                mujoco.mj_applyFT(model,d,force,np.zeros(3),d.site_xpos[sid],model.site_bodyid[sid],d.qfrc_applied)
                mujoco.mj_step(model,d)
        finally:mujoco.set_mjcb_passive(previous)
        assert d.qpos[model.jnt_qposadr[jid]]>.1,(spec['id'],d.qpos)
        assert not np.any(d.warning.number)
