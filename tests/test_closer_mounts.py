"""Connected closer load paths and exact native loop inspection regressions."""
import json
import mujoco
import numpy as np
import pytest

from doorbench import hardware as H
from doorbench.spec import generate_all
from doorbench.build import build_model,export_door
from doorbench.closer_mount_qa import run_closer_mount_qa
from doorbench.geometry.closer_mounts import resolve_closer_configuration


@pytest.fixture(scope='module')
def closers(tmp_path_factory):
    root=tmp_path_factory.mktemp('closer-mounts');rows=[]
    for spec in generate_all():
        if H.CLOSERS[spec['closer']['model']].kind not in ('surface_overhead','electromagnetic_hold'):continue
        ir=build_model(spec)
        if not ir.meta.get('closer_mounts'):continue
        ex=export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json'))
        meta=json.loads((root/'doors'/spec['id']/'model.json').read_text())['meta']
        rows.append((spec,meta,ex['files']['mjcf']['full']))
    assert len(rows)==196
    assert sum(len(meta['closer_mounts']) for _,meta,_ in rows)==233
    return rows


def test_all_233_mounted_closers_have_real_supports_and_clear_pivot_bores(closers):
    for spec,meta,path in closers:
        for filename in ('door.xml','door_simple.xml','door_minimal.xml'):
            from pathlib import Path
            model=mujoco.MjModel.from_xml_path(str(Path(path).with_name(filename)))
            report=run_closer_mount_qa(model,meta)
            assert report['ok'],(spec['id'],filename,report)
            assert report['minimum_shoe_bore_gap_m']>.0019


def test_old_shaft_and_frame_mount_float_defects_fail(closers):
    _,meta,path=next(x for x in closers if x[0]['id']=='db0061_swing_single')
    for name,offset,key in [('closer_pinion_shaft',.15,'detached_pinion'),
                            ('closer_bracket',.15,'detached_frame_shoe')]:
        model=mujoco.MjModel.from_xml_path(path);model.geom_pos[model.geom(name).id,1]+=offset
        report=run_closer_mount_qa(model,meta)
        assert not report['ok']
        assert any(key in x for x in report['failures'] if isinstance(x,dict))


def test_closed_form_inspection_does_not_move_native_leaf_or_mutate_model(closers):
    for spec,meta,path in closers:
        if spec['id'] not in ('db0012_swing_single','db0061_swing_single','db0188_cold_storage'):continue
        model=mujoco.MjModel.from_xml_path(path);original=model.qpos0.copy()
        j=model.joint(meta['primary_joint']).id;q=model.qpos0.copy();adr=model.jnt_qposadr[j];q[adr]=.6
        for e in range(model.neq):
            if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0:
                j1,j2=model.eq_obj1id[e],model.eq_obj2id[e];x=q[model.jnt_qposadr[j2]]
                q[model.jnt_qposadr[j1]]=sum(model.eq_data[e,k]*x**k for k in range(5))
        resolve_closer_configuration(model,q,meta)
        assert q[adr]==.6
        np.testing.assert_array_equal(model.qpos0,original)
        d=mujoco.MjData(model);d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        for row in meta['closer_mounts']:
            eq=model.equality(row['connect']).id;b1,b2=model.eq_obj1id[eq],model.eq_obj2id[eq]
            p1=d.xpos[b1]+d.xmat[b1].reshape(3,3)@model.eq_data[eq,:3]
            p2=d.xpos[b2]+d.xmat[b2].reshape(3,3)@model.eq_data[eq,3:6]
            assert np.linalg.norm(p1-p2)<2e-6


def test_native_spring_only_at_pinion_has_observable_leaf_effect(closers):
    _,meta,path=next(x for x in closers if x[0]['id']=='db0016_swing_single')
    angles=[]
    for remove_spring in (False,True):
        model=mujoco.MjModel.from_xml_path(path);j=model.joint(meta['primary_joint']).id
        assert model.jnt_stiffness[j]==0.
        pinion=model.joint(meta['closer_mounts'][0]['main_joint']).id
        assert model.jnt_stiffness[pinion]>0.
        if remove_spring:model.jnt_stiffness[pinion]=0.
        d=mujoco.MjData(model);d.qpos[model.jnt_qposadr[j]]=.6
        resolve_closer_configuration(model,d.qpos,meta)
        for _ in range(round(2./model.opt.timestep)):mujoco.mj_step(model,d)
        angles.append(float(d.qpos[model.jnt_qposadr[j]]))
        assert not np.any(d.warning.number)
    assert angles[0]<.5
    assert abs(angles[1]-.6)<.03


def test_authored_pinion_curve_matches_native_loop_and_spring_energy(closers):
    for spec,meta,path in closers:
        model=mujoco.MjModel.from_xml_path(path)
        for row in meta['closer_pinion_calibration']:
            j=model.joint(row['leaf_joint']).id;pinion=model.joint(row['pinion_joint']).id
            table=row['table'];theta=np.asarray(table['door_angle_rad'])
            for index in np.linspace(2,len(theta)-3,31,dtype=int):
                q=model.qpos0.copy();q[model.jnt_qposadr[j]]=theta[index]
                for e in range(model.neq):
                    if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0:
                        j1,j2=model.eq_obj1id[e],model.eq_obj2id[e];x=q[model.jnt_qposadr[j2]]
                        q[model.jnt_qposadr[j1]]=sum(model.eq_data[e,k]*x**k for k in range(5))
                resolve_closer_configuration(model,q,meta)
                phi=float(q[model.jnt_qposadr[pinion]])
                assert abs(phi-table['pinion_angle_rad'][index])<1e-5,(spec['id'],index)
                dq=(table['pinion_angle_rad'][index+1]-table['pinion_angle_rad'][index-1])/(theta[index+1]-theta[index-1])
                native=-model.jnt_stiffness[pinion]*(phi-model.qpos_spring[model.jnt_qposadr[pinion]])*dq
                assert np.isclose(-native,table['achieved_door_torque_Nm'][index],rtol=2e-4,atol=.002)
            assert np.min(table['pinion_ratio_abs'])>.05
            mount=next(x for x in meta['closer_mounts'] if x['main_joint']==row['pinion_joint'])
            assert model.geom_bodyid[model.geom(mount['shaft_geom']).id]==model.jnt_bodyid[pinion]


def test_pinion_valves_are_passive_and_do_not_change_state(closers):
    from doorbench.closer_pinion import compile_pinion_closers,apply_pinion_closers
    signs=set()
    for spec,meta,path in closers:
        if len(signs)==2:break
        if meta['closer_pinion_laws'][0]['opening_sign'] in signs:continue
        model=mujoco.MjModel.from_xml_path(path);rules=compile_pinion_closers(model,meta);rule=rules[0];signs.add(rule.opening_sign)
        for angle in (.05,.7,1.5):
            for velocity in (-.4,.4):
                d=mujoco.MjData(model);d.qpos[rule.leaf_qpos]=angle;d.qvel[rule.pinion_dof]=velocity
                pos,vel=d.qpos.copy(),d.qvel.copy();d.qfrc_passive[rule.pinion_dof]=-rule.base*velocity
                apply_pinion_closers(model,d,rules)
                assert float(np.dot(d.qfrc_passive,d.qvel))<=0
                np.testing.assert_array_equal(d.qpos,pos);np.testing.assert_array_equal(d.qvel,vel)
                assert np.count_nonzero(d.qfrc_passive)==1
                expected=rule.opening+(rule.backcheck if rule.backcheck_angle and angle>rule.backcheck_angle else 0.) if rule.opening_sign*velocity>=0 else (rule.latch if angle<rule.latch_angle else rule.sweep)
                assert np.isclose(d.qfrc_passive[rule.pinion_dof],-expected*velocity)
    assert signs=={-1.,1.}


def test_pinion_spring_native_return_has_bounded_speed_and_loop_error(closers):
    from doorbench.closer_pinion import compile_pinion_closers,apply_pinion_closers
    for spec,meta,path in closers:
        if spec['id'] not in ('db0012_swing_single','db0016_swing_single','db0188_cold_storage','db0773_swing_single','db0972_swing_single','db0206_swing_single','db0585_cold_storage','db0908_swing_single','db0937_cold_storage'):continue
        model=mujoco.MjModel.from_xml_path(path);j=model.joint(meta['primary_joint']).id;adr=model.jnt_qposadr[j]
        for start in (np.pi/2,np.pi/12):
            d=mujoco.MjData(model);d.qpos[adr]=start
            for e in range(model.neq):
                if model.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e]>=0:
                    j1,j2=model.eq_obj1id[e],model.eq_obj2id[e];x=d.qpos[model.jnt_qposadr[j2]]
                    d.qpos[model.jnt_qposadr[j1]]=sum(model.eq_data[e,k]*x**k for k in range(5))
            resolve_closer_configuration(model,d.qpos,meta)
            rules=compile_pinion_closers(model,meta);previous=mujoco.get_mjcb_passive();maxerr=0.
            try:
                mujoco.set_mjcb_passive(lambda m,d:apply_pinion_closers(m,d,rules))
                for step in range(round(20./model.opt.timestep)):
                    mujoco.mj_step(model,d)
                    if step%10==0:
                        check=mujoco.MjData(model);check.qpos[:]=d.qpos;mujoco.mj_kinematics(model,check)
                        eq=model.equality(meta['closer_mounts'][0]['connect']).id;b1,b2=model.eq_obj1id[eq],model.eq_obj2id[eq]
                        a=check.xpos[b1]+check.xmat[b1].reshape(3,3)@model.eq_data[eq,:3]
                        b=check.xpos[b2]+check.xmat[b2].reshape(3,3)@model.eq_data[eq,3:6]
                        maxerr=max(maxerr,np.linalg.norm(a-b))
                    if d.qpos[adr]<np.pi/180:break
            finally:mujoco.set_mjcb_passive(previous)
            assert not np.any(d.warning.number),(spec['id'],d.warning.number)
            assert d.qpos[adr]<np.pi/180,(spec['id'],start,float(d.qpos[adr]))
            assert maxerr<.001,(spec['id'],maxerr)
            assert abs(d.qvel[model.jnt_dofadr[j]])<.6


def test_static_projection_reads_actual_pinion_spring_without_mutating_state(closers):
    from doorbench.closer_pinion import projected_static_resistance
    for spec,meta,path in closers:
        if spec['id'] not in ('db0012_swing_single','db0188_cold_storage'):continue
        model=mujoco.MjModel.from_xml_path(path);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        before={name:getattr(data,name).copy() for name in ('qpos','qvel','qacc','qfrc_passive','qfrc_applied')}
        report=projected_static_resistance(model,data,meta)
        for name,expected in before.items():np.testing.assert_array_equal(getattr(data,name),expected)
        achieved=meta['closer_pinion_calibration'][0]['table']['achieved_door_torque_Nm'][0]
        # Gravity lift contributes separately and must not erase actual spring resistance.
        assert report['static_resistance']>=achieved*.995
        if spec['family']!='cold_storage':assert np.isclose(report['static_resistance'],achieved,rtol=.001)
        assert report['frictionloss']>0


def test_real_delayed_valve_slowly_traverses_zone_without_a_pose_hold(closers,tmp_path):
    from dataclasses import replace
    from doorbench.closer_pinion import compile_pinion_closers,apply_pinion_closers
    covered=[]
    for spec,meta,path in closers:
        if spec['closer']['model']!='lcn_4040_delayed':continue
        # An independently engaged delayed-egress lock is a separate device.
        # Use an explicitly unlocked generated fixture to isolate its closer.
        if spec['lock']['engaged']:
            import copy
            source=copy.deepcopy(spec);source['lock']['engaged']=False
            export=export_door(source,str(tmp_path/'doors'),str(tmp_path/'hardware'),formats=('mjcf','json'))
            path=export['files']['mjcf']['full']
        model=mujoco.MjModel.from_xml_path(path);primary=model.joint(meta['primary_joint']).id;adr=model.jnt_qposadr[primary]
        rules=compile_pinion_closers(model,meta);assert rules[0].delay_angle is not None
        durations=[]
        for enabled in (True,False):
            active=rules if enabled else tuple(replace(r,delay_angle=None,delay=0.) for r in rules)
            data=mujoco.MjData(model);data.qpos[adr]=np.pi/2;resolve_closer_configuration(model,data.qpos,meta)
            previous=mujoco.get_mjcb_passive();positions=[]
            try:
                mujoco.set_mjcb_passive(lambda m,d:apply_pinion_closers(m,d,active))
                for step in range(round(45./model.opt.timestep)):
                    mujoco.mj_step(model,data)
                    if step%100==0:positions.append(float(data.qpos[adr]))
                    if data.qpos[adr]<=rules[0].delay_angle:break
            finally:mujoco.set_mjcb_passive(previous)
            assert not np.any(data.warning.number)
            assert data.qpos[adr]<=rules[0].delay_angle,(spec['id'],enabled,float(data.qpos[adr]),float(data.qvel[adr]))
            durations.append(float(data.time))
            if enabled:
                assert positions[-1]<positions[1]-.2
                assert np.max(np.abs(np.diff(positions)))<.025
        target=meta['closer_pinion_laws'][0]['delay_time_target_s']
        assert .8*target<durations[0]<1.2*target,(spec['id'],durations,target)
        assert durations[0]>4*durations[1],(spec['id'],durations)
        covered.append(spec['id'])
    assert len(covered)==5
