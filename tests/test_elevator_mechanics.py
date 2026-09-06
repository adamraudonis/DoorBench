"""Landing-door load paths and event sequencing in the actual native model."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

from doorbench.build import export_door
from doorbench.spec import generate_all
from doorbench.elevator_qa import run_elevator_qa
from doorbench.benchmark.env import DoorEnv
from doorbench.benchmark.runner import AutoDoorSensor


IDS = tuple(s['id'] for s in generate_all() if s['family']=='elevator')


@pytest.fixture(scope='module')
def doors(tmp_path_factory):
    root=tmp_path_factory.mktemp('elevators');result={}
    for spec in generate_all():
        if spec['id'] not in IDS:continue
        export_door(spec,str(root/'doors'),str(root/'hardware'),formats=('mjcf','json','urdf'))
        result[spec['id']]=root/'doors'/spec['id']
    assert len(result)==8
    return result


@pytest.mark.parametrize('id',IDS)
@pytest.mark.parametrize('tier',('full','simple','minimal'))
def test_native_interlock_cycles_and_contact_removal_controls(doors,id,tier):
    path=doors[id];suffix='' if tier=='full' else '_'+tier
    meta=json.loads((path/'model.json').read_text())['meta']
    m=mujoco.MjModel.from_xml_path(str(path/f'door{suffix}.xml'))
    result=run_elevator_qa(m,meta)
    assert result['ok'],result['failures']
    assert result['cycles']==2 and len(result['negative_controls'])==2


def step(e,sensor,seconds,base=(0.,-2.),press=False):
    button=next(r for r in e.meta['wall_switches'] if r['kind']=='call_button')
    peak=0.;states=set();max_pen=0.
    for _ in range(round(seconds/e.m.opt.timestep)):
        if press:e.apply_site_force(button['site'],[0.,-button['face']*20.,0.])
        sensor.step(base,float(e.d.time));states.add(sensor.elevator_control.state)
        if sensor.elevator_control.state in ('opening','hold','closing'):
            assert all(e.d.qpos[q['hook_joint']]>=r['released_angle_rad'] for r,q,_,_ in sensor.elevator_control.rows)
        e.step()
        peak=max(peak,min(float(e.d.qpos[q['joint']]) for _,q,_,_ in sensor.elevator_control.rows))
        max_pen=max(max_pen,max((-float(c.dist) for c in e.d.contact),default=0.))
    assert not np.any(e.d.warning.number)
    return peak,states,max_pen


@pytest.mark.parametrize('id',IDS)
def test_runtime_physical_call_full_cycle_and_relock(doors,id):
    e=DoorEnv(str(doors[id]));e.reset(randomize=False);sensor=AutoDoorSensor(e)
    try:
        step(e,sensor,.6)
        assert not e.tracker.L.lock_released and not np.any(e.d.ctrl)
        step(e,sensor,.3,press=True)
        peak,states,depth=step(e,sensor,18.)
        assert {'opening','hold','closing','locking','idle'}<=states
        assert peak>=min(r['stroke_m'] for r,_,_,_ in sensor.elevator_control.rows)-.003
        assert sensor.elevator_control.state=='idle' and depth<.001
        assert all(abs(e.d.qpos[q['joint']])<r['closed_m'] and abs(e.d.qpos[q['hook_joint']])<.015
                   for r,q,_,_ in sensor.elevator_control.rows)
    finally:e.close()


def snapshot(m):
    b=np.zeros(mujoco.mj_sizeModel(m),dtype=np.uint8);mujoco.mj_saveModel(m,buffer=b)
    return hashlib.sha256(b.tobytes()).hexdigest()


def test_power_loss_removes_force_without_state_projection_or_phantom_call(doors):
    e=DoorEnv(str(doors[IDS[0]]));e.reset(randomize=False);sensor=AutoDoorSensor(e)
    try:
        step(e,sensor,.3,press=True)
        for _ in range(100):
            step(e,sensor,.01)
            if sensor.elevator_control.state=='opening' and e.d.qpos[e.m.jnt_qposadr[e.pj]]>.05:break
        assert sensor.elevator_control.state=='opening'
        binary=snapshot(e.m);q=e.d.qpos.copy();v=e.d.qvel.copy()
        e.set_elevator_power(False);sensor.step([0.,-2.],float(e.d.time))
        assert snapshot(e.m)==binary
        np.testing.assert_array_equal(q,e.d.qpos);np.testing.assert_array_equal(v,e.d.qvel)
        assert not np.any(e.d.ctrl) and not np.any(e.d.qfrc_applied)
        step(e,sensor,.4)
        assert not np.any(e.d.actuator_force)
        e.set_elevator_power(True);step(e,sensor,.8)
        assert sensor.elevator_control.state=='idle' and not np.any(e.d.ctrl)
        step(e,sensor,.3,press=True);step(e,sensor,1.3)
        assert sensor.elevator_control.state in ('opening','hold')
    finally:e.close()


class ObstructedEnv(DoorEnv):
    def _build(self,with_human):
        assert not with_human
        native=mujoco.MjSpec.from_file(self.xml_path)
        body=native.worldbody.add_body(name='test_obstruction',mocap=True,pos=[0.,-50.,1.])
        body.add_geom(name='test_obstruction_stock',type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[.035,.08,.3],rgba=[.8,.4,.1,1.])
        m=native.compile();return m,mujoco.MjData(m)


def test_closing_reopens_on_real_obstruction_then_returns_after_clearance(doors):
    e=ObstructedEnv(str(doors[IDS[0]]));e.reset(randomize=False);sensor=AutoDoorSensor(e)
    try:
        step(e,sensor,.3,press=True)
        for _ in range(80):
            step(e,sensor,.1)
            if sensor.elevator_control.state=='hold':break
        assert sensor.elevator_control.state=='hold'
        r,q,_,_=sensor.elevator_control.rows[0];leaf=e.m.body(r['leaf']).id
        # The obstruction is placed in the empty aperture, halfway along
        # closing travel. Only this independent mocap object is repositioned.
        direction=float(e.m.jnt_axis[e._jid(r['joint']),0]);W=e.spec['leaf']['width']
        x=float(e.d.xpos[leaf,0])-direction*(W/2+r['stroke_m']*.5-.035)
        mid=e.m.body_mocapid[e.m.body('test_obstruction').id]
        e.d.mocap_pos[mid]=[x,float(e.d.xpos[leaf,1]),1.]
        touched=False;was_closing=False;reopened=False;max_contact=0.;max_pen=0.
        for _ in range(round(12./e.m.opt.timestep)):
            sensor.step([0.,-2.],float(e.d.time));state=sensor.elevator_control.state
            was_closing|=state=='closing'
            if was_closing and state=='opening':reopened=True;break
            e.step()
            for ci,c in enumerate(e.d.contact):
                names={e.m.geom(c.geom1).name,e.m.geom(c.geom2).name}
                if 'test_obstruction_stock' in names:
                    touched=True;f=np.zeros(6);mujoco.mj_contactForce(e.m,e.d,ci,f)
                    max_contact=max(max_contact,float(np.linalg.norm(f[:3])))
                    max_pen=max(max_pen,-float(c.dist))
        assert was_closing and touched and reopened and max_contact>1.
        assert max_pen<.001 and not np.any(e.d.warning.number)
        e.d.mocap_pos[mid]=[0.,-50.,1.]
        _,states,depth=step(e,sensor,18.)
        assert 'hold' in states and sensor.elevator_control.state=='idle' and depth<.001
    finally:e.close()


def test_close_only_fixture_and_presence_hold(doors):
    e=DoorEnv(str(doors[IDS[0]]));e.reset(scenario='close_only',randomize=False);sensor=AutoDoorSensor(e)
    try:
        lane=float(e.d.xpos[e.m.body(sensor.elevator_control.rows[0][0]['leaf']).id,1])
        peak,states,depth=step(e,sensor,10.,base=(0.,lane))
        assert 'hold' in states and 'closing' not in states and depth<.001
        assert peak>=sensor.elevator_control.rows[0][0]['stroke_m']-.003
        _,states,depth=step(e,sensor,18.)
        assert 'closing' in states and sensor.elevator_control.state=='idle' and depth<.001
    finally:e.close()
