import mujoco
import pytest
from doorbench.dexterous.hand_surface_audit import native_hand_surface_loads


@pytest.mark.parametrize('surface_first',[True,False])
def test_palm_is_separate_from_finger_load_for_both_geom_orders(surface_first):
    surface='<body name="leaf"><geom name="panel" type="box" size="1 .1 1"/></body>'
    hands=''.join(f'<body name="robot/lh_{name}" pos="{x} -.201 0"><freejoint/><geom type="box" size=".08 .1 .08" mass="1"/></body>' for name,x in [('palm',-.2),('ffdistal',.2)])
    model=mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".002" gravity="0 9.81 0"/><worldbody>{surface+hands if surface_first else hands+surface}</worldbody></mujoco>')
    data=mujoco.MjData(model)
    for _ in range(500):mujoco.mj_step(model,data)
    receipt=native_hand_surface_loads(model,data)
    assert 9<receipt['palm_normal_load_N']<11
    assert 18<receipt['total_normal_load_N']<22
    assert set(receipt['body_normal_loads_N'])=={'robot/lh_palm','robot/lh_ffdistal'}
    assert native_hand_surface_loads(model,data,side='rh')['total_normal_load_N']==0
