import mujoco
from doorbench.dexterous.transfer_contact_geometry import receiving_palm_gap


def scene(palm_y):
    m=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
    <body name="leaf"><geom name="leaf_slab" type="box" size=".5 .022 1"/></body>
    <body name="robot/lh_palm" pos="0 {palm_y} 0"><geom type="sphere" size=".02"/></body>
    <body name="robot/lh_ffdistal" pos="0 -.042 0"><geom type="sphere" size=".02"/></body>
    </worldbody></mujoco>''')
    return m,mujoco.MjData(m)


def test_finger_contact_cannot_qualify_palm_ten_centimeters_short():
    result=receiving_palm_gap(*scene(-.142))
    assert not result['passed']
    assert abs(result['minimum_palm_slab_distance_m']-.1)<1e-8


def test_palm_within_capture_range_and_penetration_rejection():
    assert receiving_palm_gap(*scene(-.046))['passed']
    assert not receiving_palm_gap(*scene(-.02))['passed']
