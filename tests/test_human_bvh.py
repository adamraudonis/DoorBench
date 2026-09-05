import numpy as np
import pytest

from doorbench.human_reference.bvh import forward_kinematics, parse_bvh


FIXTURE = '''HIERARCHY
ROOT pelvis {
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT child {
  OFFSET 0 0 10
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site { OFFSET 0 5 0 }
 }
}
MOTION
Frames: 2
Frame Time: 0.01
100 200 300 90 90 0 0 0 0
100 200 300 90 90 0 90 0 0
'''


def test_declared_intrinsic_order_and_parent_relative_offsets():
    capture = parse_bvh(FIXTURE)
    pos, rot = forward_kinematics(capture, length_scale=.01)
    np.testing.assert_allclose(pos[:, 0], [[1, 2, 3]] * 2)
    # Ry(90) Rx(90) maps local +Z to world -Y, not +X.
    np.testing.assert_allclose(pos[:, 1], [[1, 1.9, 3]] * 2)
    np.testing.assert_allclose(pos[0, 2], [1.05, 1.9, 3])
    np.testing.assert_allclose(pos[1, 2], [1, 1.9, 3.05])
    np.testing.assert_allclose(rot @ rot.swapaxes(-1, -2), np.broadcast_to(np.eye(3), rot.shape), atol=1e-12)
    np.testing.assert_array_equal(capture.times, [0, .01])
    assert capture.joints[-1].end_site


def test_coordinate_change_preserves_rotated_points_and_handedness():
    capture = parse_bvh(FIXTURE)
    basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    p, r = forward_kinematics(capture, length_scale=.01, basis=basis)
    p0, r0 = forward_kinematics(capture, length_scale=.01)
    np.testing.assert_allclose(p, p0 @ basis.T)
    for frame in range(2):
        for bone in range(3):
            np.testing.assert_allclose(r[frame, bone], basis @ r0[frame, bone] @ basis.T)
    with pytest.raises(ValueError, match='proper'):
        forward_kinematics(capture, length_scale=.01, basis=np.diag([-1, 1, 1]))


@pytest.mark.parametrize('text', [
    FIXTURE.replace('Frames: 2', 'Frames: 3'),
    FIXTURE.replace('0.01', 'nan'),
    FIXTURE.replace('100 200 300', 'nan 200 300'),
    FIXTURE.replace('JOINT child', 'JOINT pelvis'),
    FIXTURE.replace('Yrotation Xrotation Zrotation', 'Yrotation Yrotation Zrotation'),
])
def test_rejects_inconsistent_or_nonfinite_capture(text):
    with pytest.raises(ValueError):
        parse_bvh(text)
