import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.palm_panel_geometry import twist_palm_goal


def test_twist_preserves_collision_vertex_plane_depths():
    rng=np.random.default_rng(42)
    vertices=rng.normal(size=(100,3))*.05
    for angle in [-.6,-.35,0.,.35,.6]:
        initial=Rotation.random(random_state=rng).as_matrix()
        target=twist_palm_goal(initial,angle)
        np.testing.assert_allclose((vertices@target.T)[:,1],(vertices@initial.T)[:,1],atol=1e-15)
        np.testing.assert_allclose(target.T@target,np.eye(3),atol=1e-14)


@pytest.mark.parametrize('angle',[float('nan'),float('inf'),.601,-.601])
def test_rejects_unbounded_twist(angle):
    with pytest.raises(ValueError):twist_palm_goal(np.eye(3),angle)
