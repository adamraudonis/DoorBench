import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from doorbench.dexterous.palm_panel_geometry import flatten_palm_goal


def test_entire_collision_support_plane_is_preserved_during_flattening():
    rng=np.random.default_rng(4);vertices=rng.uniform(-.04,.04,(70,3));position=np.array([.2,-.05,.99])
    base=Rotation.from_euler('xyz',[80,15,25],degrees=True).as_matrix()
    support=max((vertices@base.T+position)[:,1])
    for u in np.linspace(0,1,101):
        p,r,shift=flatten_palm_goal(position,base,vertices,float(u))
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        assert np.linalg.det(r)>0
        assert abs(max((vertices@r.T+p)[:,1])-support)<1e-12
        np.testing.assert_array_equal(p[[0,2]],position[[0,2]])
    np.testing.assert_allclose(r[:,2],[0,-1,0],atol=1e-12)


def test_initial_pose_is_preserved_and_invalid_geometry_rejected():
    vertices=np.array([[0,0,0],[.03,0,0],[0,.04,0],[0,0,-.02]])
    r=Rotation.from_euler('x',1.).as_matrix();p=np.array([1.,2.,3.])
    pp,rr,shift=flatten_palm_goal(p,r,vertices,0.)
    np.testing.assert_array_equal(pp,p);np.testing.assert_allclose(rr,r,atol=1e-12);assert shift==0
    with pytest.raises(ValueError):flatten_palm_goal(p,np.ones((3,3)),vertices,1.)
