import numpy as np
from doorbench.dexterous.panel_chain_projection import project_panel_chain


def fixture():
    rng=np.random.default_rng(56);a=rng.normal(size=(8,8))
    return np.arange(10,18),a.T@a+np.eye(8),rng.normal(size=8),rng.normal(size=8)


def test_waist_is_included_in_complete_normal_acceleration_identity():
    indices,mass,jac,bias=fixture();forces=np.zeros(61);forces[indices[0]]=150
    caps=np.tile([-200.,200.],(61,1))
    result,info=project_panel_chain(forces,indices,mass,jac,bias,8.,1.,caps)
    inverse=np.linalg.solve(mass,jac)
    assert np.isclose(inverse@(result[indices]-bias),8.*(jac@inverse))
    assert result[indices[0]]!=forces[indices[0]]
    assert np.array_equal(result[:10],forces[:10])


def test_complete_force_assembly_preserves_projected_waist_after_leg_override():
    indices,mass,jac,bias=fixture();forces=np.linspace(-8,8,61)
    caps=np.tile([-200.,200.],(61,1))
    projected,_=project_panel_chain(forces,indices,mass,jac,bias,8.,1.,caps)
    consumed=projected.copy();consumed[:10]=np.arange(10)*.2
    np.testing.assert_array_equal(consumed[indices],projected[indices])


def test_original_caps_and_zero_blend_preserve_force_contract():
    indices,mass,jac,bias=fixture();forces=np.linspace(-8,8,61)
    caps=np.tile([-200.,200.],(61,1));caps[indices[1:]]=[-1.,1.]
    result,_=project_panel_chain(forces,indices,mass,jac,bias,12.,1.,caps)
    assert np.all(result<=caps[:,1]) and np.all(result>=caps[:,0])
    initial=np.clip(forces,caps[:,0],caps[:,1])
    result,_=project_panel_chain(initial,indices,mass,jac,bias,12.,0.,caps)
    np.testing.assert_array_equal(initial,result)
