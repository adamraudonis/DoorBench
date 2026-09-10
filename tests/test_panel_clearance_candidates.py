import numpy as np
from scripts.dexterous.screen_whole_body_panel import clearance_candidate_pairs


def test_plane_pruning_keeps_straddling_and_below_plane_shapes():
    centers=np.array([[0.,0.,1.],[0.,0.,.03],[0.,0.,-1.]])
    pairs=clearance_candidate_pairs(centers,np.full(3,.02),np.zeros((1,3)),
                                    np.array([np.inf]),[(0,np.array([0.,0.,1.]))],.04)
    assert pairs.tolist()==[[1,0],[2,0]]


def test_rotated_translated_plane_uses_world_normal():
    centers=np.array([[2.,0.,0.],[.5,0.,0.],[1.03,0.,0.]])
    pairs=clearance_candidate_pairs(centers,np.full(3,.02),np.array([[1.,0.,0.]]),
                                    np.array([np.inf]),[(0,np.array([1.,0.,0.]))],.04)
    assert pairs.tolist()==[[1,0],[2,0]]


def test_pruning_preserves_capped_exact_sphere_and_plane_minimum():
    rng=np.random.default_rng(23)
    centers=rng.uniform(-2,2,(200,3));radii=rng.uniform(.001,.1,200)
    scene=np.array([[0.,0.,0.],[.8,.2,-.1]])
    sr=np.array([np.inf,.2]);normal=np.array([.6,0.,.8]);cap=.05
    pairs=clearance_candidate_pairs(centers,radii,scene,sr,[(0,normal)],cap)
    distances=np.column_stack([centers@normal-radii,np.linalg.norm(centers-scene[1],axis=1)-radii-.2])
    expected=set(map(tuple,np.argwhere(distances<=cap)))
    assert expected.issubset(set(map(tuple,pairs)))
    assert min(cap,*[distances[i,j] for i,j in pairs])==min(cap,distances.min())
