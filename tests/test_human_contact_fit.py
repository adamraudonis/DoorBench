import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.human_reference.contact_fit import (
    fit_capture_legs, smooth_clearance_lift, two_bone_knee, transported_bend_direction,
)


def test_two_bone_preserves_lengths_and_captured_bend_side():
    h = np.array([.2, -.3, 1.])
    k = h + [0., .2, -.4]
    a = h + [0., 0., -.8]
    target = a + [.07, .03, .04]
    result = two_bone_knee(h, k, a, target)
    assert np.linalg.norm(result-h) == pytest.approx(np.linalg.norm(k-h), abs=1e-12)
    assert np.linalg.norm(target-result) == pytest.approx(np.linalg.norm(a-k), abs=1e-12)
    direction = (target-h) / np.linalg.norm(target-h)
    old_bend = k-h-direction*np.dot(k-h, direction)
    new_bend = result-h-direction*np.dot(result-h, direction)
    assert np.dot(old_bend, new_bend) > 0


def test_unreachable_does_not_stretch_or_move_hip():
    with pytest.raises(ValueError, match='unreachable'):
        two_bone_knee([0,0,1], [0,.1,.6], [0,0,.2], [0,0,-1])


def test_smooth_lift_respects_each_surface_bound_and_reduces_sharpness():
    time = np.arange(60)*.01
    minimum = np.full(60, .03)
    minimum[30] = -.02
    lift = smooth_clearance_lift(minimum, time)
    required = np.maximum(0, .001-minimum)
    assert np.all(minimum+lift >= .001-1e-10)
    assert np.all(lift >= 0)
    assert np.linalg.norm(np.diff(lift, n=2)) < np.linalg.norm(np.diff(required, n=2))*.2
    assert lift[29] > 0 and lift[31] > 0


def fixture(yaw=0.):
    names = ['root']
    parent = [-1]
    heads = [np.array([0.,0.,0.])]
    source_names, source_heads = [], []
    for side, sign, label in [('L',-1,'Left'), ('R',1,'Right')]:
        hip = np.array([sign*.08,0.,.90])
        knee = np.array([sign*.08,.08,.51])
        ankle = np.array([sign*.08,0.,.08])
        points = [hip,(hip+knee)/2,knee,(knee+ankle)/2,ankle,ankle+[0,.12,-.05]]
        last = 0
        for name, point in zip(['upperleg01','upperleg02','lowerleg01','lowerleg02','foot','toe2-3'], points):
            names.append(name+'.'+side);parent.append(last);heads.append(point);last=len(names)-1
        source_names.extend(['Character1_'+label+'Foot', 'Character1_'+label+'Foot__end'])
        source_heads.extend([ankle+[sign*.025,0.,0.],ankle+[sign*.025,.13,-.06]])
    world = Rotation.from_euler('Z', yaw).as_matrix()
    heads = np.array(heads)@world.T
    n, b = 10, len(names)
    positions = np.tile(heads,(n,1,1))
    rotations = np.tile(world,(n,b,1,1))
    offsets = np.zeros((b,3))
    for j,p in enumerate(parent):
        if p>=0:offsets[j] = world.T@(heads[j]-heads[p])
    matrix = np.tile(np.eye(4),(n,b,1,1));matrix[:,:,:3,:3] = rotations;matrix[:,:,:3,3] = positions
    return dict(time=np.arange(n)*.01,source_time=np.arange(n)*.01+.02,
                source_frame=np.arange(n)+2,bone_names=np.array(names),parent_index=np.array(parent),
                bone_pos=positions,bone_rot=rotations,bone_tail=positions+rotations[:,:,:,1]*.1,
                bone_matrix=matrix,bone_basis=np.tile(np.eye(4),(n,b,1,1)),
                calibration_parent_offsets=offsets,pelvis_pos=np.tile([0,0,.9],(n,1)),
                source_joint_names=np.array(source_names),
                source_joint_pos=np.tile(np.array(source_heads)@world.T,(n,1,1)),
                source_joint_rot=np.tile(world,(n,4,1,1)))


@pytest.mark.parametrize('yaw', [0., np.pi/2])
def test_leg_fit_restores_source_spacing_in_actual_coordinates_without_side_shift(yaw):
    raw = fixture(yaw)
    originals = {k:v.copy() for k,v in raw.items()}
    adjusted, report = fit_capture_legs(raw, np.full((10,2),-.008))
    names = raw['bone_names'].tolist()
    for side, source_index in [('L',0),('R',2)]:
        foot = names.index('foot.'+side)
        assert np.allclose(adjusted['bone_pos'][:,foot,:2], raw['source_joint_pos'][:,source_index,:2], atol=1e-12)
        assert np.array_equal(adjusted['bone_rot'][:,foot],raw['bone_rot'][:,foot])
        for group in ['upperleg','lowerleg']:
            a,b = [names.index(f'{group}{i:02d}.{side}') for i in [1,2]]
            qa = adjusted['bone_rot'][:,a] @ np.swapaxes(raw['bone_rot'][:,a],-1,-2)
            qb = adjusted['bone_rot'][:,b] @ np.swapaxes(raw['bone_rot'][:,b],-1,-2)
            assert np.allclose(qa,qb,atol=1e-12)
        h,k,a = [names.index(s+'.'+side) for s in ['upperleg01','lowerleg01','foot']]
        for i,j in [(h,k),(k,a)]:
            assert np.allclose(np.linalg.norm(adjusted['bone_pos'][:,i]-adjusted['bone_pos'][:,j],axis=1),
                               np.linalg.norm(raw['bone_pos'][:,i]-raw['bone_pos'][:,j],axis=1),atol=1e-12)
    for key,value in originals.items():assert np.array_equal(value,raw[key])
    for key in ['time','source_time','source_frame','pelvis_pos','source_joint_pos','source_joint_rot']:
        assert np.array_equal(adjusted[key],raw[key])
    assert np.array_equal(adjusted['bone_matrix'][:,0],raw['bone_matrix'][:,0])
    assert report['ankle_endpoint_max_coordinate_error_m'] < 1e-12


def test_invalid_surface_clock_and_nonfinite_data_rejected():
    raw = fixture()
    with pytest.raises(ValueError,match='shoe minima'):
        fit_capture_legs(raw,np.full((10,2),np.nan))
    time = np.arange(10)*.01;time[3] += .001
    with pytest.raises(ValueError,match='uniform capture clock'):
        smooth_clearance_lift(np.zeros(10),time)


def test_shorter_target_leg_gets_explicit_reach_lift_without_stretching():
    raw = fixture()
    raw['source_joint_pos'][:,0,0] = -.28
    raw['source_joint_pos'][:,2,0] = .28
    adjusted, report = fit_capture_legs(raw,np.full((10,2),.05))
    for side in ['L','R']:
        assert report[side]['maximum_fixed_length_reach_lift_m'] > .001
        assert report[side]['minimum_extension_margin_m'] >= .0001-1e-12
        names = raw['bone_names'].tolist()
        h,k,a = [names.index(s+'.'+side) for s in ['upperleg01','lowerleg01','foot']]
        before = raw['bone_pos']
        after = adjusted['bone_pos']
        for u,v in [(h,k),(k,a)]:
            assert np.allclose(np.linalg.norm(before[:,u]-before[:,v],axis=1),
                               np.linalg.norm(after[:,u]-after[:,v],axis=1),atol=1e-12)


def test_submillimetre_straight_knee_noise_does_not_flip_captured_bend_plane():
    hip, ankle, target = np.array([0.,0.,1.]), np.array([0.,0.,.2]), np.array([.03,0.,.23])
    history = np.array([0.,1.,0.])
    poles = []
    for noise in [-.0003,.0003,-.0002]:
        knee = np.array([noise,0.,.6])
        pole, history, report = transported_bend_direction(
            hip,knee,ankle,target,np.eye(3),history)
        poles.append(pole)
        assert report['history_blend_weight'] == 0
        assert pole[1] > .99999
        solved = two_bone_knee(hip,knee,ankle,target,bend_direction=pole)
        assert solved[1] > 0
        assert np.linalg.norm(solved-hip) == pytest.approx(np.linalg.norm(knee-hip),abs=1e-12)
    assert np.max(np.abs(np.diff(poles,axis=0))) < 1e-12
