import copy
import numpy as np
import pytest

from doorbench.dexterous.isaac_standing_continuation_measurements import (
    BODY_NAMES,pack_standing_continuation,sparse_contact_buffer)
from test_isaac_post_opening_measurements import fixture


def inputs():
    base=fixture();capacity=base['capacity']
    # Friction slots intentionally differ from normal slots and include empty rows.
    friction=np.full((capacity,3),np.nan);friction[3]=[.5,0.,0.]
    friction_points=np.full((capacity,3),np.nan);friction_points[3]=[1.,2.,3.]
    counts=np.array([[0],[0],[1],[0]],dtype=int);starts=np.array([[0],[0],[3],[0]])
    normal=np.zeros((4,1,3))
    for i in range(4):normal[i,0]=base['normal_forces'][i,0]*base['normals'][i]
    points=np.full((capacity,3),np.nan);points[:4]=0.
    return dict(time_s=.002,pose_time_s=.002,sensor_paths=base['sensor_paths'],filter_paths=base['filter_paths'],
        normal_matrix=normal,normal_buffers=(base['normal_forces'],points,base['normals'],base['distances'],base['counts'],base['starts']),
        friction_buffers=(friction,friction_points,counts,starts),capacity=capacity,physics_qualified=True,
        body_poses={name:np.array([0.,0.,0.,1.,0.,0.,0.]) for name in BODY_NAMES})


def test_observer_preserves_independent_contact_buffers_and_actual_body_origins():
    args=inputs();row=pack_standing_continuation(**args)
    assert row['contact_interval_s']==[0.,.002]
    assert row['foot_loads_N']==[200.,201.]
    assert row['hand_forces_world_N']['/World/H1/lh_palm']==[.5,-4.,0.]
    assert row['evidence']['right_environment_contacts']==0
    assert row['raw']['normal']['slots']==[0,1,2,3]
    assert row['raw']['friction']['slots']==[3]
    assert row['raw']['friction']['point_world']==[[1.,2.,3.]]
    assert row['raw']['normal_force_pairs']==[[0,0,0.,0.,200.],[1,0,0.,0.,201.],[2,0,0.,-4.,0.]]
    assert list(row['body_poses'])==list(BODY_NAMES) and row['authorized_stages']==0
    args['body_poses'][BODY_NAMES[0]][0]=999.
    assert row['body_poses'][BODY_NAMES[0]][0]==0.


@pytest.mark.parametrize('bad',['stale','off_grid','missing_pose','nonfinite_point','overlap','truncated'])
def test_invalid_current_measurement_is_not_published(bad):
    args=inputs()
    if bad=='stale':args['pose_time_s']=.001
    elif bad=='off_grid':args['time_s']=args['pose_time_s']=.003
    elif bad=='missing_pose':args['body_poses'].pop('leaf')
    elif bad=='nonfinite_point':args['normal_buffers'][1][0,0]=np.nan
    elif bad=='overlap':args['normal_buffers'][-1][1,0]=0
    else:args['normal_buffers'][-2][-1,0]=2
    with pytest.raises((ValueError,KeyError)):pack_standing_continuation(**args)


def test_capture_does_not_promote_failed_backend_evidence():
    args=inputs();args['physics_qualified']=False
    row=pack_standing_continuation(**args)
    assert not row['evidence']['physics_qualified'] and row['authorized_stages']==0
