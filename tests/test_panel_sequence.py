from types import SimpleNamespace
import numpy as np
import pytest
from doorbench.dexterous.panel_sequence import AttainedPanelSchedule


def panels():
    return [SimpleNamespace(start_time=1.,started=1.,plan=dict(initial_leaf_angle_rad=.3,final_leaf_angle_rad=.75)),
            SimpleNamespace(start_time=2.,started=None,plan=dict(initial_leaf_angle_rad=.745,final_leaf_angle_rad=1.2))]


def test_transition_requires_actual_supported_hold_and_records_attained_aperture():
    c=AttainedPanelSchedule(panels())
    assert not c.advance(1.,.3,2.5)
    for i in range(301):c.observe(1.398+i*.002,1.,.745,2.5)
    assert c.advance(2.,.745,2.5)
    assert c.index==1 and c.completed[0]['attained_aperture_rad']==.745
    assert not c.advance(3.,1.2,2.5)


@pytest.mark.parametrize('failure',['short','contact','position','stale','nan'])
def test_target_references_alone_cannot_authorize_a_handoff(failure):
    c=AttainedPanelSchedule(panels())
    for i in range(301):c.observe(1.398+i*.002,1.,.745,2.5)
    if failure=='short':c.held_since=1.8
    if failure=='stale':c.last_time=1.9
    with pytest.raises(ValueError):c.advance(2.,.7 if failure=='position' else .745,1.9 if failure=='contact' else float('nan') if failure=='nan' else 2.5)


def test_reordered_or_discontinuous_segments_are_rejected():
    p=panels();p[1].start_time=.5
    with pytest.raises(ValueError):AttainedPanelSchedule(p)
    p=panels();p[1].plan['initial_leaf_angle_rad']=.3
    with pytest.raises(ValueError):AttainedPanelSchedule(p)


def test_continuous_stance_snapshot_precedes_fallback_without_authorizing_hold():
    p=panels();p[1].preserve_stance_reference=True;c=AttainedPanelSchedule(p)
    stance=SimpleNamespace(target_root=np.array([.01,-.006,1.]),
        target_rotation=np.array([[1.,0,0],[0,.999,-.044],[0,.044,.999]]),joint_target=np.array([.1,.2]))
    assert c.pending_stance_reference(1.9,stance) is None
    saved=c.pending_stance_reference(2.,stance)
    stance.target_root[:]=0;stance.target_rotation[:]=0;stance.joint_target[:]=0
    assert saved['target_root'][2]==1. and saved['target_rotation'][2,1]==.044
    assert saved['joint_target'].tolist()==[.1,.2]
    with pytest.raises(ValueError):c.advance(2.,.745,2.5)
    # Legacy segments neither capture nor alter their established references.
    assert AttainedPanelSchedule(panels()).pending_stance_reference(2.,stance) is None
