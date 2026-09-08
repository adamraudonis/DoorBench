from types import SimpleNamespace

import numpy as np
import pytest

from doorbench.dexterous.walking_opening_teacher import WalkingOpeningTeacher


def test_physical_handoff_retains_episode_interval_and_starts_fresh_hold_clock():
    received=[]
    def force(*args,**kwargs):
        received.append((args,kwargs))
        return np.zeros(61),{}
    obj=WalkingOpeningTeacher.__new__(WalkingOpeningTeacher)
    obj.opening=SimpleNamespace(force=force)
    obj.acquisition_started=25.102
    pose=np.array([0.,0.,0.,1.,0.,0.,0.])
    for tick in range(2):
        t=tick*.002;absolute=obj.acquisition_started+t
        obj._measurement=dict(episode_time_s=absolute,pose_time_s=absolute,
            right_palm_pose=pose,evidence={'hand_contact_count':0},
            contact_interval_s=np.array([absolute-.002,absolute]))
        _,info=obj._manipulation_force(t,np.zeros(13),{}, {},pose,pose,{}, {},False)
        args,kwargs=received[-1]
        assert args[0]==t
        assert kwargs['pose_time_s']==pytest.approx(t)
        assert kwargs['contact_interval_s']==pytest.approx([max(0.,t-.002),t])
        assert info['episode_contact_interval_s']==pytest.approx([absolute-.002,absolute])
    # The source measurements retain the preceding preparation interval; only
    # the manipulation-local qualification clock starts at zero.
    assert received[0][1]['evidence']['hand_contact_count']==0


def test_stale_handoff_measurement_cannot_start_manipulation():
    obj=WalkingOpeningTeacher.__new__(WalkingOpeningTeacher)
    obj.acquisition_started=25.
    obj._measurement=dict(episode_time_s=25.)
    with pytest.raises(ValueError,match='skip or repeat'):
        obj._manipulation_force(.002,None,{}, {},None,None,{}, {},False)


def test_composition_rejects_force_epochs_from_a_different_episode_step():
    obj=WalkingOpeningTeacher.__new__(WalkingOpeningTeacher)
    obj.opening=SimpleNamespace(physics_dt=.002)
    with pytest.raises(ValueError,match='contact interval'):
        obj.force(3.,None,{}, {},[],None,None,{}, {},evidence={},right_palm_pose=None,
                  pose_time_s=3.,contact_interval_s=[2.996,2.998])
