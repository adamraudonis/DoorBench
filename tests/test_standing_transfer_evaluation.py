import copy
from doorbench.dexterous.standing_transfer_evaluation import standing_transfer_checks


def rows():
    return [dict(time_s=i*.1,leaf_pose=[0,0,0,1,0,0,0],surface=dict(body_panel_forces_world_N={'lh_palm':[0,-2,0]},palm_normal_load_N=2),stance_status='solved') for i in range(1,11)]


def test_full_clock_and_actual_palm_support_required():
    assert all(standing_transfer_checks(iter(rows()),seconds=1,dt=.1,started_s=.2).values())
    for data,start,key in [(rows()[:-1],.2,'complete_transfer_clock'),(rows(),None,'standing_transfer_started')]:
        assert not standing_transfer_checks(iter(data),seconds=1,dt=.1,started_s=start)[key]
    data=rows();data[-1]['surface']['body_panel_forces_world_N']['lh_palm']=[0,0,0]
    result=standing_transfer_checks(iter(data),seconds=1,dt=.1,started_s=.2)
    assert not result['measured_palm_load_accounting'] and not result['final_left_palm_support']


def test_missing_interval_and_failed_stance_cannot_pass():
    data=rows();data[3]['time_s']=.3;data[4]['stance_status']='maximum iterations reached'
    result=standing_transfer_checks(iter(data),seconds=1,dt=.1,started_s=.2)
    assert not result['complete_transfer_clock'] and not result['stance_solves_every_interval']
