"""Regression for dense-reward termination incentives, independent of assets."""
from types import SimpleNamespace
import numpy as np
from doorbench.dexterous.reach_training import ReachTeacherEnv


def task(continue_after_success):
    env=ReachTeacherEnv.__new__(ReachTeacherEnv)
    diag={'torso_tilt_deg':0.,'root_height_m':.95,'finite':True,'numerical_warnings':0}
    env.sim=SimpleNamespace(step=lambda *a,**k:None,
        d=SimpleNamespace(site=lambda name:SimpleNamespace(xpos=np.zeros(3)),qvel=np.zeros(3)),
        diagnostics=lambda:diag,root_vadr=0)
    env.controller=SimpleNamespace(body_action=lambda a:a,observation=lambda:np.zeros(61))
    env.steps=0;env.horizon=300;env.success_steps=0;env.reached_success=False
    env.targets=np.zeros((2,3));env.previous_action=np.zeros(19);env.standing_weight=4
    env.continue_after_success=continue_after_success
    return env,diag


def test_training_success_keeps_reward_but_evaluation_stops():
    for continuing in (True,False):
        env,_=task(continuing)
        for _ in range(25):_,reward,terminated,truncated,info=env.step(np.zeros(19))
        assert info['is_success']
        assert terminated is (not continuing)
        assert not truncated and reward>0


def test_fall_after_training_reach_is_not_reported_as_success():
    env,diag=task(True)
    for _ in range(25):env.step(np.zeros(19))
    diag['root_height_m']=.3
    _,_,terminated,_,info=env.step(np.zeros(19))
    assert terminated and info['fell'] and not info['is_success']
