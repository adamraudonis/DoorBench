import importlib.util
from pathlib import Path
import numpy as np
import pytest
from doorbench.dexterous.sensor_actor import ActorDimensions
spec=importlib.util.spec_from_file_location('replay_balance',Path(__file__).parents[1]/'scripts/dexterous/replay_isaac_sensor_balance.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    dims=ActorDimensions(tactile=6)
    initial={k:np.zeros(shape,np.uint8 if k.startswith('rgb_') else np.float32) for k,shape in dims.shapes.items()}
    initial.update(previous_action=np.zeros(61,np.float32),sensor_time_s=np.full(7,-1.),sensor_valid=np.zeros(7,bool),time_s=np.array(0.),motor_forces=np.zeros(61))
    numeric={k:np.repeat(v[None],3,axis=0) for k,v in initial.items() if k not in ('time_s','motor_forces','rgb_left','rgb_right')}
    numeric['time_s']=np.array([.002,.004,.006]);numeric['sensor_valid'][:,:5]=True
    numeric['sensor_time_s'][:,:5]=numeric['time_s'][:,None]
    numeric['joint_position'][:,0]=[.1,.2,.3]
    rgb=dict(time_s=np.array([.002,.004]),rgb_left=np.zeros((2,128,128,3),np.uint8),rgb_right=np.zeros((2,128,128,3),np.uint8))
    rgb['rgb_left'][0]=10;rgb['rgb_left'][1]=20
    return initial,numeric,rgb,dims


def test_cold_decision_and_prior_endpoint_are_separate():
    args=fixture();cold,t=m.decision_packet(0,*args);first,now=m.decision_packet(1,*args)
    assert t==0. and not cold['sensor_valid'].any() and cold['joint_position'][0]==0.
    assert now==.002 and first['joint_position'][0]==np.float32(.1)
    first['joint_position'][0]=999
    assert args[1]['joint_position'][0,0]==np.float32(.1)


def test_future_camera_never_replaces_missing_actual_frame():
    initial,numeric,rgb,dims=fixture();numeric['sensor_valid'][0,5]=True;numeric['sensor_time_s'][0,5]=.002
    packet,_=m.decision_packet(1,initial,numeric,rgb,dims)
    assert np.all(packet['rgb_left']==10)
    rgb={k:v[1:] for k,v in rgb.items()}
    with pytest.raises(ValueError,match='causal camera'):m.decision_packet(1,initial,numeric,rgb,dims)


def test_gapped_command_clock_is_rejected():
    args=fixture();args[1]['time_s'][0]=.004
    with pytest.raises(ValueError,match='clock'):m.decision_packet(1,*args)
