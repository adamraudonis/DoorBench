"""Recurrent vision/tactile/proprioception actor with a strict numeric boundary.

The model receives no task geometry, object identity, phase or absolute episode
clock. Acquisition times become relative sensor ages. Training this architecture
does not establish closed-loop success; that requires independent plant trials.
"""
import numpy as np
import torch
from torch import nn

from .sensor_contract import SENSOR_KEYS, validate_actor_packet, ActorDimensions



def prepare_actor_packet(packet, now_s, dimensions):
    """Copy only validated sensors; normalize physical units with fixed scales."""
    validate_actor_packet(packet,dimensions.shapes,dimensions.actions)
    if not np.isfinite(now_s) or now_s<0:
        raise ValueError('A finite local sensor clock is required')
    times=packet['sensor_time_s']
    if np.any(times>now_s+1e-8):
        raise ValueError('An actor cannot consume future sensor observations')
    if any(packet[key].dtype!=np.uint8 for key in ('rgb_left','rgb_right')):
        raise ValueError('Policy pixels must be unannotated uint8 sensor images')
    valid=packet['sensor_valid'].astype(bool)
    ages=np.where(times>=0,np.clip(now_s-times,0.,1.),1.)
    def sensor(key,scale):
        # The validity mask must actually remove dropped/stale payloads.
        value=packet[key] if valid[SENSOR_KEYS.index(key)] else np.zeros_like(packet[key])
        return np.clip(value/scale,-5.,5.)
    proprio=np.r_[sensor('joint_position',np.pi),sensor('joint_velocity',10.),
                  sensor('imu_gyro',10.),sensor('imu_accelerometer',20.),
                  np.clip(packet['previous_action'],-1.,1.),ages,valid.astype(float)].astype(np.float32)
    tactile=np.arcsinh(sensor('tactile',20.)*20.)/np.arcsinh(100.)
    images=np.stack([sensor(key,255.).transpose(2,0,1) for key in ('rgb_left','rgb_right')])
    return dict(proprio=proprio,tactile=tactile.astype(np.float32),images=images.astype(np.float32))


class SensorActor(nn.Module):
    """Shared stereo CNN, local touch encoder and recurrent capped-force actor.

    Hidden state belongs to one episode and must be reset between episodes.
    Outputs are normalized native motor forces, not direct door/root commands.
    """
    def __init__(self,dimensions=ActorDimensions()):
        super().__init__();self.dimensions=dimensions
        self.vision=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,32,3,2,1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(256,64),nn.SiLU())
        self.touch=nn.Sequential(nn.Linear(dimensions.tactile,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.proprio=nn.Sequential(nn.Linear(dimensions.proprio_dimension,128),nn.LayerNorm(128),nn.SiLU())
        self.memory=nn.GRU(320,dimensions.hidden,batch_first=True)
        self.action=nn.Sequential(nn.Linear(dimensions.hidden,dimensions.actions),nn.Tanh())

    def forward(self,proprio,tactile,images,hidden=None):
        if proprio.ndim!=3 or tactile.ndim!=3 or images.ndim!=6:
            raise ValueError('Expected batch, time and sensor dimensions')
        batch,steps=proprio.shape[:2];d=self.dimensions
        if proprio.shape!=(batch,steps,d.proprio_dimension) or tactile.shape!=(batch,steps,d.tactile) or images.shape!=(batch,steps,2,3,d.image_size,d.image_size):
            raise ValueError('Actor tensor dimensions differ from its frozen sensor contract')
        vision=self.vision(images.reshape(batch*steps*2,3,d.image_size,d.image_size)).reshape(batch,steps,128)
        features=torch.cat((vision,self.touch(tactile),self.proprio(proprio)),dim=-1)
        sequence,hidden=self.memory(features,hidden)
        return self.action(sequence),hidden

    @torch.inference_mode()
    def act(self,packet,now_s,hidden=None):
        values=prepare_actor_packet(packet,now_s,self.dimensions)
        device=next(self.parameters()).device
        inputs={key:torch.as_tensor(value,device=device)[None,None] for key,value in values.items()}
        action,hidden=self(**inputs,hidden=hidden)
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite sensor actor output')
        return action[0,0].cpu().numpy().copy(),hidden


def native_motor_forces(normalized,force_ranges):
    action=np.asarray(normalized,dtype=float);caps=np.asarray(force_ranges,dtype=float)
    if caps.shape!=(action.size,2) or action.ndim!=1 or not np.isfinite(np.r_[action,caps.ravel()]).all() or np.any(caps[:,1]<=caps[:,0]):
        raise ValueError('Invalid native motor-force contract')
    return caps[:,0]+(np.clip(action,-1.,1.)+1.)*.5*(caps[:,1]-caps[:,0])
