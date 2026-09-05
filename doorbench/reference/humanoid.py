"""Deterministic metric skeleton fitted to task-space targets, never a dynamics controller.

Z up; 1.72 m nominal stature. Limb lengths are fixed. Unreachable hand targets are
clamped and measured rather than stretching the arms. Feet use alternating planted
supports and swing arcs; collision, balance and actuator feasibility are NOT certified.
"""
from __future__ import annotations
import math
import numpy as np

JOINTS = ['pelvis', 'chest', 'neck', 'head', 'shoulder_l', 'elbow_l', 'wrist_l',
          'shoulder_r', 'elbow_r', 'wrist_r', 'hip_l', 'knee_l', 'ankle_l',
          'hip_r', 'knee_r', 'ankle_r']
BONES = [[0,1],[1,2],[2,3],[1,4],[4,5],[5,6],[1,7],[7,8],[8,9],
         [0,10],[10,11],[11,12],[0,13],[13,14],[14,15]]


def unit(x, fallback=(0.,1.,0.)):
    x=np.asarray(x,float); n=np.linalg.norm(x)
    return x/n if n>1e-8 else np.asarray(fallback,float)


def two_bone(root, target, a, b, pole):
    """Analytic IK with fixed segment lengths and explicit target residual."""
    delta=np.asarray(target)-root; d=float(np.linalg.norm(delta)); axis=unit(delta)
    reach=float(np.clip(d, abs(a-b)+1e-5, a+b-1e-5))
    bend=unit(np.asarray(pole)-axis*np.dot(pole,axis), (1.,0.,0.))
    along=(a*a-b*b+reach*reach)/(2*reach)
    elbow=root+axis*along+bend*math.sqrt(max(0.,a*a-along*along))
    end=root+axis*reach
    return elbow,end,float(np.linalg.norm(end-target))


def fit_motion(times, bases, targets, active, initial_base, lead=2.):
    """Return actor joints and measured wrist residuals. Door time excludes lead-in.

    Actor roots approach working distance then follow the target while operating.
    Once the benchmark base advances through the opening the actor follows it.
    This display path is separate from the benchmark's synthetic base trajectory.
    """
    n=len(times); poses=np.zeros((n,len(JOINTS),3)); errors=np.zeros(n)
    roots=np.zeros((n,3)); contact=np.zeros((n,2),dtype=np.uint8)
    first=np.asarray(targets[0]); work=np.array([first[0]-.12,min(first[1]-.40,-.40),0.])
    position=np.array([*initial_base[:2],0.],float)
    feet=None; phase=0.; oldroot=position.copy(); heading=np.array([0.,1.,0.])
    initial_y=float(initial_base[1]); last_time=float(times[0])
    for i,t in enumerate(times):
        dt=max(0.,float(t)-last_time); last_time=float(t)
        base=np.asarray(bases[i]); target=np.asarray(targets[i]); manipulating=bool(active[i])
        if t < lead:
            s=np.clip(t/lead,0,1); s=s*s*(3-2*s)
            desired=np.asarray(initial_base,float).copy(); desired[2]=0
            desired=desired*(1-s)+work*s
        elif base[1]>max(initial_y+.45,-.65):
            desired=np.array([base[0],base[1],0.])
        elif manipulating:
            # Work on the near face. Never use an unreachable high target to lift the body.
            desired=np.array([target[0]-.12,min(target[1]-.40,-.40),0.])
        else:
            desired=work.copy()
        step=desired-position; dist=float(np.linalg.norm(step[:2])); limit=dt*.95
        position+=step*min(1.,limit/max(dist,1e-8)) if i else np.zeros(3)
        velocity=position-oldroot; speed=np.linalg.norm(velocity)/max(dt,1e-6); oldroot=position.copy()
        facing=unit(target-position) if manipulating else unit(velocity,heading)
        facing[2]=0; facing=unit(facing); heading=unit(heading*(1-min(1,dt*6))+facing*min(1,dt*6))
        side=np.array([-heading[1],heading[0],0.])  # anatomical left, Z-up forward heading
        # Mild crouch for low hardware. Minimum pelvis leaves fixed legs room to bend.
        crouch=np.clip((.82-target[2])*.7,0,.30) if manipulating else 0
        pelvis=position+np.array([0.,0.,.94-crouch])
        chest=pelvis+np.array([0.,0.,.35]); neck=chest+[0,0,.16]; head=neck+[0,0,.13]
        p=poses[i]; p[:4]=[pelvis,chest,neck,head]
        for sign,sh,el,wr in [(1,4,5,6),(-1,7,8,9)]:
            shoulder=chest+sign*side*.18+[0,0,.06]
            rest=shoulder+[0,0,-.51]+heading*.06
            wanted=target if sign==-1 and manipulating and t>=lead else rest
            p[sh]=shoulder
            p[el],p[wr],err=two_bone(shoulder,wanted,.30,.28,sign*side*.65+[0,0,-1])
            if sign==-1 and manipulating and t>=lead: errors[i]=err
        phase+=dt*min(1.8,speed/.48) if speed>.03 else 0
        if feet is None: feet=[position+side*.10,position-side*.10]
        for k,(sign,hip,knee,ankle) in enumerate([(1,10,11,12),(-1,13,14,15)]):
            h=pelvis+side*sign*.105+[0,0,-.06]
            f=(phase+k*.5)%1; moving=speed>.035
            nominal=position+side*sign*.11
            if moving and f>=.5:
                swing=(f-.5)*2
                feet[k]=nominal+heading*(.16*(2*swing-1))+[0,0,.10*math.sin(math.pi*swing)]
                contact[i,k]=0
            else:
                feet[k][2]=0; contact[i,k]=1
                if np.linalg.norm(feet[k][:2]-position[:2])>.32: feet[k]=nominal.copy()
            p[hip]=h; p[knee],p[ankle],_=two_bone(h,feet[k]+[0,0,.055],.43,.43,heading)
        roots[i]=position
    return poses.astype(np.float32),errors.astype(np.float32),contact,roots.astype(np.float32)
