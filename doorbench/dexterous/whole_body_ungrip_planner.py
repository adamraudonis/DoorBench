"""Unstepped withdrawal solve from copied state and measured hand samples.

This extracted numerical core accepts no active plant, file paths or controller.
Its original geometry/IK settings remain unchanged. Actual imported observations,
dense path qualification and motor execution are separate admission stages.
"""
import copy
import re
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

PROFILES = ('recorded', 'clearance-lift-v1', 'clearance-lift-v2',
            'clearance-lift-v3', 'clearance-lift-v4')


def copy_ungrip_inputs(model, qpos, qvel, time_s, release_samples, *,
                       finger_lead_seconds=0., early_lift_m=.002,
                       withdrawal_profile='recorded'):
    """Copy coordinates and validate sample shape/clock/rotation contracts.

    This function does not invent missing observed poses from the recorded
    terminal coordinates. A runtime caller must separately verify its complete
    same-epoch body/site mapping before invoking this numerical core.
    """
    if (withdrawal_profile not in PROFILES or
            not np.isfinite([time_s,finger_lead_seconds,early_lift_m]).all() or
            time_s < 0 or not 0 <= finger_lead_seconds <= .5 or
            not 0 <= early_lift_m <= .01):
        raise ValueError('Require declared finite clock, bounded lead/lift and withdrawal profile')
    q=np.array(qpos,float,copy=True);v=np.array(qvel,float,copy=True)
    if q.shape!=(model.nq,) or v.shape!=(model.nv,) or not np.isfinite(np.r_[q,v]).all():
        raise ValueError('Require complete finite attained coordinates and velocities')
    rq=int(model.jnt_qposadr[model.joint('robot/free_base').id])
    if not np.isclose(np.linalg.norm(q[rq+3:rq+7]),1.,atol=1e-10,rtol=0):
        raise ValueError('Require the actual normalized root quaternion')
    samples=copy.deepcopy(release_samples)
    required={'finger_joint_names','finger_joint_delta_rad','palm_position_handle',
              'time_s','palm_rotation_handle','source_finger_joint_positions'}
    if not isinstance(samples,dict) or set(samples)!=required:
        raise ValueError('Require exactly the numeric measured withdrawal sample contract')
    names=samples['finger_joint_names']
    if (not isinstance(names,(list,tuple)) or len(names)!=len(set(names)) or not names or
            any(not isinstance(n,str) or not re.fullmatch(r'rh_(?:FF|MF|RF|LF|TH)J[1-5]',n) for n in names)):
        raise ValueError('Require unique authored right finger joint names')
    for name in names:
        joint=model.joint('robot/'+name).id
        if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('Measured fingers must use authored scalar hinges')
    if any(f'rh_{digit}J{k}' not in names for digit in ('FF','MF','RF','LF') for k in (1,2)):
        raise ValueError('Require both original coupled coordinates of every finger')
    times=np.array(samples['time_s'],float,copy=True);count=len(times) if times.ndim==1 else 0
    if count<2 or not np.isfinite(times).all() or times[0]!=0 or not np.all(np.diff(times)>0):
        raise ValueError('Require complete strictly increasing measured sample times')
    for key,shape in [('finger_joint_delta_rad',(count,len(names))),
                      ('source_finger_joint_positions',(count,len(names))),
                      ('palm_position_handle',(count,3)),('palm_rotation_handle',(count,3,3))]:
        a=np.array(samples[key],float,copy=True)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError('Invalid finite measured sample dimensions: '+key)
        samples[key]=a
    rotations=samples['palm_rotation_handle']
    if (not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-8,rtol=0)
            or not np.allclose(np.linalg.det(rotations),1.,atol=1e-8,rtol=0)):
        raise ValueError('Measured palm rotations must be proper')
    if withdrawal_profile!='recorded' and (times[-1]<=4.1 or min(abs(times-4.1))>1e-8):
        raise ValueError('Clearance profiles require their original4.1s measured handoff')
    samples['time_s']=times;samples['finger_joint_names']=list(names)
    d=mujoco.MjData(model);d.qpos[:]=q;d.qvel[:]=v;d.time=float(time_s)
    return d,samples


def iter_whole_body_ungrip(model, *, qpos, qvel, time_s, release_samples,
                          finger_lead_seconds=0., early_lift_m=.002,
                          withdrawal_profile='recorded', maximum_torso_tilt_deg=None,
                          retarget_attained_grasp=False):
    """Yield the original candidate; no physical or dense-path pass is implied."""
    m=model
    d,release_samples=copy_ungrip_inputs(m,qpos,qvel,time_s,release_samples,
        finger_lead_seconds=finger_lead_seconds,early_lift_m=early_lift_m,
        withdrawal_profile=withdrawal_profile)
    base=d.qpos.copy()
    d.qpos[:]=base;mujoco.mj_kinematics(m,d);hb=m.body('leaf_handle').id;rh=m.site('robot/rh_palm_touch').id;lh=m.site('robot/lh_palm_touch').id;H=d.xmat[hb].reshape(3,3).copy();P=d.site_xpos[rh].copy();R=d.site_xmat[rh].reshape(3,3).copy();PL=d.site_xpos[lh].copy();RL=d.site_xmat[lh].reshape(3,3).copy();feet=[m.body('robot/'+side+'_ankle_link').id for side in ('left','right')];FP=d.xpos[feet].copy();FR=d.xmat[feet].reshape(2,3,3).copy();rq=m.jnt_qposadr[m.joint('robot/free_base').id];rootP=base[rq:rq+3].copy();rootR=Rotation.from_quat([*base[rq+4:rq+7],base[rq+3]]);lever=m.geom('leaf_handle_lever_col_n').id;floor=m.geom('floor').id
    right=['right_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['rh_WRJ2','rh_WRJ1'];left=['left_'+n for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_yaw')]+['lh_WRJ2','lh_WRJ1'];legs=[side+'_'+j for side in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')];names=legs+['torso']+right+left;js=np.array([m.joint('robot/'+n).id for n in names]);qa=m.jnt_qposadr[js];start=base[qa].copy()
    src=release_samples;fn=list(src['finger_joint_names']);fjs=np.array([m.joint('robot/'+n).id for n in fn]);fqa=m.jnt_qposadr[fjs];fds=np.asarray(src['finger_joint_delta_rad']);pp=np.asarray(src['palm_position_handle']);times=np.asarray(src['time_s']);trials=[]
    source_fingers=np.asarray(src['source_finger_joint_positions']);source_rotations=np.asarray(src['palm_rotation_handle']);source_initial_p=pp[0];source_initial_r=source_rotations[0];initial_relative_p=H.T@(P-d.xpos[hb]);initial_relative_r=H.T@R;handleP=d.xpos[hb].copy()
    if retarget_attained_grasp:
        # Transfer recorded withdrawal increments, avoiding a preliminary move
        # into an old embodiment posture. This is a new geometric candidate,
        # not the original measured trajectory or a physical qualification.
        delta_rotation=initial_relative_r@source_initial_r.T
        pp=(pp-source_initial_p)@delta_rotation.T+initial_relative_p
        source_rotations=delta_rotation@source_rotations
        source_fingers=source_fingers-source_fingers[0]+base[fqa]
        source_initial_p=initial_relative_p.copy();source_initial_r=initial_relative_r.copy()
    rotation_bound=.12
    if maximum_torso_tilt_deg is not None:
        initial_tilt=float(np.degrees(np.arccos(np.clip(d.xmat[m.body('robot/torso_link').id].reshape(3,3)[2,2],-1,1))))
        if not np.isfinite(maximum_torso_tilt_deg) or not initial_tilt<maximum_torso_tilt_deg<=4:
            raise ValueError('Upright withdrawal requires a positive rotation margin within the standing 4 degree design target')
        # Bound horizontal rotation; yaw remains available without extra lean.
        # Independent dense FK still checks the actual torso, feet and contacts.
        rotation_bound=min(.12,.99*np.radians(maximum_torso_tilt_deg-initial_tilt)/np.sqrt(2))
    for dx,dy,roll in [(0,-.16056,0)]:
     rows=[];previous=np.r_[np.zeros(6),start]
     nodes=[('grasp_adjustment',float(u)) for u in np.linspace(0,1,31)]+[('measured_release',float(u)) for u in np.linspace(0,1,len(times))[1:] if withdrawal_profile=='recorded' or float(times[round(u*(len(times)-1))])<=4.1+1e-8]
     if withdrawal_profile.startswith('clearance-lift-'):nodes += [('clearance_lift',float(u)) for u in np.linspace(0,1,41)[1:]]
     for phase,u in nodes:
      if phase=='grasp_adjustment':
       fraction=u;relative_p=(1-u)*initial_relative_p+u*source_initial_p;relative_r=Rotation.from_rotvec(u*Rotation.from_matrix(source_initial_r@initial_relative_r.T).as_rotvec()).as_matrix()@initial_relative_r;f=(1-u)*base[fqa]+u*source_fingers[0];clock=4*u
      elif phase=='clearance_lift':
       start_i=int(np.argmin(abs(times-4.1)));fraction=u;blend=u**3*(10+u*(-15+6*u));relative_p=pp[start_i]+np.array([0.,0.,early_lift_m if withdrawal_profile in ('clearance-lift-v2','clearance-lift-v3','clearance-lift-v4') else 0.])+np.array([-.006 if withdrawal_profile=='clearance-lift-v4' else 0.,-.016 if withdrawal_profile in ('clearance-lift-v3','clearance-lift-v4') else -.010,.040])*blend;relative_r=Rotation.from_rotvec(blend*Rotation.from_matrix(source_rotations[-1]@source_rotations[start_i].T).as_rotvec()).as_matrix()@source_rotations[start_i];source_t=float(times[start_i])+blend*(times[-1]-times[start_i]);f=np.array([np.interp(source_t,times,source_fingers[:,k]) for k in range(len(fn))]);clock=4+float(times[start_i])+.7*u
      else:
       i=round(u*(len(fds)-1));fraction=u;relative_p=pp[i];relative_r=source_rotations[i];finger_t=float(times[i])+finger_lead_seconds*np.sin(np.pi*u)**2;f=np.array([np.interp(finger_t,times,source_fingers[:,k]) for k in range(len(fn))]);clock=4+float(times[i])
      if withdrawal_profile in ('clearance-lift-v2','clearance-lift-v3','clearance-lift-v4') and phase=='measured_release':
       lift_u=float(np.clip((clock-6.)/1.,0.,1.));relative_p=relative_p+np.array([0.,0.,early_lift_m*lift_u**3*(10+lift_u*(-15+6*lift_u))])
      PR=handleP+H@relative_p;RR=H@relative_r;f=np.clip(f,m.jnt_range[fjs,0],m.jnt_range[fjs,1])
      for digit in ('FF','MF','RF','LF'):
       a,b=[fn.index(f'rh_{digit}J{k}') for k in (1,2)]
       if f[a]>f[b]:f[[a,b]]=f[[a,b]].mean()
      target_base=base.copy();target_base[fqa]=f;lo=np.r_[[-.08,-.08,-.04],[-.12,-.12,-.12],m.jnt_range[js,0]+.001];hi=np.r_[[.08,.08,.04],[.12,.12,.12],m.jnt_range[js,1]-.001]
      lo[3:5]=-rotation_bound;hi[3:5]=rotation_bound
      def fun(x):
       d.qpos[:]=target_base;d.qpos[rq:rq+3]=rootP+x[:3];r=(Rotation.from_rotvec(x[3:6])*rootR).as_quat();d.qpos[rq+3:rq+7]=np.r_[r[3],r[:3]];d.qpos[qa]=x[6:];mujoco.mj_kinematics(m,d)
       hand=np.r_[100*(d.site_xpos[rh]-PR),10*Rotation.from_matrix(RR@d.site_xmat[rh].reshape(3,3).T).as_rotvec(),100*(d.site_xpos[lh]-PL),10*Rotation.from_matrix(RL@d.site_xmat[lh].reshape(3,3).T).as_rotvec()]
       foot=np.concatenate([np.r_[100*(d.xpos[b]-FP[i]),10*Rotation.from_matrix(FR[i]@d.xmat[b].reshape(3,3).T).as_rotvec()] for i,b in enumerate(feet)])
       return np.r_[hand,foot,.01*(x[6:]-start),.02*x[:6]]
      fit=least_squares(fun,np.clip(previous,lo,hi),bounds=(lo,hi),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11);previous=fit.x.copy();res=fun(fit.x);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);cols={};invalid={}
      for cc in d.contact[:d.ncon]:
       bs=[m.body(m.geom_bodyid[g]).name for g in cc.geom]
       if cc.dist<-.003 and any(b.startswith('robot/') for b in bs) and not (floor in cc.geom and any(b.endswith('_ankle_link') for b in bs)):key=' + '.join(bs);cols[key]=max(cols.get(key,0),-float(cc.dist))
       if cc.dist<-.00005 and lever in cc.geom:
        g=int(cc.geom[1] if cc.geom[0]==lever else cc.geom[0]);b=m.geom_bodyid[g];name=m.body(b).name
        if name.startswith('robot/rh_'):
         br=d.xmat[b].reshape(3,3);p=br.T@(cc.pos-d.xpos[b]);normal=br.T@((1 if cc.geom[0]==g else -1)*cc.frame[:3]);axis=d.geom_xmat[lever].reshape(3,3)[:,2];v=cc.pos-d.geom_xpos[lever];a=np.dot(v,axis);rad=v-a*axis;align=np.dot(br@normal,-rad/max(np.linalg.norm(rad),1e-9));valid=name.endswith('distal') and p[1]<-.001 and .002<=p[2]<=.040 and -normal[1]>.5 and m.geom_size[lever,1]-abs(a)>=.001 and align>.8
         if not valid:invalid[name]=max(invalid.get(name,0),-float(cc.dist))
      up=d.xmat[m.body('robot/torso_link').id].reshape(3,3)[:,2];tilt=float(np.degrees(np.arccos(np.clip(up[2],-1,1))));rows.append(dict(phase=phase,time_s=clock,progress=float(u),fraction=fraction,palm_position=PR.tolist(),palm_rotation=RR.tolist(),right_position_error_m=float(np.linalg.norm(res[:3])/100),right_rotation_error_rad=float(np.linalg.norm(res[3:6])/10),left_position_error_m=float(np.linalg.norm(res[6:9])/100),left_rotation_error_rad=float(np.linalg.norm(res[9:12])/10),foot_position_errors_m=[float(np.linalg.norm(res[12+6*i:15+6*i])/100) for i in range(2)],foot_rotation_errors_rad=[float(np.linalg.norm(res[15+6*i:18+6*i])/10) for i in range(2)],root_delta_xyz_m=fit.x[:3].tolist(),root_delta_rotvec_rad=fit.x[3:6].tolist(),joints=dict(zip(names,fit.x[6:].tolist())),finger_joints=dict(zip(fn,f.tolist())),joint_margins_rad=dict(zip(names,np.minimum(fit.x[6:]-lo[6:],hi[6:]-fit.x[6:]).tolist())),forbidden_collisions=cols,invalid_patches=invalid,torso_tilt_deg=tilt,root_qpos_address=int(rq),qpos=d.qpos.tolist()))
     result=dict(dx=dx,dy=dy,roll=roll,maximum_palm_error_m=max(max(x['right_position_error_m'],x['left_position_error_m']) for x in rows),maximum_palm_rotation_error_rad=max(max(x['right_rotation_error_rad'],x['left_rotation_error_rad']) for x in rows),maximum_torso_tilt_deg=max(x['torso_tilt_deg'] for x in rows),maximum_forbidden_depth_m=max([v for x in rows for v in x['forbidden_collisions'].values()]+[0]),invalid_pad_samples=sum(bool(x['invalid_patches']) for x in rows),root_delta=rows[-1]['root_delta_xyz_m'],rows=rows);yield result
