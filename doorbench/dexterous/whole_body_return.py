"""Development motor-target orchestration for a candidate lever return.

This module accepts no active plant handle. A screened whole-body path changes
only the existing stance QP and arm teacher targets. The caller must retain the
attained physical foot references and validate every actual contact/state step.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .right_release_return import ControlledLeverReturn


class WholeBodyLeverReturn(ControlledLeverReturn):
    def __init__(self, teacher, path, whole_body_path):
        super().__init__(teacher, path)
        self.plan = json.loads(Path(whole_body_path).read_text())
        rows = self.plan['results']
        if len(rows)<2:
            raise ValueError('Require the complete screened whole-body path')
        self.progress = np.array([r['progress'] for r in rows])
        if not np.all(np.diff(self.progress)>0) or self.progress[0]!=0 or self.progress[-1]!=1:
            raise ValueError('Require monotonic complete screened path progress')
        self.roots=[]; self.joint_targets=[]
        self.body_names=list(rows[0]['joints'])
        self.torso_index=teacher.names.index('torso')
        for r in rows:
            if (r['forbidden_collisions'] or r['torso_tilt_deg']>=12
                or r['right_position_error_m']>.0001 or r['left_position_error_m']>.0001
                or r['right_rotation_error_rad']>.001 or r['left_rotation_error_rad']>.001
                or max(r['foot_position_errors_m'])>.0001 or max(r['foot_rotation_errors_rad'])>.001
                or min(r['joint_margins_rad'].values())< -1e-8
                or set(r['joints'])!=set(self.body_names)):
                raise ValueError('Whole-body path did not meet the declared geometric screen')
            qa=r['root_qpos_address'];self.roots.append(r['qpos'][qa:qa+7])
            self.joint_targets.append([r['joints'][n] for n in self.body_names])
        self.roots=np.asarray(self.roots,float);self.joint_targets=np.asarray(self.joint_targets,float)
        if not np.isfinite(np.r_[self.roots.ravel(),self.joint_targets.ravel()]).all():
            raise ValueError('Nonfinite screened target')

    def begin(self,t,joints,root,handle_pose):
        if not np.allclose(root[:7],self.roots[0],atol=1e-5,rtol=0):
            raise ValueError('This diagnostic path requires the exact attained root, not a reset substitute')
        if not np.allclose([joints[n] for n in self.body_names],self.joint_targets[0],atol=1e-5,rtol=0):
            raise ValueError('This diagnostic path requires the exact attained joint configuration')
        super().begin(t,joints,root,handle_pose)

    def body_goal(self,t):
        if self.started is None:
            return None
        if not np.isfinite(t) or t<self.started:
            raise ValueError('Body target requires a finite current release clock')
        u=float(np.clip((t-self.started)/self.return_seconds,0,1));blend=u**3*(10+u*(-15+6*u))
        i=min(len(self.progress)-2,max(0,int(np.searchsorted(self.progress,blend,side='right')-1)))
        f=float((blend-self.progress[i])/(self.progress[i+1]-self.progress[i]))
        a,b=self.roots[i],self.roots[i+1]
        ra=Rotation.from_quat([*a[4:],a[3]]);rb=Rotation.from_quat([*b[4:],b[3]])
        rotation=Rotation.from_rotvec(f*(rb*ra.inv()).as_rotvec())*ra
        joints=(1-f)*self.joint_targets[i]+f*self.joint_targets[i+1]
        return dict(position=(1-f)*a[:3]+f*b[:3],rotation=rotation.as_matrix(),
                    joints=dict(zip(self.body_names,joints)),progress=blend)

    def update(self,t):
        super().update(t)
        if self.started is None:return
        target=self.body_goal(t)
        self.teacher.path[-1,self.torso_index]=target['joints']['torso']
        self.info.update(whole_body_target_progress=target['progress'],
            target_pelvis_position_m=target['position'].tolist(),
            target_pelvis_rotation=target['rotation'].tolist(),
            target_torso_rad=target['joints']['torso'],
            whole_body_status='screened_geometrically_not_physically_qualified')


def apply_stance_goal(body_controller,goal):
    """Only modify existing analytic QP targets; retain attained foot references."""
    if body_controller.stance is None or body_controller.stage!='low stance hold':
        raise ValueError('Require the actual attained, active landed-foot stance')
    stance=body_controller.stance
    body_controller.height=float(goal['position'][2])
    stance.target_root[:]=goal['position']
    stance.target_rotation[:]=goal['rotation']
    names=[stance.sim.m.joint(int(j)).name.removeprefix('robot/') for j in stance.joints]
    stance.joint_target[:]=[goal['joints'][n] for n in names]
