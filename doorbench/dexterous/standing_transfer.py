"""Privileged coordinated standing transfer through original robot motors.

Consumes an independently screened, attained-state geometric route. The right
hand operation stays active while the actual feet support a small root shift and
left-palm approach. A geometric plan never substitutes for a physical test.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .bimanual_transfer import LeftPalmContact
from .operation_teacher import smooth_phase


class StandingTransferTeacher:
    def __init__(self,operation,motors,path,*,start_seconds=22.):
        self.operation=operation;self.acquisition=operation.acquisition
        self.path=Path(path);self.config=json.loads(self.path.read_text())
        c=self.config
        if c.get('schema')!='doorbench.standing-transfer.v1' or c.get('geometric_screen_passed') is not True:
            raise ValueError('Explicit screened standing-transfer route required')
        if c['robot_xml_sha256']!=motors['source_xml_sha256']:
            raise ValueError('Standing route robot differs from controller')
        self.names=self.acquisition.names
        if c['joint_names']!=self.names:raise ValueError('Standing route joint order changed')
        self.roots=np.asarray(c['root_path'],float);self.joints=np.asarray(c['joint_path'],float)
        if self.roots.shape!=(101,7) or self.joints.shape!=(101,len(self.names)) or not np.isfinite(np.r_[self.roots.ravel(),self.joints.ravel()]).all():raise ValueError('Expected complete finite 101-node route')
        if not np.allclose(np.linalg.norm(self.roots[:,3:],axis=1),1,atol=1e-6):raise ValueError('Unit root orientations required')
        proof=Path(c['dense_audit_path'])
        if hashlib.sha256(proof.read_bytes()).hexdigest()!=c['dense_audit_sha256'] or json.loads(proof.read_text()).get('passed') is not True:raise ValueError('Independent route audit changed')
        leftnames=c['left_joint_names'];rows=[]
        for row in c['left_targets']:
            rows.append({**row,**{k:np.asarray(row[k],float) for k in ('position','normal','nominal')}})
        self.left=LeftPalmContact(self.acquisition,motors,(leftnames,rows),fixed_waist=False,reach_seconds=8.,contact_force=8.)
        self.start_seconds=start_seconds;self.started=None;self.info={}
        self.rotations=Slerp(np.linspace(0,1,101),Rotation.from_quat(self.roots[:,[4,5,6,3]]))

    def force(self,t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,*,grasp_qualified,left_panel_load):
        teacher=self.acquisition
        if self.started is None and t>=self.start_seconds-1e-8 and grasp_qualified:
            root=np.asarray(root,float);actual=np.array([joints[n] for n in self.names])
            delta=Rotation.from_quat(root[[4,5,6,3]])*Rotation.from_quat(self.roots[0,[4,5,6,3]]).inv()
            if np.linalg.norm(root[:3]-self.roots[0,:3])>.003 or delta.magnitude()>.01 or np.max(abs(actual-self.joints[0]))>.02:
                raise ValueError('Actual standing state differs from the independently screened route start')
            if self.operation.open_started is None or not .075<=angles['leaf']<=.10:
                raise ValueError('Qualified partial opening required before standing transfer')
            self.left.begin(t,root,joints,leaf_pose,handle_pose);self.started=t
        if self.started is not None:
            self.left.update_targets(t,root,joints,leaf_pose,left_panel_load,handle_pose)
            u=float(smooth_phase(self.left.progress));coordinate=u*100;i=min(int(coordinate),99);f=coordinate-i
            root_goal=(1-f)*self.roots[i,:3]+f*self.roots[i+1,:3]
            q=(1-f)*self.joints[i]+f*self.joints[i+1]
            teacher.stance.target_root[:]=root_goal
            teacher.stance.target_rotation=self.rotations(u).as_matrix()
            teacher.stance.joint_target[:]=[q[self.names.index(teacher.m.joint(int(j)).name)] for j in teacher.stance.joints]
        forces,info=self.operation.force(t,root,joints,velocities,handle_pose,leaf_pose,angles,hand_loads,grasp_qualified=grasp_qualified)
        if self.started is not None:
            forces=self.left.apply_forces(forces,joints,velocities)
            info={**info,**self.left.info, 'standing_transfer_started_s':self.started}
        self.info=info
        return forces,info
