"""Independent kinematic fixtures for the bounded body-reference projection."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from doorbench.dexterous.coupled_release_motion import CoupledReferenceMotion
from doorbench.dexterous.coupled_release_reference import CoupledReleaseReference
import doorbench.dexterous.coupled_release_reference as reference_module


def fixture_reference():
    # Four articulated chains have independently fixed world-frame end poses.
    # No robot assets, historical trajectories, simulator step, or motor exists.
    branches=[];names=[]
    for i,(x,y,z) in enumerate([(0,.2,-.6),(0,-.2,-.6),(.3,.2,.2),(.3,-.2,.2)]):
        names += [f'j{i}a',f'j{i}b']
        branches.append(f'''<body name="b{i}" pos="{x} {y} {z}">
          <joint name="robot/j{i}a" axis="0 1 0"/><geom type="sphere" size=".02"/>
          <body name="tip{i}" pos=".1 0 0"><joint name="robot/j{i}b" axis="1 0 0"/>
          <geom type="sphere" size=".02"/><site name="site{i}" pos=".03 0 .02"/></body></body>''')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 .8"><freejoint name="robot/free_base"/><geom type="sphere" size=".1"/>'+''.join(branches)+'</body></worldbody></mujoco>')
    data=mujoco.MjData(model);initial=data.qpos.copy()
    quat=Rotation.from_euler('xyz',[.13,-.07,.21]).as_quat();initial[3:7]=quat[[3,0,1,2]]
    data.qpos[:]=initial;mujoco.mj_kinematics(model,data)
    ref=CoupledReleaseReference.__new__(CoupledReleaseReference)
    ref.names=names;ref.qa=np.array([model.joint('robot/'+name).qposadr[0] for name in names])
    ref.body_columns=np.arange(6+len(names));ref.feet=[model.body('tip0').id,model.body('tip1').id]
    ref.geometry=SimpleNamespace(m=model,d=data,initial=initial,rq=0,names=names,
        initial_rotation=Rotation.from_quat(initial[3:7][[1,2,3,0]]),
        lh=model.site('site2').id,rh=model.site('site3').id,
        c={'initial_feet_positions':[data.xpos[b].copy() for b in ref.feet],
           'initial_feet_rotations':[data.xmat[b].reshape(3,3).copy() for b in ref.feet]})
    result={}
    for label,site in [('left',ref.geometry.lh),('right',ref.geometry.rh)]:
        result[label+'_palm_position']=data.site_xpos[site].copy()
        result[label+'_palm_rotation']=data.site_xmat[site].reshape(3,3).copy()
    return ref,result


def test_world_rotvec_jacobian_matches_independent_central_differences():
    ref,result=fixture_reference();value=np.linspace(-.04,.035,len(ref.body_columns))
    value[:3]*=.1
    error,jac=ref._pose_residual_jacobian(value,result);numeric=np.zeros_like(jac)
    for i in range(len(value)):
        minus=value.copy();plus=value.copy();minus[i]-=1e-6;plus[i]+=1e-6
        em,_=ref._pose_residual_jacobian(minus,result);ep,_=ref._pose_residual_jacobian(plus,result)
        numeric[:,i]=(ep-em)/2e-6
    assert np.max(abs(jac-numeric))<1e-5
    assert np.linalg.norm(error)>1.  # Nonzero rotation/translation residual, not an identity-only check.


def test_coupled_projection_preserves_fixed_poses_and_original_motion_bounds():
    ref,result=fixture_reference();initial=np.zeros(len(ref.body_columns))
    motion=CoupledReferenceMotion(initial);motion.update(0.,initial)
    desired=np.full_like(initial,.1)
    unprojected=CoupledReferenceMotion(initial);unprojected.update(0.,initial)
    clipped,_=unprojected.update(.002,desired)
    error_before,_=ref._pose_residual_jacobian(clipped,result)
    def project(*args):return ref._project_reference(*args,result)
    projected,info=motion.update(.002,desired,project=project)
    error_after,_=ref._pose_residual_jacobian(projected,result)
    assert np.linalg.norm(error_after)<np.linalg.norm(error_before)*.01
    assert info['root_speed_m_s']<=.02+1e-12
    assert info['root_rotation_speed_rad_s']<=.03+1e-12
    assert info['joint_speed_rad_s']<=1.2+1e-12
    assert info['joint_acceleration_rad_s2']<=3.+1e-9
    assert info['projection']['method']=='fixed-poses-v1'


def test_projector_cannot_change_finger_reference_coordinates():
    ref,result=fixture_reference();initial=np.zeros(len(ref.body_columns)+2)
    # The two final columns represent fingers absent from the body solve.
    # Install only actual fixture joints while retaining those extra columns.
    actual_install=ref._install_reference
    ref._install_reference=lambda value:actual_install(value[:-2])
    candidate=initial.copy();candidate[-2:]=[1e-5,-1e-5]
    projected,_=ref._project_reference(candidate,initial,np.zeros_like(initial),.002,result)
    assert np.array_equal(projected[-2:],candidate[-2:])


def test_finite_feasible_stalled_solve_keeps_diagnostic_and_exact_motion_bounds(monkeypatch):
    ref,result=fixture_reference();initial=np.zeros(len(ref.body_columns))
    def stalled(fun,x,**kwargs):
        return SimpleNamespace(x=np.zeros_like(x),success=False,status=8,message='fixture line search stop',nit=1)
    monkeypatch.setattr(reference_module,'minimize',stalled)
    motion=CoupledReferenceMotion(initial);motion.update(0.,initial)
    projected,info=motion.update(.002,np.ones_like(initial),project=lambda *args:ref._project_reference(*args,result))
    assert np.array_equal(projected,initial)
    assert not info['projection']['solver_status'][0]['success']
    assert info['projection']['solver_status'][0]['status']==8
    assert info['root_speed_m_s']==0.
    assert info['joint_acceleration_rad_s2']==0.
    error,_=ref._pose_residual_jacobian(projected,result)
    assert np.max(abs(error))<1e-10


def test_nonfinite_projection_result_is_terminal_before_motion_commit(monkeypatch):
    ref,result=fixture_reference();initial=np.zeros(len(ref.body_columns))
    def invalid(fun,x,**kwargs):return SimpleNamespace(x=np.full_like(x,np.nan),success=True)
    monkeypatch.setattr(reference_module,'minimize',invalid)
    motion=CoupledReferenceMotion(initial);motion.update(0.,initial)
    with pytest.raises(ValueError,match='nonfinite'):
        motion.update(.002,np.ones_like(initial),project=lambda *args:ref._project_reference(*args,result))
    assert motion.time==0.
    assert np.array_equal(motion.value,initial)
