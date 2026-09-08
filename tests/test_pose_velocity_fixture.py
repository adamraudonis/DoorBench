"""The pose-integration scorer must detect drift without measuring an actor."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.dexterous.isaac_pose_velocity_fixture import integrate_pose_errors, VARIANTS


def test_each_solver_variant_changes_one_declared_setting():
    base=VARIANTS['baseline']
    assert base==dict(solver_type=1,velocity_iterations=8,stabilization=False,position_iterations=32)
    assert all(sum(v[k]!=base[k] for k in base)==1 for n,v in VARIANTS.items() if n!='baseline')


def test_endpoint_integration_recovers_exact_world_rate_and_rejects_wrong_velocity():
    dt=.002;n=101;w=np.array([.01,.02,.03]);poses=np.zeros((n,1,7));velocities=np.zeros((n,1,6))
    poses[:,0,3:]=Rotation.from_rotvec(np.arange(n)[:,None]*dt*w).as_quat();velocities[:,0,3:]=w
    errors,residual=integrate_pose_errors(poses,velocities,dt)
    assert np.max(abs(errors['current']))<1e-15
    assert np.max(abs(residual))<1e-16
    velocities[:,0,3]+= .001
    errors,_=integrate_pose_errors(poses,velocities,dt)
    assert np.linalg.norm(errors['current'][-1,0])>.00019
