"""Post-hoc numerical QP diagnostic, with no physics or qualification changes."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from doorbench.dexterous.sensor_balance_runtime import SensorBalanceRuntime

run=Path(sys.argv[1]);robot=Path(sys.argv[2]);output=Path(sys.argv[3])
assert not output.exists()
motors=json.loads((run/'motor-contract.json').read_text());layout=json.loads((run/'sensors/layout.json').read_text())
with np.load(run/'sensors/actor-initial-decision.npz') as z:
    actual=z['motor_forces'].copy();packet={k:z[k].copy() for k in z.files if k not in ('time_s','motor_forces')}
results=[];model_hashes={}
cases=[('default-'+str(i),{}) for i in range(4)]+[(str(i),dict(adaptive_rho_interval=i)) for i in (25,50,75,100,150,200)]+[('fixed-rho',dict(adaptive_rho=False))]
for name,settings in cases:
    runtime=SensorBalanceRuntime(robot,motors,layout,run/'sensor-balance-calibration.json');runtime.reset_episode()
    controller=runtime._controller
    controller.sim.stance_solver_settings.update(settings)
    if not model_hashes:
        for field in ('body_mass','body_inertia','body_ipos','body_iquat','mesh_vert','mesh_face','geom_size','dof_armature','dof_damping'):
            a=np.asarray(getattr(controller.m,field));model_hashes[field]=dict(shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    force=runtime.force(packet,0.)
    diff=force-actual
    results.append(dict(case=name,posthoc_settings=settings,maximum_motor_error_Nm=float(abs(diff).max()),
        leg_difference_Nm=diff[controller.stance.local].tolist(),metadata=runtime.last_info.get('qp_solver'),
        calculator_time_s=controller.d.time))
report=dict(scope=__doc__,cases=results,compiled_robot=model_hashes,robot_xml_sha256=hashlib.sha256(robot.read_bytes()).hexdigest(),physics_steps=0,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(sys.modules['doorbench.dexterous.locomotion_manipulation'].__file__),Path(sys.modules['doorbench.dexterous.sensor_balance'].__file__))})
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([dict(case=r['case'],maximum_motor_error_Nm=r['maximum_motor_error_Nm'],iterations=r['metadata']['iterations'],rho_updates=r['metadata']['rho_updates']) for r in results]),flush=True)
