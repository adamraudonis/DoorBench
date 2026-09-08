#!/usr/bin/env python3
"""Isolated Shadow-inertia hinges: native dry friction versus legacy smooth law.

External fixture excitation is not permitted assistance in a robot task. This
tests the passive adapter, not a grasp or whole-robot simulator parity claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

CASES = [('static_under', .005, False), ('static_over', .02, False),
         ('smooth_under', .005, True), ('smooth_over', .02, True)]
DT = .002
STEPS = 500
DAMPING = .05
FRICTION = .01
ARMATURE = .0002
INERTIA = .0000027 + .017 * .0125**2 + ARMATURE


def native():
    import mujoco
    import numpy as np
    rows, metadata = [], []
    for name, excitation, smooth in CASES:
        xml = f'''<mujoco><compiler angle="radian"/>
          <option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"/>
          <worldbody><body name="middle"><joint name="hinge" axis="1 0 0"
          range="-1.5708 1.5708" armature="{ARMATURE}"
          damping="{0 if smooth else DAMPING}" frictionloss="{0 if smooth else FRICTION}"/>
          <inertial pos="0 0 .0125" mass=".017"
          diaginertia=".0000027 .0000026 .00000087"/>
          <geom size=".004" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>'''
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
        d.qpos[:] = .4
        mujoco.mj_forward(m, d)
        series = []
        for step in range(STEPS):
            applied = excitation if step < STEPS // 2 else 0.
            passive = -DAMPING*d.qvel[0]-FRICTION*np.tanh(d.qvel[0]/.001) if smooth else 0.
            d.qfrc_applied[0] = applied + passive
            mujoco.mj_step(m, d)
            series.append([d.time, d.qpos[0], d.qvel[0], applied, passive])
        rows.append(series); metadata.append(dict(name=name, xml=xml))
    return np.asarray(rows), dict(mujoco=mujoco.__version__, fixtures=metadata)


def isaac():
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        import torch
        import numpy as np
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=DT, device='cuda:0', gravity=(0., 0., 0.), render_interval=1000,
            physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=32,
                                    min_velocity_iteration_count=8)))
        if getattr(sim, '_app_control_on_stop_handle', None) is not None:
            sim._app_control_on_stop_handle.unsubscribe(); sim._app_control_on_stop_handle = None
        stage = sim.stage; robots = []
        for i, (name, _, _) in enumerate(CASES):
            path = '/World/' + name; UsdGeom.Xform.Define(stage, path)
            for body in ('base', 'middle'):
                shape = UsdGeom.Cube.Define(stage, path+'/'+body)
                shape.CreateSizeAttr(.008); shape.AddTranslateOp().Set(Gf.Vec3d(i*.3, 0, 0))
                UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
                mass = UsdPhysics.MassAPI.Apply(shape.GetPrim()); mass.CreateMassAttr(.017)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.0000027, .0000026, .00000087))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, .0125))
            root = stage.GetPrimAtPath(path+'/base'); UsdPhysics.ArticulationRootAPI.Apply(root)
            settings = PhysxSchema.PhysxArticulationAPI.Apply(root)
            settings.CreateEnabledSelfCollisionsAttr(False)
            settings.CreateSolverPositionIterationCountAttr(32); settings.CreateSolverVelocityIterationCountAttr(8)
            anchor = UsdPhysics.FixedJoint.Define(stage, path+'/anchor')
            anchor.CreateBody1Rel().SetTargets([path+'/base']); anchor.CreateLocalPos0Attr(Gf.Vec3f(i*.3, 0, 0))
            joint = UsdPhysics.RevoluteJoint.Define(stage, path+'/hinge'); joint.CreateAxisAttr('X')
            joint.CreateBody0Rel().SetTargets([path+'/base']); joint.CreateBody1Rel().SetTargets([path+'/middle'])
            joint.CreateLowerLimitAttr(-90.); joint.CreateUpperLimitAttr(90.)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
            drive.CreateStiffnessAttr(0.); drive.CreateDampingAttr(0.)
            robots.append(Articulation(ArticulationCfg(
                prim_path=path, spawn=None, articulation_root_prim_path='/base',
                actuators={'fixture': ImplicitActuatorCfg(joint_names_expr=['.*'],
                    stiffness=0., damping=0., effort_limit_sim=.1)})))
        sim.reset(); readback = []; rows = [[] for _ in CASES]
        for (_, _, smooth), robot in zip(CASES, robots):
            q = torch.full_like(robot.data.default_joint_pos, .4)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.write_joint_armature_to_sim(torch.full_like(q, ARMATURE))
            # Tensor API uses SI coordinates; the fixture verifies viscous units
            # against the predicted 0.2 rad/s terminal speed, independently.
            properties = torch.tensor([[[0., 0., 0.] if smooth else
                                        [FRICTION, FRICTION, DAMPING]]], dtype=torch.float32)
            view = robot.root_physx_view
            view.set_dof_friction_properties(properties, torch.tensor([0], dtype=torch.int32))
            actual = view.get_dof_friction_properties().cpu().numpy()
            if not np.allclose(actual, properties.numpy(), rtol=0, atol=1e-8):
                raise ValueError('Backend friction properties differ from the fixture')
            readback.append(actual.tolist()); robot.update(DT)
        for step in range(STEPS):
            commands = []
            for (_, excitation, smooth), robot in zip(CASES, robots):
                velocity = float(robot.data.joint_vel[0, 0])
                applied = excitation if step < STEPS // 2 else 0.
                passive = -DAMPING*velocity-FRICTION*np.tanh(velocity/.001) if smooth else 0.
                robot.set_joint_effort_target(torch.full_like(robot.data.joint_pos, applied+passive))
                robot.write_data_to_sim(); commands.append((applied, passive))
            sim.step(render=False)
            for i, robot in enumerate(robots):
                robot.update(DT)
                rows[i].append([(step+1)*DT, float(robot.data.joint_pos[0, 0]),
                                float(robot.data.joint_vel[0, 0]), *commands[i]])
        return np.asarray(rows), dict(backend_friction_properties=readback,
                                     device='cuda:0', solver='TGS 32/8')
    finally:
        app.close()


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['native', 'isaac'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    rows, metadata = native() if a.backend == 'native' else isaac()
    np.savez_compressed(a.output/'trace.npz', rows=rows)
    metrics = {}
    for i, (name, _, _) in enumerate(CASES):
        metrics[name] = dict(final_angle_rad=float(rows[i, -1, 1]),
            driven_displacement_rad=float(rows[i, 249, 1]-.4),
            late_driven_velocity_rad_s=float(np.mean(rows[i, 200:250, 2])),
            late_coast_peak_velocity_rad_s=float(np.max(np.abs(rows[i, 450:, 2]))))
    checks = dict(all_steps=rows.shape == (4, STEPS, 5), finite=bool(np.isfinite(rows).all()),
        no_joint_limit_hit=bool(np.max(np.abs(rows[:, :, 1])) < 1.5),
        backend_static_resists_small_load=abs(metrics['static_under']['driven_displacement_rad']) < .002,
        backend_viscous_units=abs(metrics['static_over']['late_driven_velocity_rad_s']-.2) < .01,
        backend_coast_stops=metrics['static_over']['late_coast_peak_velocity_rad_s'] < .001)
    report = dict(scope=__doc__, backend=a.backend, checks=checks, passed=all(checks.values()),
        physics_dt_s=DT, steps_per_fixture=STEPS, original_link_effective_inertia=INERTIA,
        excitation='Each case receives 0.005 or 0.02 Nm for 0.5 s, then zero for 0.5 s.',
        reset_angle_rad=.4, runtime_state_writes=0, metadata=metadata, metrics=metrics,
        columns=['t', 'q', 'v', 'external_excitation_Nm', 'explicit_passive_Nm'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
