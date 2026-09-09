"""Read-only native packet and final-report adapter for continuous development."""
import copy
import numpy as np

from .native_post_opening_measurements import measured_state,measured_contacts,outward_release_normal
from .post_opening_teacher import PostOpeningTeacher
from .passage import RobotBounds


def packet(sim,previous_raw,physical_ok,names,actuators):
    _,_,_,state=measured_state(sim,names,('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide'),
        ('left_ankle_link','right_ankle_link','lh_palm','rh_palm','leaf','leaf_handle'))
    _,_,evidence=measured_contacts(sim.m,previous_raw,physics_qualified=physical_ok)
    normal=None
    if state['door_positions']['leaf_hinge']>=1.2 and evidence['left_hand_contacts']:
        normal=outward_release_normal(sim.m,previous_raw)
    return dict(body_poses=state['body_poses'],door_velocities=state['door_velocities'],
        continuation_evidence=evidence,applied_motor_forces=sim.d.actuator_force[actuators].copy(),
        release_normal_world=normal)


def annotate(sim,row,bounds):
    row['root_xyz']=sim.d.qpos[sim.root_qadr:sim.root_qadr+3].tolist()
    row['root_velocity_world']=sim.d.qvel[sim.root_vadr:sim.root_vadr+3].tolist()
    row['minimum_body_y_m']=float(bounds(sim.d)[0][1])


def report(controller,legacy,physics,sim):
    result=copy.deepcopy(legacy)
    # The legacy opening runner evaluates palm load at its terminal state.
    # Intentional release after an independently frozen opening changes that
    # question. Retain those old values separately, never rename them a pass.
    terminal_keys=('sustained_left_panel_load','usable_aperture_under_palm_load','sustained_left_palm_load')
    result['legacy_terminal_palm_checks']={k:result['checks'].pop(k) for k in terminal_keys}
    opening=controller.opening_audit
    result['checks']['independently_qualified_opening_handoff']=bool(opening and opening['passed'])
    if opening:
        result['checks'].update({'opening_'+k:v for k,v in opening['checks'].items()})
    end=float(sim.d.time);tail=[r for r in physics if r['sim_time_s']>=end-1.-1e-8]
    speed=max(float(np.linalg.norm(r['root_velocity_world'][:2])) for r in tail)
    origin=np.array(tail[0]['root_xyz'][:2]);excursion=max(float(np.linalg.norm(np.array(r['root_xyz'][:2])-origin)) for r in tail)
    result['checks'].update(continuous_controller_completed=controller.done,
        continuation_solver=controller.post.controller is not None and controller.post.controller.solver_failures==0,
        whole_body_beyond_doorway=physics[-1]['minimum_body_y_m']>.2,
        quiet_final_second=len(tail)>=500 and speed<.03 and excursion<.03,
        actual_force_delivery=controller.maximum_motor_delivery_error<=1e-5,
        no_continuous_controller_failure=controller.failure is None)
    result.update(passed=all(result['checks'].values()),opening_handoff=opening,
        continuous_handoffs=controller.handoffs,continuous_failure=controller.failure,
        handoff_policy=controller.handoff_policy,
        final_second_max_horizontal_speed_m_s=speed,final_second_horizontal_excursion_m=excursion,
        maximum_actual_motor_delivery_error_Nm=controller.maximum_motor_delivery_error,
        scope='Uninterrupted privileged native approach, handle operation, opening, release, stow and traversal development; no Isaac, repeatability or learned sensor-policy claim')
    return result
