"""Evaluate a continuous privileged teacher on actual student-visited states.

The native model is an analytic FK/dynamics calculator only. Expert actions are
counterfactual labels, never delivered to a simulator. Invalid or infeasible
queries are retained and cannot be admitted as expert supervision.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from doorbench.dexterous.acquisition_teacher import AcquisitionTeacher
from doorbench.dexterous.offline_teacher_queries import load_query_evidence,sha,correction_quality
from doorbench.dexterous.provenance import capture


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','robot','reference','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--solver-max-iterations',type=int)
    ap.add_argument('--solver-rho',type=float)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing correction experiments')
    queries,contract,physical,source=load_query_evidence(args.run)
    motors=json.loads((args.run/'motor-contract.json').read_text());reference=json.loads(args.reference.read_text())
    provenance=json.loads((args.run/'provenance.json').read_text())
    original=[v for k,v in provenance['files'].items() if k.endswith('/reference.json')]
    if original!=[sha(args.reference)] or sha(args.robot)!=motors['source_xml_sha256']:
        raise ValueError('Teacher model/reference differ from actual actor reset source')
    settings={k:v for k,v in (('max_iter',args.solver_max_iterations),('rho',args.solver_rho)) if v is not None}
    capture(Path(__file__).resolve().parents[2],args.output,dict(run=str(args.run),robot=str(args.robot),reference=str(args.reference),stance_solver_settings=settings,scope=__doc__),timing='before_offline_teacher_queries')
    args.output.joinpath('reference.json').write_bytes(args.reference.read_bytes())
    teacher=AcquisitionTeacher(args.robot,motors,reference,stance_solver_settings=settings)
    def forbidden_step(*args,**kwargs):raise RuntimeError('A teacher-query diagnostic must never step a physics model')
    mujoco.mj_step=forbidden_step
    if hasattr(mujoco,'mj_step1'):mujoco.mj_step1=forbidden_step
    if hasattr(mujoco,'mj_step2'):mujoco.mj_step2=forbidden_step
    source_checks=json.loads((args.run/'report.json').read_text())['checks']
    globally_physical=all(source_checks.get(k) is True for k in ('joint_stops','documented_loopbacks','self_collision','environment_collision',
        'working_hand_collision','plant_parameters_unchanged','finite','motor_delivery_matches_command','native_motor_caps'))
    names=contract['joint_order'];body_names=contract['hand_body_order'];caps=teacher.caps
    rows=[];labels=[];previous_valid=True
    for i,t in enumerate(queries['time_s']):
        state=queries['root_state'][i];q=queries['joint_position'][i];v=queries['joint_velocity'][i]
        error=None
        try:
            force,info=teacher.force(float(t),state,dict(zip(names,q)),dict(zip(names,v)),queries['handle_pose'][i],dict(zip(body_names,queries['right_hand_forces_world'][i])))
        except Exception as exc:
            force=np.zeros(len(caps));info={};error=type(exc).__name__+': '+str(exc)
        quality=correction_quality(state,force,caps,solver_status=info.get('stance_status'),source_physical=globally_physical,teacher_exception=error)
        valid=quality['candidate_label_valid']
        prefix_valid=previous_valid and valid;previous_valid=prefix_valid
        labels.append(force);rows.append(dict(time_s=float(t),**quality,solver_status=info.get('stance_status'),teacher_exception=error,
            contiguous_valid_prefix=prefix_valid,teacher_info=info,
            maximum_correction_from_student_motor_force=float(np.max(abs(force-physical['motor_forces'][i])))))
    labels=np.asarray(labels);normalized=2*(labels-caps[:,0])/(caps[:,1]-caps[:,0])-1
    valid=np.array([r['candidate_label_valid'] for r in rows],bool);prefix=np.array([r['contiguous_valid_prefix'] for r in rows],bool)
    np.savez_compressed(args.output/'counterfactual-teacher-actions.npz',time_s=queries['time_s'],motor_force=labels,
        normalized_force=normalized,label_valid=valid,contiguous_valid_prefix=prefix)
    (args.output/'queries.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(schema='doorbench.counterfactual-acquisition-teacher.v1',scope=__doc__,source=source,
        reference_sha256=sha(args.reference),robot_sha256=sha(args.robot),source_actor_task_passed=source['source_actor_task_passed'],
        queries=len(rows),candidate_valid_labels=int(valid.sum()),contiguous_valid_prefix_labels=int(prefix.sum()),
        first_invalid_query=next((r for r in rows if not r['candidate_label_valid']),None),
        maximum_teacher_internal_sim_time_s=float(teacher.d.time),teacher_force_calls=teacher.ticks,
        teacher_actions_delivered_to_physics=False,student_actions_used_as_expert_labels=False,physical_recovery_evaluated=False,
        teacher_contract='Original continuous AcquisitionTeacher, strict solved QP; no resets between visited states; measured normal hand loads only',
        stance_solver_settings=settings,stance_residual_tolerances=dict(eps_abs=1e-4,eps_rel=1e-4),
        files_sha256={name:sha(args.output/name) for name in ('reference.json','counterfactual-teacher-actions.npz','queries.json','manifest.json','source.tar.gz')},
        limitations=['Counterfactual analytic labels are not physically validated corrective trajectories.',
            'Global source mechanical bounds are preserved; upright validity and solver feasibility are checked for each actual visited state.',
            'The teacher uses the same body-origin normal-contact approximation as its qualified Isaac adapter.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('queries','candidate_valid_labels','contiguous_valid_prefix_labels','first_invalid_query','teacher_force_calls','maximum_teacher_internal_sim_time_s')}),flush=True)


if __name__=='__main__':main()
