"""Verify the first fitting-qualified checkpoint for a new physical experiment.

This only runs the sensor-policy inference boundary on recorded observations.
It never runs a simulator, delivers teacher commands, or claims robot success.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch

from doorbench.dexterous.sensor_training_bundle import load_bundle,digest
from doorbench.dexterous.sensor_policy_controller import SensorPolicyController


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--run-directory',type=Path,required=True)
    p.add_argument('--runtime-protocol-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    episodes,bundle=load_bundle(args.dataset_manifest);e=episodes[0]
    verification=json.loads((args.run_directory/'download-verification.json').read_text())
    for name,expected in verification['files_sha256'].items():
        if digest(args.run_directory/name)!=expected:raise ValueError('Downloaded run changed after independent verification')
    thresholds=bundle['readiness_limits'];candidates=[];rows=[]
    for path in sorted(args.run_directory.glob('fit-step-*.json')):
        f=json.loads(path.read_text());step=f['completed_optimizer_steps']
        checkpoint=args.run_directory/f'actor-step-{step:06d}.pt'
        if digest(checkpoint)!=f['checkpoint_sha256']:raise ValueError('Fit report and weights differ')
        checks={n:f['prediction_mse_normalized_force'][n]['learned']<v*(1-1e-6) for n,v in thresholds['dataset_mse'].items()}
        checks.update({n:f['startup'][n]<v*(1-1e-6) for n,v in thresholds['startup'].items()})
        if checks!=f['readiness_checks'] or len(checks)!=6:raise ValueError('Frozen six checks differ')
        rows.append(dict(step=step,checks=checks,checkpoint_sha256=f['checkpoint_sha256']))
        if all(checks.values()):candidates.append((step,checkpoint,f,path))
    if not candidates:raise ValueError('No checkpoint meets every original fitting criterion')
    step,checkpoint,fit,fit_path=min(candidates,key=lambda x:x[0])
    actual=args.runtime_protocol_run
    motors=json.loads((actual/'motor-contract.json').read_text());layout=json.loads((actual/'sensors/layout.json').read_text())
    if layout!=e.layout or motors!=json.loads((e.path/'motor-contract.json').read_text()):
        raise ValueError('Proposed existing runtime plant/calibration differs from training evidence')
    if digest(actual/'acquisition-reset.json')!=digest(e.path/'acquisition-reset.json'):
        raise ValueError('Existing runtime reset differs from the qualified teacher reset')
    kwargs=dict(motor_contract=motors,sensor_layout=layout,physics_dt_s=.002)
    def make():return SensorPolicyController(checkpoint,**kwargs)
    checks={};rejections={}
    def rejected(name,action):
        try:action()
        except ValueError as error:checks[name]=True;rejections[name]=str(error)
        else:checks[name]=False
    actor=make();rejected('reset_required',lambda:actor.force(e.packet(0),float(e.times[0])))
    actor.reset_episode();forces=[]
    for i in range(251):
        packet=e.packet(i);packet['previous_action']=actor.previous_action
        forces.append(actor.force(packet,float(e.times[i])))
    forces=np.asarray(forces);caps=actor.force_ranges
    checks['finite_251_consecutive_inferences']=bool(np.isfinite(forces).all())
    checks['original_motor_caps']=bool(np.all(forces>=caps[:,0]) and np.all(forces<=caps[:,1]))
    actor.reset_episode();packet=e.packet(0);packet['previous_action']=actor.previous_action
    checks['reset_reproduces_first_action']=bool(np.array_equal(actor.force(packet,0.),forces[0]))
    packet=e.packet(1);packet['previous_action']=actor.previous_action
    rejected('skipped_2ms_tick_rejected',lambda:actor.force(packet,.004))
    checks['failed_clock_check_does_not_advance_state']=bool(np.array_equal(actor.force(packet,.002),forces[1]))
    changed=copy.deepcopy(motors);changed['actuators'][0]['force_range'][1]*=.99
    rejected('changed_valid_motor_caps_rejected',lambda:SensorPolicyController(checkpoint,**dict(kwargs,motor_contract=changed)))
    changed=copy.deepcopy(motors);changed['actuators'][0],changed['actuators'][1]=changed['actuators'][1],changed['actuators'][0]
    rejected('reordered_motors_rejected',lambda:SensorPolicyController(checkpoint,**dict(kwargs,motor_contract=changed)))
    rejected('different_physics_timestep_rejected',lambda:SensorPolicyController(checkpoint,**dict(kwargs,physics_dt_s=.004)))
    actor.reset_episode();packet=e.packet(0);packet['previous_action']=actor.previous_action
    for forbidden in ('root_state','door_pose','teacher_action','task_phase','goal_pose'):
        malformed=dict(packet);malformed[forbidden]=np.zeros(7)
        rejected('forbidden_'+forbidden+'_rejected',lambda:actor.force(malformed,0.))
    checks['failed_input_checks_do_not_advance_state']=bool(np.array_equal(actor.force(packet,0.),forces[0]))
    # Global clock offsets change neither relative sensor age nor actor output.
    shifted=make();shifted.reset_episode();shifted_forces=[]
    for i in range(4):
        packet=e.packet(i);packet['previous_action']=shifted.previous_action
        packet['sensor_time_s'][packet['sensor_time_s']>=0]+=10.
        shifted_forces.append(shifted.force(packet,float(e.times[i])+10.))
    checks['absolute_clock_offset_not_actor_feature']=bool(np.allclose(np.array(shifted_forces),forces[:4],rtol=0,atol=1e-5))
    receipt=dict(scope=__doc__,selection_rule='First checkpoint whose same-weight report meets all six original frozen fitting checks; final weights do not replace it.',
        selected_optimizer_step=step,checkpoint_path=str(checkpoint),checkpoint_sha256=digest(checkpoint),fit_report_sha256=digest(fit_path),
        all_checkpoint_checks=rows,selected_fitting_checks=fit['readiness_checks'],readiness_thresholds=thresholds,runtime_checks=checks,
        rejection_evidence=rejections,passed=all(checks.values()),eligible_for_root_controlled_physical_experiment=all(checks.values()),
        source_sha256=digest(Path(__file__)),runtime_class_sha256=digest(Path(__file__).resolve().parents[2]/'doorbench/dexterous/sensor_policy_controller.py'),
        dataset_manifest_sha256=digest(args.dataset_manifest),motor_contract_sha256=actor.motor_contract_sha256,robot_xml_sha256=actor.robot_xml_sha256,
        actual_protocol_run=str(actual),actual_protocol_files_sha256={n:digest(actual/n) for n in ('configuration.json','acquisition-reset.json','motor-contract.json','sensors/layout.json')},
        consecutive_inferences=251,physics_dt_s=.002,maximum_absolute_motor_command=float(np.abs(forces).max()),
        forbidden_actor_input_keys_tested=['root_state','door_pose','teacher_action','task_phase','goal_pose'],
        closed_loop_evaluated=False,task_success_rate=None,teacher_fallback=False,physical_state_steps=0,
        limitation='Inference and admission checks only. The robot has not executed this checkpoint; recorded sensors do not respond to its commands.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('all_checkpoint_checks','rejection_evidence','scope','actual_protocol_files_sha256')}))


if __name__=='__main__':main()
