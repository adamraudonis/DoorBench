#!/usr/bin/env python3
"""Prepare a sensor-feedback Door55 reach or grasp on an existing ready v2 host.

This command never launches Isaac or allocates a GPU. --dry-run reads and hashes
inputs only. --check-only executes fresh native qualification, packet replay and
reset preflight, then writes the exact future Isaac invocation and audit command.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from doorbench.dexterous.isaac_readiness import load_ready_receipt, ready_directory, file_sha256

REFERENCE = 'configs/dexterous/door55-precurl-v2/reference.json'
CALIBRATION = 'configs/dexterous/sensor-balance-v1.json'
SCHEDULES = {'reach':'configs/dexterous/sensor-reach-balance-feedforward-v3.json',
             'grasp':'configs/dexterous/sensor-acquisition-balance-v1.json'}


def usd_dependencies(paths):
    """Resolve binary/text layer and asset closure using USD's CPU resolver."""
    from pxr import Sdf, UsdUtils
    files={};sdk=set()
    for path in paths:
        layers,assets,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(Path(path).resolve())))
        if not layers:raise ValueError('USD dependency resolver did not load root layer: '+str(path))
        for name in unresolved:
            # The authored imported robot references this installed rendering
            # shader. It is not a geometry/physics payload. No arbitrary missing
            # layer, texture, URI or provider is silently omitted.
            if name!='OmniPBR.mdl':raise ValueError('Unresolved USD dependency: '+name)
            sdk.add(name)
        for name in [*(layer.realPath for layer in layers),*assets]:
            if name=='OmniPBR.mdl':sdk.add(name);continue
            p=Path(name).resolve(strict=True)
            if not p.is_file():raise ValueError('USD dependency is not a regular local file: '+name)
            files[str(p)]=file_sha256(p)
    return dict(file_sha256=files,installed_sdk_shader_dependencies=sorted(sdk),
        limitation='OmniPBR.mdl is an installed rendering shader, bound by the Isaac runtime rather than source-asset bytes; missing geometry/texture/layer dependencies fail.')


def write_json(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def build_plan(receipt_path, output, *, task='reach', passive_profile='legacy-tanh-v1',
               root=ROOT, native_python=sys.executable, isaac_python=None):
    from doorbench.dexterous.motor_contract_identity import motor_contract_fingerprint
    from doorbench.dexterous.robot_design_identity import robot_design_identity
    root=Path(root).resolve();output=Path(output).resolve();receipt_path=Path(receipt_path).resolve()
    if output.exists():raise FileExistsError('Use a fresh preparation directory; retain failures')
    if task not in SCHEDULES or passive_profile not in ('legacy-tanh-v1','backend-dry-v2'):
        raise ValueError('Unknown component or passive profile')
    receipt=load_ready_receipt(receipt_path,expected_profile='shadow-loopback-v2')
    robot=Path(receipt['native_robot']);motors_path=Path(receipt['motor_contract'])
    door=Path(receipt['door_usd']).with_name('door.xml')
    ref=root/REFERENCE;calib=root/CALIBRATION;schedule=root/SCHEDULES[task]
    motors=json.loads(motors_path.read_text());reference=json.loads(ref.read_text())
    calibration=json.loads(calib.read_text());parameters=json.loads(schedule.read_text())
    if (parameters['source_reference_sha256']!=file_sha256(ref) or
            calibration['provenance']['source_reference_sha256']!=file_sha256(ref)):
        raise ValueError('Frozen reference/schedule/posture provenance differs')
    names=motors['joint_names'];source=reference['acquisition']
    if (source['joint_names']!=names or len(names)!=69 or len(set(names))!=69 or len(motors['actuators'])!=61 or
            calibration['desired_posture']!=dict(zip(names,source['path_qpos'][0]))):
        raise ValueError('Only the original named69-joint/61-motor fixed posture is supported')
    # A path-dependent identity change creates a *candidate* calibration. It is
    # never admitted using the historical trial: the same native phases run
    # even on a byte-identical host. No posture, gain or physical term changes.
    rebound=(calibration['robot_xml_sha256']!=file_sha256(robot) or
             calibration['motor_contract_sha256']!=motor_contract_fingerprint(motors))
    calibration['robot_xml_sha256']=file_sha256(robot)
    calibration['motor_contract_sha256']=motor_contract_fingerprint(motors)
    native=output/'native';inputs=output/'inputs';runtime=inputs/'runtime'
    calibration_path=inputs/'calibration.json';preflight=output/'reset-preflight.json'
    trial=output/'isaac-trial';kind='reach' if task=='reach' else 'acquisition'
    seconds=11 if task=='reach' else 19
    # Executable symlinks inside a venv must keep their venv path. Resolving
    # them to /usr/bin/python loses pyvenv.cfg and the installed dependencies.
    python=os.path.abspath(os.path.expanduser(os.fspath(native_python)))
    isaac=os.path.abspath(os.path.expanduser(os.fspath(isaac_python or Path(os.environ.get('DOORBENCH_WORK','/workspace'))/'venv/bin/python')))
    def command(script, *args):return [python,str(root/'scripts/dexterous'/script),*map(str,args)]
    common=['--robot',robot,'--door',door.parent,'--motors',motors_path,'--reference',ref,
            '--calibration',calibration_path,'--schedule',schedule,'--seconds',seconds,'--output',native]
    evaluation=['--run',native,'--robot',robot,'--calibration',calibration_path,
                '--protocol',runtime/'protocol.json','--joint-route',runtime/'joint-route.json']
    phases=[dict(name='project_joint_only_inputs',argv=command('prepare_sensor_'+kind+'_runtime.py',
                    '--reference',ref,'--schedule',schedule,'--motors',motors_path,'--output',runtime)),
        dict(name='fresh_native_component_qualification',argv=command('probe_sensor_'+kind+'_balance.py',*common),
             report=str(native/'report.json'),gate='passed',expected_duration_s=seconds),
        dict(name='independent_actual_state_qualification',argv=command('evaluate_native_sensor_'+kind+'.py',*evaluation,
                    '--output',output/'native-independent.json'),report=str(output/'native-independent.json'),gate='independent_evaluation.passed',expected_duration_s=seconds),
        dict(name='complete_causal_packet_replay',argv=command('check_sensor_'+kind+'_runtime.py',*evaluation,
                    '--output',output/'native-replay.json'),report=str(output/'native-replay.json'),gate='replay_passed',expected_duration_s=seconds),
        dict(name='fresh_closed_contact_free_reset_preflight',argv=command('preflight_sensor_reset.py',
                    '--reference',ref,'--motors',motors_path,'--native-robot',robot,'--native-door',door,
                    '--robot-usd',receipt['robot_usd'],'--door-usd',receipt['door_usd'],'--output',preflight),
             report=str(preflight),gate='passed')]
    argv=[isaac,str(root/'scripts/dexterous/isaac_opening.py'),'--robot-usd',receipt['robot_usd'],
        '--door-usd',receipt['door_usd'],'--motors',str(motors_path),'--reference',str(ref),
        '--sensor-balance-robot',str(robot),'--sensor-balance-calibration',str(calibration_path),
        '--sensor-layout',str(native/'sensor-layout.json'),'--sensor-reset-preflight',str(preflight),
        '--reset-from-acquisition-path','--sensor-'+kind+'-protocol',str(runtime/'protocol.json'),
        '--sensor-'+kind+'-route',str(runtime/'joint-route.json'),'--grasp-profile','distal-pad-v1',
        '--joint-passive-profile',passive_profile,'--seconds',str(seconds),'--output',str(trial),
        '--headless','--device','cuda:0','--record','--enable_cameras']
    hashes=dict(receipt['input_hashes'])
    usd_closure=usd_dependencies([receipt['robot_usd'],receipt['door_usd']])
    hashes.update(usd_closure['file_sha256'])
    source_files=sorted({*list((root/'doorbench').rglob('*.py')),*list((root/'scripts/dexterous').glob('*.py')),
                         *list((root/'scripts/isaac').glob('*.py'))})
    for path in [receipt_path,door,ref,calib,schedule,*source_files]:hashes[str(path.resolve())]=file_sha256(path)
    return dict(schema='doorbench.sensor-demo-preparation.v1',root=str(root),output=str(output),task=task,
        scope='Native-qualified preparation only; future actual Isaac component must independently pass; no vision learning, opening or traversal claim',
        ready_receipt=str(receipt_path),physics_dt_s=.002,duration_s=seconds,joint_passive_profile=passive_profile,
        calibration_rebound=rebound,candidate_calibration=calibration,original_calibration=str(calib),
        candidate_calibration_path=str(calibration_path),native_qualification_required=True,
        input_sha256=hashes,usd_dependency_closure=usd_closure,source_files=[str(p) for p in source_files],
        source_design={str(p):robot_design_identity(p) for p in (robot,door)},
        phases=phases,isaac_argv=argv,isaac_audit_argv=command('audit_isaac_sensor_balance.py',
            '--run',trial,'--robot',robot,'--output',output/'isaac-independent.json'),
        native_preparation_passed=False,isaac_launched=False,isaac_qualified=False)


def verify_inputs(plan):
    from doorbench.dexterous.robot_design_identity import verify_robot_design_identity
    for path,digest in plan['input_sha256'].items():
        if file_sha256(path)!=digest:raise ValueError('Frozen preparation input/source changed: '+path)
    for path,identity in plan['source_design'].items():verify_robot_design_identity(path,identity)


def verify_generated(hashes):
    for path,digest in hashes.items():
        if file_sha256(path)!=digest:raise ValueError('Generated qualification evidence changed: '+path)


def verify_launch(path):
    """Read-only launch-time guard; never starts the future GPU command."""
    plan=json.loads(Path(path).read_text())
    if (plan.get('schema')!='doorbench.sensor-demo-preparation.v1' or
            plan.get('native_preparation_passed') is not True or not plan.get('generated_input_sha256')):
        raise ValueError('Require completed native preparation manifest')
    verify_inputs(plan);verify_generated(plan['generated_input_sha256'])
    for phase in plan['phases']:require_phase_report(phase)
    return plan


def require_phase_report(phase):
    if 'report' not in phase:return
    report=json.loads(Path(phase['report']).read_text());value=report
    for key in phase['gate'].split('.'):value=value[key]
    if value is not True:raise ValueError('Qualification failed: '+phase['name'])
    seconds=phase.get('expected_duration_s')
    if seconds is not None:
        if phase['gate']=='replay_passed':
            if (report.get('replayed_decisions')!=round(seconds/.002) or report.get('failure') is not None or
                    report.get('physics_steps')!=0):raise ValueError('Incomplete packet replay')
        else:
            result=report['independent_evaluation'] if phase['gate'].startswith('independent_') else report
            checks=result.get('checks');duration=result.get('duration_s')
            if (type(checks) is not dict or not checks or any(v is not True for v in checks.values()) or
                    result.get('physics_steps')!=round(seconds/.002) or type(duration) not in (int,float) or
                    not math.isfinite(duration) or abs(duration-seconds)>1e-8):
                raise ValueError('Incomplete native physical qualification')


def execute(plan):
    """Only native CPU phases. The Isaac argv is data, never executed here."""
    root=Path(plan['root']);output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    progress=dict(schema='doorbench.sensor-demo-preparation-progress.v1',passed=False,completed_phases=[],
                  current_phase='freeze_inputs',isaac_launched=False)
    write_json(output/'plan.json',plan);write_json(output/'progress.json',progress)
    write_json(output/'pipeline.json',dict(stage='Prepare portable sensor demo',
        engine='MuJoCo CPU qualification',report_file='preparation-report.json',scope=plan['scope']))
    (output/'run.pid').write_text(str(os.getpid()))
    def status(message):
        print(message,flush=True)
        with (output/'run.log').open('a') as log:log.write(message+'\n')
    try:
        verify_inputs(plan)
        for src in plan['source_files']:
            dest=output/'frozen-source'/Path(src).relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        calibration=Path(plan['candidate_calibration_path']);calibration.parent.mkdir(parents=True,exist_ok=True)
        if plan['calibration_rebound']:write_json(calibration,plan['candidate_calibration'])
        else:shutil.copy2(plan['original_calibration'],calibration)
        write_json(output/'calibration-binding.json',dict(candidate=plan['calibration_rebound'],
            original_sha256=file_sha256(plan['original_calibration']),candidate_sha256=file_sha256(calibration),
            changed_fields=['robot_xml_sha256','motor_contract_sha256'] if plan['calibration_rebound'] else [],
            posture_or_gain_changed=False,historical_qualification_reused=False))
        generated={str(p):file_sha256(p) for p in (calibration,output/'calibration-binding.json')}
        env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        for index,phase in enumerate(plan['phases'],1):
            verify_inputs(plan);verify_generated(generated)
            progress['current_phase']=phase['name'];write_json(output/'progress.json',progress)
            status(f"== [{index}/{len(plan['phases'])}] {phase['name'].replace('_',' ')}")
            with (output/(phase['name']+'.log')).open('x') as log:
                subprocess.run(phase['argv'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            verify_inputs(plan);verify_generated(generated)
            require_phase_report(phase)
            # Bind outputs as soon as their producing phase completes, before
            # a later evaluator can accidentally consume edited/mixed evidence.
            for directory in (output/'inputs',output/'native'):
                if directory.exists():
                    for p in directory.rglob('*'):
                        if p.is_file():generated[str(p)]=file_sha256(p)
            if 'report' in phase:generated[phase['report']]=file_sha256(phase['report'])
            progress['completed_phases'].append(phase['name']);write_json(output/'progress.json',progress)
        verify_inputs(plan);verify_generated(generated)
        launch=dict(plan,native_preparation_passed=True)
        launch['generated_input_sha256']=generated
        write_json(output/'isaac-launch.json',launch)
        # POSIX quoting is only for a human-readable command; no shell execution.
        guard=[plan['phases'][0]['argv'][0],str(root/'scripts/isaac/prepare_sensor_demo.py'),
               '--verify-launch',str(output/'isaac-launch.json')]
        (output/'isaac-command.txt').write_text(shlex.join(guard)+' && '+shlex.join(launch['isaac_argv'])+'\n')
        progress.update(passed=True,current_phase='prepared_native_only');write_json(output/'progress.json',progress)
        write_json(output/'preparation-report.json',dict(passed=True,
            checks={phase['name']:True for phase in plan['phases']},scope=plan['scope'],isaac_launched=False))
        status('Native preparation complete; Isaac command is ready for a separate run.')
        return launch
    except BaseException as exc:
        progress.update(error=type(exc).__name__+': '+str(exc));write_json(output/'progress.json',progress)
        write_json(output/'preparation-report.json',dict(passed=False,
            checks={phase['name']:phase['name'] in progress['completed_phases'] for phase in plan['phases']},
            scope=plan['scope'],error=progress['error'],isaac_launched=False))
        status('PREPARATION_FAILED: '+progress['current_phase']+'; '+progress['error'])
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--task',choices=('reach','grasp'),default='reach')
    p.add_argument('--joint-passive-profile',choices=('legacy-tanh-v1','backend-dry-v2'),default='legacy-tanh-v1')
    p.add_argument('--native-python',type=Path,default=Path(sys.executable));p.add_argument('--isaac-python',type=Path)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--dry-run',action='store_true');mode.add_argument('--check-only',action='store_true');mode.add_argument('--verify-launch',type=Path)
    a=p.parse_args()
    if a.verify_launch:
        verify_launch(a.verify_launch);print('Launch inputs and native qualifications verified; no Isaac process started.');return
    if a.output is None:p.error('--output is required for preparation')
    ready=Path(os.environ.get('DOORBENCH_READY_DIR',ready_directory(ROOT,'shadow-loopback-v2')))
    plan=build_plan(a.receipt or ready/'ready.json',a.output,task=a.task,passive_profile=a.joint_passive_profile,
                    native_python=a.native_python,isaac_python=a.isaac_python)
    if a.check_only:
        plan=execute(plan)
        print(json.dumps(dict(native_preparation_passed=True,isaac_launched=False,
            output=plan['output'],launch_command=str(Path(plan['output'])/'isaac-command.txt'),
            qualification_manifest=str(Path(plan['output'])/'isaac-launch.json')),indent=2))
    else:print(json.dumps(plan,indent=2,allow_nan=False))


if __name__=='__main__':main()
