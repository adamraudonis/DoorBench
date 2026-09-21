#!/usr/bin/env python3
"""Run one local standing operation from verified runtime and native receipts.

This tests privileged grasp/lever/partial opening, not traversal. The default
prints commands only. A failed experiment keeps its logs and independent audit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def repo_path(value):
    path=Path(str(value).replace('\\','/'))
    return (path if path.is_absolute() else ROOT/path).resolve()


def verified_hashes(records,*,base,required=()):
    if not isinstance(records,dict) or not records:
        raise ValueError('Missing recorded input hashes')
    bound={}
    for name,digest in records.items():
        path=Path(name.replace('\\','/'))
        path=(path if path.is_absolute() else base/path).resolve()
        if sha(path)!=digest:raise ValueError('Recorded prerequisite input changed: '+str(path))
        bound[str(path)]=digest
    if any(str(Path(path).resolve()) not in bound for path in required):
        raise ValueError('Prerequisite audit does not bind the selected trial inputs')
    return bound


def verify_hub_geometry(path,door_xml):
    """Compare the recorded body-local descriptor with its hashed door source."""
    geom=ET.parse(door_xml).find(".//body[@name='leaf_handle']/geom[@name='leaf_handle_hub_col_n']")
    if geom is None or geom.get('type')!='cylinder':raise ValueError('Declared handle hub geometry is unavailable')
    size=[float(v) for v in geom.get('size','').split()]
    position=[float(v) for v in geom.get('pos','0 0 0').split()]
    quaternion=[float(v) for v in geom.get('quat','1 0 0 0').split()]
    if len(size)!=2 or len(position)!=3 or len(quaternion)!=4 or not all(math.isfinite(v) for v in size+position+quaternion):
        raise ValueError('Explicit finite body-local hub cylinder required')
    norm=math.sqrt(sum(v*v for v in quaternion))
    if norm==0:raise ValueError('Invalid hub orientation')
    expected=dict(size=size+[0.],position=position,quaternion_wxyz=[v/norm for v in quaternion])
    actual=json.loads(path.read_text())
    for key,wanted in expected.items():
        values=actual.get(key)
        if (not isinstance(values,list) or len(values)!=len(wanted)
                or not all(type(v) in (int,float) and math.isfinite(v) and abs(v-w)<=1e-7 for v,w in zip(values,wanted))):
            raise ValueError('Recorded hub geometry differs from the native door')


def verified_inputs(args):
    from doorbench.dexterous.isaac_readiness import load_ready_receipt
    ready=load_ready_receipt(args.ready,expected_profile='shadow-loopback-v2')
    receipts=[args.ready]
    hashes=dict(ready['input_hashes'])
    reports={}
    for name in ('report.json','independent-contact-audit.json','independent-whole-handle-audit.json'):
        path=args.native_trial/name
        report=json.loads(path.read_text())
        if name=='independent-whole-handle-audit.json':
            checks_passed=(report.get('schema')=='doorbench.native-whole-handle-audit.v1'
                and report.get('extra_loaded_patches')==0 and report.get('affected_physics_intervals')==0
                and report.get('maximum_extra_force_N')==0.)
        else:
            checks_passed=bool(report.get('checks')) and all(v is True for v in report['checks'].values())
        if report.get('passed') is not True or not checks_passed:
            raise ValueError('Native physical/contact prerequisite failed: '+name)
        reports[name]=report
        receipts.append(path)
    manifest=args.native_trial/'manifest.json'
    native=json.loads(manifest.read_text());cfg=native['configuration'];receipts.append(manifest)
    if native['inputs']['robot']['sha256']!=ready['native_robot_sha256']:
        raise ValueError('Native trial and Isaac readiness identify different robots')
    if sha(Path(ready['door_usd']).with_name('door.xml'))!=native['inputs']['door']['door.xml']:
        raise ValueError('Native trial and Isaac door differ')
    if not cfg.get('portable_wrapper') or cfg.get('stance_profile')!='landed-foot-v1':
        raise ValueError('Standalone portable landed-foot native operation required')
    if any(cfg.get(k) for k in ('standing_transfer_path','standing_return_path','standing_withdrawal_path','jev_progress_plan','hold_attained_grasp')):
        raise ValueError('This launcher requires an unchanged standalone native baseline')
    if cfg.get('press_seconds')!=5. or cfg.get('landed_foot_max_iterations',50000)!=50000:
        raise ValueError('Isaac operation requires the original five-second press and 50000-iteration stance configuration')
    raw_manifest=args.native_trial/'raw-transitions/manifest.json'
    hashes.update(verified_hashes(reports['independent-contact-audit.json'].get('input_sha256'),base=args.native_trial,
        required=(manifest,args.native_trial/'report.json',args.native_trial/'physics-steps.json.gz',raw_manifest)))
    hashes.update(verified_hashes(reports['independent-whole-handle-audit.json'].get('input_sha256'),base=ROOT,
        required=(manifest,raw_manifest,repo_path(cfg['robot']),repo_path(cfg['door'])/'door.xml')))
    archive=json.loads(raw_manifest.read_text())
    if archive.get('complete') is not True or not archive.get('chunks'):raise ValueError('Complete native transition archive required')
    hashes.update(verified_hashes({c['file']:c['sha256'] for c in archive['chunks']},base=raw_manifest.parent))
    reference=repo_path(cfg['reference'])
    geometry_path=reference.with_name('geometry-audit.json')
    geometry=json.loads(geometry_path.read_text())
    if geometry.get('passed') is not True:
        raise ValueError('Local acquisition geometry screen is required')
    hashes.update(verified_hashes(geometry.get('input_sha256'),base=ROOT,
        required=(repo_path(cfg['robot']),repo_path(cfg['door'])/'door.xml')))
    if sha(reference)!=sha(args.native_trial/'reference.json'):
        raise ValueError('Acquisition reference differs from native trial')
    profile=cfg.get('grasp_profile','distal-pad-v1')
    if profile not in ('distal-pad-v1','volar-phalange-v1') or any(
            reports[n].get('grasp_profile','distal-pad-v1')!=profile for n in ('report.json','independent-contact-audit.json')):
        raise ValueError('Native report, contact audit and selected grasp profile differ')
    declaration=None
    if profile!='distal-pad-v1':
        declaration=args.native_trial/'grasp-profile-definition.json'
        spec=json.loads(declaration.read_text())
        if spec.get('profile')!=profile or spec.get('robot_xml_sha256')!=ready['native_robot_sha256']:
            raise ValueError('Prospective profile differs from calibrated robot')
        receipts.append(declaration)
    snapshots=args.native_trial/'controller-inputs/manifest.json'
    documents=json.loads(snapshots.read_text())
    if documents.get('schema')!='doorbench.controller-input-documents.v1':raise ValueError('Frozen native controller inputs required')
    records={repo_path(name):record for name,record in documents['documents'].items()}
    for path in (reference,geometry_path,repo_path(cfg['motors']),*([repo_path(cfg['grasp_profile_definition'])] if declaration else [])):
        record=records.get(path)
        if record is None or sha(path)!=record['sha256'] or sha(snapshots.parent/record['snapshot'])!=record['sha256']:
            raise ValueError('Native controller document changed or lacks a frozen snapshot: '+str(path))
        if path==repo_path(cfg['motors']) and sha(ready['motor_contract'])!=record['sha256']:
            raise ValueError('Ready motor contract differs from the native controller')
        if declaration and path==repo_path(cfg['grasp_profile_definition']) and sha(declaration)!=record['sha256']:
            raise ValueError('Recorded grasp declaration differs from the frozen native controller')
        receipts.extend((path,snapshots.parent/record['snapshot']))
    receipts.append(snapshots)
    if cfg.get('operation_handle_hub_avoidance'):
        hub=args.native_trial/'hub-geometry.json'
        verify_hub_geometry(hub,repo_path(cfg['door'])/'door.xml')
        receipts.append(hub)
    if args.jev_progress_plan is not None:
        from doorbench.dexterous.isaac_jev_progress import read_jev_plan
        read_jev_plan(args.jev_progress_plan)
        receipts.append(args.jev_progress_plan)
    receipts.extend((reference,reference.with_name('geometry-audit.json')))
    hashes.update({str(p.resolve()):sha(p) for p in receipts})
    return ready,cfg,reference,profile,declaration,hashes


def commands(args,ready,cfg,reference,profile,declaration):
    trial=args.output/'trial'
    argv=[str(args.isaac_python),'-u',str(ROOT/'scripts/dexterous/isaac_opening.py'),
        '--robot-usd',ready['robot_usd'],'--door-usd',ready['door_usd'],
        '--motors',ready['motor_contract'],'--native-robot',ready['native_robot'],
        '--reference',str(reference),'--output',str(trial),'--seconds',str(args.seconds),
        '--headless','--device','cuda:0','--record','--enable_cameras','--acquisition',
        '--reset-from-acquisition-path','--acquisition-stance-profile','landed-foot-v1',
        '--operate-after-acquisition','--joint-passive-profile','backend-dry-v2',
        '--grasp-profile',profile]
    mapped={'pressure_segment':'acquisition-pressure-segment','index_finger_force':'acquisition-index-finger-force',
        'middle_finger_force':'acquisition-middle-finger-force','operator_compliance_gain':'operator-compliance-gain',
        'min_acquisition_seconds':'operation-min-acquisition-seconds','index_proximal_offset_rad':'operation-index-proximal-offset-rad',
        'index_tendon_offset_rad':'operation-index-tendon-offset-rad','operation_leaf_target_rad':'operation-leaf-target-rad',
        'operation_opening_trigger_rad':'operation-opening-trigger-rad','operation_operator_lead_limit_rad':'operation-operator-lead-limit-rad',
        'operation_leaf_lead_limit_rad':'operation-leaf-lead-limit-rad','operation_operator_follow_after_leaf_rad':'operation-operator-follow-after-leaf-rad'}
    for key,option in mapped.items():
        if cfg.get(key) is not None:argv+=['--'+option,str(cfg[key])]
    argv+=['--operation-grasp-offset-in-handle-m',*map(str,cfg['grasp_offset_in_handle_m'])]
    if cfg.get('operation_fixed_pad_control'):
        argv+=['--operation-actual-pad-control','--operation-material-profile',cfg['operation_pad_control_profile']]
    if cfg.get('operation_handle_hub_avoidance'):
        argv+=['--operation-hub-geometry',str(args.native_trial/'hub-geometry.json'),
               '--operation-hub-clearance-m',str(cfg['operation_hub_clearance_m'])]
    if declaration is not None:argv+=['--grasp-profile-definition',str(declaration)]
    if args.open_on_latch_clear or cfg.get('open_on_latch_clear'):argv.append('--open-on-latch-clear')
    if args.review_render_profile:argv+=['--review-render-profile',args.review_render_profile]
    if args.jev_progress_plan is not None:
        argv+=['--jev-progress-plan',str(args.jev_progress_plan),'--jev-sample-period',str(args.jev_sample_period)]
    audit=[str(args.asset_python),str(ROOT/'scripts/dexterous/audit_isaac_acquisition_contacts.py'),
        '--trial',str(trial),'--output',str(args.output/'independent-contact-audit.json'),'--grasp-profile',profile]
    return argv,audit


def passed_checks(report):
    return bool(report.get('checks')) and all(value is True for value in report['checks'].values())


def verify_runtime_binding(trial,argv,receipt):
    configuration=json.loads((trial/'configuration.json').read_text())['args']
    provenance=json.loads((trial/'provenance.json').read_text())['files']
    recorded={str(Path(name).resolve()):digest for name,digest in provenance.items()}
    source=str((ROOT/'scripts/dexterous/isaac_opening.py').resolve())
    if recorded.get(source)!=receipt['source_sha256'][source]:
        raise ValueError('Runtime source differs from the launched source')
    for flag in ('robot-usd','door-usd','motors','native-robot','reference'):
        path=Path(argv[argv.index('--'+flag)+1]).resolve()
        if (recorded.get(str(path))!=receipt['input_sha256'].get(str(path))
                or Path(configuration[flag.replace('-','_')]).resolve()!=path):
            raise ValueError('Runtime input differs from verified launch: '+flag)
    for flag in ('grasp-profile','review-render-profile','jev-progress-plan'):
        expected=argv[argv.index('--'+flag)+1] if '--'+flag in argv else None
        if configuration.get(flag.replace('-','_'))!=expected:
            raise ValueError('Runtime experiment mode differs from launch: '+flag)
    if configuration.get('open_on_latch_clear') is not ('--open-on-latch-clear' in argv):
        raise ValueError('Runtime latch transition differs from launch')
    if '--jev-progress-plan' in argv:
        plan=Path(argv[argv.index('--jev-progress-plan')+1]).resolve()
        if sha(trial/'jev-progress-plan-input.json')!=receipt['input_sha256'].get(str(plan)):
            raise ValueError('Runtime Jev plan differs from the verified plan')


def stop_child(process,args,receipt):
    trial=args.output/'trial'
    if trial.exists():(trial/'stop.request').write_text('Local launcher stopped this trial; preserve failed prefix.\n')
    try:return process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        receipt['forced_termination']=True
        if os.name=='nt':
            stopped=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=FLAGS)
            receipt['termination_returncode']=stopped.returncode
        else:process.kill()
        return process.wait(timeout=30)


def execute(args,argv,audit,hashes):
    """Always finalize a failed receipt; completed evidence is audited on failure."""
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if args.jev_progress_plan is None:env.pop('TYPESAFE_API_KEY',None)
    # Argument validation and independent scoring never need the model credential.
    offline_env=dict(env);offline_env.pop('TYPESAFE_API_KEY',None)
    sources=(Path(__file__),ROOT/'scripts/dexterous/isaac_opening.py',ROOT/'scripts/dexterous/audit_isaac_acquisition_contacts.py')
    receipt=dict(scope='Local privileged standing acquisition, lever operation and partial opening; no traversal claim',
        input_sha256=hashes,source_sha256={str(path.resolve()):sha(path) for path in sources},
        argv=argv,audit_argv=audit,started_unix=time.time(),api_key_saved=False,
        passed=False,runtime_passed=False,independent_passed=False,physics_started=False)
    for source in sources:(args.output/('source-'+source.name)).write_bytes(source.read_bytes())
    save(args.output/'launch.json',receipt)
    try:
        with (args.output/'preflight.log').open('w') as log:
            preflight=subprocess.run(argv+['--validate-arguments-only'],cwd=ROOT,env=offline_env,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,timeout=90)
        receipt['preflight_returncode']=preflight.returncode
        if preflight.returncode:raise RuntimeError('Argument preflight failed')
        # Catch files changed during preflight instead of launching from stale proof.
        verified_hashes(hashes,base=ROOT)
        verified_hashes(receipt['source_sha256'],base=ROOT)
        with (args.output/'isaac.log').open('w') as log:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS)
            receipt.update(pid=process.pid,physics_started=True);save(args.output/'launch.json',receipt)
            try:code=process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                receipt['timeout']=True
                code=stop_child(process,args,receipt)
            except BaseException:
                receipt['interrupted']=True
                stop_child(process,args,receipt)
                raise
            receipt['returncode']=code
    except BaseException as error:
        receipt['launch_error_type']=type(error).__name__
        # The command and log paths suffice; arbitrary exception text can contain credentials.
    trial=args.output/'trial'
    report_path=trial/'operation-report.json'
    receipt['report_emitted']=report_path.exists()
    try:
        if report_path.exists():
            report=json.loads(report_path.read_text())
            receipt['report_sha256']=sha(report_path)
            try:
                verify_runtime_binding(trial,argv,receipt)
                receipt['runtime_binding_passed']=True
            except Exception as error:
                receipt['runtime_binding_passed']=False
                receipt['runtime_binding_error_type']=type(error).__name__
            receipt['runtime_passed']=bool(receipt['runtime_binding_passed'] and report.get('passed') is True and passed_checks(report)
                and not (trial/'early-stop.json').exists() and not (trial/'error.txt').exists()
                and abs(report.get('duration_s',0.)-args.seconds)<1e-8)
        # Even an acquisition-only failure may contain a useful complete raw prefix.
        audit_inputs=(trial/'configuration.json',trial/'provenance.json',trial/'acquisition-pad-steps.json.gz')
        if all(path.exists() for path in audit_inputs) and (report_path.exists() or (trial/'acquisition-report.json').exists()):
            auditor=str((ROOT/'scripts/dexterous/audit_isaac_acquisition_contacts.py').resolve())
            if sha(auditor)!=receipt['source_sha256'][auditor]:raise ValueError('Independent auditor changed after launch')
            with (args.output/'independent-audit.log').open('w') as log:
                result=subprocess.run(audit,cwd=ROOT,env=offline_env,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,timeout=600)
            receipt['audit_returncode']=result.returncode
            audit_path=args.output/'independent-contact-audit.json'
            if audit_path.exists():
                data=json.loads(audit_path.read_text());receipt['audit_sha256']=sha(audit_path)
                named_report=report_path if report_path.exists() else trial/'acquisition-report.json'
                verified_hashes(data.get('input_sha256'),base=trial,required=(*audit_inputs,named_report))
                receipt['independent_passed']=bool(result.returncode==0
                    and data.get('schema')=='doorbench.isaac-acquisition-contact-audit.v1'
                    and data.get('accounting_passed') is True and data.get('independent_raw_contact_audit_complete') is True
                    and passed_checks(data) and data.get('invalid_loaded_patches')==0)
    except Exception as error:
        receipt['audit_error_type']=type(error).__name__
    receipt['passed']=bool(receipt.get('returncode')==0 and receipt['runtime_passed'] and receipt['independent_passed']
        and not receipt.get('timeout') and not receipt.get('launch_error_type') and not receipt.get('audit_error_type'))
    receipt['finished_unix']=time.time()
    save(args.output/'result.json',receipt)
    print(json.dumps(receipt))
    return 0 if receipt['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ready','native-trial','output','asset-python','isaac-python'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=float,default=24.)
    p.add_argument('--timeout-seconds',type=float,default=3600.)
    p.add_argument('--open-on-latch-clear',action='store_true')
    p.add_argument('--review-render-profile',choices=('native-materials-v1',))
    p.add_argument('--jev-progress-plan',type=Path)
    p.add_argument('--jev-sample-period',type=float,default=.2)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not 20<=args.seconds<=60 or not 300<=args.timeout_seconds<=7200:
        p.error('Use a 20–60 second trial and 300–7200 second wall budget')
    if not math.isfinite(args.jev_sample_period) or not .05<=args.jev_sample_period<=10.:
        p.error('Jev sample period must be 0.05..10 wall-clock seconds')
    if args.jev_progress_plan is None and args.jev_sample_period!=.2:
        p.error('An explicit Jev sample period requires --jev-progress-plan')
    for name in ('ready','native_trial','output','asset_python','isaac_python','jev_progress_plan'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.absolute()) # Preserve short Windows interpreter alias.
    ready,cfg,reference,profile,declaration,hashes=verified_inputs(args)
    argv,audit=commands(args,ready,cfg,reference,profile,declaration)
    if not args.execute:
        print(json.dumps(dict(physics_started=False,operation=argv,independent_audit=audit),indent=2));return 0
    if args.jev_progress_plan is not None and not os.environ.get('TYPESAFE_API_KEY'):
        raise ValueError('TYPESAFE_API_KEY must be supplied through the process environment')
    return execute(args,argv,audit,hashes)


if __name__=='__main__':raise SystemExit(main())
