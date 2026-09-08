"""Copy minimal immutable evidence and source for the frozen CUDA fit experiment."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile

from doorbench.dexterous.legacy_teacher_provenance import FILES, SOURCE_HASHES
from doorbench.dexterous.sensor_training_bundle import SCHEMA, digest, load_bundle, bundled_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh immutable bundle directory')
    config=json.loads(args.experiment.read_text());root=Path(__file__).resolve().parents[2]
    if config['schema']!='doorbench.sensor-convergence-experiment.v1':raise ValueError('Unknown experiment')
    args.output.mkdir(parents=True)
    def copy(source, relative):
        destination=bundled_path(args.output,relative);destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(source,destination)
        if digest(source)!=digest(destination):raise ValueError('Evidence copy differs')
        return relative
    def evidence(run,names):
        for name in sorted(set(names)):copy(args.base/run/name,f'data/{run}/{name}')
        return f'data/{run}'
    teacher=config['teacher_run'];cold=config['cold_reset_run']
    teacher_path=evidence(teacher,list(FILES)+['source-'+Path(n).name for n in SOURCE_HASHES]+['acquisition-reset.json'])
    receipt=copy(args.base/config['teacher_receipt'],'data/'+config['teacher_receipt'])
    cold_path=evidence(cold,['acquisition-reset.json','configuration.json','motor-contract.json',
        'sensors/layout.json','sensors/report.json','sensors/actor-initial-decision.npz'])
    datasets=[dict(name='nominal',kind='qualified_teacher',run=teacher_path,
        qualification='acquisition-report.json',legacy_teacher_receipt=receipt,reset_observation_run=cold_path,examples=6457)]
    for row in config['corrections']:
        report=json.loads((args.base/row['labels']/'report.json').read_text())
        labels=evidence(row['labels'],['report.json']+list(report['files_sha256']))
        run=evidence(row['run'],list(report['source']['files_sha256']))
        datasets.append(dict(name=row['name'],kind='counterfactual_correction',labels=labels,run=run,examples=row['examples']))
    baseline=config['baseline_run']
    evidence(baseline,['manifest.json','source.tar.gz','report.json','early-force-comparison.json',
        'prediction-evaluation.json']+[f'actor00{i}-model{m}-prediction.json' for i in (2,3,4) for m in (4,5)])
    copy(args.experiment,'experiment.json')
    sources=sorted(set(root.glob('doorbench/**/*.py'))|set(root.glob('scripts/dexterous/*.py'))|{root/'pyproject.toml'})
    for source in sources:
        if source.is_file() and not source.is_symlink():copy(source,'source/'+str(source.relative_to(root)))
    hashes={str(f.relative_to(args.output)):digest(f) for f in args.output.rglob('*') if f.is_file()}
    manifest=dict(schema=SCHEMA,datasets=datasets,files_sha256=hashes,
        training_settings=config['training_settings'],readiness_limits=config['readiness_limits'],
        immutable_evidence=True,source_reports_rewritten=False,
        scope='Training inputs and actor/runtime calibration only. Simulator meshes are unnecessary for fitting and are not bundled. No physical execution or success claim.')
    manifest_path=args.output/'dataset-manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    episodes,_=load_bundle(manifest_path)
    frozen=json.loads((args.base/baseline/'report.json').read_text())['training_datasets']
    for episode,expected in zip(episodes,frozen,strict=True):
        for key in ('files','motor_contract_sha256','physics_dt_s','examples'):
            if episode.metadata.get(key)!=expected.get(key):raise ValueError(f'Frozen005 dataset differs: {key}')
        if episode.metadata.get('correction_report_sha256')!=expected.get('correction_report_sha256'):
            raise ValueError('Frozen005 correction labels differ')
    settings=config['training_settings']|config['execution']
    argv=['python','source/scripts/dexterous/train_sensor_imitation.py','--dataset-manifest','dataset-manifest.json','--output','run']
    for key,value in settings.items():argv+=['--'+key.replace('_','-'),str(value)]
    launch=dict(schema='doorbench.sensor-training-launch.v1',argv=argv,
        working_directory='Extracted bundle root',environment=dict(PYTHONPATH='source',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        dataset_manifest_sha256=digest(manifest_path),source_and_data_files=len(hashes),
        loaded_examples=[len(e) for e in episodes],frozen005_data_identity_verified=True,
        python_dependencies=['numpy','torch','scipy'],device_requirement='Existing CUDA-enabled PyTorch; record actual versions. No Isaac simulator process or native assets needed.',
        no_gpu_provisioning=True,maximum_wall_seconds=settings['max_wall_seconds'],
        result_scope='Prediction fitting only; no actor rollout, teacher fallback, physical state writes or successful task claim.')
    (args.output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    archive_path=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive_path,'w:gz') as archive:
        archive.add(args.output,arcname=args.output.name)
    print(json.dumps(dict(bundle=str(args.output),archive=str(archive_path),archive_sha256=digest(archive_path),
        archive_bytes=archive_path.stat().st_size,loaded_examples=launch['loaded_examples']),indent=2))


if __name__=='__main__':main()
