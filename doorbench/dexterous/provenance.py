"""Credential-free provenance for moving experiments between clusters."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(root,output,configuration,*,timing='before_training'):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def git(*args):
        r=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else None
    # Deliberately archive source only, never environment files, credentials or assets.
    files=sorted(set(root.glob('doorbench/**/*.py')) | set(root.glob('scripts/dexterous/*.py')) |
                 set(root.glob('scripts/dexterous/*.sh')) | {root/'pyproject.toml',root/'scripts/generate_dataset.py'})
    files=[p for p in files if p.is_file() and not p.is_symlink()]
    hashes={str(p.relative_to(root)):sha256(p) for p in files}
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
    deps={}
    for name in ('mujoco','torch','numpy','scipy','stable-baselines3','gymnasium','mujoco-warp','warp-lang'):
        try:deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    # Avoid pip-freeze direct URLs: private package URLs can embed credentials.
    packages=sorted({d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions()
                     if d.metadata['Name'] and d.metadata['Name'].lower()!='doorbench'})
    (output/'requirements.txt').write_text('\n'.join(packages)+'\n')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True) if __import__('shutil').which('nvidia-smi') else None
    inputs={}
    for name in ('robot','door','upstream'):
        p=Path(configuration[name])
        if name=='robot':inputs[name]={'sha256':sha256(p),'audit':json.loads(p.with_suffix('.audit.json').read_text())}
        elif name=='door':inputs[name]={f.name:sha256(f) for f in p.glob('*') if f.suffix in ('.xml','.json')}
        else:
            rev=subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True)
            inputs[name]={'revision':rev.stdout.strip(), 'checkpoint_files':
                {name:sha256(p/'data/reach_two_hands'/name) for name in ('torch_model.pt','mean.npy','var.npy')}}
    if configuration.get('checkpoint'):
        inputs['resume_checkpoint']={'sha256':sha256(configuration['checkpoint'])}
    report={'schema_version':'doorbench.run-manifest.v1','captured_at_unix':time.time(),
            'capture_timing':timing,'source_commit':git('rev-parse','HEAD'),
            'source_hashes':hashes,'source_archive_sha256':sha256(output/'source.tar.gz'),
            'configuration':configuration,'inputs':inputs,'dependencies':deps,
            'python':platform.python_version(),'platform':platform.platform(), 'cpu_count':os.cpu_count(),
            'gpu':gpu.stdout.strip() if gpu and gpu.returncode==0 else None,
            'portable_backend_claim':False}
    (output/'manifest.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    return report
