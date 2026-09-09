"""Verified source identity for both Git worktrees and deployed source bundles."""
import hashlib
import json
from pathlib import Path
import subprocess


def source_provenance(root):
    root=Path(root).resolve()
    manifest=root/'source-manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text());files=record.get('files')
        if not isinstance(files,dict) or not files:raise ValueError('Require a nonempty source bundle manifest')
        identity=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        if identity!=record.get('sha256'):raise ValueError('Source bundle manifest identity differs')
        for name,digest in files.items():
            path=root/name
            if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
                raise ValueError('Source bundle contains an invalid file path')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Source bundle file changed: '+name)
        return dict(source_git_revision=record.get('source_git_revision'),source_bundle_sha256=identity)
    top=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=root,text=True,capture_output=True)
    if top.returncode or Path(top.stdout.strip()).resolve()!=root:
        raise ValueError('Require a Git worktree root or verified deployed source manifest')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    return dict(source_git_revision=revision,source_bundle_sha256=None)
