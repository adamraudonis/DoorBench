"""Preserve controller documents separately from large source episode archives."""
import hashlib
import json
from pathlib import Path


def snapshot_controller_inputs(paths, output, *, timing='before controller initialization'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records={};episodes={};pending=[Path(p) for p in paths if p is not None]
    # Only declared control-document references are followed. An audit may name
    # gigabytes of raw transitions; those belong to the source episode archive.
    references={'screen_path','audit_path','planner_source_path','panel_plan_path',
                'plan_path','dense_audit_path','scene_path_source','leaf_motion_audit_path'}
    def mappings(value):
        if isinstance(value,dict):
            yield value
            for child in value.values():yield from mappings(child)
        elif isinstance(value,list):
            for child in value:
                if isinstance(child,(dict,list)):yield from mappings(child)
    while pending:
        path=pending.pop().resolve()
        if str(path) in records:continue
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
        filename=digest[:16]+'-'+path.name
        destination=output/filename
        if destination.exists() and destination.read_bytes()!=data:
            raise ValueError('Conflicting content-addressed controller document')
        destination.write_bytes(data)
        records[str(path)]=dict(sha256=digest,bytes=len(data),snapshot=filename)
        if path.suffix!='.json':continue
        document=json.loads(data)
        if not isinstance(document,dict):continue
        for mapping in mappings(document):
            for key in references:
                value=mapping.get(key)
                if value is not None:pending.append(Path(value))
            for entry in mapping.get('panel_continuations',[]):pending.append(Path(entry['path']))
            for key in ('source_run','attained_trial'):
                if key not in mapping:continue
                episode=Path(mapping[key]).resolve()
                if str(episode) in episodes:continue
                episodes[str(episode)]={'scope':'Retain this separate complete source episode archive; it is not copied here'}
                trajectory=episode/'trajectory.npz'
                episodes[str(episode)]['available_locally']=trajectory.is_file()
                if not trajectory.is_file():continue
                for name in ('manifest.json','report.json','independent-pad-audit.json','independent-whole-handle-audit.json'):
                    if (episode/name).is_file():pending.append(episode/name)
                with trajectory.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                episodes[str(episode)]['trajectory_sha256']=digest
    result=dict(schema='doorbench.controller-input-documents.v1',capture_timing=timing,
                scope='Controller documents only; physical model assets and complete source episode archives must be retained separately',
                documents=records,source_episodes=episodes)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
