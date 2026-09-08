"""Lossless, bounded-memory numeric storage for privileged transition evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

STATE_FIELDS = ('interval_start_s','interval_end_s','geometry_time_s','qpos_before',
                'qvel_before','qpos_after','qvel_after','controls','actuator_force')
CONTACT_FIELDS = {'geom':(2,np.int32),'body':(2,np.int32),
    'position_world_m':(3,np.float64),'frame_world':((3,3),np.float64),
    'wrench_contact_frame':(6,np.float64),'distance_m':((),np.float64)}
BODY_FIELDS = {'body_ids':((),np.int32),'body_positions_world_m':(3,np.float64),
               'body_rotations_world':((3,3),np.float64)}


def packed(rows):
    """Use float64 throughout; ragged offsets preserve even zero-load contacts."""
    out={key:np.asarray([row[key] for row in rows],dtype=np.float64) for key in STATE_FIELDS}
    out['contact_offsets']=np.r_[0,np.cumsum([len(row['contacts']) for row in rows])]
    out['body_offsets']=np.r_[0,np.cumsum([len(row['body_ids']) for row in rows])]
    contacts=[c for row in rows for c in row['contacts']]
    for key,(shape,dtype) in CONTACT_FIELDS.items():
        shape=(shape,) if isinstance(shape,int) else shape
        out['contact_'+key]=np.asarray([c[key] for c in contacts],dtype=dtype).reshape((-1,*shape))
    for key,(shape,dtype) in BODY_FIELDS.items():
        shape=(shape,) if isinstance(shape,int) else shape
        out[key]=np.asarray([v for row in rows for v in row[key]],dtype=dtype).reshape((-1,*shape))
    return out


def unpacked(arrays):
    for i in range(len(arrays['interval_start_s'])):
        row={key:arrays[key][i].tolist() for key in STATE_FIELDS}
        start,end=arrays['contact_offsets'][i:i+2]
        row['contacts']=[{key:arrays['contact_'+key][j].tolist() for key in CONTACT_FIELDS}
                         for j in range(start,end)]
        start,end=arrays['body_offsets'][i:i+2]
        row.update({key:arrays[key][start:end].tolist() for key in BODY_FIELDS})
        yield row


class NativeTransitionArchive:
    """Write each completed chunk atomically; the manifest tracks durable rows."""
    def __init__(self,path,*,chunk_size=250):
        if chunk_size<1:raise ValueError('Expected a positive chunk size')
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.chunk_size=chunk_size;self.rows=[];self.chunks=[];self.closed=False
        self._manifest(False)

    def _manifest(self,complete):
        data=dict(schema='doorbench.native-transitions.v1',complete=complete,
            scope='Privileged evaluator-only actual mj_step solution; float64 lossless',
            chunks=self.chunks,rows=sum(v['rows'] for v in self.chunks))
        temporary=self.path/'manifest.pending'
        temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(self.path/'manifest.json')

    def write(self,row):
        if self.closed:raise ValueError('Archive already closed')
        self.rows.append(row)
        if len(self.rows)>=self.chunk_size:self.flush()

    def flush(self):
        if not self.rows:return
        name=f'transitions-{len(self.chunks):05d}.npz';path=self.path/name
        temporary=self.path/(name+'.pending')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**packed(self.rows))
        temporary.replace(path)
        self.chunks.append(dict(file=name,rows=len(self.rows),bytes=path.stat().st_size,
            sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
            interval_start_s=self.rows[0]['interval_start_s'],interval_end_s=self.rows[-1]['interval_end_s']))
        self.rows=[];self._manifest(False)

    def close(self,*,complete=True):
        if self.closed:return
        self.flush();self._manifest(complete);self.closed=True

    @staticmethod
    def read(path,*,allow_incomplete=False):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if manifest['schema']!='doorbench.native-transitions.v1':raise ValueError('Unknown archive schema')
        if not manifest['complete'] and not allow_incomplete:raise ValueError('Incomplete transition archive')
        count=0
        for chunk in manifest['chunks']:
            file=path/chunk['file']
            if hashlib.file_digest(file.open('rb'),'sha256').hexdigest()!=chunk['sha256']:
                raise ValueError('Transition archive hash mismatch')
            with np.load(file,allow_pickle=False) as arrays:
                rows=list(unpacked({key:arrays[key] for key in arrays.files}))
            if len(rows)!=chunk['rows']:raise ValueError('Transition archive row count mismatch')
            count+=len(rows);yield from rows
        if count!=manifest['rows']:raise ValueError('Transition manifest count mismatch')
