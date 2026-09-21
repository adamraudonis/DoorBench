"""Detached same-process pause handshake; no Kit, controller or source authority.

The producer owns the complete live-state inventory and zero-update pause seam.
This helper never waits by pumping application events and never restores state.
Its receipt is a handshake, not physical-source or next-stage admission.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import secrets
import struct
import time

import numpy as np

DT = .002
SNAPSHOT_SCHEMA = 'doorbench.paused-isaac-transfer-snapshot.v1'
REQUEST_SCHEMA = 'doorbench.live-isaac-planning-request.v1'
RESPONSE_SCHEMA = 'doorbench.live-isaac-planning-response.v1'
ANCHOR_SCHEMA = 'doorbench.live-isaac-pause-anchor.v1'
MAX_RESPONSE_BYTES = 65536


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):value.update(block)
    return value.hexdigest()


def _digest_text(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _hex(value):
    return type(value) is str and len(value)==64 and all(v in '0123456789abcdef' for v in value)


def _identity(objects):
    if type(objects) is not dict or not objects or any(type(k) is not str or not k for k in objects):
        raise ValueError('Explicit retained live object inventory required')
    return {name:dict(python_id=id(value),type=type(value).__module__+'.'+type(value).__qualname__)
        for name,value in sorted(objects.items())}


def _encode(value):
    if isinstance(value,(np.ndarray,np.generic)):
        a=np.asarray(value)
        if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():raise ValueError('Finite numeric live arrays required')
        return dict(kind='array',dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
    if value is None or type(value) in (bool,int,str):return dict(kind=type(value).__name__,value=value)
    if type(value) is bytes:return dict(kind='bytes',length=len(value),sha256=hashlib.sha256(value).hexdigest())
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('Finite live scalar required')
        return dict(kind='float64',bytes_hex=struct.pack('<d',value).hex())
    if type(value) in (list,tuple):return dict(kind=type(value).__name__,items=[_encode(v) for v in value])
    if type(value) is dict and all(type(k) is str for k in value):
        return dict(kind='mapping',items={k:_encode(v) for k,v in sorted(value.items())})
    raise ValueError('Live fingerprint inventory must contain detached numeric/plain values')


def capture_pause_anchor(*,episode_id,step_index,epoch_s,physics_clock,
                         measured,controller,retained_objects,evidence_counts,
                         pending_unaccepted_command):
    """Hash supplied copies; never obtains plant data or calls a controller.

    The caller must provide complete actual backend readback and mutable state,
    not just a convenient subset. Structural validity is not that completeness
    proof; the producer's named inventory and source admission must enforce it.
    """
    if (type(episode_id) is not str or not episode_id or type(step_index) is not int or step_index<=0
            or type(epoch_s) not in (int,float) or not math.isfinite(epoch_s)
            or abs(epoch_s-step_index*DT)>1e-10):
        raise ValueError('Exact completed 500 Hz prefix clock required')
    if not measured or not controller or not physics_clock or not evidence_counts:
        raise ValueError('Explicit measured, controller, clock and evidence inventories required')
    if pending_unaccepted_command is not False:raise ValueError('Complete current interval with no pending command required')
    if any(type(v) is not int or v<0 for v in evidence_counts.values()):raise ValueError('Nonnegative exact evidence counts required')
    values=dict(measurement=_encode(measured),controller=_encode(controller),
        physics_clock=_encode(physics_clock),evidence_counts=copy.deepcopy(evidence_counts),
        retained_objects=_identity(retained_objects))
    return dict(schema=ANCHOR_SCHEMA,episode_id=episode_id,step_index=step_index,epoch_s=float(epoch_s),
        physics_dt_s=DT,measurement_fingerprint=_digest_text(values['measurement']),
        controller_fingerprint=_digest_text(values['controller']),
        controller_identity=_digest_text(values['retained_objects']),inventory=values,
        pending_unaccepted_command=False,scope='Caller-supplied exact inventory; no backend read or completeness proof performed here')


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('Duplicate response key')
        result[key]=value
    return result


def _read(path,maximum):
    path=Path(path)
    if path.stat().st_size>maximum:raise ValueError('Planning document exceeds byte budget')
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=_pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite planning JSON')))


class LivePlanningPause:
    """Fresh request -> file-only polling -> checked handshake or sticky abort.

    No sleep/event loop is owned here. Call ``poll`` on a bounded wall-clock
    cadence without app.update/render/physics calls. A ready response still
    requires ``validate_resume`` and the producer's actual stage authorization.
    """
    def __init__(self,directory,*,episode_id,retained_objects,timeout_seconds,monotonic=time.monotonic):
        if (type(episode_id) is not str or not episode_id
                or type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds)
                or not 0<timeout_seconds<=7200 or not callable(monotonic)):
            raise ValueError('Explicit live episode and bounded planning deadline required')
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=False)
        self.episode_id=episode_id;self.pause_token=secrets.token_hex(32)
        self._objects=dict(retained_objects);self._identity=_digest_text(_identity(self._objects))
        self._clock=monotonic;self.started_wall=float(monotonic());self.last_wall=self.started_wall
        if not math.isfinite(self.started_wall):raise ValueError('Finite monotonic clock required')
        self.timeout=float(timeout_seconds);self.deadline=self.started_wall+self.timeout
        self.state='capturing';self.failure=None;self.anchor=None;self.snapshot=None;self.response=None
        self.request_path=self.directory/'request.json';self.response_path=self.directory/(self.pause_token+'.response.json')

    def _time(self):
        now=float(self._clock())
        if not math.isfinite(now) or now<self.last_wall:raise ValueError('Planning monotonic clock moved backward or became nonfinite')
        self.last_wall=now
        if now>=self.deadline:raise TimeoutError('Live planning deadline expired')
        return now

    def _persist(self,name,value):
        with (self.directory/name).open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

    def abort(self,reason):
        if self.state=='aborted':return self.receipt()
        if self.state=='resumed':raise ValueError('Completed pause handshake cannot be reused')
        self.state='aborted';self.failure=str(reason)
        try:self._persist('terminal.json',self.receipt())
        except Exception:pass  # Parent finalization also preserves receipt(); never mask the primary failure.
        return self.receipt()

    def _guard(self,operation):
        if self.state=='aborted':raise ValueError('Planning pause is terminal after abort: '+self.failure)
        try:return operation()
        except Exception as error:
            self.abort(type(error).__name__+': '+str(error));raise

    def _anchor(self,value):
        if (type(value) is not dict or value.get('schema')!=ANCHOR_SCHEMA
                or value.get('episode_id')!=self.episode_id or value.get('controller_identity')!=self._identity
                or value.get('physics_dt_s')!=DT or value.get('pending_unaccepted_command') is not False
                or _digest_text(value['inventory']['retained_objects'])!=self._identity
                or _digest_text(value['inventory']['measurement'])!=value['measurement_fingerprint']
                or _digest_text(value['inventory']['controller'])!=value['controller_fingerprint']
                or type(value['step_index']) is not int or value['step_index']<=0
                or type(value['epoch_s']) not in (int,float) or not math.isfinite(value['epoch_s'])
                or abs(value['epoch_s']-value['step_index']*DT)>1e-10):
            raise ValueError('Exact same-live-process anchor required')
        if self.anchor is not None and value!=self.anchor:raise ValueError('Live state, controller, clock or evidence changed during planning pause')

    def _inside(self,path):
        path=Path(path)
        if not path.is_absolute() or path.is_symlink():raise ValueError('Absolute fresh pause-local file required')
        resolved=path.resolve()
        if not resolved.is_relative_to(self.directory):raise ValueError('Plan/snapshot file escapes fresh pause directory')
        return resolved

    def _files(self,entries):
        if type(entries) is not dict or not entries or len(entries)>64:raise ValueError('Bounded explicit planning file manifest required')
        result={}
        for role,item in entries.items():
            if type(role) is not str or type(item) is not dict or set(item)!={'path','sha256'} or not _hex(item['sha256']):
                raise ValueError('Exact file role/path/hash required')
            path=self._inside(item['path'])
            if str(path) in result or _sha(path)!=item['sha256']:raise ValueError('Duplicate or changed pause-local file')
            result[str(path)]=item['sha256']
        return result

    def publish(self,snapshot_path,anchor):
        def operation():
            if self.state!='capturing':raise ValueError('A pause request may be published only once')
            self._time();self._anchor(anchor)
            path=self._inside(snapshot_path);before=_sha(path);snapshot=_read(path,4*1024*1024)
            pause=snapshot['live_pause']
            if (snapshot.get('schema')!=SNAPSHOT_SCHEMA or snapshot.get('source_kind')!='paused-live-isaac-transfer-v1'
                    or snapshot.get('source_engine')!='isaac-physx' or 'engine' in snapshot
                    or snapshot.get('closed_prefix') is not True
                    or snapshot.get('episode_complete') is not False or pause.get('pause_token')!=self.pause_token
                    or any(pause.get(key)!=anchor[key] for key in ('episode_id','controller_identity','step_index','epoch_s',
                        'physics_dt_s','measurement_fingerprint','controller_fingerprint'))):
                raise ValueError('Explicit unfinished same-live transfer snapshot required')
            hashes=self._files(snapshot['files'])
            if _sha(path)!=before:raise ValueError('Snapshot changed while publishing request')
            self.anchor=copy.deepcopy(anchor);self.snapshot=dict(path=str(path),sha256=before,files=hashes)
            request=dict(schema=REQUEST_SCHEMA,pause_token=self.pause_token,episode_id=self.episode_id,
                snapshot_path=str(path),snapshot_sha256=before,live_pause=copy.deepcopy(pause),
                response_path=str(self.response_path),timeout_seconds=self.timeout,
                instructions_scope='Detached CPU planning/audits only; no plant/controller handles, completed-run fabrication or resume authority',
                authorized_stages=0,episode_complete=False)
            self._persist('request.json',request);self.state='waiting';return copy.deepcopy(request)
        return self._guard(operation)

    def poll(self):
        def operation():
            if self.state not in ('waiting','response_ready'):raise ValueError('Pause is not waiting for planning')
            self._time()
            if self.state=='response_ready':return copy.deepcopy(self.response)
            if not self.response_path.exists():return None
            self._inside(self.response_path);before=_sha(self.response_path);document=_read(self.response_path,MAX_RESPONSE_BYTES)
            common={'schema','pause_token','episode_id','snapshot_path','snapshot_sha256','decision'}
            expected=common|({'reason'} if document.get('decision')=='abort' else {'source_context_sha256','files'})
            if (type(document) is not dict or set(document)!=expected or document['schema']!=RESPONSE_SCHEMA
                    or document['pause_token']!=self.pause_token or document['episode_id']!=self.episode_id
                    or document['snapshot_path']!=self.snapshot['path'] or document['snapshot_sha256']!=self.snapshot['sha256']):
                raise ValueError('Stale or mismatched planning response')
            if document['decision']=='abort':
                if type(document['reason']) is not str or not document['reason']:raise ValueError('Explicit planner abort reason required')
                raise ValueError('Detached planner requested abort: '+document['reason'])
            if document['decision']!='ready' or not _hex(document['source_context_sha256']):raise ValueError('Explicit fresh context identity required')
            if not {'runtime','phase_audit','context'}<=set(document['files']):raise ValueError('Runtime, fresh phase audit and source context must be bound')
            files=self._files(document['files'])
            if _sha(self.response_path)!=before:raise ValueError('Planning response changed during read')
            self.response=dict(document=document,sha256=before,files=files);self.state='response_ready'
            return copy.deepcopy(self.response)
        return self._guard(operation)

    def validate_resume(self,*,capture_anchor,validate_plan):
        """Return a one-use handshake after trusted admission and live rechecks.

        capture_anchor must perform the producer's independent backend readback
        without update/render calls. validate_plan must fresh-validate original
        phase and new geometry/runtime proofs; a worker's passed flag is not it.
        """
        def operation():
            if self.state!='response_ready':raise ValueError('Fresh completed planning response required')
            self._time();self._anchor(capture_anchor())
            bindings={**self.snapshot['files'],self.snapshot['path']:self.snapshot['sha256'],
                **self.response['files'],str(self.response_path):self.response['sha256']}
            for path,value in bindings.items():
                if _sha(path)!=value:raise ValueError('Paused source or planning input changed')
            admission=validate_plan(copy.deepcopy(self.response['document']))
            if (type(admission) is not dict or admission.get('passed') is not True
                    or admission.get('snapshot_sha256')!=self.snapshot['sha256']
                    or admission.get('source_context_sha256')!=self.response['document']['source_context_sha256']
                    or any(admission.get('input_sha256',{}).get(path)!=value for path,value in bindings.items())):
                raise ValueError('Fresh trusted runtime admission did not bind every pause/response input')
            self._anchor(capture_anchor());self._time()
            for path,value in bindings.items():
                if _sha(path)!=value:raise ValueError('Admission input changed before resume')
            self.admission=copy.deepcopy(admission)
            receipt=self.receipt();receipt.update(state='resumed',resume_handshake_passed=True)
            self._persist('terminal.json',receipt)
            self.state='resumed';return receipt
        return self._guard(operation)

    def receipt(self):
        return dict(schema='doorbench.live-isaac-planning-pause-receipt.v1',state=self.state,
            pause_token=self.pause_token,episode_id=self.episode_id,snapshot=copy.deepcopy(self.snapshot),
            anchor=copy.deepcopy(self.anchor),failure=self.failure,
            resume_handshake_passed=self.state=='resumed',trusted_admission=copy.deepcopy(getattr(self,'admission',None)),
            elapsed_wall_seconds=self.last_wall-self.started_wall,timeout_seconds=self.timeout,
            authorized_stages=0,physics_steps=0,active_state_writes=0,episode_complete=False,
            scope='Same-process handshake only. Producer retains objects; separate trusted runtime admission and pre-motor geometry guards own stage authority.')
