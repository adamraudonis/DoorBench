"""Producer-owned transfer pause and continuation; no physics/event-loop calls."""
import copy
import json
from pathlib import Path
import time

import numpy as np

from .isaac_live_planning_pause import LivePlanningPause
from .isaac_live_snapshot_assembly import assemble_paused_transfer_snapshot
from .isaac_paused_transfer_audit import CHECKS, MECHANICAL_LIMITS
from .isaac_withdrawal_runtime import (
    admit_paused_isaac_withdrawal_runtime,create_paused_isaac_withdrawal_controller)
from .standing_transfer_evaluation import standing_transfer_checks


class LiveEpisodeClock:
    """One original prefix, followed by one admitted suffix at the next index."""
    def __init__(self,maximum_seconds,dt=.002):
        if dt!=.002 or not np.isfinite(maximum_seconds) or maximum_seconds<=0:
            raise ValueError('Positive original 500 Hz episode budget required')
        self.dt=dt;self.limit=round(maximum_seconds/dt);self.current=-1;self.suffix=False

    def __iter__(self):
        while self.current+1<self.limit:
            self.current+=1
            yield self.current

    def append_suffix(self,start,duration):
        end=start+duration
        if (self.suffix or start!=(self.current+1)*self.dt
                or not np.isfinite([start,duration,end]).all() or duration<=0
                or abs(end/self.dt-round(end/self.dt))>1e-7):
            raise ValueError('One finite grid-aligned suffix at the completed interval required')
        self.limit=round(end/self.dt);self.suffix=True
        return self.limit*self.dt


def transfer_phase_declarations(*,physics_checks,states,pad_steps,transfer_steps,
        transfer,operation,door_names,epoch,prefix,rest):
    """Apply original producer reductions; independent snapshot audit follows."""
    checks=dict(physics_checks)
    positions=np.asarray(states['door']);times=np.asarray(states['time_s'])
    leaf=positions[:,door_names.index('leaf_hinge')]
    handle=positions[:,door_names.index('leaf_handle_hinge')]
    bolt=positions[:,door_names.index('leaf_latch_bolt_slide')]
    tail=[row for row in pad_steps if row['sim_time_s']>=epoch-.5-1e-8]
    checks.update(sustained_pad_grasp=bool(len(tail)>=251 and all(row['valid_pad_grasp'] for row in tail)),
        acquisition_precedes_operation=operation.started is not None,
        operator_driven_to_release=bool(handle.max()>=.80 and bolt.max()>=.011),
        partial_leaf_opening_held=bool(np.all((leaf[times>=epoch-.5]>=.075)&(leaf[times>=epoch-.5]<=.10))),
        opening_bounded_for_transfer=bool(leaf.max()<=.12),
        live_exact_source_prefix=prefix.receipt()['passed'],
        qualified_transfer_rest_endpoint=bool(rest.receipt()['triggered'] and rest.receipt()['terminal_time_s']==epoch))
    checks.update(standing_transfer_checks(transfer_steps,seconds=epoch,dt=.002,started_s=transfer.started))
    if set(checks)!=CHECKS or any(type(value) is not bool for value in checks.values()):
        raise ValueError('Exactly the original 26 transfer phase declarations required')
    return checks


def plan_from_live_transfer(*,directory,episode_id,observer,motors,timeout_seconds,
        capture_anchor,snapshot_inputs,sleep=time.sleep):
    """Called only after the complete interval; wait without pumping Kit."""
    observer.require_ready(snapshot_inputs['rest_detector']['terminal_time_s'])
    pause=LivePlanningPause(directory,episode_id=episode_id,
        retained_objects=observer.retained_objects(),timeout_seconds=timeout_seconds)
    try:
        anchor=capture_anchor()
        assembled=assemble_paused_transfer_snapshot(pause.directory/'snapshot',
            pause_token=pause.pause_token,anchor=anchor,observer_state=observer.snapshot(),**snapshot_inputs)
        pause.publish(assembled['snapshot_path'],anchor)
        pause._anchor(capture_anchor())
        print('LIVE_TRANSFER_PLANNING_REQUEST '+str(pause.request_path),flush=True)
        while pause.poll() is None:sleep(.25)
        admitted=[]
        def validate(response):
            runtime=response['files']['runtime']['path']
            admission=admit_paused_isaac_withdrawal_runtime(runtime,motors)
            result=admission.planning_receipt(pause);admitted.append(admission)
            return result
        pause.validate_resume(capture_anchor=capture_anchor,validate_plan=validate)
        admission=admitted[0]
        controller=create_paused_isaac_withdrawal_controller(observer,motors,admission.path,
            admission=admission,pause=pause)
        return controller,pause
    except BaseException as error:
        if pause.state!='resumed':pause.abort(type(error).__name__+': '+str(error))
        (pause.directory/'producer-failure.json').write_text(json.dumps(dict(
            error_type=type(error).__name__,error=str(error),pause=pause.receipt()),indent=2)+'\n')
        raise


def phase_mechanical_audit(mechanical,checks):
    result=copy.deepcopy(mechanical)
    result['checks']={name:checks[name] for name in (*MECHANICAL_LIMITS,'plant_parameters_unchanged')}
    result['passed']=all(result['checks'].values())
    return result


def source_paths():
    from .isaac_paused_release_context import paused_release_source_paths
    from .isaac_withdrawal_runtime import withdrawal_runtime_source_paths
    return tuple(dict.fromkeys((Path(__file__).resolve(),
        Path(__file__).with_name('isaac_live_snapshot_assembly.py'),
        Path(__file__).with_name('isaac_live_evidence_snapshot.py'),
        Path(__file__).with_name('isaac_live_pause_readback.py'),
        *paused_release_source_paths(),*withdrawal_runtime_source_paths())))
