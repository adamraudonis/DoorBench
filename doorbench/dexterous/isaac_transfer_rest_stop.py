"""Prospective first resting-transfer window from existing actual measurements.

Pure observer: no plant, command, wall clock, archive replay or contact inference.
The selected pad audit owns anatomy/opposition; the independent raw audit still
owns final physical qualification. A trigger only chooses the episode endpoint.
"""
import copy
import math

import numpy as np

from .grasp_verification import grasp_profile
from .isaac_opening_measurements import pose_parts


SCHEMA='doorbench.isaac-transfer-rest-stop.v1'
DT=.002
WINDOW_SAMPLES=251
CHECK_NAMES=('complete_clock','raw_evidence_complete','selected_right_grasp',
    'all_loaded_right_patches_qualified','no_non_digit_handle_force',
    'measured_palm_load_accounting','left_palm_support','stance_solved',
    'resting_mechanism','transfer_started_at_declared_epoch','route_reach_complete')


def _number(value,label):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+label+' required')
    return float(value)


def _flag(value,label):
    if not isinstance(value,(bool,np.bool_)):raise ValueError('Boolean '+label+' required')
    return bool(value)


def _count(value,label):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError('Integer '+label+' required')
    return int(value)


class TransferRestStop:
    def __init__(self,start_seconds,dt=DT):
        self.start_seconds=_number(start_seconds,'transfer start')
        self.dt=_number(dt,'physics interval')
        if (self.dt!=DT or self.start_seconds<=0
                or abs(self.start_seconds/DT-round(self.start_seconds/DT))>1e-8):
            raise ValueError('Positive 500 Hz transfer start and original .002 second interval required')
        self._start_tick=round(self.start_seconds/DT)
        self._count=0;self._window=0;self._maximum=0;self._window_start=None
        self._last=None;self._terminal=None;self._profile=None;self._failure=None;self._frozen=None
        self._checks=dict.fromkeys(CHECK_NAMES,False)

    def receipt(self):
        if self._frozen is not None:return copy.deepcopy(self._frozen)
        return dict(schema=SCHEMA,triggered=self._terminal is not None,terminal_time_s=self._terminal,
            window_samples=self._window,required_window_samples=WINDOW_SAMPLES,window_start_s=self._window_start,
            maximum_window_samples=self._maximum,maximum_window_duration_s=max(0,self._maximum-1)*DT,
            start_seconds=self.start_seconds,dt=self.dt,minimum_terminal_time_s=(self._start_tick+4250)*DT,
            observed_samples=self._count,last_time_s=self._last,grasp_profile=self._profile,
            checks=self._checks.copy(),failed_checks=[key for key,value in self._checks.items() if not value],
            failure=self._failure,physical_qualification=False,
            scope='Prospective measured endpoint only; full independent physical/contact audits remain required')

    def observe(self,time_s,pad,surface_row,angles):
        """Consume each post-step interval from .002; freeze at first trigger.

        Later calls return True without inspecting new data once the endpoint
        has been chosen. A continuation's separate prefix witness owns its
        later intervals; this receipt never claims to have qualified them.
        """
        if self._frozen is not None:return True
        if self._failure is not None:raise ValueError('Transfer rest stop is terminal: '+self._failure)
        try:
            now=_number(time_s,'sample epoch');tick=self._count+1
            if abs(now-tick*DT)>1e-8:raise ValueError('Contiguous actual 500 Hz epochs from .002 required')
            raw=pad['raw_evidence']
            clocks=[(pad['sim_time_s'],now),(pad['physics_dt_s'],DT),(surface_row['time_s'],now),
                (raw['interval_start_s'],now-DT),(raw['interval_end_s'],now),(raw['geometry_time_s'],now)]
            if any(abs(_number(value,'synchronized evidence clock')-expected)>1e-8 for value,expected in clocks):
                raise ValueError('Same-epoch pad, raw, surface and geometry evidence required')
            profile=grasp_profile(pad['grasp_profile'])
            if self._profile is not None and profile!=self._profile:raise ValueError('Selected grasp profile changed within the episode')
            valid=_flag(pad['valid_pad_grasp'],'selected grasp')
            contacts=pad['contacts']
            if not isinstance(contacts,list) or not isinstance(raw['contacts'],list):raise ValueError('Complete copied contact lists required')
            loaded=set();patches_ok=True
            for patch in contacts:
                force=_number(patch['normal_force_N'],'loaded patch force')
                qualified=_flag(patch['pad_qualified'],'selected patch qualification')
                if force<0:patches_ok=False
                elif force>0:
                    patches_ok &= qualified
                    loaded.add(patch['digit'])
            nondigit=_number(pad['non_digit_handle_force_N'],'non-digit handle force')
            capacity=_count(pad['contact_capacity'],'pad capacity');active=_count(pad['active_contact_count'],'active pad count')
            raw_capacity=_count(raw['contact_capacity'],'raw capacity');raw_active=_count(raw['active_contact_count'],'raw active count')
            pair_error=_number(pad['normal_pair_force_consistency_error_N'],'pad pair-force error')
            raw_error=_number(raw['normal_pair_force_consistency_error_N'],'raw pair-force error')
            raw_ok=(raw.get('schema')=='doorbench.shadow-raw-pad-evidence.v1'
                and raw.get('clock')=='physx-interval-end' and raw.get('scope')=='complete-handle-body'
                and capacity==raw_capacity and active==raw_active and 0<=active<capacity
                and len(contacts)<=len(raw['contacts'])<=active and 0<=pair_error<=1e-3
                and raw_error==pair_error and not raw.get('truncated',False) and not pad.get('truncated',False)
                and not raw.get('dropped_contacts',0) and not pad.get('dropped_contacts',0))
            _,rotation=pose_parts(surface_row['leaf_pose'])
            surface=surface_row['surface'];vector=np.asarray(surface['body_panel_forces_world_N']['lh_palm'],float)
            if vector.shape!=(3,) or not np.isfinite(vector).all():raise ValueError('Finite measured palm force vector required')
            palm=max(0.,float(-rotation[:,1]@vector));reported=_number(surface['palm_normal_load_N'],'reported palm load')
            mechanism={name:_number(angles[name],name+' coordinate') for name in ('leaf','operator','latch')}
            started=surface_row['started_s']
            if started is not None:started=_number(started,'actual transfer start')
            checks=dict(complete_clock=True,raw_evidence_complete=bool(raw_ok),
                selected_right_grasp=bool(valid and pad.get('hand')=='rh' and loaded=={'ff','mf','rf','lf','th'}),
                all_loaded_right_patches_qualified=bool(patches_ok),no_non_digit_handle_force=nondigit==0.,
                measured_palm_load_accounting=abs(palm-reported)<=1e-8,left_palm_support=palm>=2.,
                stance_solved=surface_row['stance_status'] in ('solved','solved inaccurate'),
                resting_mechanism=(.075<=mechanism['leaf']<=.10 and abs(mechanism['operator'])<=.05 and abs(mechanism['latch'])<=.001),
                transfer_started_at_declared_epoch=started==self.start_seconds,
                route_reach_complete=tick>=self._start_tick+4000)
            self._profile=profile;self._checks=checks;self._count=tick;self._last=now
            if all(checks.values()):
                if self._window==0:self._window_start=now
                self._window+=1;self._maximum=max(self._maximum,self._window)
            else:self._window=0;self._window_start=None
            if self._window==WINDOW_SAMPLES:
                self._terminal=now;self._frozen=self.receipt()
                return True
            return False
        except Exception as error:
            self._failure=str(error);self._checks['complete_clock']=False
            self._window=0;self._window_start=None
            raise ValueError('Invalid transfer rest-stop evidence: '+self._failure) from error
