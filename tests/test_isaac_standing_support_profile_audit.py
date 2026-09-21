"""Synthetic saved profile histories; no physical source or load claim."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from doorbench.dexterous import isaac_standing_support_profile_audit as audit
from doorbench.dexterous.isaac_withdrawal_support import ReleaseQualifiedPalmLoadProfile, PALM_LOAD_PROFILE
from doorbench.dexterous.qualified_isaac_grasp import digest
from test_isaac_standing_reference_audit import source, admit, save_tail
from doorbench.dexterous import isaac_standing_reference_audit as reference
import test_isaac_standing_continuation_audit as observations


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def history(start, duration, end, release):
    profile = ReleaseQualifiedPalmLoadProfile(PALM_LOAD_PROFILE, 6., duration)
    result = []
    for i in range(round(end/.002)):
        t = i*.002
        event = release if release is not None and t >= release else None
        teacher = dict(withdrawal_started_s=start if t >= start else None,
                       release_started_s=event, inherited_support=None)
        if t >= start:
            _, info = profile.target(t, start, event)
            teacher['inherited_support'] = dict(info, target_N=6.)
        result.append(dict(sim_time_s=(i+1)*.002, teacher=teacher))
    return result


def gzwrite(path, rows):
    with gzip.open(path, 'wt') as stream:
        json.dump(rows, stream, allow_nan=False)


@pytest.fixture
def saved(tmp_path):
    start, duration, end, release = 42.548, 8., 50.548, 44.218
    source_path = tmp_path/'source.json'
    write(source_path, dict(start_time_s=start, duration_s=duration,
                           scope='Synthetic duration binding; no qualification'))
    trial = tmp_path/'trial'; trial.mkdir()
    (trial/'standing-withdrawal-source.json').write_bytes(source_path.read_bytes())
    captured = trial/'source-isaac_withdrawal_support.py'
    captured.write_bytes(Path(__import__('doorbench.dexterous.isaac_withdrawal_support', fromlist=['x']).__file__).read_bytes())
    runtime = dict(withdrawal_palm_load_profile=PALM_LOAD_PROFILE,
        source_config_path=str(source_path), source_config_sha256=digest(source_path))
    context = dict(withdrawal_palm_load_profile=PALM_LOAD_PROFILE, inherited_support_target_N=6.,
        source_config_path=str(source_path), start_time_s=start, duration_s=duration)
    report = dict(standing_withdrawal=dict(started_s=start, release_started_s=release))
    write(trial/'operation-report.json', report)
    records = history(start, duration, end, release)
    gzwrite(trial/'standing-withdrawal-steps.json.gz', records)
    count = len(records)
    rows = [dict(command_time_s=i*.002, withdrawal=dict(release_started_s=release),
                 support=dict(inherited_started_s=start)) for i in range(count-3, count)]
    hashes = {str(p):digest(p) for p in (source_path, captured, trial/'operation-report.json')}
    return dict(trial=trial, runtime=runtime, context=context, rows=rows, end=end,
                count=count, hashes=hashes, records=records, report=report)


def check(s):
    return audit.audit_palm_profile(**{key:s[key] for key in
        ('trial', 'runtime', 'context', 'rows', 'end', 'count', 'hashes')})


def test_actual_profile_primitive_replayed_independently_at_original_command_epochs(saved):
    result = check(saved)
    assert result['stream_intervals'] == 25274
    assert result['release_started_s'] == 44.218
    assert result['tail_active_targets_N'][-1] == 5.999999720839327
    assert result['source_target_N'] == 6.
    assert not result['physical_qualification']
    assert result['source_and_active_targets_are_distinct']
    assert str(Path(audit.__file__).resolve()) in saved['hashes']


@pytest.mark.parametrize('t,target', [(42.548, 6.), (44.218, 6.), (44.468, 4.25),
    (44.718, 2.5), (49.548, 2.5), (50.048, 4.25), (50.548, 6.)])
def test_independent_schedule_values_without_live_controller(t, target):
    assert audit.expected_palm_request(t, 42.548, 8., 44.218)[0] == target


@pytest.mark.parametrize('kind', ['unknown', 'null', 'context_profile', 'source_target',
    'source_hash', 'source_copy', 'source_duration', 'context_duration', 'report_start',
    'report_hash', 'missing_implementation', 'stream_epoch', 'stream_missing',
    'early_release_row', 'late_release_row', 'changed_release', 'changed_request',
    'tail_release', 'tail_entry', 'restore_epoch', 'source_identity', 'claimed_speed_limit'])
def test_profile_binding_and_recorded_event_tampering_rejected(saved, kind):
    if kind == 'unknown': saved['runtime']['withdrawal_palm_load_profile'] = 'other'
    elif kind == 'null': saved['runtime']['withdrawal_palm_load_profile'] = None
    elif kind == 'context_profile': saved['context']['withdrawal_palm_load_profile'] = 'other'
    elif kind == 'source_target': saved['context']['inherited_support_target_N'] = 4.
    elif kind == 'source_hash': saved['runtime']['source_config_sha256'] = 'a'*64
    elif kind == 'source_copy': (saved['trial']/'standing-withdrawal-source.json').write_text('{}')
    elif kind == 'source_duration':
        p = Path(saved['runtime']['source_config_path']); data=json.loads(p.read_text());data['duration_s']=7.
        write(p,data);(saved['trial']/'standing-withdrawal-source.json').write_bytes(p.read_bytes())
        saved['runtime']['source_config_sha256']=digest(p);saved['hashes'][str(p)]=digest(p)
    elif kind == 'context_duration': saved['context']['duration_s'] = 7.
    elif kind in ('report_start', 'report_hash'):
        saved['report']['standing_withdrawal']['started_s'] += .002
        p=saved['trial']/'operation-report.json';write(p,saved['report'])
        if kind == 'report_start':saved['hashes'][str(p)]=digest(p)
    elif kind == 'missing_implementation': saved['hashes'].pop(str(saved['trial']/'source-isaac_withdrawal_support.py'))
    elif kind == 'tail_release': saved['rows'][-1]['withdrawal']['release_started_s'] += .002
    elif kind == 'tail_entry': saved['rows'][-1]['support']['inherited_started_s'] += .002
    else:
        rows=saved['records'];event=round(44.218/.002)
        if kind == 'stream_epoch': rows[-1]['sim_time_s'] -= .002
        elif kind == 'stream_missing': rows.pop()
        elif kind == 'early_release_row': rows[event-1]['teacher']['release_started_s']=44.218
        elif kind == 'late_release_row': rows[event]['teacher']['release_started_s']=None
        elif kind == 'changed_release': rows[-1]['teacher']['release_started_s']=44.220
        elif kind == 'changed_request': rows[-1]['teacher']['inherited_support']['active_target_N']=6.
        elif kind == 'restore_epoch': rows[-1]['teacher']['inherited_support']['restore_started_s'] += .002
        elif kind == 'source_identity': rows[-1]['teacher']['inherited_support']['target_N']=2.5
        elif kind == 'claimed_speed_limit': rows[-1]['teacher']['inherited_support']['leaf_speed_limit_claimed']=True
        gzwrite(saved['trial']/'standing-withdrawal-steps.json.gz', rows)
    with pytest.raises(ValueError):check(saved)


def test_wrong_but_consistently_relabelled_report_release_is_rejected_by_first_stream_event(saved):
    saved['report']['standing_withdrawal']['release_started_s'] += .002
    p=saved['trial']/'operation-report.json';write(p,saved['report']);saved['hashes'][str(p)]=digest(p)
    for row in saved['rows']:row['withdrawal']['release_started_s'] += .002
    with pytest.raises(ValueError,match='actual release'):check(saved)


def test_default_reference_accounting_keeps_exact_target_equality(source):
    source['tail']['tail'][-1]['left']['hybrid_normal_target_N']=5.999999720839327
    save_tail(source)
    with pytest.raises(ValueError,match='source-bound support'):admit(source)


def test_algebraic_tail_check_accepts_exact_proved_active_value_but_keeps_source_six(source):
    # Unit test of the narrow algebraic seam. The independent full event/history
    # test above supplies this value; the six-row fixture is not a release run.
    tail=source['tail'];row=copy.deepcopy(tail['tail'][-1]);target=5.999999720839327
    row['left']['hybrid_normal_target_N']=target
    args=(row,tail['static_controller_data'],tail['attained_tracking_contract'],
          source['motors'],source['config']['robot_joint_names'])
    reference._consistency(*args,active_support_target=target)
    assert row['left']['support_target_N']==row['support']['maximum_target_N']==6.
    with pytest.raises(ValueError,match='source-bound support'):
        reference._consistency(*args,active_support_target=6.)
    row['left']['support_target_N']=row['support']['maximum_target_N']=4.
    with pytest.raises(ValueError,match='source-bound support'):
        reference._consistency(*args,active_support_target=target)


def test_context_profile_cannot_silently_run_through_default_accounting(source):
    source['context']['withdrawal_palm_load_profile']=PALM_LOAD_PROFILE
    write(source['trial']/'standing-withdrawal-admission.json',{'source_context':source['context']})
    with pytest.raises(ValueError,match='absent from its bound runtime'):admit(source)


def test_whole_reference_admission_uses_bound_profile_without_task_promotion(source):
    # The six-interval synthetic source is before release, not a fabricated
    # completed physical withdrawal. Run the real observation/tail admissions.
    trial=source['trial'];context=source['context'];runtime_path=Path(context['runtime_path'])
    runtime=json.loads(runtime_path.read_text());duration_source=runtime_path.with_name('duration.json')
    write(duration_source,dict(start_time_s=.006,duration_s=8.))
    (trial/'standing-withdrawal-source.json').write_bytes(duration_source.read_bytes())
    runtime.update(withdrawal_palm_load_profile=PALM_LOAD_PROFILE,
        source_config_path=str(duration_source),source_config_sha256=digest(duration_source))
    write(runtime_path,runtime);(trial/'standing-withdrawal-runtime.json').write_bytes(runtime_path.read_bytes())
    context.update(runtime_sha256=digest(runtime_path),withdrawal_palm_load_profile=PALM_LOAD_PROFILE,
        source_config_path=str(duration_source),inherited_support_target_N=6.,duration_s=8.)
    binding={k:context[k] for k in ('runtime_path','runtime_sha256','motor_contract_sha256','source_state_sha256')}
    source['tail']['source_binding']=copy.deepcopy(binding)
    for row in source['tail']['tail']:
        row['source_binding']=copy.deepcopy(binding);row['withdrawal']['release_started_s']=None
    write(trial/'standing-withdrawal-admission.json',{'source_context':context})
    implementation=runtime_path.with_name('isaac_withdrawal_support.py')
    implementation.write_bytes(Path(__import__('doorbench.dexterous.isaac_withdrawal_support',fromlist=['x']).__file__).read_bytes())
    (trial/'source-isaac_withdrawal_support.py').write_bytes(implementation.read_bytes())
    source['provenance']['files'].update({str(p):digest(p) for p in (runtime_path,duration_source,implementation)})
    source['report']['standing_withdrawal']=dict(started_s=.006,release_started_s=None)
    write(trial/'provenance.json',source['provenance']);write(trial/'operation-report.json',source['report'])
    gzwrite(trial/'standing-withdrawal-steps.json.gz',history(.006,8.,.012,None))
    save_tail(source)
    source['observation_audit']=observations.audit.audit_standing_continuation(trial)
    write(source['audit_path'],source['observation_audit'])
    result=admit(source)
    assert result['passed'] and result['support_profile_accounting']['tail_active_targets_N']==[6.,6.,6.]
    assert result['authorized_stages']==0 and not result['physical_task_qualification']
    assert result['tail']==source['tail']['tail']
    # A target of6 in the stream cannot authorize altered tail requests.
    source['tail']['tail'][-1]['left']['hybrid_normal_target_N']=5.999999720839327;save_tail(source)
    with pytest.raises(ValueError,match='source-bound support'):admit(source)
