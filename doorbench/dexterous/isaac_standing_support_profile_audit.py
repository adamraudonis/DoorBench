"""Independent saved-data accounting of the named withdrawal load schedule.

No feedback/controller is evaluated. The complete recorded event clock binds
the last three active requests; source identity and physical gates stay intact.
"""
import gzip
from pathlib import Path

import numpy as np

from .continuation_record_stream import iter_continuation_records
from .isaac_standing_continuation_audit import _json
from .qualified_isaac_grasp import digest

PROFILE = 'release-unload-restore-v1'
DT = .002


def _number(value, label):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError('Finite recorded palm-profile '+label+' required')
    return float(value)


def expected_palm_request(t, start, duration, release):
    """Independent scalar algebra; does not invoke the live profile helper."""
    t, start, duration = [_number(v, k) for v, k in
                         ((t, 'command epoch'), (start, 'entry'), (duration, 'duration'))]
    complete = start+duration
    restore = complete-1.
    if start < 0 or t < start or duration <= 1.5 or not np.isfinite([complete, restore]).all():
        raise ValueError('Original finite palm-profile duration/entry required')
    if release is not None:
        release = _number(release, 'actual release epoch')
        if release < start or release+.5 > restore:
            raise ValueError('Original nonoverlapping palm-profile release required')
    def smooth(u):
        u = min(1., max(0., u))
        return u*u*u*(10.+u*(-15.+6.*u))
    if release is None or t < release:
        return 6., 'awaiting_qualified_release'
    if t < release+.5:
        return 6.-3.5*smooth((t-release)/.5), 'unloading_after_release'
    if t < restore:
        return 2.5, 'withdrawal_low_load'
    return 2.5+3.5*smooth(t-restore), ('restoring_terminal_support'
        if t < complete else 'terminal_support')


def audit_palm_profile(trial, runtime, context, rows, *, end, count, hashes):
    """Bind runtime/source duration and independently replay recorded release.

    Called only when the optional profile is explicit. ``hashes`` is the fresh
    observation admission's input inventory and is extended with this proof.
    Passing this accounting grants no physical or controller authority.
    """
    if (runtime.get('withdrawal_palm_load_profile') != PROFILE
            or type(runtime.get('withdrawal_palm_load_profile')) is not str
            or context.get('withdrawal_palm_load_profile') != PROFILE
            or _number(context.get('inherited_support_target_N'), 'source target') != 6.):
        raise ValueError('Exact declared six-newton source palm profile required')
    trial = Path(trial)
    source_path = Path(runtime['source_config_path']).resolve()
    if (not Path(runtime['source_config_path']).is_absolute()
            or Path(context['source_config_path']).resolve() != source_path
            or hashes.get(str(source_path)) != runtime.get('source_config_sha256')
            or digest(source_path) != runtime['source_config_sha256']):
        raise ValueError('Captured runtime must bind the original palm-profile duration source')
    saved_source = trial/'standing-withdrawal-source.json'
    if digest(saved_source) != runtime['source_config_sha256']:
        raise ValueError('Saved palm-profile duration source differs')
    source = _json(saved_source)
    start = _number(context['start_time_s'], 'entry')
    duration = _number(context['duration_s'], 'duration')
    if source.get('start_time_s') != start or source.get('duration_s') != duration:
        raise ValueError('Original source-bound palm-profile epochs differ')
    # The source record can be incomplete/failed: observation accounting does
    # not promote it. A release must nevertheless be an actual recorded event.
    report_path = trial/'operation-report.json'
    if digest(report_path) != hashes.get(str(report_path)):
        raise ValueError('Original measured release report changed')
    reported = _json(report_path)['standing_withdrawal']
    if reported.get('started_s') != start:
        raise ValueError('Actual withdrawal entry differs from palm-profile source')
    release = reported.get('release_started_s')
    if release is not None:
        release = _number(release, 'actual release epoch')
        if release >= end or abs(release/DT-round(release/DT)) > 1e-7:
            raise ValueError('Actual release must be an earlier recorded 500 Hz command')
    expected_palm_request(start, start, duration, release)
    captured = trial/'source-isaac_withdrawal_support.py'
    if hashes.get(str(captured)) != digest(captured):
        raise ValueError('Original palm-profile feedback implementation must be captured')
    stream_path = trial/'standing-withdrawal-steps.json.gz'
    inputs = (saved_source, stream_path, Path(__file__).resolve(),
              Path(__file__).with_name('continuation_record_stream.py').resolve())
    for path in inputs:
        current = digest(path)
        if str(path) in hashes and hashes[str(path)] != current:
            raise ValueError('Palm-profile evidence changed: '+str(path))
        hashes[str(path)] = current
    target_by_time = {}
    first_release = None
    seen = 0
    requested = {row['command_time_s'] for row in rows}
    with gzip.open(stream_path, 'rt') as stream:
        for i, record in enumerate(iter_continuation_records(stream)):
            post = (i+1)*DT
            command = i*DT
            seen += 1
            if _number(record['sim_time_s'], 'post-step epoch') != post or seen > count:
                raise ValueError('Complete contiguous actual palm-profile stream required')
            teacher = record['teacher']
            actual_release = teacher['release_started_s']
            expected_release = release if release is not None and command >= release else None
            if actual_release != expected_release:
                raise ValueError('Recorded actual release event differs from its report/command clock')
            if actual_release is not None and first_release is None:
                first_release = command
                if first_release != release:
                    raise ValueError('First actual release must occur at its recorded command')
            if command < start:
                continue
            if teacher['withdrawal_started_s'] != start:
                raise ValueError('Recorded withdrawal start changed during palm profile')
            active, phase = expected_palm_request(command, start, duration, release)
            info = teacher['inherited_support']
            expected = dict(palm_load_profile=PROFILE, active_target_N=active,
                inherited_source_target_N=6., inherited_maximum_target_N=6., target_N=6.,
                palm_load_phase=phase, qualified_release_started_s=expected_release,
                unload_duration_s=.5, restore_started_s=start+duration-1., restore_duration_s=1.,
                withdrawal_complete_s=start+duration, leaf_speed_limit_claimed=False)
            if any(type(info.get(k)) is not type(v) and not
                   (type(info.get(k)) in (int, float) and type(v) in (int, float))
                   or info.get(k) != v for k, v in expected.items()):
                raise ValueError('Recorded active palm request differs from the independent named schedule')
            if command in requested:
                target_by_time[command] = active
    if seen != count or first_release != release or len(target_by_time) != len(rows):
        raise ValueError('Complete palm-profile command/event history and every tail epoch required')
    for row in rows:
        t = row['command_time_s']
        actual_release = release if release is not None and t >= release else None
        if (row['withdrawal']['release_started_s'] != actual_release
                or row['support']['inherited_started_s'] != start):
            raise ValueError('Captured reference tail differs from actual palm-profile entry/release')
    return dict(profile=PROFILE, source_target_N=6., duration_s=duration,
        start_time_s=start, release_started_s=release, stream_intervals=seen,
        tail_active_targets_N=[target_by_time[row['command_time_s']] for row in rows],
        source_and_active_targets_are_distinct=True, physical_qualification=False,
        scope='Exact recorded numeric schedule/event accounting only; no minimum contact load or leaf-speed guarantee')
