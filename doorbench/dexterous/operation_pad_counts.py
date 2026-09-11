"""One-pass counters with the original operation-report semantics."""


def operation_pad_counts(records, start_s):
    counts=dict(operation_invalid_grasp_samples=0, operation_digit_unload_samples=0,
                operation_opposition_failure_samples=0, operation_invalid_pad_patch_samples=0)
    for row in records:
        if start_s is None or not row['sim_time_s'] >= start_s:
            continue
        counts['operation_invalid_grasp_samples'] += int(not row['valid_pad_grasp'])
        counts['operation_digit_unload_samples'] += int(any(v < .2 for v in row['digit_forces_N'].values()))
        counts['operation_opposition_failure_samples'] += int(
            row.get('minimum_pairwise_finger_alignment',1.) <= .5 or
            row.get('maximum_thumb_finger_dot',-1.) >= -.5)
        counts['operation_invalid_pad_patch_samples'] += int(any(not c['pad_qualified'] for c in row['contacts']))
    return counts
