from copy import deepcopy
from pathlib import Path
import pytest
from doorbench.dexterous.press_plan_binding import file_sha,require_press_binding,verify_acquisition_audit


def test_changed_schedule_or_controller_cannot_reuse_attained_press_plan(tmp_path):
    for n in ('calibration','schedule','motors'):(tmp_path/n).write_text(n)
    root=Path(__file__).resolve().parents[1]
    names=['doorbench/dexterous/'+n+'.py' for n in ('sensor_reach_balance','sensor_balance','sensor_acquisition_schedule','locomotion_manipulation','stance')]
    args={n:tmp_path/n for n in ('calibration','schedule','motors')}
    args.update(gravity_correction=.2,initial_velocity=0.)
    binding=dict(calibration_sha256=file_sha(args['calibration']),arm_schedule_sha256=file_sha(args['schedule']),motor_contract_sha256=file_sha(args['motors']),gravity_correction=.2,initial_velocity=0.,controller_source_sha256={n:file_sha(root/n) for n in names})
    plan=dict(schema='doorbench.offline-sensor-press-plan.v2',acquisition_binding=binding)
    require_press_binding(plan,**args)
    changed=deepcopy(plan);changed['acquisition_binding']['controller_source_sha256'][names[0]]='0'*64
    with pytest.raises(ValueError,match='exact attained'):require_press_binding(changed,**args)
    (tmp_path/'schedule').write_text('new pressure profile')
    with pytest.raises(ValueError,match='exact attained'):require_press_binding(plan,**args)


def test_passed_audit_does_not_admit_changed_physics_archive(tmp_path):
    names=['report.json','provenance.json','trajectory.npz','controller.jsonl.gz']
    for n in names:(tmp_path/n).write_bytes(n.encode())
    audit=dict(independent_evaluation=dict(passed=True,checks={'physical':True}),physics_steps_in_evaluator=0,source_sha256={n:file_sha(tmp_path/n) for n in names})
    verify_acquisition_audit(tmp_path,audit)
    (tmp_path/'trajectory.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed acquisition'):verify_acquisition_audit(tmp_path,audit)
