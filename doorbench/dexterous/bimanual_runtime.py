"""Strict runtime-specific rescreen receipt validation for frozen teacher targets."""
import json
from pathlib import Path

SCHEMA='doorbench.bimanual-runtime-contact-rescreen.v1'
CHECKS=('expected_source_design','finite_complete_state_matrix',
        'original_robot_joint_limits','documented_loopback_limits',
        'all_nonfoot_collision_depth')


def validate_runtime_rescreen(path, *, config, config_sha256, compiled_identity,
                             design_identity, door_xml_sha256):
    report=json.loads(Path(path).read_text())
    if report.get('schema')!=SCHEMA or report.get('passed') is not True:
        raise ValueError('Destination contact geometry rescreen did not pass')
    if any(report.get('checks',{}).get(k) is not True for k in CHECKS):
        raise ValueError('Missing destination geometry rescreen gates')
    expected=dict(target_config_sha256=config_sha256,
        trajectory_sha256=config.get('runtime_rescreen_trajectory_sha256'),
        source_design_sha256=design_identity['sha256'],
        source_compiled_robot_sha256=config['compiled_robot_identity']['sha256'],
        destination_compiled_robot_sha256=compiled_identity['sha256'],
        destination_mujoco_version=compiled_identity['mujoco_version'],
        door_xml_sha256=door_xml_sha256)
    if any(not value or report.get(key)!=value for key,value in expected.items()):
        raise ValueError('Runtime rescreen belongs to different targets, trace, robot, or door')
    if report.get('recorded_samples',0)<2 or report.get('physics_steps')!=0:
        raise ValueError('Malformed static runtime geometry receipt')
    return report
