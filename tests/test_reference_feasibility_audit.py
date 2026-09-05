"""Independent failing fixtures for the read-only feasibility audit."""
import numpy as np
import pytest
from scripts.audit_reference_feasibility import collision_metrics, support_metrics, capability


def pose():
    p = np.array([[0, 0, .94], [0, 0, 1.29], [0, 0, 1.45], [0, 0, 1.58],
                  [-.18, 0, 1.35], [-.18, 0, 1.05], [-.18, 0, .77],
                  [.18, 0, 1.35], [.18, 0, 1.05], [.18, 0, .77],
                  [-.105, 0, .88], [-.105, 0, .45], [-.105, 0, .02],
                  [.105, 0, .88], [.105, 0, .45], [.105, 0, .02]])
    return p


def test_stance_anchor_accumulates_slide_and_resets_only_on_release():
    p = np.repeat(pose()[None], 5, axis=0)
    p[:, 12, 0] += [0, .004, .008, .3, .3]
    contact = np.zeros((5, 2), bool); contact[:, 0] = [1, 1, 1, 0, 1]
    report = support_metrics(p, contact)
    assert report['max_stance_anchor_drift_m'] == pytest.approx(.008)
    assert report['stance_foot_frames_drift_gt_5mm'] == 1
    assert report['declared_stance_segments'] == 2


def test_moving_avatar_distance_queries_follow_actual_world_pose(tmp_path):
    xml = tmp_path/'scene.xml'
    xml.write_text('<mujoco><worldbody><geom name="wall" type="box" '
                   'pos="0 0 1.58" size=".1 .1 .1"/></worldbody></mujoco>')
    p = np.repeat(pose()[None], 3, axis=0)
    p[0, :, 1] -= 3
    p[2, :, 1] += 3
    report = collision_metrics(xml, {'bodies': [{'geoms': [{'name': 'wall', 'semantic': 'wall'}]}]},
                               p, np.empty((3, 0)), np.array([0., .05, .10]),
                               np.zeros(3, bool), np.zeros((3, 3)), 0)
    assert report['evaluated_frames'] == 3
    assert report['frames_with_head_penetration_gt_3mm'] == 1
    assert all(e['frame'] == 1 for e in report['examples'])
    # Sphere center coincides with box center: radius .108 + nearest face .1.
    head = next(e for e in report['examples'] if e['avatar_part'] == 'audit_joint_3')
    assert head['signed_distance_m'] == pytest.approx(-.208, abs=1e-6)


def test_pet_aperture_does_not_claim_every_crawling_posture_impossible():
    tiny = capability({'family': 'pet_door', 'opening': {'width': .16, 'height': .18}})
    large = capability({'family': 'pet_door', 'opening': {'width': .39, 'height': .65}})
    assert tiny['intrinsic_rigid_head_obstruction']
    assert not large['intrinsic_rigid_head_obstruction']
    assert large['requires_nonwalking_traversal']
    assert any('crawl' in note for note in large['notes'])


def test_cylinder_orientation_and_length_match_current_bone(tmp_path):
    xml = tmp_path/'scene.xml'
    xml.write_text('<mujoco><worldbody><geom name="wall" type="box" '
                   'pos="0 0 1.35" size=".02 .02 .02"/></worldbody></mujoco>')
    p = pose()[None]; p[0, 5] = [.5, 0, 1.35]
    report = collision_metrics(xml, {'bodies': [{'geoms': [{'name': 'wall', 'semantic': 'wall'}]}]},
                               p, np.empty((1, 0)), np.array([0.]), np.zeros(1, bool), np.zeros((1, 3)), 0)
    hit = next(e for e in report['examples'] if e['avatar_part'] == 'audit_bone_4_5')
    assert hit['signed_distance_m'] == pytest.approx(-.057, abs=1e-5)
