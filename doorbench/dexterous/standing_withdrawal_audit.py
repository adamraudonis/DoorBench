"""Phase-specific acceptance for intentional hand release, preserving grasp gates."""
import numpy as np
import mujoco


def clearance_pairs(model):
    active=[g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    hand=[g for g in active if model.body(model.geom_bodyid[g]).name.startswith('robot/rh_')]
    scene=[g for g in active if not model.body(model.geom_bodyid[g]).name.startswith('robot/')]
    if not hand or not scene:raise ValueError('Explicit colliding hand and scene geometry required')
    return [(g,h) for g in hand for h in scene]


def environment_clearance(model,data,pairs):
    return min(float(mujoco.mj_geomDistance(model,data,g,h,.5,None)) for g,h in pairs)


def withdrawal_checks(original,rows,*,dt,duration,started,release_started,completed):
    checks=dict(original);checks.pop('sustained_pad_grasp',None)
    def held(at,predicate):
        if at is None or not np.isfinite(at) or not .5<=at<=duration:return False
        window=[r for r in rows if at-.5-1e-8<=r['sim_time_s']<=at+1e-8]
        return bool(len(window)==round(.5/dt)+1 and all(predicate(r) for r in window))
    # The partial-opening envelope applies before the intentional release.
    # Once the right hand lets go, left-palm pushing can physically open the
    # leaf farther. Retain the original joint/contact/motor safety checks.
    checks.pop('partial_leaf_opening_held',None);checks.pop('opening_bounded_for_transfer',None)
    before=lambda:(r for r in rows if release_started is None or r['sim_time_s']<=release_started+1e-8)
    checks.update(
        partial_opening_held_before_intentional_release=held(release_started,lambda r:.075<=r['door_q']<=.10),
        opening_bounded_before_intentional_release=any(True for _ in before()) and all(r['door_q']<=.12 for r in before()),
        leaf_remains_open_after_withdrawal=held(duration,lambda r:r['door_q']>=.075),
        resting_grip_before_withdrawal=held(started,lambda r:r['pad_grasp']['valid_pad_grasp'] and abs(r['handle_angle_rad'])<=.05 and abs(r.get('bolt_slide_m',1.))<=.001),
        opposed_grip_before_intentional_release=held(release_started,lambda r:r['pad_grasp']['valid_pad_grasp']),
        left_support_before_intentional_release=held(release_started,lambda r:r.get('left_surface',{}).get('palm_normal_load_N',0)>=2.),
        no_invalid_loaded_right_surfaces=all(c['pad_qualified'] for r in rows for c in r['pad_grasp']['contacts']),
        withdrawal_route_completed=bool(completed and release_started is not None),
        final_hand_clear_of_environment=held(duration,lambda r:r.get('right_environment_clearance_m',-1.)>=.04))
    return checks
