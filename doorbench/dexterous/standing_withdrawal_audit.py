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
    checks.update(
        resting_grip_before_withdrawal=held(started,lambda r:r['pad_grasp']['valid_pad_grasp'] and abs(r['handle_angle_rad'])<=.05 and abs(r.get('bolt_slide_m',1.))<=.001),
        opposed_grip_before_intentional_release=held(release_started,lambda r:r['pad_grasp']['valid_pad_grasp']),
        left_support_before_intentional_release=held(release_started,lambda r:r.get('left_surface',{}).get('palm_normal_load_N',0)>=2.),
        no_invalid_loaded_right_surfaces=all(c['pad_qualified'] for r in rows for c in r['pad_grasp']['contacts']),
        withdrawal_route_completed=bool(completed and release_started is not None),
        final_hand_clear_of_environment=held(duration,lambda r:r.get('right_environment_clearance_m',-1.)>=.04))
    return checks
