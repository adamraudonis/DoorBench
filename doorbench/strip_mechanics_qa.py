"""Native load-cycle evidence for discrete flexible strip curtains.

This applies finite face loads to a fresh MjData, retaining every neighboring
strip contact. It does not prescribe flexure poses or certify human traversal,
PVC fatigue, temperature-dependent material response, or a deformable shell.
"""
from __future__ import annotations

import math

import numpy as np


def _energy(m, d, mujoco):
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    return float(sum(d.energy))


def run_strip_mechanics_qa(model, metadata, *, repetitions=2):
    """Check authored geometry, then push/release from both sides natively.

    The model is never mutated. Twenty-newton point loads act at declared real
    sheet-face sites, with fresh data and no pose assignment after reset. These
    are assembly stress loads, not a claim about a controller or grasp.
    """
    import mujoco

    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError('repetitions must be an integer from 1 through 3')
    m = model; d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    failures = []; measurements = {}
    def fail(check, **details):
        failures.append({'check': check, **details})
    record = metadata.get('strip_curtain')
    if not record:
        return {'ok': False, 'failures': [{'check': 'missing_flexible_strip_metadata'}],
                'n_failures': 1, 'measurements': measurements}
    try:
        controls = record['controls']; layout = record['layout']
        thickness = float(layout['thickness'])
        if not math.isfinite(thickness) or thickness <= 0:
            raise ValueError('strip thickness must be positive and finite')
        tolerance = min(.001, thickness/2)
        if m.opt.timestep > metadata.get('native_timestep_s', 0)+1e-12:
            fail('native_timestep_exceeds_authored_bound')
        if len(controls) != layout['count'] or len(controls) < 2:
            fail('strip_count_mismatch')
        bonded_signatures=set()
        for control in controls:
            tab=m.body(control['fixed_tab_body']).id
            first=m.body(control['segment_bodies'][0]).id
            a,b=sorted((tab,first));bonded_signatures.add((a<<16)+b)
        if set(map(int,m.exclude_signature))!=bonded_signatures:
            fail('neighbor_contact_exclusion_or_missing_material_bond')
        geom_ids = []
        for control in controls:
            names = control['segment_bodies']
            if len(names) != layout['segments']:
                fail('strip_segment_count', strip=control['strip'])
            previous = m.body(control['fixed_tab_body']).id
            tab=m.geom(control['fixed_tab_geom']).id
            if m.body_weldid[previous]!=0 or m.body_jntnum[previous]!=0 or m.geom_bodyid[tab]!=previous:
                fail('clamping_tab_not_world_fixed',strip=control['strip'])
            first=m.geom(names[0]+'_pvc').id
            tab_bottom=d.geom_xpos[tab,2]-m.geom_size[tab,2]
            moving_top=d.geom_xpos[first,2]+m.geom_size[first,2]
            if abs(tab_bottom-moving_top)>1e-6 or not np.allclose(d.geom_xpos[tab,:2],d.geom_xpos[first,:2],atol=1e-6):
                fail('unbonded_clamping_tab',strip=control['strip'])
            if not np.allclose(m.geom_size[tab,:2],m.geom_size[first,:2],atol=1e-6):
                fail('clamping_tab_cross_section',strip=control['strip'])
            for name in control['clamp_geoms']:
                clamp=m.geom(name).id
                gap=mujoco.mj_geomDistance(m,d,tab,clamp,.1,None)
                if m.body_weldid[m.geom_bodyid[clamp]]!=0 or abs(gap)>1e-6:
                    fail('clamp_not_bearing_on_fixed_tab',strip=control['strip'],clamp=name,gap_m=float(gap))
                rail=m.geom('hanger_rail').id
                if mujoco.mj_geomDistance(m,d,clamp,rail,.1,None)>1e-6:
                    fail('clamp_disconnected_from_rail',clamp=name)
            for name in names:
                body = m.body(name).id; geom = m.geom(name+'_pvc').id
                geom_ids.append(geom)
                if m.body_parentid[body] != previous:
                    fail('disconnected_material_chain', body=name)
                previous = body
                if not m.geom_contype[geom] or not m.geom_conaffinity[geom]:
                    fail('disabled_strip_contact', geom=m.geom(geom).name)
                if m.geom_bodyid[geom] != body or m.body_jntnum[body] != 1:
                    fail('missing_material_flexure', body=name)
                joint = int(m.body_jntadr[body]); dof = int(m.jnt_dofadr[joint])
                if m.dof_armature[dof] != 0 or m.body_mass[body] <= 0:
                    fail('nonphysical_material_inertia', body=name)
                if m.geom_margin[geom] != 0:
                    fail('preloaded_strip_contact_margin', geom=m.geom(geom).name)
                if m.jnt_stiffness[joint] <= 0 or m.dof_damping[dof] <= 0:
                    fail('missing_elastic_material_response', body=name)
                expected_stiffness = record['bending_stiffness_Nm_per_rad']*(2 if name == names[0] else 1)
                if abs(m.jnt_stiffness[joint]-expected_stiffness) > 1e-6:
                    fail('wrong_material_bending_stiffness', body=name)
        center = len(controls)//2
        chosen = []
        for layer, direction, index in ((0, 1, 0), (1, -1, 1)):
            control = min((c for c in controls if c['layer'] == layer),
                          key=lambda c: abs(c['strip']-center))
            sid = m.site(control['push_sites'][index]).id
            selected_body = int(m.site_bodyid[sid])
            if selected_body not in [m.body(name).id for name in control['segment_bodies']]:
                fail('detached_push_site', site=m.site(sid).name)
            ray = np.array([0., float(direction), 0.])
            hit = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(m, d, d.site_xpos[sid]-.08*ray, ray, None, True, -1, hit)
            if abs(distance-.08) > 1e-5 or hit[0] < 0 or m.geom_bodyid[hit[0]] != selected_body:
                fail('occluded_or_off_surface_push_site', site=m.site(sid).name,
                     hit=m.geom(int(hit[0])).name if hit[0] >= 0 else None,
                     ray_distance_m=float(distance))
            chosen.append((sid, direction))
        measurements.update(material_segments=len(geom_ids), native_timestep_s=float(m.opt.timestep),
                            penetration_limit_m=tolerance,
                            contact_sites=[m.site(sid).name for sid, _ in chosen])
    except (KeyError, ValueError, IndexError, TypeError) as error:
        fail('invalid_strip_geometry_contract', reason=str(error))
    if failures:
        return {'ok': False, 'failures': failures, 'n_failures': len(failures), 'measurements': measurements}

    dt = float(m.opt.timestep); energy0 = _energy(m, d, mujoco)
    work = 0.; maximum_unaccounted = 0.; max_penetration = 0.; max_speed = 0.
    peak_pair = None; peak_phase = None; phases = []
    load_phases = [('settle', .25, None, 0)]
    for repeat in range(repetitions):
        for (site, direction), label in zip(chosen, ('forward', 'reverse')):
            load_phases.extend([(f'{label}_{repeat+1}', 4., site, direction*20.),
                                (f'release_{label}_{repeat+1}', 6., None, 0.)])
    for phase, duration, sid, force in load_phases:
        phase_start_energy = _energy(m, d, mujoco)
        peak_energy = phase_start_energy; peak_displacement = 0.
        start_y = float(d.site_xpos[sid, 1]) if sid is not None else 0.
        for _ in range(round(duration/dt)):
            d.qfrc_applied[:] = 0
            if sid is not None:
                mujoco.mj_applyFT(m, d, np.array([0., force, 0.]), np.zeros(3),
                                 d.site_xpos[sid], int(m.site_bodyid[sid]), d.qfrc_applied)
                work += float(d.qfrc_applied @ d.qvel)*dt
            mujoco.mj_step(m, d)
            if sid is not None:
                peak_displacement = max(peak_displacement,
                    float((d.site_xpos[sid, 1]-start_y)*np.sign(force)))
            energy = _energy(m, d, mujoco)
            peak_energy = max(peak_energy, energy)
            maximum_unaccounted = max(maximum_unaccounted, energy-energy0-work)
            max_speed = max(max_speed, float(np.max(np.abs(d.qvel))))
            if d.ncon:
                contact = d.contact[int(np.argmin(d.contact.dist))]
                if -contact.dist > max_penetration:
                    max_penetration = -float(contact.dist)
                    peak_pair = [m.geom(contact.geom1).name, m.geom(contact.geom2).name]
                    peak_phase = phase
            if any(w.number for w in d.warning):
                fail('native_warning', phase=phase, warning_counts=[int(w.number) for w in d.warning])
                break
            if max_penetration > tolerance:
                fail('thin_sheet_penetration', phase=peak_phase, pair=peak_pair,
                     penetration_m=max_penetration, limit_m=tolerance)
                break
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                fail('nonfinite_native_state', phase=phase)
                break
            if maximum_unaccounted > 1.:
                fail('unaccounted_energy_growth', phase=phase, joules=maximum_unaccounted)
                break
        final_energy = _energy(m, d, mujoco)
        phases.append({'phase': phase, 'peak_displacement_m': peak_displacement,
                       'energy_change_J': final_energy-phase_start_energy,
                       'passive_peak_energy_gain_J': peak_energy-phase_start_energy if sid is None else None,
                       'kinetic_end_J': float(d.energy[1])})
        if failures: break
        if sid is not None and peak_displacement < .10:
            fail('surface_load_did_not_open_material', phase=phase, displacement_m=peak_displacement)
        if sid is None and phase != 'settle' and final_energy > phase_start_energy+.25:
            fail('passive_energy_growth', phase=phase, joules=final_energy-phase_start_energy)
        if failures: break
    measurements.update(max_penetration_m=max_penetration, peak_pair=peak_pair, peak_phase=peak_phase,
                        max_joint_speed_rad_s=max_speed, max_unaccounted_energy_J=maximum_unaccounted,
                        net_applied_work_J=work, max_arena_bytes=int(d.maxuse_arena), phases=phases)
    return {'ok': not failures, 'failures': failures, 'n_failures': len(failures),
            'measurements': measurements,
            'scope': 'Finite 20 N sheet-face loads and passive native release, with all strip contacts retained. '
                     'Planar material approximation; not a traversal, human-strength, material-life or temperature certification.'}
