"""Deterministic horizontal rail coverage and modeled wheel-contact checks.

The nominal unlocked range is swept even when an engaged lock narrows the MJCF
joint range. A rail-only result is explicitly distinguished from wheel contact;
it is not evidence that a missing suspension mechanism has been modeled.

Scope: leaves that run along a HORIZONTAL rail. A vertically guided assembly -
a coiling curtain in its side guides, a sectional door on its tracks - is the
same requirement about a different axis and a different kind of hardware, and
is checked by ``doorbench/enclosure_qa.run_guided_travel_qa``
(``checks["guided_travel"]``), which sweeps the nominal travel the same way.
"""
from __future__ import annotations

import numpy as np

SLIDING_FAMILIES = {"sliding_single", "sliding_bypass", "automatic_sliding", "elevator", "gate_sliding"}


def run_sliding_track_qa(model, metadata, samples=25, tolerance=0.003):
    import mujoco

    if samples < 2:
        raise ValueError("At least two samples are required")
    supports = metadata.get("sliding_track_supports", [])
    expected = metadata.get("family") in SLIDING_FAMILIES
    failures, rail_only = [], []
    wheels_checked, worst_contact_gap, worst_overhang = 0, 0.0, 0.0
    guide_stations_checked, worst_guide_gap = 0, 0.0
    if expected and not supports:
        failures.append({"check": "metadata", "reason": "Horizontal slider has no track support definitions"})
    data = mujoco.MjData(model)
    for support in supports:
        name = support["body"]
        try:
            joint = model.joint(support["joint"]).id
            rail = model.geom(support["rail"]).id
            body = model.body(name).id
            wheels = [model.geom(n).id for n in support.get("rollers", [])]
            stops = [model.geom(n).id for n in support.get("end_stops", [])]
            guides = [([model.geom(n).id for n in station["jaws"]], [model.geom(n).id for n in station["feet"]])
                      for station in support.get("floor_guides", [])]
            guide_leaf_geoms = [model.geom(n).id for n in support.get("guide_leaf_geoms", [])]
        except KeyError as error:
            failures.append({"check": "missing_geometry", "body": name, "reason": str(error)})
            continue
        if int(model.geom_type[rail]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            failures.append({"check": "rail_type", "body": name})
            continue
        if not wheels:
            rail_only.append(name)
        if support.get("floor_guides_required") and (not guides or not guide_leaf_geoms):
            failures.append({"check": "floor_guide_missing", "body": name})
        guide_stations_checked += len(guides)
        wheels_checked += len(wheels)
        endpoint_stop_gaps = []
        for index, q in enumerate(np.linspace(*support["nominal_range"], samples)):
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[joint]] = q
            mujoco.mj_forward(model, data)
            if guides:
                if index == 0:
                    # Each jaw must meet a static floor-mounted foot; a detached
                    # decorative guide is not evidence of lateral restraint.
                    for jaws, feet in guides:
                        mounted = len(jaws) == len(feet) == 2
                        for jaw, foot in zip(jaws, feet):
                            mounted &= int(model.geom_bodyid[jaw]) == int(model.geom_bodyid[foot]) == 0
                            mounted &= abs(float(data.geom_xpos[foot, 2] - model.geom_size[foot, 2])) <= tolerance
                            mounted &= mujoco.mj_geomDistance(model, data, jaw, foot, 10.0, None) <= tolerance
                        if not mounted:
                            failures.append({"check": "floor_guide_mount", "body": name})
                station_gaps = []
                for jaws, _ in guides:
                    if len(jaws) != 2 or not guide_leaf_geoms:
                        continue
                    ys = [data.geom_xpos[jaw, 1] for jaw in jaws]
                    if not min(ys) < data.xpos[body, 1] < max(ys):
                        continue  # jaws must straddle this lane, not its neighbor
                    station_gaps.append(max(min(mujoco.mj_geomDistance(model, data, jaw, geom, 10.0, None)
                                                for geom in guide_leaf_geoms) for jaw in jaws))
                gap = max(0.0, min(station_gaps)) if station_gaps else 10.0
                worst_guide_gap = max(worst_guide_gap, float(gap))
                if gap > tolerance:
                    failures.append({"check": "floor_guide_engagement", "body": name, "q": float(q), "gap_m": float(gap)})
            rp = data.geom_xpos[rail]
            rs = model.geom_size[rail]
            rail_low, rail_high = rp[0] - rs[0], rp[0] + rs[0]
            leaf_low = data.xpos[body, 0] - support["leaf_width_m"] / 2
            leaf_high = data.xpos[body, 0] + support["leaf_width_m"] / 2
            overhang = max(rail_low - leaf_low, leaf_high - rail_high, 0.0)
            worst_overhang = max(worst_overhang, float(overhang))
            lane_gap = max(abs(float(data.xpos[body, 1] - rp[1])) - float(rs[1]) - 0.02, 0.0)
            if overhang > tolerance or lane_gap > tolerance:
                failures.append({"check": "rail_coverage", "body": name, "q": float(q), "overhang_m": float(overhang), "lane_gap_m": lane_gap})
                break
            for wheel in wheels:
                wp = data.geom_xpos[wheel]
                radius, half_width = model.geom_size[wheel][:2]
                # Wheels have their cylinder axis along y and roll on the rail's upper x face - or, on a top-hung
                # leaf, hang from its underside (a carrier wheel in the header): tangency to either face counts.
                vertical_gap = min(abs(float(wp[2] - radius - (rp[2] + rs[2]))),
                                   abs(float((rp[2] - rs[2]) - (wp[2] + radius))))
                lateral_gap = max(abs(float(wp[1] - rp[1])) - float(rs[1] + half_width), 0.0)
                tread_overhang = max(float(rail_low - (wp[0] - radius)), float(wp[0] + radius - rail_high), 0.0)
                gap = max(vertical_gap, lateral_gap, tread_overhang)
                worst_contact_gap = max(worst_contact_gap, gap)
                if gap > tolerance:
                    failures.append({"check": "wheel_contact", "body": name, "wheel": model.geom(wheel).name, "q": float(q), "gap_m": gap})
            if wheels and stops and index in (0, samples - 1):
                # At each end at least one terminal wheel is tangent to an end-stop face.
                gaps = []
                for wheel in wheels:
                    for stop in stops:
                        delta = np.abs(data.geom_xpos[wheel] - data.geom_xpos[stop])
                        dx, dz = np.maximum(delta[[0, 2]] - model.geom_size[stop, [0, 2]], 0.0)
                        radial_gap = abs(float(np.hypot(dx, dz) - model.geom_size[wheel, 0]))
                        lateral_gap = max(float(delta[1] - model.geom_size[stop, 1] - model.geom_size[wheel, 1]), 0.0)
                        gaps.append(max(radial_gap, lateral_gap))
                endpoint_stop_gaps.append(min(gaps))
        if endpoint_stop_gaps and max(endpoint_stop_gaps) > tolerance:
            failures.append({"check": "end_stop", "body": name, "endpoint_gaps_m": endpoint_stop_gaps})
    return {"ok": not failures, "n_supports": len(supports), "wheels_checked": wheels_checked,
            "guide_stations_checked": guide_stations_checked, "max_guide_gap_m": worst_guide_gap,
            "rail_only_bodies": rail_only, "samples_per_joint": samples,
            "max_contact_gap_m": worst_contact_gap, "max_leaf_overhang_m": worst_overhang,
            "n_failures": len(failures), "failures": failures[:20]}
