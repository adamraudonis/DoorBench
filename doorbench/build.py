"""Build pipeline: spec -> physics -> IR model -> exports."""
from __future__ import annotations

import json
import os
import time

from .ir import Model
from . import physics as P
from .geometry import hinged as GH
from .geometry import other as GO
from .geometry import meshes as MESH


MIN_MODELLED_T = 0.003   # m; the thinnest a leaf may be MODELLED.  spec["leaf"]["thickness"] is the mass parameter
#                          (materials.SlabConstruction.area_density smears the whole leaf into it), and for a
#                          chain-link gate that is the 0.3 mm of the mesh wire - which builds a gate leaf, its
#                          stiles, rails and pickets and its collision proxy 0.3 mm thick: a degenerate collider in
#                          both engines and a membrane on screen.  The clamp is applied to the geometry only, after
#                          physics has been derived, so no mass or QA number moves.


def build_model(spec: dict, phys: dict | None = None) -> Model:
    phys = phys or P.derive(spec)
    if float(spec["leaf"].get("thickness", 1.0)) < MIN_MODELLED_T:
        spec = {**spec, "leaf": {**spec["leaf"], "thickness": MIN_MODELLED_T, "mass_thickness": spec["leaf"]["thickness"]}}
    model = Model(spec["id"])
    model.meta.update({"door_id": spec["id"], "family": spec["family"], "task": spec.get("task"), "notes": []})
    fam = spec["family"]
    if fam in ("swing_single", "automatic_swing", "cold_storage", "baby_gate"):
        GH.build_swing_single(spec, phys, model)
    elif fam == "swing_double":
        GH.build_swing_double(spec, phys, model)
    elif fam == "dutch":
        GH.build_dutch(spec, phys, model)
    elif fam == "saloon":
        GH.build_saloon(spec, phys, model)
    elif fam == "pivot":
        GH.build_swing_single(spec, phys, model)
    elif fam == "ship_watertight":
        GH.build_ship(spec, phys, model)
    elif fam in ("vault", "blast"):
        GH.build_vault(spec, phys, model)
    elif fam == "gate_swing":
        GH.build_gate_or_fence(spec, phys, model)
    elif fam == "stall":
        GH.build_stall(spec, phys, model)
    elif fam in ("sliding_single", "sliding_bypass", "automatic_sliding", "elevator", "gate_sliding"):
        GO.build_sliding(spec, phys, model)
    elif fam in ("bifold", "accordion"):
        GO.build_folding(spec, phys, model)
    elif fam == "revolving":
        GO.build_revolving(spec, phys, model)
    elif fam == "turnstile_tripod":
        GO.build_turnstile(spec, phys, model, full_height=False)
    elif fam == "turnstile_fullheight":
        GO.build_turnstile(spec, phys, model, full_height=True)
    elif fam in ("garage_sectional", "rollup"):
        GO.build_vertical(spec, phys, model)
    elif fam in ("hatch_floor", "hatch_ceiling", "pet_door", "strip_curtain", "garage_tiltup"):
        GO.build_horizontal(spec, phys, model)
    else:
        raise ValueError(f"unknown family {fam}")
    if fam == "automatic_swing":
        act = spec["kinematics"].get("actuator", {})
        model.meta.setdefault("actuators", []).append({"name": "swing_operator", "joint": model.meta["primary_joint"], "kind": "position", "kp": 150.0, "kv": 40.0, "forcerange": (-act.get("max_torque_Nm", 60), act.get("max_torque_Nm", 60)), "ctrlrange": (0.0, 1.6)})
    # Armature floors: reflected inertia of internal lock/latch mechanisms (gears, springs, spindles).  Also required so
    # MuJoCo's mass-scaled soft constraints (equalities, tendon & joint limits) can act on very light mechanism bodies.
    ARM_HINGE = {"operator": 0.01, "lock": 0.005, "latch": 0.005, "mechanism": 0.002, "secondary": 0.005, "decor": 0.002, "primary": 0.01}
    ARM_SLIDE = {"operator": 0.15, "lock": 0.1, "latch": 0.1, "mechanism": 0.05, "secondary": 0.1, "decor": 0.05, "primary": 0.5}
    for b in model.bodies:
        j = b.joint
        if j is None:
            continue
        floor = (ARM_HINGE if j.type == "hinge" else ARM_SLIDE).get(j.role, 0.005)
        if j.role == "primary" and j.type == "hinge":
            floor = 0.02
        if j.role == "primary" and j.type == "slide":
            floor = 0.5
        j.armature = max(j.armature, floor)
    op = spec["opening"]
    # --- mass reconciliation: the derived slab + glass mass (physics.py, calibrated to manufacturer weight tables) is
    # distributed over the leaf bodies by volume, so the simulated moving mass matches the spec whatever the geometry
    # builder did (glass panes, split leaves, strips and rotors otherwise carried density-based masses)
    leaf_bodies = [b for b in model.bodies if getattr(b, "semantic", "") == "leaf" and not b.static]
    hw_now = float(sum(b.inertial("full")[0] for b in model.bodies if not b.static and getattr(b, "semantic", "") != "leaf"))
    slab_glass = float(phys["mass"].get("slab_kg", 0.0) + phys["mass"].get("glass_kg", 0.0))
    # hardware that is not modelled as its own body (tracks, hangers, straps, plates) rides on the leaf
    tgt_mass = max(0.5 * slab_glass, float(phys["mass"]["total_kg"]) - hw_now)
    if tgt_mass > 0 and leaf_bodies:
        vols = [max(sum((g.volume() or 0.0) for g in b.geoms), 1e-6) for b in leaf_bodies]
        vt = sum(vols)
        for b, vol in zip(leaf_bodies, vols):
            b.mass_override = tgt_mass * vol / vt
        model.meta["mass_reconciled_kg"] = tgt_mass
    model.meta["scene_extent"] = max(1.5, 0.75 * max(op["width"], op["height"]) + 0.5)
    model.meta["cam_target_z"] = 0.5 * op["height"] + float(op.get("elevation", 0.0) or 0.0) + float(op.get("sill_height", 0.0) or 0.0) * 0.5
    model.meta["cam_target_x"] = 0.0
    model.meta["handle_cam_x"] = float(model.meta.get("hinge_x", 0.0) + model.meta.get("u", 1.0) * (spec["leaf"]["width"] - 0.1)) if model.meta.get("u") is not None else 0.3
    # handle-detail camera: aim at the first grip site (world position via the parent chain; rotations are identity)
    def _world_pos(body_name):
        p = [0.0, 0.0, 0.0]
        seen = 0
        while body_name and seen < 12:
            b = model.body(body_name)
            p = [p[i] + float(b.pos[i]) for i in range(3)]
            body_name = b.parent
            seen += 1
        return p
    grip = None
    for b in model.bodies:
        for s_ in b.sites:
            if getattr(s_, "role", "") == "grip":
                wp = _world_pos(b.name)
                grip = [wp[i] + float(s_.pos[i]) for i in range(3)]
                break
        if grip:
            break
    if grip is None:
        grip = [model.meta["handle_cam_x"], float(model.meta.get("wall_y", 0.0)), float(model.meta.get("handle_height", 1.0))]
    fam = spec["family"]
    if fam == "hatch_floor":
        off = (0.35, -0.55, 0.85)
    elif fam == "hatch_ceiling":
        off = (0.35, -0.55, -0.85)
    elif fam in ("garage_sectional", "garage_tiltup", "rollup", "gate_sliding", "turnstile_tripod", "turnstile_fullheight", "revolving"):
        off = (0.25, -1.1, 0.35)
    else:
        off = (0.18, -0.8, 0.28)
    model.meta["handle_cam_target"] = grip
    model.meta["handle_cam_pos"] = [grip[0] + off[0], grip[1] + off[1], grip[2] + off[2]]
    model.bake_initial()
    model.uniquify()
    model.validate()
    return model


def export_door(spec: dict, out_root: str, hardware_dir: str, formats=("mjcf", "urdf", "usd", "json"), tiers=("full", "simple", "minimal")) -> dict:
    """Export one door.  Returns a summary dict."""
    from .export import mjcf as XM
    t0 = time.time()
    phys = P.derive(spec)
    model = build_model(spec, phys)
    out_dir = os.path.join(out_root, spec["id"])
    os.makedirs(out_dir, exist_ok=True)
    rel_hw = os.path.relpath(hardware_dir, out_dir)
    rel_tex = os.path.relpath(os.path.join(os.path.dirname(hardware_dir), "textures"), out_dir)
    summary = {"id": spec["id"], "family": spec["family"], "files": {}, "mass_kg": phys["mass"]["total_kg"], "n_bodies": len(model.bodies_in_tier("full")), "n_joints": len(model.joints("full"))}
    # meshes -> shared hardware library
    written = write_hardware_meshes(model, hardware_dir)
    summary["meshes"] = written
    if "mjcf" in formats:
        summary["files"]["mjcf"] = XM.write_mjcf(model, out_dir, tiers=tiers, mesh_dir_rel=rel_hw, texture_dir_rel=rel_tex)
    if "urdf" in formats:
        from .export import urdf as XU
        summary["files"]["urdf"] = XU.write_urdf(model, out_dir, mesh_dir_rel=rel_hw)
    if "usd" in formats:
        from .export import usd as XS
        try:
            summary["files"]["usd"] = XS.write_usd(model, out_dir, hardware_dir=hardware_dir)
        except Exception as e:  # pragma: no cover
            summary["files"]["usd"] = f"ERROR: {e!r}"
        try:
            # canonical articulation for Isaac Lab multi-door training (same link/joint names for every door)
            summary["files"]["usd_rl"] = XS.write_usd_rl(model, out_dir, hardware_dir=hardware_dir, spec={**spec, "physics": phys})
        except Exception as e:  # pragma: no cover
            summary["files"]["usd_rl"] = f"ERROR: {e!r}"
    if "json" in formats:
        from .benchmark.scenarios import build_benchmark, benchmark_summary
        model_dict = json.loads(json.dumps(model.to_dict("full"), default=_json_default))
        bench = build_benchmark(spec, phys, model_dict)      # scenarios + rewards (docs/BENCHMARK.md)
        summary["benchmark"] = benchmark_summary(bench)
        with open(os.path.join(out_dir, "spec.json"), "w") as f:
            json.dump({**spec, "physics": phys, "benchmark": bench}, f, indent=1, default=_json_default)
        with open(os.path.join(out_dir, "model.json"), "w") as f:
            json.dump(model_dict, f)
    summary["build_time_s"] = time.time() - t0
    return summary


def write_hardware_meshes(model: Model, hardware_dir: str) -> list:
    """Write every shared mesh used by the model to hardware_dir/<key>.obj (once)."""
    os.makedirs(hardware_dir, exist_ok=True)
    out = []
    for b in model.bodies:
        for g in b.geoms:
            if g.type == "mesh" and g.mesh is not None:
                path = os.path.join(hardware_dir, f"{g.mesh_name}.obj")
                if not os.path.exists(path):
                    g.mesh.export(path, include_normals=False, include_texture=False)
                    out.append(g.mesh_name)
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(repr(o))
