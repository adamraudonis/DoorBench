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


def build_model(spec: dict, phys: dict | None = None) -> Model:
    phys = phys or P.derive(spec)
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
    approach=spec.get('robot',{}).get('approach_side','-y')
    if approach not in ('-y','+y'):raise ValueError('Unsupported approach side')
    model.meta['approach_face']=1 if approach=='+y' else -1
    if approach=='+y':
        plane=float(model.meta.get('wall_y',0.))
        for body in model.bodies:
            for site in body.sites:
                if site.name in ('approach_point','goal_point'):
                    site.pos=(site.pos[0],2*plane-site.pos[1],site.pos[2])
    from .geometry.closer_mounts import finish_deferred_closers
    finish_deferred_closers(model)
    if fam == "automatic_swing":
        act = spec["kinematics"].get("actuator", {})
        model.meta.setdefault("actuators", []).append({"name": "swing_operator", "joint": model.meta["primary_joint"], "kind": "position", "kp": 150.0, "kv": 40.0, "forcerange": (-act.get("max_torque_Nm", 60), act.get("max_torque_Nm", 60)), "ctrlrange": (0.0, 1.6)})
    if fam in ("automatic_swing", "automatic_sliding", 'garage_sectional'):
        from .geometry.automatic_controls import add_automatic_controls
        add_automatic_controls(model, spec)
    # Armature floors: reflected inertia of internal lock/latch mechanisms (gears, springs, spindles).  Also required so
    # MuJoCo's mass-scaled soft constraints (equalities, tendon & joint limits) can act on very light mechanism bodies.
    ARM_HINGE = {"operator": 0.01, "lock": 0.005, "latch": 0.005, "mechanism": 0.002, "secondary": 0.005, "decor": 0.002, "primary": 0.01}
    ARM_SLIDE = {"operator": 0.15, "lock": 0.1, "latch": 0.1, "mechanism": 0.05, "secondary": 0.1, "decor": 0.05, "primary": 0.5}
    physical_names=model.meta.get('physical_inertia_joints',[])
    if physical_names:
        import numpy as np
        by_joint={b.joint.name:b for b in model.bodies if b.joint}
        if (not isinstance(physical_names,list) or any(not isinstance(n,str) for n in physical_names)
                or len(set(physical_names))!=len(physical_names) or not set(physical_names)<=by_joint.keys()):
            raise ValueError('physical_inertia_joints must name distinct existing joints')
        for name in physical_names:
            body=by_joint[name]
            for tier in body.tiers:
                geometric_mass=sum(g.mass() for g in body.geoms if tier in g.tiers)
                mass,_,inertia=body.inertial(tier)
                if not (np.isfinite(geometric_mass) and geometric_mass>0 and np.isfinite(mass) and mass>0
                        and np.isfinite(inertia).all() and np.linalg.eigvalsh(inertia).min()>0):
                    raise ValueError(f'Physical-inertia joint {name} needs positive geometric mass and inertia in {tier}')
    for b in model.bodies:
        j = b.joint
        if j is None:
            continue
        if j.type == 'free':
            # A free mechanism body has six physical rigid-body DOFs and no
            # drive train whose reflected inertia could justify armature.
            j.armature = 0.
            continue
        if j.name in model.meta.get('material_flexure_joints',[]) or j.name in model.meta.get('physical_inertia_joints',[]):
            # Passive material, shells and linkages carry actual geometry
            # inertia; none has a geared drive requiring reflected armature.
            j.armature=0.
            continue
        floor = (ARM_HINGE if j.type == "hinge" else ARM_SLIDE).get(j.role, 0.005)
        if j.role == "primary" and j.type == "hinge":
            floor = 0.02
        if j.role == "primary" and j.type == "slide":
            floor = 0.5
        j.armature = max(j.armature, floor)
    op = spec["opening"]
    # Explicit physical panel budgets; duplicate proxies/decor never set allocation.
    from .mass_reconciliation import reconcile_moving_mass
    reconcile_moving_mass(model, phys)
    # A bidirectional lift lever must centre under its own actual weight.
    # Specify a real torsion-spring preload, rather than holding the handle
    # with an invisible runtime torque or a test fixture's neutral servo.
    if model.meta.get('gravity_balanced_operators'):
        import numpy as np
        from .ir import quat_to_mat
        def orientation(body):
            R=quat_to_mat(body.quat)
            return orientation(model.body(body.parent))@R if body.parent else R
        for name in model.meta['gravity_balanced_operators']:
            body=next(b for b in model.bodies if b.joint and b.joint.name==name)
            joint=body.joint
            if joint.type!='hinge' or joint.stiffness<=0:
                raise ValueError('Gravity-balanced operators require an actual torsion spring')
            mass,com,_=body.inertial('full')
            gravity=orientation(body).T@np.array([0.,0.,-9.81])
            torque=float(np.dot(joint.axis,np.cross(np.asarray(com)-joint.pos,mass*gravity)))
            joint.springref=-torque/joint.stiffness
            model.meta.setdefault('operator_return_preloads',[]).append({'joint':name,'springref_rad':joint.springref,
                'stiffness_Nm_rad':joint.stiffness,'closed_gravity_torque_Nm':torque,
                'scope':'Torsion spring preload balances the actual closed-pose operator geometry; no runtime servo'})
    model.meta["scene_extent"] = max(1.5, 0.75 * max(op["width"], op["height"]) + 0.5)
    model.meta["cam_target_z"] = 0.5 * op["height"] + float(op.get("elevation", 0.0) or 0.0) + float(op.get("sill_height", 0.0) or 0.0) * 0.5
    model.meta["cam_target_x"] = 0.0
    model.meta["handle_cam_x"] = float(model.meta.get("hinge_x", 0.0) + model.meta.get("u", 1.0) * (spec["leaf"]["width"] - 0.1)) if model.meta.get("u") is not None else 0.3
    # Use the full authored parent transforms, including wound slat rest poses.
    def _world_pos(body_name,local):
        from .ir import quat_rotate
        p = list(local)
        seen=set()
        while body_name:
            if body_name in seen:raise ValueError('Cyclic body hierarchy')
            seen.add(body_name)
            b = model.body(body_name)
            rotated=quat_rotate(b.quat,p)
            p = [float(rotated[i])+float(b.pos[i]) for i in range(3)]
            body_name = b.parent
        return p
    grip = None
    for b in model.bodies:
        for s_ in b.sites:
            if getattr(s_, "role", "") == "grip":
                grip = _world_pos(b.name,s_.pos)
                break
        if grip:
            break
    if grip is None:
        grip = [model.meta["handle_cam_x"], float(model.meta.get("wall_y", 0.0)), float(model.meta.get("handle_height", 1.0))]
    if model.meta.get('rollup_hoist'):
        names=set(model.meta['rollup_hoist']['material_grip_sites'])
        candidates=[_world_pos(b.name,s.pos) for b in model.bodies for s in b.sites if s.name in names]
        grip=min(candidates,key=lambda p:abs(p[2]-1.2))
    fam = spec["family"]
    if fam == "hatch_floor":
        off = (0.35, -0.55, 0.85)
    elif fam == "hatch_ceiling":
        off = (0.35, -0.55, -0.85)
    elif fam in ("garage_sectional", "garage_tiltup", "rollup", "gate_sliding", "turnstile_tripod", "turnstile_fullheight", "revolving"):
        off = (0.25, -1.1, 0.35)
    else:
        off = (0.18, -0.8, 0.28)
    if model.meta['approach_face']>0:off=(off[0],-off[1],off[2])
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
                content = g.mesh.export(file_type="obj", include_normals=False, include_texture=False)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                from pathlib import Path
                target = Path(path)
                if not target.exists() or target.read_bytes() != content:
                    # Parallel generators may share a mesh key. Readers must
                    # see one complete OBJ, never a partially rewritten file.
                    import tempfile
                    with tempfile.NamedTemporaryFile(dir=hardware_dir, suffix='.obj.tmp', delete=False) as mesh_tmp:
                        mesh_tmp.write(content)
                        temporary = mesh_tmp.name
                    os.replace(temporary, target)
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
