"""Retrospective volar-phalange assessment; never rewrites task qualification.

The frozen distal-only score remains authoritative for its original experiment.
This candidate contract additionally accepts four-finger middle/proximal volar
surfaces, while requiring the already verified distal thumb, cylindrical lever
contact, all five digit loads and opposition. It is not universal hand anatomy.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

from doorbench.dexterous.environment import DexterousDoorEnv
from doorbench.dexterous.grasp_verification import pad_opposition


def qualified_patch(contact):
    match = re.fullmatch(r"rh_(ff|mf|rf|lf|th)(proximal|middle|distal)", contact["body"].rsplit("/",1)[-1])
    if match is None:
        return False
    digit, segment = match.groups()
    # FK/mesh-checked four-finger lengths: proximal45mm, middle25mm. Thumb
    # proximal/middle axes and surfaces are NOT generalized by this contract.
    if digit == "th" and segment != "distal":
        return False
    upper = dict(proximal=.045, middle=.025, distal=.040)[segment]
    p = contact["body_position_m"]; n = contact["hand_outward_normal_body"]
    return bool(p[1] < -.001 and .002 <= p[2] <= upper and -n[1] > .5 and
        contact["axial_clearance_m"] >= .001 and contact["on_lever_cylindrical_side"] and
        contact["inward_radial_normal_alignment"] > .8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("robot", "door", "run", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--render", action="store_true", help="Save annotated FK reconstruction views; no physics replay")
    args = parser.parse_args()
    result = json.loads((args.run/"operation-report.json").read_text())
    config = json.loads((args.run/"configuration.json").read_text())
    with gzip.open(args.run/"acquisition-pad-steps.json.gz", "rt") as f:
        pads = json.load(f)
    with np.load(args.run/"acquisition-physics.npz", allow_pickle=False) as archive:
        trajectory = {k:archive[k] for k in archive.files}
    sim = DexterousDoorEnv(args.door, args.robot, json.loads(args.robot.with_suffix(".audit.json").read_text()))
    m,d = sim.m,sim.d
    qa = [m.jnt_qposadr[m.joint("robot/"+name).id] for name in config["robot_joint_names"]]
    da = [m.jnt_qposadr[m.joint(name).id] for name in config["door_joint_names"]]
    gid = m.geom("leaf_handle_lever_col_n").id
    if len(pads) != len(trajectory["time_s"])+1:
        raise ValueError("Incomplete synchronized pad and physical trajectory")
    rows=[]; maximum_fk_error=0.; body_counts=Counter(); rejected_counts=Counter()
    renderer = None
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.render:
        from PIL import Image, ImageDraw
        for g in range(m.ngeom):
            name=m.body(m.geom_bodyid[g]).name
            if name.startswith('robot/rh_'):
                m.geom_matid[g]=-1
                m.geom_rgba[g]=[.05,.4,.85,1.] if name.startswith('robot/rh_th') else [.5,.55,.60,1.]
                if name in ('robot/rh_ffmiddle','robot/rh_mfmiddle'):
                    m.geom_rgba[g]=[1.,.35,.02,1.]
            if m.geom(g).name.startswith('leaf_handle'):
                m.geom_matid[g]=-1;m.geom_rgba[g]=[.72,.51,.08,1.]
        renderer=mujoco.Renderer(m,height=720,width=960)
        options=mujoco.MjvOption();options.sitegroup[:]=0
    for i,t in enumerate(trajectory["time_s"]):
        if abs(t-pads[i+1]["sim_time_s"]) > 1e-8:
            raise ValueError("Pad timestamps differ from physical states")
        d.qpos[sim.root_qadr:sim.root_qadr+7] = trajectory["root"][i,:7]
        d.qpos[qa]=trajectory["joints"][i];d.qpos[da]=trajectory["door"][i]
        # Reconstructed measured state in a never-stepped diagnostic model only.
        mujoco.mj_kinematics(m,d)
        center=d.geom_xpos[gid];axis=d.geom_xmat[gid].reshape(3,3)[:,2]
        contacts=[]
        for patch in pads[i+1]["contacts"]:
            body=m.body("robot/"+patch["body"].rsplit("/",1)[-1]).id
            world=d.xpos[body]+d.xmat[body].reshape(3,3)@patch["body_position_m"]
            maximum_fk_error=max(maximum_fk_error,float(np.linalg.norm(world-patch["position"])))
            contact=dict(patch,pad_qualified=qualified_patch(patch))
            contacts.append(contact)
            if patch["normal_force_N"]>1e-6 and not patch["pad_qualified"]:
                body_counts[patch["body"].rsplit("/",1)[-1]]+=1
                if not contact["pad_qualified"]:
                    rejected_counts[patch["body"].rsplit("/",1)[-1]]+=1
        audit=pad_opposition(contacts,center,axis)
        audit['reason']=audit['reason'].replace('distal volar pad','volar phalange')
        rows.append(dict(time_s=float(t),**audit))
        if renderer is not None and any(abs(t-wanted)<1e-8 for wanted in (16.,18.9,20.,22.)):
            for azimuth in (90,150,210):
                camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.4
                camera.azimuth=azimuth;camera.elevation=-20
                renderer.update_scene(d,camera=camera,scene_option=options);sim.hide_sensor_overlays(renderer.scene)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,960,32),fill='black')
                draw.text((8,10),f'FK RECONSTRUCTION OF RECORDED ISAAC STATE | t={t:.3f}s | no physics replay | middle: orange / thumb: blue',fill='white')
                image.save(args.out.parent/f'fk-hand-{t:.1f}-{azimuth}.png')
    tail=[r for r in rows if r["time_s"]>=result["duration_s"]-.5-1e-8]
    active=[r for r in rows if r["time_s"]>=result["operation_reference"]["operation_start_s"]]
    original_bytes=(args.run/"operation-report.json").read_bytes()
    report=dict(scope=__doc__,run=str(args.run),contract="candidate-shadow-four-finger-volar-phalanges-distal-thumb-v1",
        original_strict_task_passed=result["passed"],original_strict_final_grasp=result["checks"]["sustained_pad_grasp"],
        original_report_sha256=hashlib.sha256(original_bytes).hexdigest(),
        contract_parameters=dict(finger_segments=['proximal','middle','distal'],thumb_segments=['distal'],
            local_surface_y_less_than_m=-.001,minimum_local_z_m=.002,
            maximum_local_z_m=dict(proximal=.045,middle=.025,distal=.040),
            minimum_outward_minus_y=.5,minimum_lever_axial_clearance_m=.001,
            minimum_inward_radial_alignment=.8,minimum_force_per_digit_N=.2,
            maximum_misplaced_force_fraction=.05,minimum_finger_pair_dot=.5,maximum_thumb_finger_dot=-.5),
        checked_2ms_rows=len(rows),max_recorded_local_point_to_native_FK_world_error_m=maximum_fk_error,
        actual_state_FK_consistent=maximum_fk_error<.0005,
        retrospective_candidate_final_hold=bool(all(r["valid_pad_grasp"] for r in tail)),
        candidate_operation_invalid_or_unloaded_steps=sum(not r["valid_pad_grasp"] for r in active),
        candidate_final=rows[-1],originally_rejected_patch_body_counts=dict(body_counts),
        still_rejected_patch_body_counts=dict(rejected_counts),
        original_mechanical_checks={k:v for k,v in result["checks"].items() if k not in ["sustained_pad_grasp"]},
        limitations=["Retrospective diagnostic only; does not relabel this task or authorize training on a failed original report.",
            "Thumb middle/proximal surfaces are outside this candidate contract.",
            "A future qualification must declare this contract before execution and retain both distal and volar scores."])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if renderer is not None:renderer.close()
    sim.close()


if __name__ == "__main__":
    main()
