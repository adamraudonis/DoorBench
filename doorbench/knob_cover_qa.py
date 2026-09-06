"""Native independent-shell and finite-finger access checks."""
from __future__ import annotations
import math
import numpy as np
from .ir import quat_z_to


def run_knob_cover_qa(xml_path,metadata):
    import mujoco as mj
    records=metadata.get('knob_covers',[])
    if not records:return {'ok':True,'failures':[],'scope':'No covered knobs'}
    source=mj.MjSpec.from_file(str(xml_path))
    probe=source.worldbody.add_body(name='qa_knob_finger',mocap=True,pos=[10,10,10])
    probe.add_geom(name='qa_knob_finger',type=mj.mjtGeom.mjGEOM_CAPSULE,size=[.006,.020,0],contype=0,conaffinity=0)
    m=source.compile();d=mj.MjData(m);pg=m.geom('qa_knob_finger').id
    mocap=int(m.body_mocapid[m.body('qa_knob_finger').id]);failures=[];rows=[]
    for record in records:
        op=m.joint(record['operator_joint']).id;oa=int(m.jnt_qposadr[op]);od=int(m.jnt_dofadr[op])
        for face in record['faces']:
            cj=m.joint(face['cover_joint']).id;ca=int(m.jnt_qposadr[cj]);cd=int(m.jnt_dofadr[cj])
            shell=[m.geom(g).id for g in face['shell_geoms']];knob=m.geom(face['knob_geom']).id
            if cj==op or m.jnt_limited[cj] or m.jnt_stiffness[cj] or m.dof_armature[cd]:
                failures.append({'check':'cover_not_independent_free_shell','joint':face['cover_joint']})
            minimum=math.inf
            # A 12mm diameter, 52mm end-to-end finger must reach the knob
            # through the aperture throughout the full required knob turn.
            for angle in np.linspace(0.,record['required_turn_rad'],41):
                mj.mj_resetData(m,d);d.qpos[oa]=angle;mj.mj_forward(m,d)
                for site in face['grip_sites']:
                    sid=m.site(site).id;normal=d.site_xmat[sid].reshape(3,3)[:,2].copy()
                    d.mocap_pos[mocap]=d.site_xpos[sid]+.026*normal
                    d.mocap_quat[mocap]=quat_z_to(normal);mj.mj_kinematics(m,d)
                    gap=mj.mj_geomDistance(m,d,pg,knob,.02,None)
                    if abs(gap)>1e-5:failures.append({'check':'finger_not_on_knob','site':site,'gap_m':float(gap)})
                    for geom in shell:
                        distance=float(mj.mj_geomDistance(m,d,pg,geom,.1,None));minimum=min(minimum,distance)
            if minimum<.0005:
                failures.append({'check':'finger_path_blocked_by_cover','cover':face['cover_body'],'minimum_gap_m':minimum})
            # Turning the outside of the cover cannot retract the inner latch.
            mj.mj_resetData(m,d);sid=m.site(face['cover_site']).id
            for _ in range(round(.8/m.opt.timestep)):
                mj.mj_forward(m,d);d.qfrc_applied[:]=0.
                axis=d.xaxis[cj];radial=d.site_xpos[sid]-d.xanchor[cj]
                tangent=np.cross(axis,radial);tangent/=np.linalg.norm(tangent)
                force=float(np.clip(.02*(6.-d.qvel[cd])/.041,-8.,8.))*tangent
                mj.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[sid],m.site_bodyid[sid],d.qfrc_applied)
                mj.mj_step(m,d)
                if np.any(d.warning.number):
                    failures.append({'check':'cover_spin_native_warning','counts':d.warning.number.tolist()});break
            outside_turn=float(d.qpos[ca]);inner_turn=float(d.qpos[oa])
            if outside_turn<1. or abs(inner_turn)>.005:
                failures.append({'check':'outer_shell_drives_knob','cover_q':outside_turn,'knob_q':inner_turn})
            # Opposed 20N tangential fingertip loads transfer through the
            # actual inner knob. These are ideal contact inputs, not a hand.
            mj.mj_resetData(m,d)
            for _ in range(round(1./m.opt.timestep)):
                mj.mj_forward(m,d);d.qfrc_applied[:]=0.
                for site in face['grip_sites']:
                    sid=m.site(site).id;normal=d.site_xmat[sid].reshape(3,3)[:,2]
                    force=20.*np.cross(d.xaxis[op],normal)
                    mj.mj_applyFT(m,d,force,np.zeros(3),d.site_xpos[sid],m.site_bodyid[sid],d.qfrc_applied)
                mj.mj_step(m,d)
            operated=float(d.qpos[oa])
            if operated<.95*record['required_turn_rad']:
                failures.append({'check':'inner_knob_does_not_actuate','q':operated})
            if np.any(d.warning.number):failures.append({'check':'native_warning','counts':d.warning.number.tolist()})
            rows.append({'face':face['face'],'cover':face['cover_body'],'finger_minimum_shell_gap_m':minimum,
                'outside_turn_rad':outside_turn,'inner_turn_under_shell_load_rad':inner_turn,
                'inner_knob_actuated_rad':operated})
    return {'ok':not failures,'failures':failures,'measurements':rows,
        'scope':'Independent cover rotation, full-turn finite-finger aperture clearance, bounded loads on actual inner knob. Ideal bearings; no humanoid grasp or child-resistance certificate.'}
