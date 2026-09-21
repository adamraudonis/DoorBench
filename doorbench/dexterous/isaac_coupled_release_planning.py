"""Fresh unstepped coupled-map generation from an admitted actual Isaac source.

Only bounded scalar solver/design preferences are borrowed. Initial state,
material frames, references and all qualification come from the newly admitted
actual PhysX source and its independently audited Isaac RH release candidate.
"""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .isaac_coupled_release_geometry import (IsaacCoupledReleaseGeometry,JOINT_NAMES,
    SCHEMA,load_isaac_coupled_source,validate_envelope_binding,validate_map)
from .landed_left_planner import LandedLeftScene
from .operation_teacher import smooth_phase
from .qualified_isaac_grasp import digest


DEFAULT_PREFERENCES = dict(time_nodes=81,max_nfev=180,
    follow_handle_through_route_seconds=3.5,operator_margin_rad=.01,
    initial_aperture_headroom_rad=.015,final_aperture_upper_rad=.4)


def numeric_coupled_preferences(document):
    """Old documents can contribute these bounded numbers, never coordinates."""
    if type(document) is not dict:
        raise ValueError('Explicit numeric coupled preference object required')
    values = document.get('configuration',document)
    if type(values) is not dict:
        raise ValueError('Explicit coupled preference configuration required')
    result = {name:values.get(name,default) for name,default in DEFAULT_PREFERENCES.items()}
    for name,low,high in [('time_nodes',81,401),('max_nfev',1,500)]:
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError('Bounded integer solver preference required: '+name)
    for name,low,high in [('follow_handle_through_route_seconds',0.,7.5),
            ('operator_margin_rad',.001,.05),('initial_aperture_headroom_rad',0.,.05),
            ('final_aperture_upper_rad',.1,.4)]:
        if (type(result[name]) not in (int,float) or not np.isfinite(result[name])
                or not low <= result[name] <= high):
            raise ValueError('Bounded finite coupled preference required: '+name)
    return result


def make_isaac_withdrawal_source_config(*,source,candidate,dense_audit,robot,door_xml,door_usd,
                                       profile='volar-phalange-v1',source_kind=None,phase_audit_path=None):
    """Build and fully validate detached configuration; never a runtime route."""
    from .isaac_release_source_dispatch import admit_release_planning_source
    from .withdrawal_source_context import load_withdrawal_source_context
    context = admit_release_planning_source(source,robot=robot,door_xml=door_xml,door_usd=door_usd,
        profile=profile,source_kind=source_kind,phase_audit_path=phase_audit_path)
    source = Path(source).resolve();candidate = Path(candidate).resolve();dense_audit = Path(dense_audit).resolve()
    admission = context.admission
    audit = json.loads(dense_audit.read_text())
    result = dict(schema='doorbench.standing-withdrawal.v1',source_engine='isaac-physx',
        source_run=str(source),screen_path=str(candidate),screen_sha256=digest(candidate),
        audit_path=str(dense_audit),audit_sha256=digest(dense_audit),measured_rest_transfer=True,
        grasp_profile=profile,contact_audit_name='independent-contact-audit.json',
        robot_path=admission['robot_path'],door_xml_path=admission['door_xml_path'],
        door_usd_path=admission['door_usd_path'],source_state_sha256=admission['source_qualification']['state_sha256'],
        start_time_s=context.terminal_time_s,duration_s=audit['duration_s'],
        authorized_stages=0,runtime_route_exported=False,
        scope='Detached actual-Isaac release planning inputs. No live constructor or stage permission exported.')
    if source_kind is not None:
        result.pop('contact_audit_name')
        result.update(source_kind=source_kind,source_run=admission['source_run'],
            source_snapshot_path=str(source),source_snapshot_sha256=digest(source),
            motor_contract_path=str(context.motor_contract_path),configuration_path=str(context.configuration_path))
        if phase_audit_path is not None:
            result.update(phase_audit_path=str(Path(phase_audit_path).resolve()),phase_audit_sha256=digest(phase_audit_path))
    motor_path=context.motor_contract_path if source_kind is not None else source/'trial/motor-contract.json'
    motors = json.loads(motor_path.read_text())
    checked = load_withdrawal_source_context(result,motors,measured_rest=True)
    if checked.data['source_admission'] != admission:
        raise ValueError('Actual source changed between configuration admissions')
    context.verify_inputs()
    return result


def _seed_model(source_config,preferences):
    source_config = Path(source_config).resolve()
    context,hashes = load_isaac_coupled_source(source_config)
    data = context.data
    scene = LandedLeftScene(data['robot_path'],data['door_xml_path'])
    m = scene.m;initial = np.asarray(data['initial_qpos'],float)
    if initial.shape != (m.nq,) or not np.isfinite(initial).all():
        raise ValueError('Complete finite admitted actual source coordinates required')
    qa = np.array([m.joint('robot/'+name).qposadr[0] for name in JOINT_NAMES])
    lq,oq,bq = [int(m.joint(name).qposadr[0]) for name in ('leaf_hinge','leaf_handle_hinge','leaf_latch_bolt_slide')]
    duration = data['duration_s']
    elapsed = np.linspace(0,duration,preferences['time_nodes'])
    angles = np.sort(np.unique(np.r_[np.linspace(.08,.4,17),initial[lq]]))
    coordinates = np.tile(np.r_[np.zeros(6),initial[qa]],(len(elapsed),len(angles),1))
    margin = preferences['operator_margin_rad'];final = preferences['final_aperture_upper_rad']
    initial_upper = min(final,initial[lq]+preferences['initial_aperture_headroom_rad'])
    if initial_upper < initial[lq]:
        raise ValueError('Prospective aperture domain must contain the actual source')
    plan = dict(schema=SCHEMA,source_engine='isaac-physx',source_admission=data['source_admission'],
        source_context_sha256=data['source_context_sha256'],source_state_sha256=data['source_state_sha256'],
        initial_episode_time_s=data['start_time_s'],initial_qpos=initial.tolist(),duration_s=duration,
        grasp_profile=data['source_admission']['grasp_profile'],motor_contract_sha256=data['motor_contract_sha256'],
        source_config=str(source_config),right_hand_candidate=data['screen_path'],right_hand_dense_audit=data['audit_path'],
        source_physics_archive=data['source_archive_path'],input_sha256={**data['input_sha256'],**hashes},
        elapsed_s=elapsed.tolist(),leaf_rad=angles.tolist(),coordinates=coordinates.tolist(),joint_names=list(JOINT_NAMES),
        operator_reference_rad=float(initial[oq]),operator_envelope_rad=[max(-.05,float(initial[oq])-margin),min(.05,float(initial[oq])+margin)],
        follow_handle_through_route_seconds=preferences['follow_handle_through_route_seconds'],
        admitted_leaf_upper_nodes=[[0.,float(initial_upper)],[duration,float(final)]],
        configuration=preferences.copy(),physics_steps=0,source_sample_playback=0,active_state_writes=0,
        authorized_stages=0,physical_admission=False,runtime_route_exported=False,geometric_admission=False,
        preferences_are_numeric_only=True,
        aperture_boundary_semantics='Prospective admissible measured-angle bounds, not a commanded or recorded door trajectory')
    if 'source_kind' in data:plan['source_kind']=data['source_kind']
    validate_map(plan,initial,qa,lq,oq,bq)
    validate_envelope_binding(plan,data,source_config)
    # The unsolved tensor lives only inside this private workspace. It is never
    # serialized as a completed map, screen, or source qualification.
    model = object.__new__(IsaacCoupledReleaseGeometry)
    model.path = None;model.source_config_path = source_config;model.plan = plan
    model.source_context = context;model.source_data = data;model._file_hashes = plan['input_sha256'].copy()
    model._initialize_geometry(data,plan,scene=scene)
    model.verify_inputs()
    return model


def _reference_targets(model,t,angle):
    """Same geometry as unchanged evaluate, without restricting solver knots.

    The tensor covers the whole declared rectangular interpolation grid, while
    the independent audit/runtime later use the narrower prospective domain.
    """
    m,d = model.m,model.d
    clock = float(smooth_phase(t/model.plan['duration_s']))*model.times[-1]
    k = min(len(model.times)-2,max(0,int(np.searchsorted(model.times,clock,side='right')-1)))
    f = (clock-model.times[k])/(model.times[k+1]-model.times[k])
    reference = (1-f)*model.qs[k]+f*model.qs[k+1]
    rp = (1-f)*model.positions[k]+f*model.positions[k+1]
    rr = model.rotations(clock).as_matrix()
    d.qpos[:] = reference;mujoco.mj_kinematics(m,d)
    hp = d.xpos[model.handle].copy();hr = d.xmat[model.handle].reshape(3,3).copy()
    d.qpos[model.lq] = angle;d.qpos[model.oq] = model.initial[model.oq];d.qpos[model.bq] = model.initial[model.bq]
    mujoco.mj_kinematics(m,d)
    follow = 1.-float(smooth_phase((max(0.,clock-.5)-model.through)/(8.-model.through)))
    transformed_p = d.xpos[model.handle]+d.xmat[model.handle].reshape(3,3)@hr.T@(rp-hp)
    transformed_r = d.xmat[model.handle].reshape(3,3)@hr.T@rr
    rp = rp+follow*(transformed_p-rp)
    rr = Rotation.from_rotvec(follow*Rotation.from_matrix(transformed_r@rr.T).as_rotvec()).as_matrix()@rr
    lp = d.xpos[model.leaf]+d.xmat[model.leaf].reshape(3,3)@model.localp
    lr = d.xmat[model.leaf].reshape(3,3)@model.localr
    root_rotation = Rotation.from_quat(reference[model.rq+3:model.rq+7][[1,2,3,0]])
    preference = np.r_[reference[model.rq:model.rq+3]-model.initial[model.rq:model.rq+3],
        (root_rotation*model.initial_rotation.inv()).as_rotvec(),reference[model.qa]]
    return reference,rp,rr,lp,lr,preference


def _solve_map(model,preferences,*,progress=None):
    m,d = model.m,model.d;initial = model.initial;rq = model.rq;qa = model.qa
    source_coordinate = np.r_[np.zeros(6),initial[qa]]
    lower = np.minimum(np.r_[[-.03,-.03,-.03],[-.04,-.04,-.12],m.jnt_range[model.joints,0]+.01],source_coordinate)
    upper = np.maximum(np.r_[[.03,.03,.012],[.04,.04,.12],m.jnt_range[model.joints,1]-.01],source_coordinate)
    feet = [m.body('robot/'+side+'_ankle_link').id for side in ('left','right')]
    torso = m.body('robot/torso_link').id;body = int(m.jnt_bodyid[m.joint('robot/free_base').id])
    fp,fr,com = [np.asarray(model.c[name]) for name in ('initial_feet_positions','initial_feet_rotations','initial_com')]
    coordinates = np.zeros((len(model.elapsed),len(model.angles),len(source_coordinate)))
    residuals = np.zeros(coordinates.shape[:2]);converged = np.zeros(residuals.shape,dtype=bool)
    evaluations = np.zeros(residuals.shape,dtype=int)
    for j,angle in enumerate(model.angles):
        previous = None
        for i,t in enumerate(model.elapsed):
            reference,rp,rr,lp,lr,preference = _reference_targets(model,float(t),float(angle))
            neighbor = coordinates[i,j-1] if j else preference
            if previous is None:previous = neighbor.copy()
            def install(x):
                d.qpos[:] = reference;d.qpos[model.lq] = angle
                d.qpos[model.oq] = initial[model.oq];d.qpos[model.bq] = initial[model.bq]
                d.qpos[rq:rq+3] = initial[rq:rq+3]+x[:3]
                q = (Rotation.from_rotvec(x[3:6])*model.initial_rotation).as_quat()
                d.qpos[rq+3:rq+7] = q[[3,0,1,2]];d.qpos[qa] = x[6:]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            def residual(x):
                install(x)
                hands = np.r_[100*(d.site_xpos[model.lh]-lp),10*Rotation.from_matrix(lr@d.site_xmat[model.lh].reshape(3,3).T).as_rotvec(),
                    100*(d.site_xpos[model.rh]-rp),10*Rotation.from_matrix(rr@d.site_xmat[model.rh].reshape(3,3).T).as_rotvec()]
                foot = np.concatenate([np.r_[100*(d.xpos[b]-fp[k]),10*Rotation.from_matrix(fr[k]@d.xmat[b].reshape(3,3).T).as_rotvec()] for k,b in enumerate(feet)])
                tilt = np.arccos(np.clip(d.xmat[torso].reshape(3,3)[2,2],-1,1))
                barriers = 50*np.maximum([np.linalg.norm(x[:3])-.029,np.linalg.norm(d.subtree_com[body,:2]-com[:2])-.014,tilt-np.radians(3.9)],0.)
                return np.r_[hands,foot,.008*(x-preference),.006*(x-previous),.01*(x-neighbor),barriers]
            if i==0 and angle==initial[model.lq]:
                x = source_coordinate.copy();converged[i,j] = True
            else:
                seed = neighbor if j else previous
                fit = least_squares(residual,np.clip(seed,lower+1e-12,upper-1e-12),bounds=(lower,upper),
                    max_nfev=preferences['max_nfev'],ftol=1e-10,xtol=1e-10,gtol=1e-10)
                x = np.asarray(fit.x);converged[i,j] = bool(fit.success);evaluations[i,j] = int(fit.nfev)
            if x.shape != source_coordinate.shape or not np.isfinite(x).all():
                raise ValueError('Nonfinite/incomplete coupled solve cannot be exported')
            coordinates[i,j] = x;residuals[i,j] = np.linalg.norm(residual(x)[:24]);previous = x.copy()
            if not np.isfinite(residuals[i,j]):
                raise ValueError('Nonfinite coupled geometric residual cannot be exported')
            if progress is not None and (i+1)%20==0:
                progress(dict(column=j+1,columns=len(model.angles),completed_nodes=i+1,
                    nodes_in_column=len(model.elapsed),leaf_rad=float(angle),elapsed_s=float(t),
                    pose_residual=float(residuals[i,j])))
        if progress is not None:
            progress(dict(column=j+1,columns=len(model.angles),leaf_rad=float(angle),
                maximum_pose_residual=float(residuals[:,j].max()),unconverged_solves=int(np.count_nonzero(~converged[:,j]))))
    return coordinates,residuals,converged,evaluations


def generate_isaac_coupled_envelope(source_config,preferences=None,*,progress=None):
    """Return one freshly solved unadmitted tensor; never alter source files."""
    preferences = numeric_coupled_preferences({} if preferences is None else preferences)
    from . import (isaac_coupled_release_geometry,coupled_release_geometry,
        isaac_release_geometry_audit,landed_left_planner,operation_teacher)
    code = {str(Path(path).resolve()):digest(path) for path in [__file__]+[module.__file__ for module in
        (isaac_coupled_release_geometry,coupled_release_geometry,isaac_release_geometry_audit,
         landed_left_planner,operation_teacher)]}
    model = _seed_model(source_config,preferences)
    from .isaac_release_source_dispatch import source_paths
    code.update({str(path):digest(path) for path in source_paths(model.source_data.get('source_kind'))})
    try:
        coordinates,residuals,converged,evaluations = _solve_map(model,preferences,progress=progress)
        plan = copy.deepcopy(model.plan)
        plan.update(coordinates=coordinates.tolist(),pose_residual_norm=residuals.tolist(),
            solver_converged=converged.tolist(),solver_evaluations=evaluations.tolist(),
            unconverged_solves=int(np.count_nonzero(~converged)),solver_grid_points=int(residuals.size),
            interpolation='Original tensor cubic spline and bounded measured-frame arm correction',
            scope='Fresh unstepped actual-Isaac coupled geometry candidate. Numerical convergence is not geometric or physical admission; independent dense map audit and live physical gates remain required.')
        validate_map(plan,model.initial,model.qa,model.lq,model.oq,model.bq)
        validate_envelope_binding(plan,model.source_data,model.source_config_path)
        model.verify_inputs()
        if any(digest(name)!=expected for name,expected in code.items()):
            raise ValueError('Coupled planner/evaluator code changed during generation')
        plan['input_sha256'].update(code)
        return plan
    finally:
        model.close()
