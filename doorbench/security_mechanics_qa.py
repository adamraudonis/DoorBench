"""Native inside-service tests for the original contact-operated security guards.

This is a mechanical test fixture, not a robot-access or benchmark result.
The ordinary latch is held retracted; the guard is tested independently. Every
motion follows mj_step under bounded loads, without intermediate qpos writes.
"""
from __future__ import annotations

import math
import numpy as np
import copy
import hashlib
import json
from collections import OrderedDict

_SERVICE_CACHE=OrderedDict()


def _model_sha256(model):
    """Fingerprint native model storage, not a caller's XML filename."""
    import hashlib
    import mujoco
    buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def run_security_service_qa(model, metadata, *, opening_preload=0., opening_stiffness=0.):
    """Process-local reuse bound to the exact native model, metadata and inputs.

    No receipt from another process or source implementation is reused. This
    avoids repeating the same native cycles for geometry and operation gates.
    """
    from .closer_pinion import compile_pinion_closers,apply_pinion_closers
    from .closer_track_hold import compile_track_holds,apply_track_holds
    digest=hashlib.sha256((_model_sha256(model)+json.dumps({'meta':metadata,
        'preload':opening_preload,'stiffness':opening_stiffness},sort_keys=True)).encode()).hexdigest()
    key=(digest,run_security_mechanics_qa,compile_pinion_closers,apply_pinion_closers,
         compile_track_holds,apply_track_holds)
    if key in _SERVICE_CACHE:
        _SERVICE_CACHE.move_to_end(key);result=copy.deepcopy(_SERVICE_CACHE[key]);result['cache_hit']=True;return result
    result=_run_security_service_qa(model,metadata,opening_preload=opening_preload,opening_stiffness=opening_stiffness)
    result['cache_hit']=False;_SERVICE_CACHE[key]=copy.deepcopy(result)
    while len(_SERVICE_CACHE)>12:_SERVICE_CACHE.popitem(last=False)
    return result


def _run_security_service_qa(model, metadata, *, opening_preload=0., opening_stiffness=0.):
    """Run the inside-service fixture on a private model and restore callbacks.

    Automatic actuator gains/biases are disabled only on the copy; authored
    pinion closers and track-hold electrical fields stay active. This owns the
    process-global passive callback synchronously, like other native QA tools;
    simultaneous MuJoCo calls in other threads are not supported.
    """
    if not metadata.get('security_guards'):
        return run_security_mechanics_qa(model,metadata,
            opening_preload=opening_preload,opening_stiffness=opening_stiffness)
    import copy
    import mujoco
    from .closer_pinion import compile_pinion_closers,apply_pinion_closers
    from .closer_track_hold import compile_track_holds,apply_track_holds
    before=_model_sha256(model)
    private=copy.copy(model);private_metadata=copy.deepcopy(metadata)
    pinions=compile_pinion_closers(private,private_metadata)
    tracks=compile_track_holds(private,private_metadata)
    disabled=[{'name':private.actuator(k).name,
        'old_gain':private.actuator_gainprm[k].tolist(),
        'old_bias':private.actuator_biasprm[k].tolist()} for k in range(private.nu)]
    private.actuator_gainprm[:]=0.;private.actuator_biasprm[:]=0.
    previous=mujoco.get_mjcb_passive()
    def callback(native,data):
        if native is private:
            apply_pinion_closers(native,data,pinions)
            apply_track_holds(native,data,tracks)
        elif previous is not None:
            previous(native,data)
    try:
        mujoco.set_mjcb_passive(callback)
        result=run_security_mechanics_qa(private,private_metadata,
            opening_preload=opening_preload,opening_stiffness=opening_stiffness)
    finally:
        mujoco.set_mjcb_passive(previous)
    after=_model_sha256(model)
    result['service_fixture']={'private_model':True,'disabled_actuators':disabled,
        'pinion_closer_count':len(pinions),'track_hold_count':len(tracks),
        'source_model_mjb_sha256_before':before,'source_model_mjb_sha256_after':after,
        'source_model_unchanged':before==after,
        'scope':'Inside-service isolation with ordinary latch withdrawn and automatic drives disabled only on a private model; authored passive closer/track fields retained. Not robot access, powered operation or security strength certification.'}
    if before!=after:
        result['ok']=False
        result.setdefault('failures',[]).append('Source native model changed during security service QA')
    return result


class _CycleFailure(ValueError):
    def __init__(self,message,measurement):
        super().__init__(message);self.measurement=measurement


def _chain_neck_slot_clearance(m,d,record):
    """Actual cylinder section through both keeper-lip faces, in leaf space.

    A correct head height alone does not establish clearance for a tilted
    neck. This measures its elliptical section at each lip face and requires
    the finite cylinder to span both faces before lateral slot engagement.
    """
    import mujoco
    leaf=m.body(record['leaf_body']).id;neck=m.geom(record['neck_geom']).id
    lips=[m.geom(name).id for name in record['keeper_geoms']
          if '_keeper_lip_' in name and '_keyhole_' not in name]
    if (m.geom_type[neck]!=mujoco.mjtGeom.mjGEOM_CYLINDER or not lips or
        any(m.geom_bodyid[g]!=leaf or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX or
            not np.allclose(m.geom_quat[g],(1,0,0,0),rtol=0,atol=1e-12) for g in lips)):
        raise ValueError('Chain slot clearance requires the actual cylinder and leaf-mounted rectangular lips')
    z=record['keyhole_center_leaf'][2]
    upper=[m.geom_pos[g,2]-m.geom_size[g,2] for g in lips if m.geom_pos[g,2]>z]
    lower=[m.geom_pos[g,2]+m.geom_size[g,2] for g in lips if m.geom_pos[g,2]<z]
    if not upper or not lower:raise ValueError('Chain retaining slot lacks both physical lips')
    upper=min(upper);lower=max(lower)
    rotation=d.xmat[leaf].reshape(3,3).T
    center=rotation@(d.geom_xpos[neck]-d.xpos[leaf])
    axis=rotation@d.geom_xmat[neck].reshape(3,3)[:,2]
    if abs(axis[1])<1e-8:
        return {'spans_lip':False,'minimum_clearance_m':None,'neck_axis_leaf':axis.tolist(),'faces':[]}
    faces=sorted({float(m.geom_pos[g,1]+sign*m.geom_size[g,1]) for g in lips for sign in(-1,1)})
    radius=float(m.geom_size[neck,0])*math.sqrt(1+(axis[2]/axis[1])**2)
    rows=[]
    for y in faces:
        t=float((y-center[1])/axis[1]);height=center[2]+t*axis[2]
        rows.append({'face_y_m':y,'axis_parameter_m':t,
                     'clearance_m':float(min(upper-height,height-lower)-radius)})
    return {'spans_lip':all(abs(row['axis_parameter_m'])<=m.geom_size[neck,1] for row in rows),
            'minimum_clearance_m':min(row['clearance_m'] for row in rows),
            'neck_axis_leaf':axis.tolist(),'faces':rows}


def run_security_mechanics_qa(model, metadata, *, opening_preload=0., opening_stiffness=0.):
    import mujoco
    from .native_warnings import capture_native_warnings
    records=metadata.get('security_guards',[])
    if not records:return {'ok':True,'applicable':False,'failures':[]}
    failures=[];measurements=[]
    with capture_native_warnings() as messages:
        for record in records:
            try:
                result=_exercise(mujoco,model,metadata,record,float(opening_preload),float(opening_stiffness))
                measurements.append(result);failures.extend(result.get('failures',[]))
            except (ValueError,KeyError,AssertionError) as exc:
                failures.append(str(exc))
                if isinstance(exc,_CycleFailure):measurements.append(exc.measurement)
    failures.extend('Native warning: '+message for message in messages)
    return {'ok':not failures,'applicable':True,'failures':failures,'measurements':measurements,
        'native_warning_messages':list(messages),
        'scope':'Inside-service mechanical fixture with ordinary latch withdrawn; not robot approach accessibility, security strength certification or benchmark success.'}


def _exercise(mj,m,meta,r,preload,stiffness):
    if not all(math.isfinite(x) and x>=0 for x in (preload,stiffness)):raise ValueError('Invalid fixture effort coefficients')
    d=mj.MjData(m);mj.mj_forward(m,d)
    leaf=m.body(r['leaf_body']).id;j=m.joint(meta['primary_joint']).id;qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j])
    if not m.jnt_limited[j] or m.jnt_range[j,1]<.75:raise ValueError('Guard proof requires full leaf travel, not an artificial security range')
    ceiling=.000025 if r['kind']=='chain' else .0001
    if m.opt.timestep>ceiling+1e-12:raise ValueError(f'Guard contact proof needs authored {ceiling:g} s maximum timestep')
    if r['kind']=='chain' and m.opt.integrator!=mj.mjtIntegrator.mjINT_IMPLICIT:
        raise ValueError('Coupled chain proof requires full implicit integration')
    if r['kind']=='chain':
        declared=r.get('adjacent_wire_contact_pairs')
        if not declared:raise ValueError('Chain requires explicit adjacent wire contacts')
        prefix=r['head_geom'].removesuffix('_head_ball')
        expected={frozenset((prefix+f'_anchor_eye_{a}',prefix+f'_link_0_wire_{b}')) for a in range(4) for b in range(4)}
        expected|={frozenset((prefix+f'_link_{i}_wire_{a}',prefix+f'_link_{i+1}_wire_{b}'))
                   for i in range(7) for a in range(4) for b in range(4)}
        compiled={frozenset((int(a),int(b))) for a,b in zip(m.pair_geom1,m.pair_geom2)}
        observed=set()
        for pair in declared:
            if set(pair)!={'geom1','geom2'} or pair['geom1']==pair['geom2']:
                raise ValueError('Malformed adjacent wire contact pair')
            names=[pair['geom1'],pair['geom2']]
            observed.add(frozenset(names))
            if frozenset(m.geom(name).id for name in names) not in compiled:
                raise ValueError('Missing adjacent wire contact override: '+' / '.join(names))
        if len(declared)!=128 or observed!=expected:
            raise ValueError('Incomplete or duplicated adjacent wire contact inventory')
        scope=r.get('contact_solver_scope',{})
        steel={m.geom(g).name for g in range(m.ngeom) if m.geom(g).name.startswith(prefix)}
        names=scope.get('priority_geoms',[])
        if (scope.get('priority')!=1 or len(names)!=len(steel) or set(names)!=steel or
            scope.get('solref_s')!=[.0002,1.] or scope.get('solimp')!=[.95,.95,.0001]):
            raise ValueError('Incomplete stiff steel chain contact material declaration')
        if any(m.geom_priority[m.geom(name).id]!=1 or
               ((m.geom_contype[m.geom(name).id] or m.geom_conaffinity[m.geom(name).id]) and
                (not np.allclose(m.geom_solref[m.geom(name).id],(.0002,1.),rtol=0,atol=1e-12) or
                 not np.allclose(m.geom_solimp[m.geom(name).id,:3],(.95,.95,.0001),rtol=0,atol=1e-12)))
               for name in names):
            raise ValueError('Native steel chain contact material differs from declared priority/stiffness')
    for name in r['keeper_geoms']+[r['head_geom']]:
        g=m.geom(name).id
        if not(m.geom_contype[g] or m.geom_conaffinity[g]):raise ValueError(f'Guard contact disabled: {name}')
    for name in r['guard_joints']:
        k=m.joint(name).id;v=m.jnt_dofadr[k];b=m.jnt_bodyid[k]
        if m.dof_armature[v]!=0 or m.body_mass[b]<=0 or min(m.body_inertia[b])<=0:
            raise ValueError(f'Guard {name} lacks physical-only positive inertia')
    if r['kind']=='chain' and not r['slot_width_m']<r['head_diameter_m']<r['keyhole_width_m']:
        raise ValueError('Chain head must be retained by slot and pass the release opening')
    if r['kind']=='chain':
        handoff={'keyhole_lateral':(r['keyhole_width_m']-r['head_diameter_m'])/2-.0001,
                 'slot_vertical':(r['slot_width_m']-r['neck_diameter_m'])/2-.0001,
                 'seated_position':.003}
        if r.get('handoff_tolerances_m')!=handoff:
            raise ValueError('Chain handoff tolerances must match physical slot/keyhole clearances')
    sid=m.site(r['release_site']).id;graspbody=int(m.site_bodyid[sid]);head=m.geom(r['head_geom']).id
    guardswing=m.joint(r['guard_joint']).id if r['kind']=='swing_bar_guard' else None
    latch_targets={k:float(m.jnt_range[k,1]) for k in range(m.njnt)
                   if any(tag in m.joint(k).name for tag in ('handle_hinge','latch_bolt_slide')) and m.jnt_limited[k]}
    worst=0.;pair=None;max_effort=0.;max_force=0.;max_torque=0.;rows=[];inspection=[]
    velocity_peaks=np.zeros(m.nv)
    jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));velocity=np.zeros(6);rotation_error=np.zeros(4)
    def leafpoint(local):return d.xpos[leaf]+d.xmat[leaf].reshape(3,3)@np.asarray(local,float)
    calibrated=[r['table'] for r in meta.get('closer_pinion_calibration',[]) if r['leaf_joint']==meta['primary_joint']]
    def closing_bias():
        angle=max(0,float(d.qpos[qa]))
        if calibrated:
            return sum(float(np.interp(angle,t['door_angle_rad'],t['achieved_door_torque_Nm'])) for t in calibrated)
        return preload+stiffness*angle
    def phase(name,duration,door,head_path=None,bar_path=None):
        nonlocal worst,pair,max_effort,max_force,max_torque
        before=float(d.qpos[qa]);peak=before;grasp_error=0.
        for i in range(math.ceil(duration/m.opt.timestep)):
            fraction=min(1,(i+.5)*m.opt.timestep/duration);d.qfrc_applied[:]=0
            for k,target in latch_targets.items():
                q,v=int(m.jnt_qposadr[k]),int(m.jnt_dofadr[k])
                d.qfrc_applied[v]=np.clip(500*(target-d.qpos[q])-8*d.qvel[v],-20,20)
            if door=='open':
                effort=min(100.,closing_bias()+8.)
                # Brake near the actual range end, with bounded fixture force.
                remaining=max(0,float(m.jnt_range[j,1])-.025-d.qpos[qa])
                desired=min(.30,math.sqrt(2*.30*remaining))
                effort=min(effort,closing_bias()+35*(desired-d.qvel[va]))
            else:
                desired=float(np.clip(8*(.0002-d.qpos[qa]),-.30,.05))
                effort=float(np.clip(closing_bias()+250*(desired-d.qvel[va]),-100,100))
            d.qfrc_applied[va]=effort;max_effort=max(max_effort,abs(effort))
            if head_path is not None:
                start,end=head_path;goal=leafpoint(np.asarray(start)+(np.asarray(end)-start)*fraction)
                wanted=d.xmat[leaf].reshape(3,3);actual=d.xmat[graspbody].reshape(3,3)
                # Hold the actual grip surface at the position implied by the
                # desired head pose. Head-centre feedback applied 14 mm away
                # creates a competing moment and can fold the loose fitting.
                if int(m.geom_bodyid[head])!=graspbody:
                    raise ValueError('Chain head and physical grip must share a rigid body')
                grip_goal=goal+wanted@(m.site_pos[sid]-m.geom_pos[head])
                mj.mj_jacSite(m,d,jp,jr,sid);vel=jp@d.qvel
                force=np.clip(1000*(grip_goal-d.site_xpos[sid])-6*vel,-20,20)
                mj.mju_mat2Quat(rotation_error,(wanted@actual.T).reshape(-1))
                if rotation_error[0]<0:rotation_error[:]*=-1
                length=float(np.linalg.norm(rotation_error[1:]));angle=2*math.atan2(length,rotation_error[0])
                error=rotation_error[1:]*(angle/max(length,1e-12))
                mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,graspbody,velocity,0)
                torque=.2*error-.0002*velocity[:3]
                torque*=min(1.,.02/max(float(np.linalg.norm(torque)),1e-12))
                mj.mj_applyFT(m,d,force,torque,d.site_xpos[sid],graspbody,d.qfrc_applied)
                grasp_error=max(grasp_error,float(np.linalg.norm(goal-d.geom_xpos[head])))
                max_force=max(max_force,float(np.linalg.norm(force)));max_torque=max(max_torque,float(np.linalg.norm(torque)))
            if bar_path is not None:
                q,v=m.jnt_qposadr[guardswing],m.jnt_dofadr[guardswing]
                target=bar_path[0]+(bar_path[1]-bar_path[0])*fraction
                d.qfrc_applied[v]=np.clip(.5*(target-d.qpos[q])-.02*d.qvel[v],-.25,.25)
            # MuJoCo resets state after a severe acceleration warning. Keep
            # the immediately preceding coordinates so a failed receipt does
            # not misleadingly report the newly reset home configuration.
            previous_qpos=d.qpos.copy();previous_qvel=d.qvel.copy()
            mj.mj_step(m,d);peak=max(peak,float(d.qpos[qa]))
            if i%max(1,round(.1/m.opt.timestep))==0:
                inspection.append({'phase':name,'time_s':float(d.time),'qpos':d.qpos.tolist()})
            np.maximum(velocity_peaks,np.abs(d.qvel),out=velocity_peaks)
            for c in d.contact:
                if -c.dist>worst:worst=-float(c.dist);pair=[m.geom(k).name for k in c.geom]
            if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):
                raise _CycleFailure(f'{name}: nonfinite state or MuJoCo warning',
                    {'kind':r['kind'],'duration_s':float(d.time),'phases':rows,
                     'failed_phase':name,'failed_phase_step':i,'max_penetration_m':worst,
                     'worst_pair':pair,'head_center_leaf':localhead().tolist(),
                     'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),
                     'qpos_before_failed_step':previous_qpos.tolist(),
                     'qvel_before_failed_step':previous_qvel.tolist(),
                     'contacts_at_failure':[{'pair':[m.geom(k).name for k in c.geom],
                         'distance_m':float(c.dist)} for c in d.contact]})
        row={'phase':name,'start_angle_rad':before,'end_angle_rad':float(d.qpos[qa]),'max_angle_rad':peak,'max_head_tracking_error_m':grasp_error,
             'max_penetration_so_far_m':worst,'worst_pair_so_far':pair}
        if head_path is not None:
            # mj_step leaves position-dependent fields one integration step
            # behind qpos. Refresh kinematics for the terminal measurement.
            mj.mj_kinematics(m,d)
            desired=leafpoint(head_path[1])
            relative=d.xmat[leaf].reshape(3,3)@d.xmat[graspbody].reshape(3,3).T
            row.update(terminal_head_error_m=float(np.linalg.norm(desired-d.geom_xpos[head])),
                       terminal_orientation_error_rad=float(math.acos(np.clip((np.trace(relative)-1)/2,-1,1))),
                       terminal_neck_axis_error_rad=float(math.acos(np.clip(d.xmat[leaf].reshape(3,3)[:,1]@d.xmat[graspbody].reshape(3,3)[:,1],-1,1))),
                       terminal_head_center_leaf=localhead().tolist(),
                       terminal_contacts=[{'pair':[m.geom(k).name for k in c.geom],'distance_m':float(c.dist)} for c in d.contact if c.dist<.001])
        inspection.append({'phase':name,'time_s':float(d.time),'qpos':d.qpos.tolist()})
        rows.append(row);return row
    def localhead():return (d.xmat[leaf].reshape(3,3).T@(d.geom_xpos[head]-d.xpos[leaf])).copy()
    def require(condition,message):
        if not condition:
            raise _CycleFailure(message,{'kind':r['kind'],'duration_s':float(d.time),'phases':rows,
                'max_penetration_m':worst,'worst_pair':pair,'head_center_leaf':localhead().tolist(),
                'contacts_at_failure':[{'pair':[m.geom(k).name for k in c.geom],'distance_m':float(c.dist)} for c in d.contact]})
    def settle_keyhole(key,entering_slot=False):
        phase('align_at_keyhole',.5,'closed',(localhead(),key))
        # The spherical head must fit the actual enlarged opening before
        # commanding withdrawal; a completed timer alone is not release.
        lateral=float(np.linalg.norm((localhead()-key)[[0,2]]))
        require(lateral<handoff['keyhole_lateral'],f'Chain head missed keyhole: lateral residual {lateral:.6f} m')
        if entering_slot:
            error=abs(float(localhead()[2]-key[2]))
            require(error<handoff['slot_vertical'],f'Chain neck missed narrow slot: vertical residual {error:.6f} m')
            clearance=_chain_neck_slot_clearance(m,d,r)
            rows[-1]['neck_slot_clearance']=clearance
            require(clearance['spans_lip'] and clearance['minimum_clearance_m']>.0001,
                    'Chain neck is not aligned through both retaining lip faces with 0.1 mm clearance')
    def insert_chain(front,key,near):
        direction=np.asarray(r['release_sequence'][2]['direction'])
        def reachable_head(goal):
            rotation=d.xmat[leaf].reshape(3,3)
            origin=np.asarray(r['frame_anchor_world'])
            pivot=leafpoint(goal)-rotation@np.asarray(r['head_center_local'])
            radius=pivot-origin
            pivot=origin+radius*min(1.,(r['chain_length_m']-.002)/max(float(np.linalg.norm(radius)),1e-12))
            return rotation.T@(pivot+rotation@np.asarray(r['head_center_local'])-d.xpos[leaf])
        def reach_aligned(name,goal,duration):
            row=phase(name,duration,'closed',(localhead(),goal))
            for attempt in range(8):
                if row['terminal_head_error_m']<.003 and row['terminal_orientation_error_rad']<.15:
                    return
                row=phase(name+'_hold',.5,'closed',(localhead(),goal))
            require(False,name+': head did not reach a clear aligned pose before insertion')
        # Lift the dangling fitting away from the jamb, then carry it through
        # a clear plane before approaching the hole. Every phase starts at the
        # measured native position; no pose reset or timer-only insertion.
        outward=reachable_head(localhead()+direction*.040+[0,0,.015])
        reach_aligned('clear_regrasp',outward,2.)
        clear=reachable_head(key+direction*.060)
        require(float(np.dot(clear-key,direction))>.045,'Insufficient chain reach for a clear regrasp plane')
        reach_aligned('position_for_insertion',clear,2.5)
        lateral=float(np.linalg.norm((localhead()-key)[[0,2]]))
        require(lateral<handoff['keyhole_lateral'],'Chain head is not aligned with keyhole before approach')
        phase('insert_head',3.,'closed',(localhead(),key))
        settle_keyhole(key,entering_slot=True)
        phase('engage_slot',2.,'closed',(localhead(),near))
        error=float(np.linalg.norm(localhead()-near))
        require(error<handoff['seated_position'],f'Chain head did not reach retaining slot: residual {error:.6f} m')
    phase('settle',.5,'closed')
    initial=phase('initial_load',6.,'open')
    if r['engaged_initial'] and not .005<initial['max_angle_rad']<.20:raise ValueError(f"Engaged guard does not allow-and-limit partial opening: {initial['max_angle_rad']:.6f} rad")
    require(r['engaged_initial'] or initial['end_angle_rad']>=.30,'Disengaged installed guard blocks opening')
    phase('close_for_service',10.,'closed')
    if abs(d.qpos[qa])>.003:raise ValueError(f'Fixture cannot close leaf before servicing guard: {d.qpos[qa]:.6f} rad')
    # Both mechanisms are exercised through release and re-engagement, even
    # when the benchmark approach is outside and therefore cannot service it.
    for repetition in range(2):
        if r['kind']=='chain':
            near=np.asarray(r['seated_head_center_leaf'])
            key=np.asarray(r['keyhole_center_leaf']);front=key+np.asarray(r['release_sequence'][2]['direction'])*.025
            if repetition==0 and not r['engaged_initial']:
                insert_chain(front,key,near)
            else:
                current=localhead();phase('grasp_align',.5,'closed',(current,current))
            phase('slide_to_keyhole',2.,'closed',(localhead(),key))
            settle_keyhole(key)
            phase('withdraw_head',1.5,'closed',(localhead(),front));phase('hold_removed',.5,'closed',(localhead(),front))
        else:
            current=float(d.qpos[m.jnt_qposadr[guardswing]])
            phase('park_bar',3.,'closed',bar_path=(current,math.pi));phase('bar_parked',.5,'closed',bar_path=(math.pi,math.pi))
            if abs(d.qpos[m.jnt_qposadr[guardswing]]-math.pi)>.01:raise ValueError('Bar cannot be manually parked while closed')
        opened=phase('released_open',8.,'open')
        require(opened['end_angle_rad']>=.40,f"Released guard still blocks opening: {opened['end_angle_rad']:.6f} rad")
        phase('close_for_reengagement',10.,'closed')
        if abs(d.qpos[qa])>.003:raise ValueError('Leaf did not close before re-engagement')
        if r['kind']=='chain':
            insert_chain(front,key,near)
        else:
            phase('engage_bar',3.,'closed',bar_path=(math.pi,0.));phase('bar_engaged',.5,'closed',bar_path=(0.,0.))
        loaded=phase('reengaged_load',3.,'open')
        if not .005<loaded['max_angle_rad']<.20:raise ValueError(f"Re-engaged guard does not retain leaf: {loaded['max_angle_rad']:.6f} rad")
        phase('close_after_retention',3.,'closed')
    return {'kind':r['kind'],'duration_s':float(d.time),'phases':rows,'max_penetration_m':worst,'worst_pair':pair,
            'inspection_samples':inspection,
            'max_leaf_fixture_torque_Nm':max_effort,'max_grasp_force_N':max_force,'max_grasp_torque_Nm':max_torque,
            'peak_absolute_joint_velocity_SI':{m.joint(k).name:float(velocity_peaks[m.jnt_dofadr[k]]) for k in range(m.njnt)},
            'failures':[f'Guard cycle penetration {worst:.6f} m exceeds1mm: {pair}'] if worst>.001 else []}
