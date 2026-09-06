"""One-time reset helpers; never project coordinates during native simulation."""
import numpy as np


def prepare_vault_open_fixture(model,qpos,meta):
    """Prepare a geometrically consistent already-open scenario at reset.

    This is a prescribed fixture, not evidence that an agent unlocked a vault.
    Native episodes never project its closed-loop coordinates after reset.
    """
    from .geometry.vault_hardware import resolve_vault_configuration
    for row in meta['vault_boltwork']['groups']:
        address=int(model.jnt_qposadr[model.joint(row['operator_joint']).id])
        qpos[address]=row['operator_nominal_range'][1]
    resolve_vault_configuration(model,qpos,meta)
    return {'kind':'prescribed_open_fixture','native_release_history':False,
            'scope':'Already-open scenario with every bolt withdrawn and crank/rod loops resolved once at reset.'}


def resolve_joint_followers(model,qpos,drivers):
    """Set active polynomial followers of prescribed scalar driver joints.

    MuJoCo polynomials operate on displacement from qpos0. This is needed for
    a rising hinge when an episode starts with its leaf already open.
    """
    import mujoco
    active={model.joint(name).id for name in drivers}
    resolved=set()
    for _ in range(model.neq):
        changed=False
        for e in range(model.neq):
            if e in resolved or not model.eq_active0[e] or int(model.eq_type[e])!=int(mujoco.mjtEq.mjEQ_JOINT):continue
            child,parent=int(model.eq_obj1id[e]),int(model.eq_obj2id[e])
            if parent not in active:continue
            ca,pa=int(model.jnt_qposadr[child]),int(model.jnt_qposadr[parent])
            x=float(qpos[pa]-model.qpos0[pa]);coef=model.eq_data[e,:5]
            qpos[ca]=model.qpos0[ca]+np.polynomial.polynomial.polyval(x,coef)
            active.add(child);resolved.add(e);changed=True
        if not changed:break
    return qpos
