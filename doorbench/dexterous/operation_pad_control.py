"""Privileged fixed-material pad feedback through original robot motors.

An experimental teacher controller, not sensor-only policy input. It reduces
finger posture competition while following the palm's commanded lever motion.
"""
import numpy as np
from .pad_tracking import PadTracker
from .operation_teacher import pose_components, reproject_grasp, smooth_phase


class OperationPadControl:
    def __init__(self, teacher, handle_pose, *, profile="commanded-material-v1"):
        if profile not in ("commanded-material-v1","actual-material-v1"):raise ValueError("Unknown material contact profile")
        self.profile=profile
        self.teacher=teacher
        distal={digit:[g for g in geoms if teacher.m.body(teacher.m.geom_bodyid[g]).name.endswith('distal')]
                for digit,geoms in teacher.digit_geoms.items()}
        self.tracker=PadTracker(teacher.m,teacher.d,distal,teacher.lever,
            digits=('ff','mf','rf','lf','th'),stiffness=1200.,damping=3.,maximum_force=6. if profile=="actual-material-v1" else 12.)
        hp,hr=pose_components(handle_pose)
        self.relative={digit:hr.T@(position-hp) for digit,position in self.tracker.positions(teacher.d).items()}

    def force(self, forces, elapsed, handle_pose, leaf_pose, angles, goals, geometry):
        teacher=self.teacher;blend=float(smooth_phase(elapsed))
        # Palm recentering changes finger posture, not the material contact
        # targets on the lever. Translating these by the palm offset moves them
        # off the physical cylinder and defeats contact feedback.
        targets={digit:reproject_grasp(handle_pose,leaf_pose,angles,goals,relative,np.eye(3),geometry)[0]
                 for digit,relative in self.relative.items()}
        actual=getattr(self,'profile','commanded-material-v1')=='actual-material-v1'
        if actual:
            hp,hr=pose_components(handle_pose)
            targets={digit:hp+hr@relative for digit,relative in self.relative.items()}
        generalized,errors=self.tracker.generalized_force(teacher.d,targets)
        length=teacher.matrix@teacher.d.qpos[teacher.qa]
        # Relax only the finger positional servo, retaining its velocity damping
        # and the original acquisition controller's distal normal preload.
        posture=(teacher.kp*teacher.target+teacher.bias[:,0]+teacher.bias[:,1]*length
                 +teacher.kp*teacher.gain*(teacher.target-length))
        result=np.asarray(forces,float).copy()
        result[teacher.fingers]-=.8*blend*posture[teacher.fingers]
        result[teacher.fingers]+=blend*(teacher.finger_inverse@generalized[teacher.va])
        if not actual:result[teacher.arm_motors]+=blend*(teacher.arm_inverse@generalized[teacher.va])
        result=np.clip(result,teacher.caps[:,0],teacher.caps[:,1])
        if not np.isfinite(result).all():raise ValueError('Nonfinite contact-controller command')
        teacher.last_force=result.copy()
        return result,dict(operation_pad_control='actual-material-v1' if actual else 'fixed-material-v1',pad_tracking_errors_m=errors,
                           finger_posture_relaxation_fraction=.8*blend,pad_feedback_blend=blend)
