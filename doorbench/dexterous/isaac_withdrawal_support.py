"""Continue the actual predecessor's bounded palm-only hybrid force feedback."""
import numpy as np


class InheritedIsaacPalmSupport:
    def __init__(self,transfer,target_N):
        if (type(target_N) not in (int,float) or not np.isfinite(target_N)
                or not 2 < target_N <= 8 or transfer.hybrid_support is not True
                or transfer.left.support_load_target != target_N):
            raise ValueError('Exact source-bound predecessor palm target and hybrid support required')
        self.transfer=transfer;self.left=transfer.left;self.target_N=float(target_N)
        self.feedback=None;self.started=None;self.previous=None;self.info={};self.failure=None

    def _fail(self,error):
        if self.failure is None:self.failure=str(error)
        raise ValueError('Inherited Isaac support stopped: '+self.failure)

    def begin(self,t):
        """Observe, never reset, the feedback created during the actual prefix."""
        if self.failure is not None:self._fail(self.failure)
        try:
            f=self.transfer.support_feedback
            if (self.started is not None or not np.isfinite(t) or f is None or f.left is not self.left
                    or f.maximum_target_N != self.target_N or self.left.support_load_target != self.target_N
                    or f.previous is None or abs(float(t)-float(f.previous[0])-.002)>1e-8
                    or f.started is None or not np.isfinite(f.started)):
                raise ValueError('One fresh actual predecessor feedback state required at entry')
            fields=('filtered_palm_load','hybrid_normal_target','hybrid_blend')
            if (not np.isfinite([getattr(self.left,key,float('nan')) for key in fields]).all()
                    or self.left.hybrid_normal_target != self.target_N
                    or not 0 <= self.left.hybrid_blend <= 1
                    or self.left.filtered_palm_load < 0):
                raise ValueError('Finite unchanged predecessor force/filter/blend required')
            self.feedback=f;self.started=float(t);self.previous=float(f.previous[0])
            self.info=dict(target_N=self.target_N,reused_predecessor_state=True,
                previous_feedback_epoch=self.previous,feedback_epoch=self.previous,
                inherited_feedback_started_s=float(f.started),entry_filtered_palm_load_N=float(self.left.filtered_palm_load),
                entry_hybrid_blend=float(self.left.hybrid_blend),support_surface='left_palm_only')
        except Exception as error:self._fail(error)

    def update(self,t,root,joints,leaf_pose,palm_load):
        if self.failure is not None:self._fail(self.failure)
        try:
            if (self.started is None or self.feedback is not self.transfer.support_feedback
                    or self.feedback.left is not self.left or not np.isfinite([t,palm_load]).all()
                    or palm_load < 0 or abs(float(t)-self.previous-.002)>1e-8
                    or float(self.feedback.previous[0]) != self.previous
                    or self.left.support_load_target != self.target_N
                    or self.feedback.maximum_target_N != self.target_N):
                raise ValueError('Consecutive same-target actual palm feedback interval required')
            previous=self.previous
            self.left._read(root,joints)
            self.feedback.update(float(t),leaf_pose,float(palm_load),self.target_N)
            if (self.left.hybrid_normal_target != self.target_N
                    or float(self.feedback.previous[0]) != float(t)):
                raise ValueError('Predecessor support update changed its target or epoch')
            self.previous=float(t)
            self.info.update(previous_feedback_epoch=previous,feedback_epoch=float(t),
                palm_only_load_N=float(palm_load),filtered_palm_load_N=float(self.left.filtered_palm_load),
                hybrid_blend=float(self.left.hybrid_blend),surface_velocity_world_m_s=np.asarray(self.left.surface_velocity_world).tolist(),
                normal_offset_m=float(self.left.offset))
            return self.info.copy()
        except Exception as error:self._fail(error)
