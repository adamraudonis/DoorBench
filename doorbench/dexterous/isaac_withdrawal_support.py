"""Continue the actual predecessor's bounded palm-only hybrid force feedback."""
import numpy as np


PALM_LOAD_PROFILE='release-unload-restore-v1'


def validate_palm_load_profile_name(profile):
    if type(profile) is not str or profile!=PALM_LOAD_PROFILE:
        raise ValueError('Unknown explicit withdrawal palm-load profile')
    return profile


class ReleaseQualifiedPalmLoadProfile:
    """A prospective target schedule, never a measured leaf-speed limit.

    Source identity and the existing hybrid feedback maximum stay at 6 N.
    Only its active numeric request changes after the controller's actual
    qualified release event. No reference, filter or motor state is installed.
    """
    def __init__(self,profile,target_N,duration_s):
        validate_palm_load_profile_name(profile)
        if (type(target_N) not in (int,float) or not np.isfinite(target_N) or target_N!=6.
                or type(duration_s) not in (int,float) or not np.isfinite(duration_s)
                or duration_s<=1.5):
            raise ValueError('Palm-load profile requires source 6 N and finite duration above 1.5 s')
        self.name=profile;self.duration_s=float(duration_s);self.release_started=None

    @staticmethod
    def _smooth(value):
        u=float(np.clip(value,0.,1.))
        return u*u*u*(10.+u*(-15.+6.*u))

    def target(self,t,entry,release_started):
        if (type(t) not in (int,float) or type(entry) not in (int,float)
                or not np.isfinite([t,entry]).all() or t<entry):
            raise ValueError('Finite physical withdrawal profile clock required')
        complete=entry+self.duration_s;restore_start=complete-1.
        if not np.isfinite([restore_start,complete]).all():
            raise ValueError('Finite derived withdrawal profile epochs required')
        if self.release_started is None:
            if release_started is not None:
                if (type(release_started) not in (int,float) or not np.isfinite(release_started)
                        or release_started!=t or release_started+.5>restore_start):
                    raise ValueError('Fresh actual release event with nonoverlapping palm ramps required')
                self.release_started=float(release_started)
        elif (type(release_started) not in (int,float) or not np.isfinite(release_started)
                or release_started!=self.release_started):
            raise ValueError('Actual qualified release event cannot disappear or change')
        if self.release_started is None:
            target=6.;phase='awaiting_qualified_release'
        elif t<self.release_started+.5:
            target=6.-3.5*self._smooth((t-self.release_started)/.5);phase='unloading_after_release'
        elif t<restore_start:
            target=2.5;phase='withdrawal_low_load'
        else:
            target=2.5+3.5*self._smooth(t-restore_start)
            phase='restoring_terminal_support' if t<complete else 'terminal_support'
        return target,dict(palm_load_profile=self.name,active_target_N=target,
            inherited_source_target_N=6.,inherited_maximum_target_N=6.,
            palm_load_phase=phase,qualified_release_started_s=self.release_started,
            unload_duration_s=.5,restore_started_s=restore_start,restore_duration_s=1.,
            withdrawal_complete_s=complete,leaf_speed_limit_claimed=False)


class InheritedIsaacPalmSupport:
    def __init__(self,transfer,target_N,*,profile=None,duration_s=None):
        if (type(target_N) not in (int,float) or not np.isfinite(target_N)
                or not 2 < target_N <= 8 or transfer.hybrid_support is not True
                or transfer.left.support_load_target != target_N):
            raise ValueError('Exact source-bound predecessor palm target and hybrid support required')
        self.transfer=transfer;self.left=transfer.left;self.target_N=float(target_N)
        self.feedback=None;self.started=None;self.previous=None;self.info={};self.failure=None
        self.profile=(None if profile is None else ReleaseQualifiedPalmLoadProfile(profile,target_N,duration_s))
        self.active_target_N=float(target_N)

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

    def update(self,t,root,joints,leaf_pose,palm_load,*,release_started_s=None):
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
            target=self.target_N;profile_info=None
            if self.profile is not None:
                if self.left.hybrid_normal_target!=self.active_target_N:
                    raise ValueError('Active inherited palm request changed outside the selected profile')
                target,profile_info=self.profile.target(float(t),self.started,release_started_s)
            self.left._read(root,joints)
            self.feedback.update(float(t),leaf_pose,float(palm_load),target)
            if (self.left.hybrid_normal_target != target
                    or float(self.feedback.previous[0]) != float(t)):
                raise ValueError('Predecessor support update changed its target or epoch')
            self.previous=float(t)
            self.active_target_N=target
            self.info.update(previous_feedback_epoch=previous,feedback_epoch=float(t),
                palm_only_load_N=float(palm_load),filtered_palm_load_N=float(self.left.filtered_palm_load),
                hybrid_blend=float(self.left.hybrid_blend),surface_velocity_world_m_s=np.asarray(self.left.surface_velocity_world).tolist(),
                normal_offset_m=float(self.left.offset))
            if profile_info is not None:self.info.update(profile_info)
            return self.info.copy()
        except Exception as error:self._fail(error)
