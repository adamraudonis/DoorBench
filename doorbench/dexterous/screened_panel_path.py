"""Smooth interpolation of an unstepped panel path, with explicit derivatives.

This is a geometric target generator, not a controller or a dynamics model.
Coordinates are translational/root-rotation-vector deltas followed by scalar
joint angles. The caller retains the actual root quaternion used as the origin.
"""
import numpy as np
from scipy.interpolate import CubicSpline


class ScreenedPanelPath:
    def __init__(self, progress, coordinates, duration_s):
        x = np.asarray(progress, float)
        y = np.asarray(coordinates, float)
        if (x.ndim != 1 or len(x) < 4 or y.ndim != 2 or len(y) != len(x)
                or not np.isfinite(np.r_[x, y.ravel(), duration_s]).all()
                or x[0] != 0 or x[-1] != 1 or not np.all(np.diff(x) > 0)
                or duration_s < 3.):
            raise ValueError('Require a finite complete path and duration >=3 seconds')
        self.duration_s = float(duration_s)
        self.spline = CubicSpline(x, y, axis=0)

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError('Require a finite path clock')
        u = float(np.clip(time_s / self.duration_s, 0., 1.))
        s = u**3 * (10. + u * (-15. + 6.*u))
        ds = 30.*u*u*(1.-u)**2 / self.duration_s
        dds = 60.*u*(1.-u)*(1.-2.*u) / self.duration_s**2
        if time_s < 0 or time_s > self.duration_s:
            ds = dds = 0.
        position = self.spline(s)
        velocity = self.spline(s, 1)*ds
        acceleration = self.spline(s, 2)*ds*ds + self.spline(s, 1)*dds
        return dict(progress=s, progress_velocity=ds, progress_acceleration=dds,
                    position=position, velocity=velocity, acceleration=acceleration)


    def derivative_bounds(self):
        """Exact per-coordinate derivative bounds over every cubic segment."""
        first=np.zeros(self.spline.c.shape[-1]);second=first.copy()
        groups=[(0,3),(3,6)];norms=np.zeros(2)
        for i,span in enumerate(np.diff(self.spline.x)):
            a,b,c,_=self.spline.c[:,i,:]
            for j in range(len(first)):
                samples=[0.,float(span)]
                if abs(a[j])>1e-15:
                    critical=-b[j]/(3*a[j])
                    if 0<critical<span:samples.append(float(critical))
                first[j]=max(first[j],max(abs(3*a[j]*s*s+2*b[j]*s+c[j]) for s in samples))
                second[j]=max(second[j],abs(2*b[j]),abs(6*a[j]*span+2*b[j]))
            for k,(start,end) in enumerate(groups):
                if len(first)<end:continue
                squared=np.zeros(5)
                for j in range(start,end):squared+=np.convolve([c[j],2*b[j],3*a[j]],[c[j],2*b[j],3*a[j]])
                derivative=np.polynomial.polynomial.polyder(squared)
                candidates=[0.,float(span)]
                candidates += [float(root.real) for root in np.polynomial.polynomial.polyroots(derivative) if abs(root.imag)<1e-10 and 0<root.real<span]
                norms[k]=max(norms[k],max(np.linalg.norm(3*a[start:end]*s*s+2*b[start:end]*s+c[start:end]) for s in candidates))
        return dict(first=first,second=second,root_translation_first_norm=norms[0],root_rotvec_first_norm=norms[1])


class MeasuredAperturePhase:
    """Bound a reference aperture's speed/acceleration while following the plant.

    Exact leaf angles are privileged teacher measurements. The phase never
    commands a door joint and cannot guarantee that contact tracks its target.
    """
    def __init__(self, initial_angle, final_angle, initial_velocity,
                 maximum_speed=.149, maximum_acceleration=.08, tracking_lead=.005,
                 lead_start_angle=None, lead_ramp_rad=.1):
        if (not np.isfinite([initial_angle,final_angle,initial_velocity,
                             maximum_speed,maximum_acceleration,tracking_lead]).all()
                or not 0 <= tracking_lead <= .02
                or final_angle <= initial_angle or maximum_speed <= 0
                or maximum_acceleration <= 0 or not 0 <= initial_velocity <= maximum_speed):
            raise ValueError('Require bounded actual starting aperture velocity')
        if lead_start_angle is not None and (not np.isfinite([lead_start_angle,lead_ramp_rad]).all() or lead_start_angle<initial_angle or not .05<=lead_ramp_rad<=.5 or tracking_lead<.005):
            raise ValueError('Require a bounded later lead ramp that retains the original initial5mrad')
        self.angle=float(initial_angle);self.final=float(final_angle)
        self.velocity=float(initial_velocity);self.speed=float(maximum_speed)
        self.tracking_lead=float(tracking_lead)
        self.lead_start_angle=None if lead_start_angle is None else float(lead_start_angle)
        self.lead_ramp_rad=float(lead_ramp_rad)
        self.acceleration=float(maximum_acceleration);self.time=None;self.last_acceleration=0.

    def lead_at(self, measured_angle):
        if not np.isfinite(measured_angle):raise ValueError('Require a finite measured aperture')
        if self.lead_start_angle is None:return self.tracking_lead
        u=float(np.clip((measured_angle-self.lead_start_angle)/self.lead_ramp_rad,0.,1.))
        return .005+(self.tracking_lead-.005)*u**3*(10.+u*(-15.+6.*u))

    def update(self, time_s, measured_angle):
        if not np.isfinite([time_s,measured_angle]).all():
            raise ValueError('Require finite current aperture measurement')
        if self.time is None:
            self.time=float(time_s)
            return self.angle,self.velocity,0.
        dt=float(time_s-self.time)
        if dt < 0 or dt > .00200001:
            raise ValueError('Require consecutive physical intervals <=2ms')
        if dt == 0:return self.angle,self.velocity,self.last_acceleration
        remaining=max(0.,self.final-self.angle)
        desired=float(np.clip(4.*(measured_angle+self.lead_at(measured_angle)-self.angle),0.,self.speed))
        # Require remaining distance after this trapezoidal integration to
        # contain the complete next-state constant-acceleration stop.
        adt=self.acceleration*dt
        discriminant=adt*adt-4.*(adt*self.velocity-2.*self.acceleration*remaining)
        stopping=max(0.,.5*(-adt+np.sqrt(max(0.,discriminant))))
        desired=min(desired,stopping)
        change=float(np.clip(desired-self.velocity,-self.acceleration*dt,self.acceleration*dt))
        next_velocity=self.velocity+change
        advance=.5*(self.velocity+next_velocity)*dt
        if next_velocity <= 1e-14 and self.velocity < self.acceleration*dt:
            # A final stop can occur inside this interval; remain at rest
            # afterward instead of integrating a negative velocity.
            advance=self.velocity*self.velocity/(2.*self.acceleration)
        if advance > remaining+1e-8:
            raise ValueError('Reference exhausted its braking envelope')
        self.angle=min(self.final,self.angle+advance)
        acceleration=change/dt
        self.velocity=next_velocity;self.time=float(time_s);self.last_acceleration=acceleration
        return self.angle,self.velocity,acceleration
