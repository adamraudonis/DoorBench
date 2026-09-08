"""A whole-cycle displacement estimate removes periodic in-place pelvis sway."""
import numpy as np
from doorbench.dexterous.locomotion_approach import WaypointApproach


def test_full_cycle_recovers_net_travel_with_reused_position_buffer():
    controller=WaypointApproach([100.,100.],0.,brake_velocity_window=.8)
    position=np.zeros(2)
    for t in np.arange(0.,2.001,.02):
        position[:]=[.2*t,.03*np.sin(2*np.pi*t/.8)]
        velocity=[.2,.03*2*np.pi/.8*np.cos(2*np.pi*t/.8)]
        controller.step(position,0.,velocity,float(t))
        if t>.81:np.testing.assert_allclose(controller.brake_velocity,[.2,0.],atol=1e-12)
    assert abs(controller.velocity[1])>.01


def test_linear_travel_with_irregular_sampling_interpolates_exact_window():
    controller=WaypointApproach([100.,100.],0.,brake_velocity_window=.8)
    for t in [0.,.11,.27,.54,.91,1.02,1.25]:
        controller.step([.2*t,-.13*t],0.,[.2,-.13],t)
        if t>.8:np.testing.assert_allclose(controller.brake_velocity,[.2,-.13],atol=1e-12)
