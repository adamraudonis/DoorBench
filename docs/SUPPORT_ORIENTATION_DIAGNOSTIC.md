# Fixed-foot orientation is not a valid correction here

The detached support-foot orientation hypothesis is rejected. Both feet physically roll together, even while their tactile loads remain approximately equal. Agreement between the two leg-derived estimates therefore cannot certify an upright foot frame.

| Recorded acquisition | Existing IMU root orientation RMS | Fixed-foot root orientation RMS | Maximum fixed-foot error |
| --- | ---: | ---: | ---: |
| Native | 0.128 mrad | 2.608 mrad | 3.970 mrad |
| Isaac with backend dry friction | 3.131 mrad | 8.262 mrad | 9.401 mrad |

The calculation reconstructs each foot's rotation relative to the pelvis from the recorded leg encoders. Multiplying its inverse by the original calibrated foot rotation yields a candidate pelvis orientation. The two candidates are averaged with the same clipped local tactile weights as the existing estimator. All 9,500 decision states are compared at their actual pre-decision epoch. Actual root/foot poses are confined to the independent scoring calculation.

At one second in Isaac, the feet have physically rolled approximately 8.35–8.46 mrad. The fixed-foot assumption then produces 7.20 mm lateral root error, compared with 2.46 mm for the existing IMU estimate. Native feet roll up to approximately 4 mrad as well. This is not an encoder-order or epoch mismatch: Isaac leg encoders equal the recorded same-epoch named positions exactly, and native disagreement is only float32 rounding.

The calibrated sole anchor is 70.658 mm below and 56.430 mm forward of the ankle origin. Rotation about the sole moves the ankle origin; treating that origin as a fixed support point adds position bias even without sliding contact. Native sole-anchor horizontal drift is only tens of micrometers, despite roughly 0.26 mm ankle-origin lateral movement. Isaac also has true sole-anchor motion, up to approximately 0.60 mm lateral and 0.51 mm longitudinal on the left foot. The mesh reconstruction is based on the original collision vertices; it does not independently certify the imported PhysX mesh surface.

The actual reset starts 1.7363 mm below the mesh-derived calibration height, causing initial sole overlap. Settling removes almost all vertical estimate error by 0.1 s, while the roll and lateral errors persist. Changing the reset or assuming perfectly static feet would be a different physical protocol and was not attempted.

`scripts/dexterous/diagnose_support_orientation.py` reproduces the calculation from the original robot/calibration and native/Isaac archives. The [receipt](evidence/support-orientation-rejection-001.json) binds full numerical evidence. No physical simulator steps, active-state writes, controller changes, or qualification changes occurred. The next diagnostic concerns IMU production and integration timing, not adoption of this rejected support model.
