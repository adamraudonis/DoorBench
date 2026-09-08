# Retrospective review of operation-v2-004

**The original distal-pad experiment remains failed.** Its final contact pattern
is nevertheless an opposed grip on valid inner finger surfaces: index and middle
fingers use their middle phalanges, ring and little fingers use distal surfaces,
and the distal thumb opposes all four. It should be evaluated in a future trial
with an explicitly declared volar-phalange profile, while retaining its distal
score. This retrospective assessment is not training qualification.

The [GRASP taxonomy](https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf)
classifies both contact surfaces and opposition, and distinguishes palm-based
wraps from pad opposition even when hand shapes look similar. It supports
describing contacts explicitly instead of treating every non-fingertip contact as
anatomical failure. It does not prove that this robot trajectory is natural or a
textbook power grasp. The [Shadow finger documentation](https://shadow-robot-company-dexterous-hand.readthedocs-hosted.com/en/latest/user_guide/md_finger.html)
defines the separate proximal, middle and distal links and their unilateral
loopback relation. The local surface calibration below comes from the frozen
Shadow XML and measured contacts, rather than a claim about all human hands.

I checked all 11,000 physical samples and reconstructed the robot from its actual
Isaac root and joint states without stepping the diagnostic model. Recorded
contact locations agree with the reconstructed link frames within **1.284 µm**.
All originally disallowed contacts lie on the index or middle finger's middle
phalange. Their outward local Y normals range from −0.986 to −0.934; their lever
radial alignment exceeds 0.9992 and their axial clearance exceeds 6.09 mm.
Index contact positions span 17.51–24.00 mm along its 25 mm middle segment;
middle-finger contacts span 18.83–19.01 mm.

Under the separately defined candidate surface set, the final 0.5 seconds have
all five opposed loads. Final finger loads are 2.28, 1.64, 2.21 and 2.73 N;
the thumb carries 7.62 N. Minimum finger pair alignment is 0.707 and maximum
thumb–finger dot product is −0.716. There is no misplaced final load. The
operation still contains **701 samples** without all five qualifying loads;
these are retained. The original mechanical, collision and actuation checks
remain unchanged.

I inspected annotated reconstruction views at 16, 18.9, 20 and 22 seconds. The
90° view exposes the thumb opposite the four fingers and the two middle contact
surfaces. The 150° and 210° views are obstructed by the door and are retained as
unsuitable inspection angles. These images are labeled FK reconstructions, not
a new native simulation or a live Isaac camera capture.

Evidence and images: `DoorBench-runs/2026-09-08-shadow-loopback/volar-review-004/`.
The compact [receipt](evidence/isaac-volar-review-004.json) preserves the failed
original report hash, exact candidate thresholds and all diagnostic limits.
The reproducible script is `scripts/dexterous/review_volar_contacts.py`.
