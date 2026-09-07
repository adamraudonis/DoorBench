# Hand parameter provenance

`myohand.json` is a small parameter extract from
[MyoHub/myo_sim](https://github.com/MyoHub/myo_sim), revision
`eb327acbae0fad12279495040607f5235d962328`,
`myo_sim/models/arm/assets/myoarm_r_chain.xml`, beginning at `lunate_r`.
The source path, revision and file digest are also recorded in the JSON.
This extract is licensed under **Apache-2.0**; see `LICENSE-MyoSim.txt`.
The surrounding DoorBench MIT license does not replace this license.

The original hand combines the MoBL upper-extremity and 2nd-Hand models;
see [upstream hand documentation](https://github.com/MyoHub/myo_sim/blob/eb327acbae0fad12279495040607f5235d962328/myo_sim/models/arm/README_hand.md).

Retained: anatomical body offsets, digit axes and ranges, and primitive
contact envelopes. Omitted: meshes, muscle/tendon sites, original inertia,
and muscle actuation. DoorBench supplies approximate masses, joint servos,
contact sensors and thin skeleton geometry. This is **not the calibrated
MyoSuite musculoskeletal model**, and its forces are not human ground truth.

The builder rotates the right-hand coordinate frame and mirrors it for the
left hand. Polar vectors use `S`; hinge axes use `det(S) * S`. Both the thumb's
metacarpal and its two phalanges remain articulated. Overlapping tissue envelopes contact the environment; separate, thinner bone
capsules enforce hand/hand and hand/body separation. This does not model skin
deformation or pressure between adjacent fingers.
DoorBench strengthens numerical joint-limit constraints and audits the actual
post-step angles, since soft simulator limits can otherwise be exceeded.

The 21 landmarks follow [COCO-WholeBody's hand ordering in Sapiens](https://github.com/facebookresearch/sapiens/blob/main/pose/configs/_base_/datasets/coco_wholebody.py):
wrist; thumb CMC/MCP/IP/tip; then MCP/PIP/DIP/tip for index, middle, ring, little.
Joint centres follow the source hierarchy. Tip landmarks are authored offsets
at the distal contact envelopes, not measured landmarks from a photograph.
No Sapiens model weights, inference or captured motion are used.

Photographic pose references personally inspected during this revision:
[lever grip, diagonal wrist and thumb along the handle](https://getstrength.com/open-someones-door/)
and [a second close-up grip](https://www.tudogostoso.com.br/noticias/o-que-significa-quando-uma-pessoa-checa-varias-vezes-se-trancou-mesmo-a-porta-de-casa-ou-do-carro-segundo-um-psicologo-a15476.htm).
These guide visual review only; photos are linked, not redistributed or used
as metric 3D ground truth.
