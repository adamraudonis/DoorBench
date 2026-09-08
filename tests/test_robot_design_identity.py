import copy
import pytest
pytest.importorskip("mujoco")

from doorbench.dexterous.robot_design_identity import robot_design_identity, verify_robot_design_identity


OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def scene(directory, *, absolute=False, includes=True):
    directory.mkdir(parents=True)
    assets = directory / "assets"; assets.mkdir()
    (assets / "tetra.obj").write_text(OBJ)
    mesh = str(assets / "tetra.obj") if absolute else "tetra.obj"
    default = '<default><default class="mechanical"><joint damping="0.123456789012345"/><geom friction="0.7 0.01 0.001"/></default></default>'
    body = '<worldbody><body name="body" childclass="mechanical"><joint name="hinge"/><geom type="mesh" mesh="tetra"/></body></worldbody>'
    if includes:
        (directory / "defaults.xml").write_text("<mujocoinclude>" + default + "</mujocoinclude>")
        (directory / "body.xml").write_text("<mujocoinclude>" + body + "</mujocoinclude>")
        (directory / "nested.xml").write_text('<mujocoinclude><include file="body.xml"/></mujocoinclude>')
        elements = '<include file="defaults.xml"/><include file="nested.xml"/>'
    else:
        elements = default + body
    path = directory / "robot.xml"
    path.write_text(f'<mujoco model="identity fixture"><compiler angle="radian" meshdir="assets"/>'
                   f'<asset><mesh name="tetra" file="{mesh}"/></asset>{elements}'
                   '<actuator><motor name="motor" joint="hinge" gear="2"/></actuator></mujoco>')
    return path


def test_relocation_absolute_paths_and_inline_includes_have_identical_identity(tmp_path):
    a = scene(tmp_path / "first", includes=True)
    b = scene(tmp_path / "elsewhere", absolute=True, includes=False)
    expected = robot_design_identity(a)
    assert robot_design_identity(b) == expected
    assert verify_robot_design_identity(b, expected) == expected
    assert len(expected["referenced_assets"]) == 1
    assert str(tmp_path) not in str(expected)


@pytest.mark.parametrize("change", ["mesh_bytes", "default", "motor", "solver", "class_binding", "tiny_setting"])
def test_changed_authoring_or_assets_cannot_reuse_identity(tmp_path, change):
    p = scene(tmp_path / "scene")
    expected = robot_design_identity(p)
    if change == "mesh_bytes":
        (p.parent / "assets/tetra.obj").write_text(OBJ.replace("v 1 0 0", "v 2 0 0"))
    elif change == "default":
        q = p.parent / "defaults.xml"; q.write_text(q.read_text().replace("0.7 0.01", "0.8 0.01"))
    elif change == "tiny_setting":
        q = p.parent / "defaults.xml"; q.write_text(q.read_text().replace("0.123456789012345", "0.123456789012346"))
    elif change == "motor": p.write_text(p.read_text().replace('gear="2"', 'gear="3"'))
    elif change == "solver": p.write_text(p.read_text().replace("<actuator>", '<option timestep="0.004"/><actuator>'))
    elif change == "class_binding":
        q = p.parent / "body.xml"; q.write_text(q.read_text().replace(' childclass="mechanical"', ""))
    with pytest.raises(ValueError, match="differ"):
        verify_robot_design_identity(p, expected)


def test_whitespace_comments_attribute_order_are_not_physical_changes(tmp_path):
    p = scene(tmp_path / "scene")
    expected = robot_design_identity(p)
    p.write_text(p.read_text().replace('angle="radian" meshdir="assets"', 'meshdir="assets" angle="radian"')
                 .replace("<asset>", "\n<!-- formatting only -->\n<asset>\n"))
    assert robot_design_identity(p) == expected


def test_default_names_and_exact_numeric_strings_are_preserved(tmp_path):
    p = scene(tmp_path / "scene")
    a = robot_design_identity(p)
    q = p.parent / "defaults.xml"; q.write_text(q.read_text().replace('class="mechanical"', 'class="other"'))
    body = p.parent / "body.xml"; body.write_text(body.read_text().replace('childclass="mechanical"', 'childclass="other"'))
    assert robot_design_identity(p)["sha256"] != a["sha256"]


def test_missing_asset_include_cycle_and_entities_fail_closed(tmp_path):
    p = scene(tmp_path / "scene")
    (p.parent / "assets/tetra.obj").unlink()
    with pytest.raises(FileNotFoundError): robot_design_identity(p)
    (p.parent / "assets/tetra.obj").write_text(OBJ)
    (p.parent / "nested.xml").write_text('<mujocoinclude><include file="nested.xml"/></mujocoinclude>')
    with pytest.raises(ValueError, match="cyclic"): robot_design_identity(p)
    p.write_text('<!DOCTYPE mujoco [<!ENTITY value "bad">]><mujoco/>')
    with pytest.raises(ValueError, match="entities"): robot_design_identity(p)


def test_pinned_parser_and_complete_expected_receipt_are_required(tmp_path):
    p = scene(tmp_path / "scene")
    expected = robot_design_identity(p)
    different = copy.deepcopy(expected); different["mujoco_version"] = "unknown"
    with pytest.raises(ValueError, match="parser version"): verify_robot_design_identity(p, different)
    different = copy.deepcopy(expected); different["referenced_assets"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="differ"): verify_robot_design_identity(p, different)


def test_texture_face_files_and_texture_directory_are_all_bound(tmp_path):
    p = scene(tmp_path / "scene", includes=False)
    textures = p.parent / "textures"; textures.mkdir()
    # Parsing MJCF does not decode these bytes; the source contract must bind
    # even referenced assets which are unused by the active collision model.
    for side in ("right", "left", "up", "down", "front", "back"):
        (textures / (side + ".png")).write_bytes(("fixture " + side).encode())
    attributes = " ".join(f'file{side}="{side}.png"' for side in ("right", "left", "up", "down", "front", "back"))
    p.write_text(p.read_text().replace('meshdir="assets"', 'meshdir="assets" texturedir="textures"')
                 .replace("</asset>", f'<texture name="cube" type="cube" {attributes}/></asset>'))
    expected = robot_design_identity(p)
    assert len(expected["referenced_assets"]) == 7
    (textures / "back.png").write_bytes(b"changed back face")
    with pytest.raises(ValueError, match="differ"): verify_robot_design_identity(p, expected)
