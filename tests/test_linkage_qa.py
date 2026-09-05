"""Independent fixtures for loop feasibility, including both-side mechanisms."""
import pytest

from doorbench.linkage_qa import check_linkage_model, run_linkage_qa


def _fixture(sliding_shoe):
    mujoco = pytest.importorskip("mujoco")
    inertial = '<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>'
    shoe = f'<body name="shoe" pos="1 0 0">{inertial}<joint name="shoe_slide" type="slide" axis="0 0 1" range="0 0.02"/></body>' if sliding_shoe else ''
    xml = f'''<mujoco><compiler angle="radian"/><worldbody>
      <body name="carrier">{inertial}<joint name="rise" type="slide" axis="0 0 1" range="0 0.02"/>
        <body name="leaf">{inertial}<joint name="leaf_hinge" axis="0 0 1" range="0 1"/>
          <body name="arm">{inertial}<joint name="arm_hinge" axis="0 0 1" limited="false"/></body>
        </body>
      </body>{shoe}</worldbody><equality>
      <joint name="lift_coupling" joint1="rise" joint2="leaf_hinge" polycoef="0 0.01 0 0 0"/>
      <connect name="arm_connect" body1="arm" body2="{'shoe' if sliding_shoe else 'world'}" anchor="1 0 0"/>
      </equality></mujoco>'''
    def body(name, joint, role):
        return {"name": name, "joint": {"name": joint, "role": role, "robot_interactive": role == "primary"}}
    description = {"bodies": [body("carrier", "rise", "mechanism"), body("leaf", "leaf_hinge", "primary"), body("arm", "arm_hinge", "mechanism")],
                   "equalities": [{"kind": "joint", "a": "rise", "b": "leaf_hinge"}, {"kind": "connect", "a": "arm", "b": "shoe" if sliding_shoe else "world"}]}
    if sliding_shoe:
        description["bodies"].append(body("shoe", "shoe_slide", "mechanism"))
    return mujoco.MjModel.from_xml_string(xml), description


def test_gate_rejects_impossible_rise_without_solving_away_coupling():
    model, description = _fixture(False)
    report = check_linkage_model(model, description)
    assert not report["ok"]
    assert report["n_samples"] == 26
    assert report["max_residual_m"] == pytest.approx(0.01, abs=1e-5)
    assert all("rise" not in row["mechanism_joints"] for row in report["failures"])


def test_gate_solves_mechanisms_on_both_sides_of_connection():
    model, description = _fixture(True)
    report = check_linkage_model(model, description)
    assert report["ok"], report
    assert report["n_loops"] == 1
    assert report["max_residual_m"] < 1e-4


def test_missing_model_is_a_failure(tmp_path):
    assert not run_linkage_qa(str(tmp_path))["ok"]


def test_free_mechanism_driver_cannot_ignore_its_coupled_arm():
    """Opposed coaxial arms cancel: they cannot compensate any leaf rotation."""
    mujoco = pytest.importorskip("mujoco")
    mass = '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
    xml = f'''<mujoco><compiler angle="radian"/><worldbody>
      <body name="leaf">{mass}<joint name="leaf_hinge" axis="0 0 1" range="0 1"/>
        <body name="arm1">{mass}<joint name="arm1_hinge" axis="0 0 1" limited="false"/>
          <body name="arm2">{mass}<joint name="arm2_hinge" axis="0 0 1" limited="false"/></body>
        </body>
      </body></worldbody><equality>
      <joint name="opposed_arms" joint1="arm2_hinge" joint2="arm1_hinge" polycoef="0 -1 0 0 0"/>
      <connect name="tip" body1="arm2" body2="world" anchor="1 0 0"/>
      </equality></mujoco>'''
    description = {"bodies": [
        {"name": name, "joint": {"name": joint, "role": role}}
        for name, joint, role in [("leaf", "leaf_hinge", "primary"),
                                  ("arm1", "arm1_hinge", "mechanism"),
                                  ("arm2", "arm2_hinge", "mechanism")]],
        "equalities": [{"kind": "joint", "a": "arm2_hinge", "b": "arm1_hinge"},
                       {"kind": "connect", "a": "arm2", "b": "world"}]}
    report = check_linkage_model(mujoco.MjModel.from_xml_string(xml), description, n_steps=4)
    assert not report["ok"]
    assert report["max_residual_m"] > 0.9
    assert any(f["driver"] == "leaf_hinge" and f["q"] == 1.0 for f in report["failures"])
