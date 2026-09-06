import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize(
    "fastapi,extra,expected",
    [("0.121.0", False, 0), ("0.122.0", False, 1), ("0.121.0", True, 1)],
)
def test_only_documented_exact_dependency_exception_is_accepted(
    monkeypatch, fastapi, extra, expected
):
    path = Path(__file__).resolve().parents[1] / "scripts/isaaclab/check_g1_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def dist(name, version, requirements=()):
        return SimpleNamespace(
            metadata={"Name": name}, version=version, requires=requirements
        )

    packages = [
        dist("isaacsim-kernel", "5.1.0.0", ["fastapi==0.115.7"]),
        dist("fastapi", fastapi, ["starlette>=0.40,<0.50"]),
        dist("starlette", "0.49.1"),
    ]
    if extra:
        packages.append(dist("another-package", "1.0", ["missing-dependency>=1"]))
    monkeypatch.setattr(module.metadata, "distributions", lambda: packages)
    assert module.check() == expected
