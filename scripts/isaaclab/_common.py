"""Shared helpers for the Isaac Lab scripts (imported AFTER the AppLauncher started the simulator).

Also carries the small compatibility layer for Isaac Lab **v2.3.2** / rsl-rl-lib 3.1.x:
  * ``dump_pickle`` / ``load_pickle``: Isaac Lab removed ``isaaclab.utils.io.dump_pickle`` in isaaclab 0.47.0
    ("Removed pickle utilities ... as pickle contains security vulnerabilities"); train.py still writes params/*.pkl
    next to the yaml for our own tooling, so the helper lives here (plain ``pickle``, same file semantics).
  * ``rsl_rl_version_check``: Isaac Lab's own rsl_rl scripts require rsl-rl-lib >= 3.0.1; warn early instead of
    failing deep inside the runner.
  * ``policy_normalizer``: rsl_rl 3.x keeps the observation normalizer on the policy module
    (``actor_obs_normalizer`` / ``student_obs_normalizer``), not on the runner.
  * ``unwrap_obs``: ``RslRlVecEnvWrapper.get_observations()`` returns a TensorDict in v2.3.x (older wrappers returned
    ``(obs, extras)``).
"""
from __future__ import annotations

import importlib
import os
import pickle
import platform
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RSL_RL_MIN_VERSION = "3.0.1"   # scripts/reinforcement_learning/rsl_rl/train.py of Isaac Lab v2.3.2; v2.3.2 pins rsl-rl-lib==3.1.2


def package_version(module: str, *dist_names: str) -> str | None:
    """Version of an installed package, from whichever of the four places actually carries it.

    ``isaaclab.__version__`` / ``isaacsim.__version__`` do not exist in Isaac Sim 5.1 + Isaac Lab 2.3, which is why
    every parity record of rounds 1 and 2 says ``null`` for both.  The version does live in the distribution metadata,
    in Isaac Sim's own ``get_version()`` and in the ``VERSION`` file next to the app - try all of them and, failing
    that, say where the module was imported from, so the run is still identifiable.  Returns None only when the module
    is not installed at all."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    v = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
    if isinstance(v, str) and v:
        return v
    for dist in (dist_names or (module,)):
        try:
            from importlib.metadata import version as _dist_version
            return _dist_version(dist)
        except Exception:
            continue
    if module == "isaacsim":
        try:
            from isaacsim.core.version import get_version
            core = next((str(x) for x in get_version() if x), None)
            if core:
                return core
        except Exception:
            pass
        for base in {os.path.dirname(os.path.dirname(getattr(mod, "__file__", "") or "")), os.environ.get("ISAAC_PATH", "")}:
            vf = os.path.join(base, "VERSION") if base else ""
            if vf and os.path.isfile(vf):
                try:
                    with open(vf) as f:
                        return f.read().strip().splitlines()[0]
                except OSError:
                    pass
    path = getattr(mod, "__file__", None)
    # a namespace package with no __file__ carries no version and is usually not the package we meant (an empty
    # directory of the right name on sys.path): report it as not installed rather than as an unknown version
    return f"unknown (imported from {os.path.dirname(path)})" if path else None


def simulator_engine() -> dict:
    """The engine block every Isaac record carries: what actually produced the numbers.

    A parity report that cannot name the simulator version is not reproducible, so this resolves the versions the hard
    way and records the interpreter and platform alongside them."""
    eng = {"isaac_sim": package_version("isaacsim", "isaacsim", "isaacsim-core"), "isaac_lab": package_version("isaaclab", "isaaclab"),
           "python": platform.python_version(), "platform": platform.platform()}
    torch_v = package_version("torch")
    if torch_v:
        eng["torch"] = torch_v
    return eng


def ensure_extension_importable():
    """Make ``doorbench_isaaclab`` importable from the checkout even when it was not pip-installed."""
    try:
        import doorbench_isaaclab  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(ROOT, "isaaclab"))
        import doorbench_isaaclab  # noqa: F401
    os.environ.setdefault("DOORBENCH_ASSETS", os.path.join(ROOT, "assets"))


def add_door_args(parser):
    parser.add_argument("--doors", type=str, default=os.environ.get("DOORBENCH_DOORS", "easy-100"),
                        help="door subset: easy-100 | easy-300 | all | random-50 | family:saloon,pivot | db0002_swing_single,... | @ids.txt")
    parser.add_argument("--door_seed", type=int, default=0, help="seed for the door shuffle / random subset")
    parser.add_argument("--door_random_choice", action="store_true", help="random door per env instead of round-robin")


def apply_door_args(env_cfg, args):
    from doorbench_isaaclab.door_task_env_cfg import set_doors

    set_doors(env_cfg, args.doors, seed=args.door_seed, random_choice=args.door_random_choice)
    n = len(env_cfg.scene.door.spawn.usd_path)
    print(f"[doorbench] {n} doors selected by --doors {args.doors!r} (round-robin over {env_cfg.scene.num_envs} envs)")
    return env_cfg


# ------------------------------------------------------------------------------------ Isaac Lab v2.3.2 compatibility
def dump_pickle(filename: str, data) -> str:
    """Save ``data`` (a dict or a configclass object) as a pickle; appends ``.pkl`` and creates the directory, like the
    ``isaaclab.utils.io.dump_pickle`` that Isaac Lab <= 0.46 shipped.  Returns the path written."""
    if not filename.endswith("pkl"):
        filename += ".pkl"
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    return filename


def load_pickle(filename: str):
    """Load a pickle written by :func:`dump_pickle` (only for files this repo wrote itself)."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def rsl_rl_version_check(minimum: str = RSL_RL_MIN_VERSION) -> str | None:
    """Installed rsl-rl-lib version (None if not installed); prints a warning when older than ``minimum``."""
    try:
        from importlib import metadata

        installed = metadata.version("rsl-rl-lib")
    except Exception:  # not installed / metadata missing
        print(f"[WARN] rsl-rl-lib not found in this python; Isaac Lab v2.3.2 expects rsl-rl-lib>={minimum} (pip install rsl-rl-lib==3.1.2)")
        return None

    def key(v: str):
        return tuple(int(p) if p.isdigit() else 0 for p in v.split("+")[0].split("."))

    if key(installed) < key(minimum):
        print(f"[WARN] rsl-rl-lib {installed} < {minimum}: the runner API (TensorDict observations, obs_groups) differs; "
              f"install the version Isaac Lab v2.3.2 pins:  pip install rsl-rl-lib==3.1.2")
    return installed


def policy_normalizer(policy_nn):
    """Observation normalizer of an rsl_rl 3.x policy module (None when normalization is off)."""
    for name in ("actor_obs_normalizer", "student_obs_normalizer"):
        norm = getattr(policy_nn, name, None)
        if norm is not None:
            return norm
    return None


def unwrap_obs(obs):
    """``RslRlVecEnvWrapper.get_observations()`` -> TensorDict (v2.3.x); older wrappers returned ``(obs, extras)``."""
    return obs[0] if isinstance(obs, tuple) else obs
