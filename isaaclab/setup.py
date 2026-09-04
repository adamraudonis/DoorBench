"""Install with the Isaac Lab python:  ./isaaclab.sh -p -m pip install -e /path/to/DoorBench/isaaclab

Dependencies (isaaclab, isaaclab_assets, isaaclab_rl, isaaclab_tasks, torch, gymnasium) come from the Isaac Lab
installation and are deliberately not pinned here.
"""
from setuptools import find_packages, setup

setup(
    name="doorbench_isaaclab",
    version="0.1.0",
    description="DoorBench doors as Isaac Lab manager-based RL tasks (multi-door scenes, G1 humanoid + gantry hand agents)",
    author="Adam Raudonis",
    license="MIT",
    packages=find_packages(include=["doorbench_isaaclab", "doorbench_isaaclab.*"]),
    package_data={"doorbench_isaaclab": ["data/*.usda"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
    zip_safe=False,
)
