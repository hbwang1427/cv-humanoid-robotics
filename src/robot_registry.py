"""
Registry of available robot URDF models.
Paths are relative to the project root.
"""

import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")

def _p(*parts) -> str:
    return os.path.normpath(os.path.join(_ROOT, *parts))


ROBOTS: dict[str, dict] = {
    # --- Custom simple humanoid (always available, no meshes needed) ---
    "simple": {
        "name":        "Simple Humanoid",
        "urdf":        _p("robot/humanoid.urdf"),
        "description": "Hand-crafted box/cylinder humanoid, no mesh dependencies",
    },

    # --- Unitree Humanoids ---
    "h1": {
        "name":        "Unitree H1",
        "urdf":        _p("robot/models/unitree_ros/robots/h1_description/urdf/h1.urdf"),
        "description": "Unitree H1 humanoid — 19 DOF",
    },
    "h1_hand": {
        "name":        "Unitree H1 with Hand",
        "urdf":        _p("robot/models/unitree_ros/robots/h1_description/urdf/h1_with_hand.urdf"),
        "description": "Unitree H1 humanoid with dexterous hands",
    },
    "h2": {
        "name":        "Unitree H2",
        "urdf":        _p("robot/models/unitree_ros/robots/h2_description/H2.urdf"),
        "description": "Unitree H2 humanoid",
    },
    "g1": {
        "name":        "Unitree G1 (23 DOF)",
        "urdf":        _p("robot/models/unitree_ros/robots/g1_description/g1_23dof.urdf"),
        "description": "Unitree G1 humanoid — 23 DOF",
    },
    "g1_29": {
        "name":        "Unitree G1 (29 DOF)",
        "urdf":        _p("robot/models/unitree_ros/robots/g1_description/g1_29dof.urdf"),
        "description": "Unitree G1 humanoid — 29 DOF",
    },
    "g1_hand": {
        "name":        "Unitree G1 with Hand",
        "urdf":        _p("robot/models/unitree_ros/robots/g1_description/g1_29dof_with_hand.urdf"),
        "description": "Unitree G1 with dexterous hands — 29 DOF",
    },
    "r1": {
        "name":        "Unitree R1",
        "urdf":        _p("robot/models/unitree_ros/robots/r1_description/R1.urdf"),
        "description": "Unitree R1 arm robot",
    },

    # --- Unitree Quadrupeds (bonus) ---
    "go2": {
        "name":        "Unitree Go2",
        "urdf":        _p("robot/models/unitree_ros/robots/go2_description/urdf/go2_description.urdf"),
        "description": "Unitree Go2 quadruped",
    },
    "b2": {
        "name":        "Unitree B2",
        "urdf":        _p("robot/models/unitree_ros/robots/b2_description/urdf/b2_description.urdf"),
        "description": "Unitree B2 quadruped",
    },
}

DEFAULT_ROBOT = "simple"


def get_urdf_path(robot_key: str) -> str:
    if robot_key not in ROBOTS:
        available = ", ".join(ROBOTS.keys())
        raise ValueError(f"Unknown robot '{robot_key}'. Available: {available}")
    entry = ROBOTS[robot_key]
    path = entry["urdf"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"URDF for '{robot_key}' not found at:\n  {path}\n"
            f"Run: git clone --depth 1 https://github.com/unitreerobotics/unitree_ros robot/models/unitree_ros"
        )
    return path


def list_robots() -> None:
    print(f"\n{'KEY':<12} {'NAME':<30} DESCRIPTION")
    print("-" * 75)
    for key, info in ROBOTS.items():
        available = os.path.exists(info["urdf"])
        status = "" if available else " [NOT FOUND]"
        print(f"{key:<12} {info['name']:<30} {info['description']}{status}")
    print()
