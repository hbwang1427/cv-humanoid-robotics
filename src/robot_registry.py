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


# ---------------------------------------------------------------------------
# Joint-name remapping
# ---------------------------------------------------------------------------
# mapper.keypoints_to_joints() emits 10 *canonical* joint names (the names used
# by robot/humanoid.urdf, the "simple" robot). Every other URDF names its joints
# differently, so to drive them we remap:
#
#     target_angle = clamp( scale * canonical_angle + offset , urdf_limits )
#
# mapper produces UNSIGNED interior angles from the law of cosines, then clamps
# to the "simple" robot's limits, so in practice:
#   * shoulder_pitch (hip-shoulder-elbow angle): ~0 arms-down .. ~1.57 arms-out
#   * elbow / knee   (interior limb angle):      ~2.35 straight .. ~0 fully folded
# The (scale, offset) pairs below shift those so a relaxed standing pose sits at
# each target joint's zero and flexion drives it. Signs are approximate —
# fine-tune against the live RViz view if a joint bends the wrong way.

_CANONICAL_JOINTS = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
    "left_hip_pitch", "left_knee", "right_hip_pitch", "right_knee",
)

_IDENTITY_MAP = {name: (name, 1.0, 0.0) for name in _CANONICAL_JOINTS}

JOINT_MAPS: dict[str, dict[str, tuple[str, float, float]]] = {
    "simple": _IDENTITY_MAP,
    # Unitree H1 — 19 DOF. Drive the 4 arm + 2 knee joints; leave hips, ankles,
    # torso, shoulder-roll/yaw and hip-yaw/roll at 0 (upright stance).
    "h1": {
        "left_shoulder_pitch":  ("left_shoulder_pitch_joint",  -1.0, 0.0),
        "right_shoulder_pitch": ("right_shoulder_pitch_joint", -1.0, 0.0),
        "left_elbow":            ("left_elbow_joint",           -1.0, 2.35),
        "right_elbow":           ("right_elbow_joint",          -1.0, 2.35),
        "left_knee":             ("left_knee_joint",            -1.0, 2.35),
        "right_knee":            ("right_knee_joint",           -1.0, 2.35),
    },
}


def get_joint_map(robot_key: str) -> dict[str, tuple[str, float, float]]:
    """Canonical -> (urdf_joint, scale, offset). Empty dict => publish canonical
    names unchanged (works for 'simple' and any URDF that reuses those names)."""
    return JOINT_MAPS.get(robot_key, {})


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
