"""
Maps geometric expression features (EAR/MAR/smile/brow) to face robot
joint angles (radians for revolute, meters for the prismatic lip
corners). Joint names/limits mirror robot/face.urdf. See
docs/class03_facial_expression.md, Section 4.
"""

import numpy as np

JOINT_LIMITS = {
    "jaw_joint": (0.0, 0.5),
    "left_brow_joint": (-0.3, 0.3),
    "right_brow_joint": (-0.3, 0.3),
    "left_eyelid_joint": (0.0, 1.2),
    "right_eyelid_joint": (0.0, 1.2),
    "left_lip_corner_joint": (0.0, 0.012),
    "right_lip_corner_joint": (0.0, 0.012),
}


def _clamp(value: float, name: str) -> float:
    lo, hi = JOINT_LIMITS[name]
    return float(np.clip(value, lo, hi))


def _eyelid_angle(ear: float) -> float:
    """EAR ~0.30 (open) -> 0.0 rad; EAR ~0.05 (closed) -> max rad."""
    closed_frac = np.clip((0.30 - ear) / 0.25, 0.0, 1.0)
    return closed_frac * JOINT_LIMITS["left_eyelid_joint"][1]


def features_to_joints(features: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    jaw = _clamp((features["mar"] / 0.7) * JOINT_LIMITS["jaw_joint"][1], "jaw_joint")

    brow_ratio = features["brow"] / (baseline["brow"] + 1e-9)
    brow = _clamp((brow_ratio - 1.0) * 3.0 * JOINT_LIMITS["left_brow_joint"][1], "left_brow_joint")

    left_lid = _clamp(_eyelid_angle(features["ear_left"]), "left_eyelid_joint")
    right_lid = _clamp(_eyelid_angle(features["ear_right"]), "right_eyelid_joint")

    smile_ratio = features["smile"] / (baseline["smile"] + 1e-9)
    lip = _clamp(max(smile_ratio - 1.0, 0.0) * 5.0 * JOINT_LIMITS["left_lip_corner_joint"][1],
                 "left_lip_corner_joint")

    return {
        "jaw_joint": jaw,
        "left_brow_joint": brow,
        "right_brow_joint": brow,
        "left_eyelid_joint": left_lid,
        "right_eyelid_joint": right_lid,
        "left_lip_corner_joint": lip,
        "right_lip_corner_joint": lip,
    }
