"""
Maps 2-D COCO keypoints to robot joint angles (radians).

Strategy: compute the angle at the middle joint of each 3-point limb chain
using the law of cosines in image space, then clamp to joint limits.
"""

import numpy as np
from detector import KP

JOINT_LIMITS = {
    "left_shoulder_pitch":  (-1.57, 1.57),
    "left_shoulder_roll":   (-1.57, 0.0),
    "left_elbow":           (0.0,   2.35),
    "right_shoulder_pitch": (-1.57, 1.57),
    "right_shoulder_roll":  (0.0,   1.57),
    "right_elbow":          (0.0,   2.35),
    "left_hip_pitch":       (-1.57, 1.57),
    "left_knee":            (0.0,   2.35),
    "right_hip_pitch":      (-1.57, 1.57),
    "right_knee":           (0.0,   2.35),
}


def _angle_at_b(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle (radians) at point B in the triangle A-B-C."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _signed_angle(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> float:
    """Signed angle of vector B-A relative to ref vector."""
    v = a - b
    angle = np.arctan2(v[1], v[0]) - np.arctan2(ref[1], ref[0])
    return float(angle)


def _clamp(value: float, name: str) -> float:
    lo, hi = JOINT_LIMITS[name]
    return float(np.clip(value, lo, hi))


def _xy(kps: np.ndarray, idx: int) -> np.ndarray:
    return kps[idx, :2]


def _visible(kps: np.ndarray, *indices: int, thresh: float = 0.3) -> bool:
    return all(kps[i, 2] > thresh for i in indices)


def keypoints_to_joints(kps: np.ndarray) -> dict[str, float]:
    """
    kps: (17, 3) array of [x, y, confidence] in image pixels.
    Returns dict of joint_name -> angle in radians (0.0 when not visible).
    """
    joints: dict[str, float] = {k: 0.0 for k in JOINT_LIMITS}

    # --- Left arm ---
    ls, le, lw = KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]
    lh = KP["left_hip"]
    if _visible(kps, ls, le, lw):
        joints["left_elbow"] = _clamp(_angle_at_b(_xy(kps, ls), _xy(kps, le), _xy(kps, lw)), "left_elbow")
    if _visible(kps, lh, ls, le):
        joints["left_shoulder_pitch"] = _clamp(_angle_at_b(_xy(kps, lh), _xy(kps, ls), _xy(kps, le)), "left_shoulder_pitch")

    # --- Right arm ---
    rs, re, rw = KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]
    rh = KP["right_hip"]
    if _visible(kps, rs, re, rw):
        joints["right_elbow"] = _clamp(_angle_at_b(_xy(kps, rs), _xy(kps, re), _xy(kps, rw)), "right_elbow")
    if _visible(kps, rh, rs, re):
        joints["right_shoulder_pitch"] = _clamp(_angle_at_b(_xy(kps, rh), _xy(kps, rs), _xy(kps, re)), "right_shoulder_pitch")

    # --- Left leg ---
    lhi, lk, la = KP["left_hip"], KP["left_knee"], KP["left_ankle"]
    if _visible(kps, lhi, lk, la):
        joints["left_knee"] = _clamp(_angle_at_b(_xy(kps, lhi), _xy(kps, lk), _xy(kps, la)), "left_knee")
        joints["left_hip_pitch"] = _clamp(np.pi - joints["left_knee"] * 0.5, "left_hip_pitch")

    # --- Right leg ---
    rhi, rk, ra = KP["right_hip"], KP["right_knee"], KP["right_ankle"]
    if _visible(kps, rhi, rk, ra):
        joints["right_knee"] = _clamp(_angle_at_b(_xy(kps, rhi), _xy(kps, rk), _xy(kps, ra)), "right_knee")
        joints["right_hip_pitch"] = _clamp(np.pi - joints["right_knee"] * 0.5, "right_hip_pitch")

    return joints


def remap_joints(joints: dict[str, float],
                 joint_map: dict[str, tuple[str, float, float]]) -> dict[str, float]:
    """Rename/rescale canonical joint angles for a specific robot's URDF.

    joint_map: canonical_name -> (urdf_joint_name, scale, offset).
    Returns {urdf_joint_name: scale * angle + offset}. An empty map passes the
    canonical dict through unchanged.
    """
    if not joint_map:
        return joints
    out: dict[str, float] = {}
    for canonical, angle in joints.items():
        if canonical not in joint_map:
            continue
        target, scale, offset = joint_map[canonical]
        out[target] = scale * angle + offset
    return out
