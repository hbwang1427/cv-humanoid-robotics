"""
Geometric expression features computed from 68 dlib landmarks, plus
per-user neutral-face calibration. See docs/class03_facial_expression.md,
Section 3 (Approach A).
"""

import numpy as np

LEFT_EYE = range(42, 48)
RIGHT_EYE = range(36, 42)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: range) -> float:
    p = pts[list(idx)]
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-9)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    p = pts[60:68]  # inner mouth
    return (_dist(p[1], p[7]) + _dist(p[2], p[6]) + _dist(p[3], p[5])) / (2.0 * _dist(p[0], p[4]) + 1e-9)


def smile_ratio(pts: np.ndarray) -> float:
    return _dist(pts[48], pts[54]) / (_dist(pts[2], pts[14]) + 1e-9)


def brow_raise(pts: np.ndarray) -> float:
    brow = pts[17:27].mean(axis=0)
    eyes = pts[36:48].mean(axis=0)
    face_h = _dist(pts[27], pts[8]) + 1e-9  # nose bridge -> chin
    return (eyes[1] - brow[1]) / face_h


def extract_features(pts: np.ndarray) -> dict[str, float]:
    return {
        "ear_left": eye_aspect_ratio(pts, LEFT_EYE),
        "ear_right": eye_aspect_ratio(pts, RIGHT_EYE),
        "mar": mouth_aspect_ratio(pts),
        "smile": smile_ratio(pts),
        "brow": brow_raise(pts),
    }


class Baseline:
    """Per-user neutral-face calibration, averaged over the first N frames."""

    def __init__(self, n_frames: int = 30):
        self.n_frames = n_frames
        self._samples: list[dict[str, float]] = []
        self.values: dict[str, float] | None = None

    @property
    def ready(self) -> bool:
        return self.values is not None

    def add(self, features: dict[str, float]) -> None:
        if self.ready:
            return
        self._samples.append(features)
        if len(self._samples) >= self.n_frames:
            keys = self._samples[0].keys()
            self.values = {k: float(np.mean([s[k] for s in self._samples])) for k in keys}


def classify(f: dict[str, float], baseline: dict[str, float]) -> str:
    ear = (f["ear_left"] + f["ear_right"]) / 2.0
    if f["mar"] > 0.55 and f["brow"] > baseline["brow"] * 1.15:
        return "surprise"
    if f["smile"] > baseline["smile"] * 1.10 and f["mar"] < 0.35:
        return "happy"
    if ear < 0.12:
        return "eyes_closed"
    if f["brow"] < baseline["brow"] * 0.85:
        return "frown"
    return "neutral"
