"""
Locates 68 dense facial landmarks with dlib, seeded by YOLO's 5 sparse
face keypoints (nose, eyes, ears) instead of dlib's own (slower) face
detector. See docs/class03_facial_expression.md, Section 2.
"""

import os

import cv2
import numpy as np

SHAPE_PREDICTOR_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "labs", "models",
                 "shape_predictor_68_face_landmarks.dat")
)


class FaceLandmarkDetector:
    def __init__(self, predictor_path: str = SHAPE_PREDICTOR_PATH):
        try:
            import dlib
        except ImportError as e:
            raise RuntimeError(
                "dlib is required for facial landmarks.\n"
                "Install: mamba install -c conda-forge dlib -y"
            ) from e
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(
                f"Missing dlib shape predictor at:\n  {predictor_path}\n"
                "Download it with:\n"
                "  curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "  bunzip2 shape_predictor_68_face_landmarks.dat.bz2\n"
                f"  mkdir -p labs/models && mv shape_predictor_68_face_landmarks.dat {predictor_path}"
            )
        self._dlib = dlib
        self.predictor = dlib.shape_predictor(predictor_path)

    def face_rect_from_yolo(self, pose_kps: np.ndarray, frame_shape: tuple):
        """Build a dlib.rectangle face box from YOLO's 5 face keypoints
        (nose=0, eyes=1-2, ears=3-4). Returns None if too few are visible."""
        face_pts = pose_kps[:5]
        valid = face_pts[face_pts[:, 2] > 0.3]
        if len(valid) < 3:
            return None
        cx, cy = valid[:, 0].mean(), valid[:, 1].mean()
        size = 2.2 * max(valid[:, 0].ptp(), 40)
        x1, y1 = int(cx - size / 2), int(cy - size / 2)
        x2, y2 = int(cx + size / 2), int(cy + size / 2 + 0.3 * size)  # extend toward chin
        h, w = frame_shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return self._dlib.rectangle(x1, y1, x2, y2)

    def detect(self, frame: np.ndarray, pose_kps: np.ndarray):
        """Returns (68, 2) landmark array in pixel coords, or None."""
        rect = self.face_rect_from_yolo(pose_kps, frame.shape)
        if rect is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, rect)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float64)

    def draw(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        out = frame.copy()
        for x, y in landmarks:
            cv2.circle(out, (int(x), int(y)), 1, (0, 255, 0), -1)
        return out
