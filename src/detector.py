import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolov8n-pose.pt"  # downloaded automatically on first run

# COCO 17-keypoint indices
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class PoseDetector:
    def __init__(self, model_path: str = MODEL_PATH, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Return (17, 3) keypoint array [x, y, confidence] for the most
        confident person in the frame, or None if no detection."""
        results = self.model(frame, conf=self.conf, verbose=False)
        if not results or results[0].keypoints is None:
            return None
        kps = results[0].keypoints.data  # shape: (N_persons, 17, 3)
        if kps.shape[0] == 0:
            return None
        # pick person with highest mean keypoint confidence
        best = int(kps[:, :, 2].mean(dim=1).argmax())
        return kps[best].cpu().numpy()  # (17, 3)

    def draw(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw skeleton overlay on frame (returns annotated copy)."""
        import cv2
        SKELETON = [
            (KP["left_shoulder"], KP["right_shoulder"]),
            (KP["left_shoulder"], KP["left_elbow"]),
            (KP["left_elbow"], KP["left_wrist"]),
            (KP["right_shoulder"], KP["right_elbow"]),
            (KP["right_elbow"], KP["right_wrist"]),
            (KP["left_hip"], KP["right_hip"]),
            (KP["left_shoulder"], KP["left_hip"]),
            (KP["right_shoulder"], KP["right_hip"]),
            (KP["left_hip"], KP["left_knee"]),
            (KP["left_knee"], KP["left_ankle"]),
            (KP["right_hip"], KP["right_knee"]),
            (KP["right_knee"], KP["right_ankle"]),
        ]
        out = frame.copy()
        for i, j in SKELETON:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > 0.3 and cj > 0.3:
                cv2.line(out, (int(xi), int(yi)), (int(xj), int(yj)), (0, 255, 0), 2)
        for x, y, c in keypoints:
            if c > 0.3:
                cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
        return out
