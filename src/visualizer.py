"""Debug window showing the video/camera feed with skeleton overlay, all keypoints, and joint angles."""

import cv2
import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class Visualizer:
    WINDOW = "Pose Mimic — press Q to quit"

    def __init__(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def show(self,
             frame: np.ndarray,
             joints: dict[str, float] | None = None,
             keypoints: np.ndarray | None = None) -> bool:
        """Display frame with full keypoint list and joint angles. Returns False on Q."""
        display = frame.copy()
        h, w = display.shape[:2]

        # --- Left panel: all 17 COCO keypoints ---
        if keypoints is not None:
            self._draw_panel(display, "KEYPOINTS (x, y, conf)", 0, _keypoint_lines(keypoints), h)

        # --- Right panel: computed joint angles ---
        if joints:
            self._draw_panel(display, "JOINT ANGLES", w - 230, _joint_lines(joints), h)

        cv2.imshow(self.WINDOW, display)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def _draw_panel(self, img, title: str, x: int, lines: list[str], frame_h: int):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, pad, lh = 0.4, 6, 15

        panel_w = 230
        panel_h = pad + lh + (lh * len(lines)) + pad
        # semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (x, 0), (x + panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # title
        cv2.putText(img, title, (x + pad, pad + lh - 3),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        # rows
        for i, line in enumerate(lines):
            color = (100, 255, 100) if "*" in line else (160, 160, 160)
            cv2.putText(img, line, (x + pad, pad + lh + (i + 1) * lh),
                        font, fs, color, 1, cv2.LINE_AA)

    def close(self):
        cv2.destroyAllWindows()


def _keypoint_lines(kps: np.ndarray) -> list[str]:
    lines = []
    for i, (x, y, c) in enumerate(kps):
        flag = "*" if c > 0.3 else " "
        lines.append(f"{flag}{i:>2} {KEYPOINT_NAMES[i]:<14} {c:.2f}")
    return lines


def _joint_lines(joints: dict[str, float]) -> list[str]:
    lines = []
    for name, angle in joints.items():
        deg = np.degrees(angle)
        flag = "*" if angle != 0.0 else " "
        lines.append(f"{flag} {name:<22} {deg:5.1f} deg")
    return lines
