"""Debug window showing the video/camera feed with dlib landmarks, feature
bars, and the classified expression label."""

import cv2
import numpy as np


class FaceVisualizer:
    WINDOW = "Face Mimic — press Q to quit"

    def __init__(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def show(self,
             frame: np.ndarray,
             features: dict[str, float] | None,
             label: str,
             calibrated: bool) -> bool:
        """Display frame with a feature panel and expression label. Returns False on Q."""
        display = frame.copy()
        h, w = display.shape[:2]

        if features is not None:
            self._draw_panel(display, "FEATURES", 0, _feature_lines(features), h)

        color = (0, 255, 0) if calibrated else (0, 200, 255)
        cv2.putText(display, label.upper(), (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow(self.WINDOW, display)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def _draw_panel(self, img, title: str, x: int, lines: list[str], frame_h: int):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, pad, lh = 0.4, 6, 15

        panel_w = 220
        panel_h = pad + lh + (lh * len(lines)) + pad
        overlay = img.copy()
        cv2.rectangle(overlay, (x, 0), (x + panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        cv2.putText(img, title, (x + pad, pad + lh - 3),
                    font, fs, (255, 255, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (x + pad, pad + lh + (i + 1) * lh),
                        font, fs, (160, 255, 160), 1, cv2.LINE_AA)

    def close(self):
        cv2.destroyAllWindows()


def _feature_lines(features: dict[str, float]) -> list[str]:
    return [f"{name:<10} {value:6.3f}" for name, value in features.items()]
