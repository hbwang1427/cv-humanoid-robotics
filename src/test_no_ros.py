"""
Standalone test: YOLO pose detection + joint mapping, no ROS required.
Supports both live camera and video file input.

Run from project root:
    python src/test_no_ros.py                                  # live camera
    python src/test_no_ros.py --video labs/videos/workout.mp4  # video file
Press Q to quit.
"""

import sys
import os
import argparse
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detector import PoseDetector
from mapper import keypoints_to_joints
from visualizer import Visualizer

DEFAULT_VIDEO = "labs/videos/workout.mp4"


def open_source(video_path: str | None):
    """Return an OpenCV VideoCapture for either a video file or the camera."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        print(f"Reading from video: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera 0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Flush first frames — macOS cameras return black until warmed up
        for _ in range(30):
            cap.grab()
        print("Reading from camera 0")
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Path to video file (omit for live camera)")
    parser.add_argument("--camera", action="store_true",
                        help="Force live camera even if --video default exists")
    args = parser.parse_args()

    video_path = None if args.camera else args.video

    cap = open_source(video_path)
    detector = PoseDetector()
    viz = Visualizer()

    print("Running — press Q in the window to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if video_path:
                    print("End of video.")
                break

            keypoints = detector.detect(frame)

            joints = None
            if keypoints is not None:
                joints = keypoints_to_joints(keypoints)
                frame = detector.draw(frame, keypoints)

            if not viz.show(frame, joints, keypoints):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viz.close()


if __name__ == "__main__":
    main()
