"""
Standalone test: YOLO face locate + dlib landmarks + expression features,
no ROS required. Works on macOS. Supports live camera and video file input.

Run from project root:
    python src/test_face_no_ros.py                                  # live camera
    python src/test_face_no_ros.py --video labs/videos/workout.mp4  # video file
Press Q to quit.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))

from detector import PoseDetector
from face_detector import FaceLandmarkDetector
from face_features import Baseline, classify, extract_features
from face_visualizer import FaceVisualizer


def open_source(video_path: str | None):
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
        for _ in range(30):
            cap.grab()
        print("Reading from camera 0")
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to video file (omit for live camera)")
    parser.add_argument("--calib-frames", type=int, default=30, help="Neutral-face calibration frames")
    args = parser.parse_args()

    cap = open_source(args.video)
    pose_detector = PoseDetector()
    face_detector = FaceLandmarkDetector()
    baseline = Baseline(n_frames=args.calib_frames)
    viz = FaceVisualizer()

    print(f"Calibrating... hold a neutral face ({args.calib_frames} frames)")
    print("Running — press Q in the window to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("End of video.")
                break

            pose_kps = pose_detector.detect(frame)
            landmarks = None
            features = None
            label = "no_face"

            if pose_kps is not None:
                landmarks = face_detector.detect(frame, pose_kps)

            if landmarks is not None:
                features = extract_features(landmarks)
                if not baseline.ready:
                    baseline.add(features)
                    label = "calibrating" if not baseline.ready else "neutral"
                else:
                    label = classify(features, baseline.values)
                frame = face_detector.draw(frame, landmarks)

            if not viz.show(frame, features, label, baseline.ready):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viz.close()


if __name__ == "__main__":
    main()
