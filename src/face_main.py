"""
Full facial-expression pipeline: camera/video -> YOLO face locate -> dlib
68 landmarks -> geometric features -> face robot joints -> ROS 2 -> RViz2.

Usage:
    source /opt/ros/humble/setup.bash
    python src/face_main.py                          # webcam
    python src/face_main.py --video path/to/clip.mp4  # video file

Then, in separate terminals:
    ros2 run robot_state_publisher robot_state_publisher \\
      --ros-args -p robot_description:="$(cat robot/face.urdf)" \\
      -r joint_states:=/face/joint_states -r robot_description:=/face/robot_description
    rviz2 -d config/face_view.rviz
"""

import argparse
import os
import sys

import cv2
import rclpy

sys.path.insert(0, os.path.dirname(__file__))

from detector import PoseDetector
from face_detector import FaceLandmarkDetector
from face_features import Baseline, classify, extract_features
from face_mapper import features_to_joints
from face_publisher import FacePublisher
from face_visualizer import FaceVisualizer

FACE_URDF = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "robot", "face.urdf"))


def open_source(video_path: str | None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        print(f"Source: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera 0")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(30):
            cap.grab()
        print("Source: camera 0")
    return cap


def main():
    parser = argparse.ArgumentParser(description="Facial expression -> robot face pipeline")
    parser.add_argument("--video", default=None, help="Path to video file (default: webcam)")
    parser.add_argument("--calib-frames", type=int, default=30, help="Neutral-face calibration frames")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor for joint output")
    args = parser.parse_args()

    print(f"Face robot: {FACE_URDF}")

    rclpy.init()
    publisher = FacePublisher(FACE_URDF, alpha=args.alpha)
    cap = open_source(args.video)
    pose_detector = PoseDetector()
    face_detector = FaceLandmarkDetector()
    baseline = Baseline(n_frames=args.calib_frames)
    viz = FaceVisualizer()

    print(f"Calibrating... hold a neutral face ({args.calib_frames} frames)")

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
                    joints = features_to_joints(features, baseline.values)
                    label = classify(features, baseline.values)
                    publisher.publish_face(joints, label)
                frame = face_detector.draw(frame, landmarks)

            if not viz.show(frame, features, label, baseline.ready):
                break

            rclpy.spin_once(publisher, timeout_sec=0)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        viz.close()
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
