"""
Full pipeline: video/camera -> YOLO pose -> joint mapping -> ROS 2 -> RViz2.

Usage:
    source /opt/ros/humble/setup.bash
    python src/main.py                          # workout video (default)
    python src/main.py --camera                 # live camera
    python src/main.py --video path/to/clip.mp4 # custom video

Then in a separate terminal:
    rviz2 -d config/robot_view.rviz
"""

import argparse
import sys
import os

import cv2
import rclpy

sys.path.insert(0, os.path.dirname(__file__))

from detector import PoseDetector
from mapper import keypoints_to_joints, remap_joints
from ros_publisher import RobotPublisher
from robot_registry import get_urdf_path, get_joint_map, list_robots, DEFAULT_ROBOT
from visualizer import Visualizer

DEFAULT_VIDEO = "labs/videos/workout.mp4"


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
    parser = argparse.ArgumentParser(description="Pose mimic full pipeline")
    parser.add_argument("--video",       default=DEFAULT_VIDEO,  help="Path to video file")
    parser.add_argument("--camera",      action="store_true",    help="Use live camera instead of video")
    parser.add_argument("--robot",       default=DEFAULT_ROBOT,  help="Robot model key (see --list-robots)")
    parser.add_argument("--list-robots", action="store_true",    help="Print available robots and exit")
    args = parser.parse_args()

    if args.list_robots:
        list_robots()
        return

    video_path = None if args.camera else args.video
    urdf_path  = get_urdf_path(args.robot)
    joint_map  = get_joint_map(args.robot)
    print(f"Robot: {args.robot}  ({urdf_path})")
    if joint_map:
        print(f"Joint remap: {len(joint_map)} driven joints -> {args.robot} URDF names")

    rclpy.init()
    publisher = RobotPublisher(urdf_path)
    cap = open_source(video_path)
    detector = PoseDetector()
    viz = Visualizer()

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
                publisher.publish_joints(remap_joints(joints, joint_map))
                frame = detector.draw(frame, keypoints)

            if not viz.show(frame, joints, keypoints):
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
