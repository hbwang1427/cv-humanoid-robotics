"""
Lab A3 — Keypoint Trajectory Over Time
========================================
Tracks selected keypoints across every frame and plots their Y-position
trajectory as an ASCII chart. Useful for spotting serve tosses, swing phases,
or jump events in sports video.

Usage (from project root):
    python labs/lab_a3_trajectory.py
    python labs/lab_a3_trajectory.py --keypoints 9 10    # left+right wrist
    python labs/lab_a3_trajectory.py --keypoints 13 14   # left+right knee
    python labs/lab_a3_trajectory.py --video labs/videos/workout.mp4
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DEFAULT_VIDEO  = "labs/videos/workout.mp4"
DEFAULT_KPS    = [9, 10]   # left wrist, right wrist


def ascii_plot(times, values, label: str, frame_h: int, width: int = 50):
    """Print a simple ASCII time-series chart."""
    print(f"\n--- {label} Y-position over time (lower = higher in frame) ---")
    print(f"{'Time':>6}  {'Position (px)':^{width}}")
    print("-" * (width + 12))
    for t, y in zip(times, values):
        if y is None:
            print(f"{t:6.2f}s  [not detected]")
        else:
            pos = int((y / frame_h) * width)
            bar = " " * pos + "|"
            print(f"{t:6.2f}s  {bar:<{width}}  {y:.0f}px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     default=DEFAULT_VIDEO)
    parser.add_argument("--model",     default="yolov8n-pose")
    parser.add_argument("--keypoints", type=int, nargs="+", default=DEFAULT_KPS,
                        help="Keypoint indices to track (0-16)")
    parser.add_argument("--conf",      type=float, default=0.3)
    parser.add_argument("--stride",    type=int, default=1,
                        help="Print every Nth data point in the chart")
    args = parser.parse_args()

    model_name = args.model if args.model.endswith(".pt") else args.model + ".pt"
    model = YOLO(model_name)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Tracking keypoints: {[KEYPOINT_NAMES[k] for k in args.keypoints]}")
    print(f"Video: {args.video}  |  {total} frames @ {fps:.1f} fps\n")

    # Storage: {kp_index: [(time, y_or_None), ...]}
    trajectories = {k: [] for k in args.keypoints}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps

        results = model(frame, conf=args.conf, verbose=False)
        if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
            for k in args.keypoints:
                trajectories[k].append((t, None))
            continue

        kps = results[0].keypoints.data[0].cpu().numpy()
        for k in args.keypoints:
            x, y, c = kps[k]
            trajectories[k].append((t, float(y) if c > args.conf else None))

    cap.release()
    print(f"Processed {frame_idx} frames.\n")

    # Summary stats
    for k in args.keypoints:
        data = [y for _, y in trajectories[k] if y is not None]
        if data:
            print(f"{KEYPOINT_NAMES[k]}: detected in {len(data)}/{frame_idx} frames  "
                  f"|  Y min={min(data):.0f}  max={max(data):.0f}  "
                  f"range={max(data)-min(data):.0f}px")
        else:
            print(f"{KEYPOINT_NAMES[k]}: never detected")

    # ASCII plots (downsampled)
    for k in args.keypoints:
        times  = [t for t, _ in trajectories[k][::args.stride]]
        values = [y for _, y in trajectories[k][::args.stride]]
        ascii_plot(times, values, KEYPOINT_NAMES[k], height)


if __name__ == "__main__":
    main()
