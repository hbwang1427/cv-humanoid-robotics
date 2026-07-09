"""
Lab A2 — Keypoint Confidence Analysis
======================================
Samples every 5th frame of the workout video and reports the average
detection confidence for each of the 17 COCO keypoints.

Usage (from project root):
    python labs/lab_a2_confidence_analysis.py
    python labs/lab_a2_confidence_analysis.py --video labs/videos/workout.mp4
    python labs/lab_a2_confidence_analysis.py --model yolov8x-pose
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

DEFAULT_VIDEO = "labs/videos/workout.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default=DEFAULT_VIDEO)
    parser.add_argument("--model",  default="yolov8n-pose")
    parser.add_argument("--every",  type=int, default=5, help="Sample every Nth frame")
    parser.add_argument("--conf",   type=float, default=0.3)
    args = parser.parse_args()

    model_name = args.model if args.model.endswith(".pt") else args.model + ".pt"
    print(f"Model : {model_name}")
    print(f"Video : {args.video}\n")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    conf_sum   = np.zeros(17)
    conf_count = np.zeros(17)
    frame_idx  = 0
    detected   = 0

    print(f"Analyzing {total} frames (sampling every {args.every}th)...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.every != 0:
            continue

        results = model(frame, conf=args.conf, verbose=False)
        if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
            continue

        kps = results[0].keypoints.data[0].cpu().numpy()
        conf_sum   += kps[:, 2]
        conf_count += 1
        detected   += 1

    cap.release()
    sampled = frame_idx // args.every

    print(f"Frames sampled : {sampled}")
    print(f"Frames with detection : {detected} ({100*detected/max(sampled,1):.1f}%)\n")
    print(f"{'#':>2}  {'Keypoint':<18} {'Avg Conf':>9}  Visibility bar")
    print("-" * 58)

    for i, name in enumerate(KEYPOINT_NAMES):
        avg = conf_sum[i] / max(conf_count[i], 1)
        filled = int(avg * 30)
        bar = "[" + "#" * filled + "-" * (30 - filled) + "]"
        flag = " LOW" if avg < 0.4 else ""
        print(f"{i:>2}  {name:<18} {avg:>9.3f}  {bar}{flag}")


if __name__ == "__main__":
    main()
