"""
Lab A — Hello YOLO Pose on Video
==================================
Run YOLO pose estimation on a tennis video frame-by-frame using OpenCV.
Displays live skeleton overlay, prints joint angles, and saves an annotated clip.

Usage (from project root):
    python labs/lab_a_hello_yolo.py
    python labs/lab_a_hello_yolo.py --video labs/videos/workout.mp4
    python labs/lab_a_hello_yolo.py --model yolov8x-pose
    python labs/lab_a_hello_yolo.py --save                    # write output video
    python labs/lab_a_hello_yolo.py --every 3                 # process every 3rd frame

Controls (OpenCV window):
    Q / ESC  — quit
    SPACE    — pause / resume
    S        — save current frame as screenshot

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# COCO 17-keypoint names
# ---------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Joint angle chains: (proximal, joint, distal) keypoint indices
LIMB_CHAINS = {
    "L elbow":     (5,  7,  9),
    "R elbow":     (6,  8, 10),
    "L shoulder":  (11, 5,  7),
    "R shoulder":  (12, 6,  8),
    "L knee":      (11, 13, 15),
    "R knee":      (12, 14, 16),
}

DEFAULT_VIDEO = Path("labs/videos/workout.mp4")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def angle_at_b(kps, a: int, b: int, c: int) -> float | None:
    """Joint angle in degrees at keypoint B (chain A-B-C). None if not visible."""
    xa, ya, ca = kps[a]
    xb, yb, cb = kps[b]
    xc, yc, cc = kps[c]
    if ca < 0.3 or cb < 0.3 or cc < 0.3:
        return None
    ba = (xa - xb, ya - yb)
    bc = (xc - xb, yc - yb)
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
]
COLORS = {
    "skeleton": (0, 255, 0),
    "joint":    (0, 0, 255),
    "angle":    (255, 255, 0),
    "info":     (200, 200, 200),
    "paused":   (0, 165, 255),
}


def draw_overlay(frame, kps, angles: dict, frame_idx: int, fps: float) -> None:
    """Draw skeleton, keypoints, angles, and HUD onto frame in-place."""
    # Skeleton bones
    for i, j in SKELETON_PAIRS:
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci > 0.3 and cj > 0.3:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)),
                     COLORS["skeleton"], 2, cv2.LINE_AA)

    # Keypoint dots
    for x, y, c in kps:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 5, COLORS["joint"], -1)

    # Joint angle labels
    chain_kp = {
        "L elbow": 7, "R elbow": 8,
        "L shoulder": 5, "R shoulder": 6,
        "L knee": 13, "R knee": 14,
    }
    for name, angle in angles.items():
        if angle is None:
            continue
        kp_idx = chain_kp[name]
        x, y, c = kps[kp_idx]
        if c > 0.3:
            cv2.putText(frame, f"{angle:.0f} deg", (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["angle"], 1, cv2.LINE_AA)

    # HUD — top-left panel
    hud_lines = [f"Frame: {frame_idx}  |  FPS: {fps:.1f}"]
    for name, angle in angles.items():
        val = f"{angle:.1f} deg" if angle is not None else "--"
        hud_lines.append(f"{name}: {val}")
    hud_lines.append("Q/ESC: quit  SPACE: pause  S: screenshot")

    for i, line in enumerate(hud_lines):
        y_pos = 20 + i * 18
        cv2.putText(frame, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["info"], 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lab A — YOLO Pose on Video or Camera")
    parser.add_argument("--video",  default=str(DEFAULT_VIDEO), help="Path to video file")
    parser.add_argument("--camera", action="store_true",        help="Use live camera instead of video")
    parser.add_argument("--model",  default="yolov8n-pose",     help="YOLO model name or path")
    parser.add_argument("--conf",   type=float, default=0.5,    help="Detection confidence threshold")
    parser.add_argument("--every",  type=int,   default=1,      help="Process every Nth frame (1 = all)")
    parser.add_argument("--save",   action="store_true",        help="Save annotated output video")
    args = parser.parse_args()

    # Load model
    model_name = args.model if args.model.endswith(".pt") else args.model + ".pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Open video file or camera
    if args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera 0")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(30):   # flush black warmup frames on macOS
            cap.grab()
        source_name = "Camera 0"
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            print(f"  Download with: yt-dlp -o labs/videos/workout.mp4 https://www.youtube.com/watch?v=-hSma-BRzoo")
            sys.exit(1)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            sys.exit(1)
        source_name = video_path.name

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0 for live camera
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_label  = f"/{total_frames}" if total_frames > 0 else ""
    print(f"Source: {source_name}  |  {width}x{height}  |  {src_fps:.1f} fps")

    # Output writer
    writer = None
    if args.save:
        out_path = Path("labs/videos/workout_annotated.mp4") if not args.camera \
                   else Path("labs/videos/camera_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))
        print(f"Saving output to: {out_path}")

    WIN = "Lab A — Pose (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(width, 1280), min(height, 720))

    frame_idx = 0
    paused = False
    import time
    t_prev = time.time()

    print("\nRunning — press Q/ESC to quit, SPACE to pause, S to screenshot.\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not args.camera:
                    print("End of video.")
                break
            frame_idx += 1

        # Compute display FPS
        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now

        display = frame.copy()
       # Run YOLO on every Nth frame
        if not paused and frame_idx % args.every == 0:
            results = model(frame, conf=args.conf, verbose=False)
            kps_data = results[0].keypoints

            if kps_data is not None and kps_data.data.shape[0] > 0:
                kps = kps_data.data[0].cpu().numpy()  # (17, 3)
                angles = {name: angle_at_b(kps, a, b, c)
                          for name, (a, b, c) in LIMB_CHAINS.items()}
                draw_overlay(display, kps, angles, frame_idx, fps)

                # Print to console once per second
                if frame_idx % max(1, int(src_fps)) == 0:
                    angle_str = "  ".join(
                        f"{n}: {v:.0f} deg" if v is not None else f"{n}: --"
                        for n, v in angles.items()
                    )
                    print(f"[{frame_idx:>5}{frame_label}]  {angle_str}")
            else:
                cv2.putText(display, "No person detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if paused:
            cv2.putText(display, "PAUSED - press SPACE to resume", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["paused"], 2)

        if writer:
            writer.write(display)

        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            shot_path = Path(f"labs/videos/screenshot_frame{frame_idx:05d}.jpg")
            cv2.imwrite(str(shot_path), display)
            print(f"Screenshot saved: {shot_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
