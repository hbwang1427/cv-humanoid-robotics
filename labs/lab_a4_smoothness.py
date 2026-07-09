"""
Lab A4 — Keypoint Smoothness Analysis
=======================================
Tracks selected keypoints across all frames and measures motion smoothness
using three biomechanical metrics:

  Velocity     — speed of movement (px/s)
  Acceleration — rate of velocity change (px/s^2)
  Jerk         — rate of acceleration change (px/s^3)  ← smoothness indicator
  SPARC score  — Spectral Arc Length: -1 (very smooth) to 0 (jerky)

Lower jerk and higher (less negative) SPARC = smoother movement.
Used in rehab and sports science to quantify movement quality.

Usage (from project root):
    python labs/lab_a4_smoothness.py                        # right wrist (golf/tennis)
    python labs/lab_a4_smoothness.py --keypoints 9 10       # both wrists
    python labs/lab_a4_smoothness.py --keypoints 11 12      # hips (rotation)
    python labs/lab_a4_smoothness.py --plot                 # show matplotlib chart
    python labs/lab_a4_smoothness.py --video labs/videos/workout.mp4
"""

import argparse
import math
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
DEFAULT_KPS   = [10]   # right wrist — key for golf/tennis


# ---------------------------------------------------------------------------
# Smoothness metrics
# ---------------------------------------------------------------------------

def finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference derivative, forward/backward at edges."""
    return np.gradient(values, dt)


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def sparc(speed: np.ndarray, fps: float, cutoff: float = 10.0) -> float:
    """
    Spectral Arc Length — measures smoothness in frequency domain.
    Range: 0 (perfectly smooth) to -inf (maximally jerky).
    Padry et al. 2012: https://doi.org/10.1371/journal.pone.0031414
    """
    N = len(speed)
    if N < 4:
        return float("nan")
    freq    = np.fft.rfftfreq(N, d=1.0 / fps)
    amp     = np.abs(np.fft.rfft(speed)) / N
    amp_hat = amp / (amp[0] + 1e-9)          # normalize

    # Only use frequencies up to cutoff
    mask = freq <= cutoff
    freq = freq[mask]
    amp_hat = amp_hat[mask]

    # Arc length of the amplitude spectrum
    dfreq = freq[1] - freq[0] if len(freq) > 1 else 1.0
    darc  = np.sqrt((dfreq ** 2) + np.diff(amp_hat, prepend=amp_hat[0]) ** 2)
    return -float(np.sum(darc))


def analyze_keypoint(positions: np.ndarray, fps: float, name: str) -> dict:
    """
    positions: (N, 2) array of (x, y) in pixels, NaN where not detected.
    Returns dict of smoothness metrics.
    """
    # Interpolate over missing (NaN) frames
    t = np.arange(len(positions))
    valid = ~np.isnan(positions[:, 0])
    if valid.sum() < 10:
        return None

    x = np.interp(t, t[valid], positions[valid, 0])
    y = np.interp(t, t[valid], positions[valid, 1])

    dt = 1.0 / fps

    vx = finite_diff(x, dt)
    vy = finite_diff(y, dt)
    speed = np.sqrt(vx**2 + vy**2)

    ax_ = finite_diff(vx, dt)
    ay_ = finite_diff(vy, dt)
    accel = np.sqrt(ax_**2 + ay_**2)

    jx = finite_diff(ax_, dt)
    jy = finite_diff(ay_, dt)
    jerk = np.sqrt(jx**2 + jy**2)

    return {
        "name":          name,
        "frames":        int(valid.sum()),
        "detect_pct":    100.0 * valid.sum() / len(positions),
        "speed_rms":     rms(speed),
        "speed_max":     float(speed.max()),
        "accel_rms":     rms(accel),
        "jerk_rms":      rms(jerk),
        "jerk_max":      float(jerk.max()),
        "sparc":         sparc(speed, fps),
        "speed":         speed,
        "accel":         accel,
        "jerk":          jerk,
        "x":             x,
        "y":             y,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def smoothness_grade(jerk_rms: float) -> str:
    if jerk_rms < 500:    return "Excellent"
    if jerk_rms < 2000:   return "Good"
    if jerk_rms < 5000:   return "Fair"
    return "Jerky"


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  SMOOTHNESS REPORT")
    print("=" * 65)
    for r in results:
        if r is None:
            continue
        grade = smoothness_grade(r["jerk_rms"])
        print(f"\nKeypoint : {r['name']}")
        print(f"  Detected       : {r['frames']} frames  ({r['detect_pct']:.1f}%)")
        print(f"  Speed (RMS)    : {r['speed_rms']:8.1f} px/s")
        print(f"  Speed (max)    : {r['speed_max']:8.1f} px/s")
        print(f"  Accel (RMS)    : {r['accel_rms']:8.1f} px/s^2")
        print(f"  Jerk  (RMS)    : {r['jerk_rms']:8.1f} px/s^3   <- smoothness")
        print(f"  Jerk  (max)    : {r['jerk_max']:8.1f} px/s^3")
        print(f"  SPARC score    : {r['sparc']:8.3f}  (0=smooth, -inf=jerky)")
        print(f"  Grade          : {grade}")
    print()


def ascii_timeseries(values: np.ndarray, fps: float, label: str,
                     n_points: int = 60) -> None:
    """Print a compact ASCII time-series strip."""
    stride = max(1, len(values) // n_points)
    sampled = values[::stride]
    vmax = sampled.max() + 1e-9

    print(f"\n{label}:")
    print("  " + "─" * n_points)
    line = "  "
    for v in sampled[:n_points]:
        level = int((v / vmax) * 7)
        line += " ▁▂▃▄▅▆▇"[level]
    print(line)
    print(f"  0{'─' * (n_points//2 - 4)}time{'─' * (n_points//2 - 3)}{len(values)/fps:.1f}s")


# ---------------------------------------------------------------------------
# Optional matplotlib chart
# ---------------------------------------------------------------------------

def plot_results(results: list[dict], fps: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 8), squeeze=False)
    fig.suptitle("Keypoint Smoothness Analysis", fontsize=14)

    for col, r in enumerate(results):
        if r is None:
            continue
        t = np.arange(len(r["speed"])) / fps

        axes[0][col].plot(t, r["speed"],  color="steelblue")
        axes[0][col].set_title(f"{r['name']} — Speed")
        axes[0][col].set_ylabel("px/s")

        axes[1][col].plot(t, r["accel"], color="darkorange")
        axes[1][col].set_title("Acceleration")
        axes[1][col].set_ylabel("px/s^2")

        axes[2][col].plot(t, r["jerk"],  color="crimson")
        axes[2][col].set_title(f"Jerk  (SPARC={r['sparc']:.2f})")
        axes[2][col].set_ylabel("px/s^3")
        axes[2][col].set_xlabel("Time (s)")

    plt.tight_layout()
    out = Path("labs/videos/smoothness_plot.png")
    plt.savefig(out, dpi=120)
    print(f"Chart saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",     default=DEFAULT_VIDEO)
    parser.add_argument("--model",     default="yolov8n-pose")
    parser.add_argument("--keypoints", type=int, nargs="+", default=DEFAULT_KPS,
                        help="Keypoint indices to analyze (0–16)")
    parser.add_argument("--conf",      type=float, default=0.3)
    parser.add_argument("--plot",      action="store_true",
                        help="Show matplotlib speed/accel/jerk chart")
    args = parser.parse_args()

    model_name = args.model if args.model.endswith(".pt") else args.model + ".pt"
    print(f"Model : {model_name}")
    print(f"Video : {args.video}")
    print(f"Tracking: {[KEYPOINT_NAMES[k] for k in args.keypoints]}\n")

    model = YOLO(model_name)
    cap   = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames @ {fps:.1f} fps\n")

    # Storage: {kp_idx: [(x, y) or (nan, nan), ...]}
    trajectories = {k: [] for k in args.keypoints}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)
        has_person = (results[0].keypoints is not None and
                      results[0].keypoints.data.shape[0] > 0)

        if has_person:
            kps = results[0].keypoints.data[0].cpu().numpy()
            for k in args.keypoints:
                x, y, c = kps[k]
                if c > args.conf:
                    trajectories[k].append((float(x), float(y)))
                else:
                    trajectories[k].append((float("nan"), float("nan")))
        else:
            for k in args.keypoints:
                trajectories[k].append((float("nan"), float("nan")))

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total} frames...")

    cap.release()
    print(f"Done. {frame_idx} frames processed.\n")

    # Analyze each keypoint
    analysis = []
    for k in args.keypoints:
        pos = np.array(trajectories[k])   # (N, 2)
        result = analyze_keypoint(pos, fps, KEYPOINT_NAMES[k])
        analysis.append(result)

    # Print report
    print_report(analysis)

    # ASCII sparklines
    for r in analysis:
        if r is None:
            continue
        ascii_timeseries(r["speed"], fps, f"{r['name']} speed (px/s)")
        ascii_timeseries(r["jerk"],  fps, f"{r['name']} jerk  (px/s^3)")

    # Optional matplotlib chart
    if args.plot:
        plot_results(analysis, fps)


if __name__ == "__main__":
    main()
