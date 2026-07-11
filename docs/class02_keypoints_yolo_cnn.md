# Class 2 — Keypoint Tracking, YOLO Architecture & Real-Time Detection
**Sports Biomechanics Analysis with AI Vision & Robotics**
*Duration: 1 hour | Phase 2 of 7*

---

## Learning Objectives

By the end of this class you will be able to:
- Explain how a CNN extracts spatial features from images
- Describe the YOLO architecture: backbone → neck → head
- Distinguish heatmap-based vs regression-based keypoint detection
- Identify the 17 COCO keypoints and their biomechanical significance for sports
- Tune confidence thresholds and analyze detection quality on the workout video
- Write a script that logs keypoint trajectories over time for a full video

---

## 0. Homework Review (15 min)

Quick check from Class 1:
- To get familiar with ROS 2: Could you list the active ROS 2 topics with `ros2 topic list`?
- Under the output architecture of Yolo
- Add bounding box on top of yolo keypoint extraction

---

## 1. CNN Fundamentals — How a Network Sees an Image (15 min)

### From pixels to features

A raw image is a 3-D array: `(Height, Width, Channels)` — e.g. `(640, 640, 3)` for RGB.
A Convolutional Neural Network learns **spatial filters** that detect edges, curves, and eventually body parts.

```
Input image         Conv layer 1        Conv layer 2        Deep layer
(640x640x3)    →   (320x320x32)   →   (160x160x64)   →  (20x20x512)
 raw pixels          edges/blobs        shapes/parts       semantic concepts
```

Each convolution:
1. Slides a small kernel (e.g. 3×3) across the image
2. Computes a dot product at every position
3. Produces a **feature map** — a spatial activation grid

```python
import torch
import torch.nn as nn

# A single conv layer: 3 input channels -> 32 feature maps, 3x3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

img = torch.randn(1, 3, 640, 640)   # batch of 1 image
features = conv(img)
print(features.shape)  # torch.Size([1, 32, 640, 640])
```

Key properties:
- **Stride** — how far the kernel steps (stride=2 halves resolution)
- **Padding** — border pixels added to preserve size
- **Depth** — number of filters = number of feature maps output
- **ReLU** — activation that adds non-linearity: `f(x) = max(0, x)`

> Deep dive: https://cs231n.github.io/convolutional-networks/

---

## 2. YOLO Architecture (15 min)

https://web.cs.ucdavis.edu/~yjlee/teaching/ecs289g-winter2018/YOLO.pdf

YOLO (You Only Look Once) runs a **single forward pass** — no separate proposal stage.
Version 8 / 11 uses a clean 3-part structure:

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
│   Backbone   │ ──▶ │     Neck     │ ──▶ │       Head        │
│  (CSPDarknet │     │   (PAN-FPN)  │     │  Detection +      │
│   or C2f)    │     │              │     │  Keypoint regs.   │
└──────────────┘     └──────────────┘     └───────────────────┘
  feature extractor   multi-scale fusion   bounding box + 17 kps
```

### Backbone — feature extraction
- Stack of Conv + BatchNorm + SiLU blocks
- Outputs feature maps at 3 scales: `P3 (80x80)`, `P4 (40x40)`, `P5 (20x20)`
- Larger scale = fine details (small objects); smaller scale = coarse semantics (large objects)

### Neck — multi-scale fusion (PANet)
- **FPN** (top-down): passes high-level semantics down to fine-resolution maps
- **PAN** (bottom-up): passes fine spatial detail back up
- Result: every scale benefits from both context and resolution

### Head — predictions per grid cell
For pose estimation, each grid cell predicts:
- 1 objectness score
- 4 box coordinates (cx, cy, w, h)
- 17 × 3 keypoint values (x, y, visibility) = **51 values**

```python
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")

# Inspect model structure
print(model.model)

# Count parameters
params = sum(p.numel() for p in model.model.parameters())
print(f"Parameters: {params/1e6:.1f}M")
```

### Model size vs speed tradeoff

| Model | Params | mAP (pose) | Speed (CPU) |
|-------|--------|------------|-------------|
| yolov8n-pose | 3.3M  | 49.4 | ~30 ms/frame |
| yolov8s-pose | 11.6M | 59.7 | ~55 ms/frame |
| yolov8m-pose | 26.4M | 64.4 | ~120 ms/frame |
| yolov8x-pose | 69.4M | 69.2 | ~400 ms/frame |
| yolo11n-pose | 2.9M  | 51.0 | ~25 ms/frame |

> Model benchmarks: https://docs.ultralytics.com/tasks/pose/#models

---

## 3. History of YOLO — v1 to v8 (10 min)

### Why does the version history matter?

Each YOLO version solved a specific weakness of its predecessor. Understanding this progression helps you choose the right model and interpret the architecture decisions still present in v8/v11 today.

---

### YOLOv1 — 2015 (Redmon et al.)
> Paper: https://arxiv.org/abs/1506.02640

The original insight: **treat detection as a single regression problem**.

```
Input (448x448) → 24 Conv layers (Darknet) → 7x7 grid → [boxes + class probs]
```

- Divided image into a **7×7 grid**; each cell predicted 2 boxes + class probabilities
- Single forward pass → real-time detection (45 FPS on GPU)
- **Weakness**: each cell only predicts one class, struggles with small/clustered objects

```
┌───────┬───────┬───────┐
│  box  │  box  │       │
│  cls  │  cls  │  ...  │   7x7 grid output
└───────┴───────┴───────┘
```

---

### YOLOv2 / YOLO9000 — 2016 (Redmon & Farhadi)
> Paper: https://arxiv.org/abs/1612.08242

Key improvements over v1:

| Change | Effect |
|--------|--------|
| **Anchor boxes** (k-means on training data) | Better shape priors, higher recall |
| **Batch Normalization** on all layers | Faster convergence, dropped Dropout |
| **Multi-scale training** (320→608px) | More robust across object sizes |
| Darknet-19 backbone | Faster than v1's network |

- Introduced **YOLO9000**: joint training on ImageNet + COCO → detects 9000 classes
- Still single scale output

---

### YOLOv3 — 2018 (Redmon & Farhadi)
> Paper: https://arxiv.org/abs/1804.02767

The version that made YOLO mainstream.

```
Darknet-53 backbone (53 conv layers, residual connections)
         ↓
3 detection scales: 13x13, 26x26, 52x52
         ↓
9 anchor boxes (3 per scale, sized by k-means)
```

- **Residual skip connections** — borrowed from ResNet, enabled deeper network
- **Multi-scale detection**: large anchors on 13×13 (big objects), small anchors on 52×52 (small objects)
- **Independent logistic classifiers** instead of softmax → multi-label detection
- ~50 FPS on Titan X — the go-to real-time detector for years

---

### YOLOv4 — 2020 (Bochkovskiy, Wang, Liao)
> Paper: https://arxiv.org/abs/2004.10934

Redmon left computer vision; community took over. v4 is a comprehensive engineering effort.

```
CSPDarknet-53      PANet neck        YOLOv3 head
  (backbone)    →  (feature fusion) →  (3 scales)
```

Key innovations:
| Technique | What it does |
|-----------|-------------|
| **CSP (Cross Stage Partial)** | Splits feature map, reduces computation 20% |
| **PANet neck** | Bottom-up path augmentation — better spatial detail flow |
| **Mosaic augmentation** | 4 images tiled → richer context, smaller batch needed |
| **CIoU loss** | Better bounding box regression than MSE |
| **DropBlock** | Structured dropout more effective than random |

---

### YOLOv5 — 2020 (Ultralytics)
> Repo: https://github.com/ultralytics/yolov5

No paper — released as open-source code. First fully **PyTorch-native** YOLO.

```
CSP backbone  →  PANet neck  →  3-scale head
(4 sizes: n/s/m/l/x)
```

- Introduced the **n/s/m/l/x** model family (nano to xlarge)
- Excellent tooling: easy training, export to ONNX/TFLite/CoreML
- Anchor boxes still used (automatically tuned via k-means on your dataset)
- Still widely deployed in production today

---

### YOLOv6 — 2022 (Meituan)
> Paper: https://arxiv.org/abs/2209.02976

Industry-focused, optimized for **edge deployment**.

- **EfficientRep backbone**: hardware-friendly RepVGG-style re-parameterization
- **Rep-PAN neck**: same reparameterization trick in neck
- **TAL (Task-Aligned Learning)**: unified cls + reg loss alignment
- Strong on mobile/edge chips (not Ultralytics ecosystem)

---

### YOLOv7 — 2022 (Wang, Bochkovskiy, Liao)
> Paper: https://arxiv.org/abs/2207.02696

Focus: **maximum accuracy** while keeping real-time speed.

```
ELAN backbone  →  MP neck  →  Auxiliary + lead head
```

| Innovation | Effect |
|-----------|--------|
| **ELAN (Efficient Layer Aggregation Network)** | Deeper gradient paths, better feature reuse |
| **Compound scaling** | Co-scales depth/width/resolution |
| **Auxiliary decoupled head** | Trains with extra supervision, discarded at inference |
| **Coarse-to-fine lead guided label assignment** | Better training signal |

- State-of-the-art at time of release: 56.8 AP on COCO at 30 FPS

---

### YOLOv8 — 2023 (Ultralytics)
> Docs: https://docs.ultralytics.com/models/yolov8/

The current standard. Biggest change: **anchor-free detection**.

```
C2f backbone  →  PAN-FPN neck  →  Decoupled head (anchor-free)
```

| Change from v5 | Why |
|---------------|-----|
| **Anchor-free** | No anchor tuning needed, cleaner generalization |
| **C2f blocks** (Cross-Stage with 2 bottlenecks) | Better gradient flow than CSP |
| **Decoupled head** | Separate cls + reg branches → better accuracy |
| **Unified framework** | detect / segment / pose / classify / track in one repo |
| **Pose head** outputs 17 × (x, y, vis) | Native keypoint support → what we use |

```python
# The pose head output per anchor point:
# [cx, cy, w, h, obj_conf] + [x1,y1,v1, x2,y2,v2, ... x17,y17,v17]
#                                         ↑ 51 keypoint values
```

---

### Architecture comparison at a glance

```
Version  Year  Backbone        Neck        Head        Anchor  Scales
───────────────────────────────────────────────────────────────────────
v1       2015  Darknet-24      -           Coupled     Yes     1
v2       2016  Darknet-19      -           Coupled     Yes     1
v3       2018  Darknet-53      FPN         Coupled     Yes     3
v4       2020  CSPDarknet-53   PANet       Coupled     Yes     3
v5       2020  CSP             PANet       Coupled     Yes     3
v6       2022  EfficientRep    Rep-PAN     Decoupled   No      3
v7       2022  ELAN            MP          Aux+Lead    Yes     3
v8       2023  C2f             PAN-FPN     Decoupled   No      3
v11      2024  C2f-Attention   PAN-FPN     Decoupled   No      3
```

### What changed in YOLOv9, v10, v11?

- **v9 (2024)**: GELAN backbone + Programmable Gradient Information (PGI) — better gradient flow
- **v10 (2024)**: NMS-free dual-label assignment — removes post-processing bottleneck
- **v11 (2024)**: C2f with attention (C2fAttn) — Ultralytics default, ~10% better than v8 at same speed

For our project we use **v8n-pose** (nano, real-time on CPU) or **v11n-pose** (slightly better accuracy).

---

## 4. Keypoint Detection Deep Dive (10 min)

### Two approaches

**Heatmap-based** (classic — HRNet, OpenPose):
- Network outputs one heatmap per keypoint
- Peak of the heatmap = predicted keypoint location
- Accurate but slow (needs upsampling to full resolution)

**Regression-based** (YOLO approach):
- Network directly predicts `(x, y, visibility)` per keypoint
- Fast — no upsampling needed
- Slightly less precise on small/occluded joints

### What does confidence (visibility) mean?

YOLO keypoint output: `kps[i] = [x, y, conf]`

| conf range | Meaning |
|-----------|---------|
| `> 0.7` | Keypoint clearly visible |
| `0.3–0.7` | Partially occluded or uncertain |
| `< 0.3` | Not reliably detected — discard |

### Biomechanical keypoints for sports

For **tennis serve** analysis:
```
Critical:   right_shoulder (6), right_elbow (8), right_wrist (10)
            left_shoulder (5),  left_hip (11),   right_hip (12)
Supporting: nose (0) — head tilt during toss
            left_ankle (15), right_ankle (16) — foot position
```

For **golf swing** analysis:
```
Critical:   left_shoulder (5), right_shoulder (6)   — shoulder turn
            left_hip (11),     right_hip (12)        — hip rotation
            left_wrist (9),    right_wrist (10)      — club face angle
Supporting: left_knee (13), right_knee (14)          — knee flex
```

---

## 5. Real-Time Detection Considerations (5 min)

### Bottlenecks in our pipeline

```
Camera/Video read    ~1  ms
YOLO inference       ~30 ms  ← main bottleneck on CPU
Keypoint mapping     ~0.1 ms
ROS 2 publish        ~1  ms
OpenCV display       ~5  ms
────────────────────────────
Total                ~37 ms  ≈ 27 FPS (CPU)
With GPU:            ~8  ms  ≈ 125 FPS
```

### Speed optimizations

```python
from ultralytics import YOLO
import torch

model = YOLO("yolov8n-pose.pt")

# 1. Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Process every 2nd frame for 2x speed (motion is slow between frames)
# (already supported via --every flag in our labs)

# 3. Reduce input resolution (default 640 -> 320)
results = model(frame, imgsz=320, conf=0.5, verbose=False)

# 4. Export to ONNX or TensorRT for production
model.export(format="onnx")   # creates yolov8n-pose.onnx
```

> Optimization guide: https://docs.ultralytics.com/guides/model-optimization-triton/

---

## 6. Lab Exercises (10 min)

### Lab A2 — Keypoint confidence analysis on the workout video

```python
# labs/lab_a2_confidence_analysis.py
# Run: python labs/lab_a2_confidence_analysis.py

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

model = YOLO("yolov8n-pose.pt")
cap   = cv2.VideoCapture("labs/videos/workout.mp4")

# Accumulate confidence per keypoint across all frames
conf_sum   = np.zeros(17)
conf_count = np.zeros(17)
frame_count = 0

print("Analyzing video... (press Ctrl-C to stop early)\n")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 5 != 0:   # sample every 5th frame
        continue

    results = model(frame, conf=0.3, verbose=False)
    if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        continue

    kps = results[0].keypoints.data[0].cpu().numpy()
    conf_sum   += kps[:, 2]
    conf_count += 1

cap.release()

# Report
print(f"Sampled {int(conf_count.max())} frames\n")
print(f"{'#':>2}  {'Keypoint':<18} {'Avg Conf':>9}  {'Visibility':>10}")
print("-" * 50)
for i, name in enumerate(KEYPOINT_NAMES):
    avg = conf_sum[i] / max(conf_count[i], 1)
    bar = "#" * int(avg * 20)
    print(f"{i:>2}  {name:<18} {avg:>9.3f}  {bar}")
```

**Questions to answer:**
- Which keypoints are most/least reliably detected in this video?
- Why might ankles have lower confidence than shoulders?
- How would camera angle affect confidence for a tennis serve?

---

### Lab A3 — Keypoint trajectory over time

```python
# labs/lab_a3_trajectory.py
# Plots wrist height over time — useful for detecting the serve toss
# Run: python labs/lab_a3_trajectory.py

import cv2
import numpy as np
from ultralytics import YOLO

model  = YOLO("yolov8n-pose.pt")
cap    = cv2.VideoCapture("labs/videos/workout.mp4")
fps    = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

R_WRIST = 10   # right wrist keypoint index
L_WRIST = 9

timestamps, r_wrist_y, l_wrist_y = [], [], []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model(frame, conf=0.3, verbose=False)
    if results[0].keypoints is None or results[0].keypoints.data.shape[0] == 0:
        continue

    kps = results[0].keypoints.data[0].cpu().numpy()
    t   = frame_idx / fps

    rw_conf = kps[R_WRIST, 2]
    lw_conf = kps[L_WRIST, 2]

    timestamps.append(t)
    r_wrist_y.append(kps[R_WRIST, 1] if rw_conf > 0.3 else None)
    l_wrist_y.append(kps[L_WRIST, 1] if lw_conf > 0.3 else None)

cap.release()

# ASCII plot of wrist Y position over time
print(f"\nRight wrist Y trajectory (lower Y = higher in frame)\n")
print(f"Time(s)  Y-pos")
print("-" * 30)
for t, y in zip(timestamps[::10], r_wrist_y[::10]):
    if y is None:
        print(f"{t:6.1f}s  [not detected]")
    else:
        bar = " " * int(y / height * 40) + "|"
        print(f"{t:6.1f}s  {bar}  {y:.0f}px")
```

---

### Lab A4 — Keypoint Smoothness Analysis

File: `labs/lab_a4_smoothness.py`

Measures **velocity**, **acceleration**, **jerk**, and **SPARC score** for any keypoint
across the full video. Jerk (rate of acceleration change) is the standard biomechanical
measure of movement smoothness — used in rehab and elite sports coaching.

```bash
# Right wrist (default) — golf club / tennis racket hand
python labs/lab_a4_smoothness.py

# Both wrists
python labs/lab_a4_smoothness.py --keypoints 9 10

# Hip rotation (golf)
python labs/lab_a4_smoothness.py --keypoints 11 12

# Show matplotlib speed / accel / jerk chart
python labs/lab_a4_smoothness.py --plot

# Compare models
python labs/lab_a4_smoothness.py --model yolov8x-pose --plot
```

**Expected output:**
```
SMOOTHNESS REPORT
=================================================================
Keypoint : right_wrist
  Detected       : 412 frames  (91.8%)
  Speed (RMS)    :    184.3 px/s
  Speed (max)    :   1203.7 px/s
  Accel (RMS)    :   4821.0 px/s^2
  Jerk  (RMS)    :  98432.1 px/s^3   <- smoothness
  Jerk  (max)    : 512034.8 px/s^3
  SPARC score    :    -3.241  (0=smooth, -inf=jerky)
  Grade          : Good
```

| Jerk RMS | Grade |
|----------|-------|
| < 500 | Excellent |
| 500–2000 | Good |
| 2000–5000 | Fair |
| > 5000 | Jerky |

The **SPARC score** (Spectral Arc Length) measures smoothness in the frequency domain —
it is negative, with values closer to 0 meaning smoother movement.

---

## 7. Take-Home Tasks

1. **Confidence report**: Run `lab_a2_confidence_analysis.py` on the workout video. Screenshot or copy the output table. Which 3 keypoints have the lowest average confidence, and what does that tell you about the camera setup?

2. **Model comparison**: Modify `lab_a_hello_yolo.py` to run both `yolov8n-pose` and `yolov8x-pose` on the same 10 frames and compare:
   - Inference time per frame
   - Number of keypoints with confidence > 0.5

3. **Sport-specific keypoints**: Choose either tennis or golf. List the 6 most important keypoints for your chosen sport and explain what joint angle each pair reveals about technique. Write 5–10 sentences.

4. **Smoothness analysis**: Run `lab_a4_smoothness.py` on the workout video.
   - What is the SPARC score for the right wrist?
   - At what timestamps (seconds) does jerk spike highest? What movement is happening there?
   - Re-run with `--keypoints 11 12` for hip rotation. Is hip movement smoother or jerkier than the wrist? Why?

5. Write a simplified yolo model and train it on hundreds of images on your local laptop

---

## Resources

| Resource | URL |
|----------|-----|
| CS231n — CNNs for Visual Recognition | https://cs231n.github.io/convolutional-networks/ |
| YOLO v8 architecture paper | https://arxiv.org/abs/2305.09972 |
| Ultralytics pose docs | https://docs.ultralytics.com/tasks/pose/ |
| YOLO model comparison | https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes |
| COCO keypoints format | https://cocodataset.org/#keypoints-2020 |
| HRNet (heatmap approach) | https://arxiv.org/abs/1908.07919 |
| OpenPose paper | https://arxiv.org/abs/1812.08008 |
| PyTorch Conv2d docs | https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html |
| YOLO optimization guide | https://docs.ultralytics.com/guides/model-optimization-triton/ |

---

## Class Schedule Overview

| Class | Topic |
|-------|-------|
| 1 | System Architecture, Setup & Toolchains |
| **2** | **Keypoint Tracking, YOLO & CNN Deep Dive** ← you are here |
| 3 | Facial Expression Recognition & Robot Face Mapping |
| 4 | Kinematic Math — Angles from Keypoints |
| 5 | ROS 2 Publisher & Joint State Messages |
| 6 | RViz2 Visualization & URDF Tuning |
| 7 | Sports Analysis — Tennis Serve / Golf Swing Scoring |
