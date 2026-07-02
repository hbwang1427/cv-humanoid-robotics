# Class 1 — System Architecture, Workspace Setup, and Toolchains
**Sports Biomechanics Analysis with AI Vision & Robotics**
*Duration: 1 hour | Phase 1 of 6*

---

## Learning Objectives

By the end of this class you will be able to:
- Describe the full end-to-end pipeline from camera to RViz2
- Explain ROS 2 core concepts: Nodes, Topics, Publishers, Subscribers
- Set up a working Python environment with YOLO and ROS 2
- Run a live YOLOv8/v11 pose inference on a sample image
- Visualize a URDF robot model in RViz2

---

## 0. Project Vision (5 min)

### What are we building?

A real-time sports biomechanics coach that:
1. Watches an athlete via webcam (or video file)
2. Detects their body pose using AI
3. Computes joint angles (elbow, shoulder, hip, knee…)
4. Mirrors that motion onto a 3-D robot skeleton
5. Displays live feedback in RViz2 — coaching form in real time

### Target sports
- **Tennis** — serve angle, backswing range, hip rotation
- **Golf** — club-path, spine tilt, wrist hinge at impact

```
┌──────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────┐
│  Webcam  │───▶│  YOLO Pose Det. │───▶│  Kinematic Math  │───▶│  ROS 2   │───▶ RViz2
│ (30 fps) │    │  (17 keypoints) │    │  (joint angles)  │    │ /joint_  │     3-D Robot
└──────────┘    └─────────────────┘    └──────────────────┘    │  states  │     Skeleton
                                                                └──────────┘
```

---

## 1. The Robotics Stack — ROS 2 (15 min)

### What is ROS 2?

ROS 2 (Robot Operating System 2) is a middleware framework that lets software modules on a robot communicate over a publish/subscribe message bus. It is **not** an OS — it runs on top of Linux/macOS/Windows.

> Official docs: https://docs.ros.org/en/humble/index.html

### Core Concepts

| Concept | Analogy | Description |
|---------|---------|-------------|
| **Node** | Microservice | An independent process with a single responsibility |
| **Topic** | Slack channel | A named message bus (e.g. `/joint_states`) |
| **Publisher** | Poster | A node that sends messages onto a topic |
| **Subscriber** | Reader | A node that receives messages from a topic |
| **Message type** | Data schema | Typed struct for each topic (e.g. `sensor_msgs/JointState`) |

### How our project maps to ROS 2

```
[pose_mimic_publisher node]
        │
        ├── publishes ──▶ /robot_description  (std_msgs/String)   ← URDF XML
        └── publishes ──▶ /joint_states       (sensor_msgs/JointState)

[robot_state_publisher node]  ← from ROS 2 package
        │  subscribes to /joint_states
        └── publishes ──▶ /tf   (geometry_msgs/TransformStamped[])

[rviz2]  ← subscribes to /robot_description + /tf → renders 3-D robot
```

### Key ROS 2 CLI commands (cheat sheet)

```bash
# List all active topics
ros2 topic list

# Inspect messages on a topic (Ctrl-C to stop)
ros2 topic echo /joint_states

# Show topic publish rate
ros2 topic hz /joint_states

# Show topic message type
ros2 topic info /joint_states

# List all running nodes
ros2 node list
```

### Supported Distributions

| Distro | Ubuntu | Status |
|--------|--------|--------|
| Humble | 22.04  | LTS ✅ (recommended) |
| Iron   | 22.04  | EOL |
| Jazzy  | 24.04  | LTS ✅ |

> Installation guide: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

---

## 2. The Vision Stack — Python + PyTorch + YOLO (15 min)

### Why YOLO for pose estimation?

YOLO (You Only Look Once) is a single-shot object detector. The `-pose` variant adds a **keypoint head** that outputs 17 body landmark positions per person.

- Fast: 30+ FPS on a laptop GPU
- Accurate: COCO keypoint benchmark top performer
- Simple API: 3 lines of Python to run inference

> Ultralytics YOLO docs: https://docs.ultralytics.com/tasks/pose/

### COCO 17-Keypoint Skeleton

```
         0 (nose)
        / \
       1   2  (eyes)
       |   |
       3   4  (ears)

       5 ─── 6   (shoulders)
       |     |
       7     8   (elbows)
       |     |
       9    10   (wrists)

      11 ─── 12  (hips)
       |     |
      13    14   (knees)
       |     |
      15    16   (ankles)
```

### PyTorch + CUDA

PyTorch is the deep learning framework that runs YOLO under the hood.

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

> PyTorch install selector: https://pytorch.org/get-started/locally/

---

## 3. Environment Setup (10 min)

### Option A — Mamba/Conda (recommended for macOS + ROS 2)

```bash
# Create environment
mamba create -n ros_env python=3.10 -y
mamba activate ros_env

# ROS 2 via conda-forge (macOS / Linux)
mamba install -c conda-forge ros-humble-desktop -y   # Linux
# or
mamba install -c conda-forge ros-humble-ros-base -y  # macOS (no RViz2)

# Vision stack
pip install ultralytics opencv-python

# Verify
python -c "import cv2, ultralytics, torch; print('All OK')"
```

### Option B — Ubuntu native + pip

```bash
# ROS 2 Humble
sudo apt install ros-humble-desktop ros-humble-rviz2 \
     ros-humble-robot-state-publisher ros-humble-xacro -y
source /opt/ros/humble/setup.bash

# Vision stack
pip install ultralytics opencv-python
```

### Option C — Docker (zero-install)

```bash
docker pull osrf/ros:humble-desktop
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  osrf/ros:humble-desktop bash
# inside container:
pip install ultralytics
```

> ROS 2 Docker images: https://hub.docker.com/r/osrf/ros

### Project dependencies summary

| Package | Purpose | Install |
|---------|---------|---------|
| `ultralytics` | YOLOv8/v11 inference | `pip install ultralytics` |
| `opencv-python` | Camera capture & display | `pip install opencv-python` |
| `torch` | Deep learning backend | via pip (see pytorch.org) |
| `rclpy` | ROS 2 Python client | via ROS 2 install |
| `sensor_msgs` | JointState message type | via ROS 2 install |

---

## 4. Lab Exercises (15 min)

### Lab A — Hello YOLO (no ROS needed)

File: `labs/lab_a_hello_yolo.py`

```bash
# default — uses the pre-downloaded tennis video
python labs/lab_a_hello_yolo.py

# try a larger, more accurate model
python labs/lab_a_hello_yolo.py --model yolov8x-pose

# save an annotated output video
python labs/lab_a_hello_yolo.py --save

# process every 2nd frame (faster on slow machines)
python labs/lab_a_hello_yolo.py --every 2
```

**Controls in the OpenCV window:**

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `SPACE` | Pause / resume |
| `S` | Save screenshot of current frame |

**Expected console output (once per second):**
```
[ 30/1800]  L elbow: 142°  R elbow: 138°  L shoulder: 89°  R shoulder: 74°  L knee: —  R knee: —
[ 60/1800]  L elbow: 155°  R elbow: 121°  ...
```

The script:
1. Loads `labs/videos/workout.mp4` with OpenCV `VideoCapture`
2. Runs YOLO pose on each frame (or every Nth frame with `--every`)
3. Draws the 12-bone skeleton, joint angle labels, and a HUD overlay
4. Prints joint angles to console once per second
5. Optionally saves an annotated `.mp4` with `--save`

> Model zoo: https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes

---

### Lab B — Hello RViz2 (Linux / ROS 2 required)

File: `labs/lab_b_rviz_urdf.sh`

```bash
chmod +x labs/lab_b_rviz_urdf.sh
./labs/lab_b_rviz_urdf.sh
```

This script automatically sources ROS 2 (Humble or Jazzy), launches `robot_state_publisher` with our URDF, and opens RViz2 with the pre-configured view.

Or manually:
```bash
# Terminal 1 — publish the URDF
source /opt/ros/humble/setup.bash
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(cat robot/humanoid.urdf)"

# Terminal 2 — open RViz2
source /opt/ros/humble/setup.bash
rviz2 -d config/robot_view.rviz
```

In RViz2:
1. Confirm **Fixed Frame** = `base_link`
2. Add **RobotModel** display → Topic: `/robot_description`
3. Add **TF** display
4. You should see the grey humanoid skeleton

---

### Lab C — Full pipeline test (no ROS)

File: `src/test_no_ros.py` (already exists)

```bash
mamba activate ros_env
python src/test_no_ros.py
```

Expected: webcam window opens, green skeleton overlaid on your body, joint angles listed on the left. Press **Q** to quit.

---

## 5. Homework / Prep for Class 2

1. Get Lab A working with a tennis or golf image from the web
2. Print the shoulder and elbow angles for the detected athlete
3. Read: [ROS 2 Concepts Overview](https://docs.ros.org/en/humble/Concepts/Basic.html)
4. Watch: [YOLO Pose Estimation in 5 min](https://www.youtube.com/watch?v=nt6EcMvPVfk)
5. Think: which joints matter most for a tennis serve? A golf swing?

---

## Resources

| Resource | URL |
|----------|-----|
| ROS 2 Humble docs | https://docs.ros.org/en/humble/index.html |
| ROS 2 Concepts (Nodes/Topics) | https://docs.ros.org/en/humble/Concepts/Basic.html |
| Ultralytics YOLO docs | https://docs.ultralytics.com |
| YOLO Pose task | https://docs.ultralytics.com/tasks/pose/ |
| YOLO model zoo | https://docs.ultralytics.com/models/ |
| PyTorch install | https://pytorch.org/get-started/locally/ |
| COCO keypoint format | https://cocodataset.org/#keypoints-2020 |
| URDF tutorial | https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html |
| RViz2 user guide | https://github.com/ros2/rviz |
| Conda-forge ROS | https://robostack.github.io |

---

## Class Schedule Overview

| Class | Topic |
|-------|-------|
| **1** | **System Architecture, Setup & Toolchains** ← you are here |
| 2 | Camera Module & Real-Time YOLO Inference |
| 3 | Kinematic Math — Angles from Keypoints |
| 4 | ROS 2 Publisher & Joint State Messages |
| 5 | RViz2 Visualization & URDF Tuning |
| 6 | Sports Analysis — Tennis Serve / Golf Swing Scoring |
