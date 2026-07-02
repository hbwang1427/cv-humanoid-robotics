# CV Humanoid Robotics — Real-Time Human Pose to Robot Mirroring

Track a human via webcam or video using YOLO keypoints, remap the motion to a humanoid robot, and visualize it live in RViz2.

---

## System Overview

```
Video/Camera -> YOLO Pose (17 keypoints) -> Joint Angle Math -> ROS 2 /joint_states -> RViz2
```

| Module | File | Role |
|--------|------|------|
| Pose detector | `src/detector.py` | YOLOv8-pose inference |
| Pose mapper | `src/mapper.py` | Keypoints -> robot joint angles |
| ROS publisher | `src/ros_publisher.py` | Publishes `/joint_states` + `/robot_description` |
| Robot registry | `src/robot_registry.py` | Maps robot keys to URDF paths |
| Visualizer | `src/visualizer.py` | OpenCV overlay (keypoints + joint angles) |
| Main entry | `src/main.py` | Full pipeline with ROS 2 |
| No-ROS test | `src/test_no_ros.py` | Pipeline without ROS 2 (OpenCV only) |

---

## Prerequisites

### System
- Ubuntu 22.04 / 24.04 (RViz2 requires Linux)
- macOS — works for detection + visualization, no RViz2
- Python 3.10+
- ROS 2 Humble or Jazzy (for full pipeline)

### Install ROS 2 Humble (Ubuntu)
```bash
# Full desktop install (includes RViz2)
sudo apt install ros-humble-desktop ros-humble-robot-state-publisher ros-humble-xacro
source /opt/ros/humble/setup.bash
```
> Full guide: https://docs.ros.org/en/humble/Installation.html

### Python dependencies
```bash
# Using mamba/conda (recommended)
mamba create -n ros_env python=3.10 -y
mamba activate ros_env
pip install ultralytics opencv-python yt-dlp
```

---

## How to Run

### Option 1 — No ROS (OpenCV only, works on macOS + Linux)

```bash
mamba activate ros_env

# Default: play workout video
python src/test_no_ros.py

# Use live camera instead
python src/test_no_ros.py --camera

# Custom video file
python src/test_no_ros.py --video path/to/video.mp4
```

---

### Option 2 — Full pipeline with ROS 2 + RViz2

**Terminal 1 — run the pose publisher:**
```bash
mamba activate ros_env
source /opt/ros/humble/setup.bash

# Default: simple humanoid + workout video
python src/main.py

# Choose a different robot
python src/main.py --robot h1
python src/main.py --robot g1
python src/main.py --robot g1_hand

# Use live camera
python src/main.py --robot h1 --camera

# List all available robots
python src/main.py --list-robots
```

**Terminal 2 — open RViz2:**
```bash
source /opt/ros/humble/setup.bash
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/robot/models/unitree_ros/robots
rviz2 -d config/robot_view.rviz
```

> RViz2 first-time setup: set **Fixed Frame** = `base_link`, add **RobotModel** display on topic `/robot_description`.

---

### Option 3 — Lab exercises (students)

```bash
mamba activate ros_env

# Lab A: YOLO pose on workout video with angle overlay
python labs/lab_a_hello_yolo.py

# Lab A with live camera
python labs/lab_a_hello_yolo.py --camera

# Lab A: save annotated output video
python labs/lab_a_hello_yolo.py --save

# Lab A: try a larger model
python labs/lab_a_hello_yolo.py --model yolov8x-pose

# Lab B: visualize URDF in RViz2 (Linux only)
./labs/lab_b_rviz_urdf.sh

# Lab C: same as Option 1 above
python src/test_no_ros.py
```

---

## Available Robot Models

Run `python src/main.py --list-robots` to see all options.

| Key | Robot | DOF |
|-----|-------|-----|
| `simple` | Custom simple humanoid (default) | 10 |
| `h1` | Unitree H1 | 19 |
| `h1_hand` | Unitree H1 with hands | — |
| `h2` | Unitree H2 | — |
| `g1` | Unitree G1 | 23 |
| `g1_29` | Unitree G1 | 29 |
| `g1_hand` | Unitree G1 with hands | 29 |
| `r1` | Unitree R1 arm | — |
| `go2` | Unitree Go2 quadruped | — |
| `b2` | Unitree B2 quadruped | — |

Models downloaded from: https://github.com/unitreerobotics/unitree_ros

---

## Lab Controls (OpenCV window)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `SPACE` | Pause / resume |
| `S` | Save screenshot |

---

## Project Structure

```
cv-humanoid-robotics/
├── README.md
├── robot/
│   ├── humanoid.urdf              # Simple built-in humanoid
│   └── models/
│       └── unitree_ros/           # Unitree robot URDFs + meshes
├── config/
│   └── robot_view.rviz            # Pre-configured RViz2 layout
├── labs/
│   ├── lab_a_hello_yolo.py        # Lab A: YOLO on video/camera
│   ├── lab_b_rviz_urdf.sh         # Lab B: URDF in RViz2
│   └── videos/
│       └── workout.mp4            # Sample workout video
├── docs/
│   └── class01_introduction.md    # Class 1 teaching material
└── src/
    ├── main.py                    # Full pipeline (ROS 2)
    ├── test_no_ros.py             # Pipeline without ROS 2
    ├── detector.py                # YOLOv8-pose inference
    ├── mapper.py                  # Keypoints -> joint angles
    ├── ros_publisher.py           # ROS 2 joint state publisher
    ├── robot_registry.py          # Robot model registry
    └── visualizer.py              # OpenCV debug overlay
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'rclpy'` | `source /opt/ros/humble/setup.bash` |
| `No module named 'cv2'` | `pip install opencv-python` |
| Camera black / not opening | Grant camera permission in System Settings, or use `--video` |
| RViz shows no robot | Check topic: `ros2 topic echo /robot_description` |
| Unitree mesh not found in RViz | `export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/robot/models/unitree_ros/robots` |
| Low FPS | Use `--every 2` flag or switch to `yolov8n-pose` (nano model) |
