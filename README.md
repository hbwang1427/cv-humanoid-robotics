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

### Facial expression -> robot face (Class 3)

```
Camera -> YOLO face locate -> dlib 68 landmarks -> EAR/MAR/smile/brow -> ROS 2 /face/joint_states -> robot_state_publisher -> /tf -> RViz2
```

| Module | File | Role |
|--------|------|------|
| Face landmark detector | `src/face_detector.py` | dlib 68-landmark predictor, seeded by YOLO's 5 face keypoints |
| Feature extractor | `src/face_features.py` | EAR/MAR/smile/brow features + neutral-face baseline calibration |
| Face mapper | `src/face_mapper.py` | Features -> face robot joint angles |
| Face publisher | `src/face_publisher.py` | Publishes `/face/joint_states` + `/face/robot_description` + `/face/expression`, with EMA smoothing |
| Face visualizer | `src/face_visualizer.py` | OpenCV overlay (feature bars + expression label) |
| Face main entry | `src/face_main.py` | Full expression pipeline with ROS 2 |
| No-ROS face test | `src/test_face_no_ros.py` | Expression pipeline without ROS 2 (OpenCV only, works on macOS) |
| Face robot | `robot/face.urdf` | Primitive-shape head with jaw/eyebrow/eyelid/lip-corner joints |

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

### Facial expression pipeline — extra dependencies
```bash
mamba install -c conda-forge dlib -y

curl -LO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mkdir -p labs/models
mv shape_predictor_68_face_landmarks.dat labs/models/
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

### Option 2b — Facial expression -> robot face (ROS 2 + RViz2, Linux)

**Terminal 1 — robot_state_publisher + RViz2 (one command):**
```bash
mamba activate ros_env
./labs/lab_c3_face_ros.sh
```

**Terminal 2 — run the expression publisher:**
```bash
mamba activate ros_env
source /opt/ros/humble/setup.bash

python src/face_main.py            # webcam
python src/face_main.py --video path/to/clip.mp4
```

Hold a neutral face for the first ~1 second (30-frame calibration), then smile, raise your eyebrows, open your mouth, or blink — the jaw/brow/eyelid/lip-corner joints on the RViz2 face should track you in real time via `/face/joint_states` -> `robot_state_publisher` -> `/tf`.

---

### Option 2c — Facial expression, no ROS (macOS-friendly)

```bash
mamba activate ros_env
python src/test_face_no_ros.py            # webcam
python src/test_face_no_ros.py --video path/to/clip.mp4
```

Shows dlib landmarks, live EAR/MAR/smile/brow feature values, and the classified expression label — useful for tuning thresholds before wiring up ROS 2.

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
│   ├── face.urdf                  # Face robot (jaw/brow/eyelid/lip-corner joints)
│   └── models/
│       └── unitree_ros/           # Unitree robot URDFs + meshes
├── config/
│   ├── robot_view.rviz            # Pre-configured RViz2 layout (body)
│   └── face_view.rviz             # Pre-configured RViz2 layout (face)
├── labs/
│   ├── lab_a_hello_yolo.py        # Lab A: YOLO on video/camera
│   ├── lab_b_rviz_urdf.sh         # Lab B: URDF in RViz2
│   ├── lab_c3_face_ros.sh         # Lab C3: face robot_state_publisher + RViz2
│   ├── models/                    # dlib shape_predictor_68_face_landmarks.dat goes here
│   └── videos/
│       └── workout.mp4            # Sample workout video
├── docs/
│   ├── class01_introduction.md    # Class 1 teaching material
│   ├── class02_keypoints_yolo_cnn.md
│   └── class03_facial_expression.md
└── src/
    ├── main.py                    # Full pipeline (ROS 2)
    ├── test_no_ros.py             # Pipeline without ROS 2
    ├── detector.py                # YOLOv8-pose inference
    ├── mapper.py                  # Keypoints -> joint angles
    ├── ros_publisher.py           # ROS 2 joint state publisher
    ├── robot_registry.py          # Robot model registry
    ├── visualizer.py              # OpenCV debug overlay
    ├── face_main.py               # Full expression pipeline (ROS 2)
    ├── test_face_no_ros.py        # Expression pipeline without ROS 2
    ├── face_detector.py           # dlib 68-landmark detection (YOLO-seeded)
    ├── face_features.py           # EAR/MAR/smile/brow + baseline calibration
    ├── face_mapper.py             # Features -> face robot joint angles
    ├── face_publisher.py          # ROS 2 face joint state publisher
    └── face_visualizer.py         # OpenCV debug overlay (face)
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
| `No module named 'dlib'` | `mamba install -c conda-forge dlib -y` (pip install requires `cmake` and compiles from source — conda is faster, especially on Apple Silicon where DeepFace-style TensorFlow installs tend to fail) |
| `FileNotFoundError: shape_predictor...dat` | Download it — see "Facial expression pipeline — extra dependencies" above |
| Face robot doesn't move in RViz2 | Check `ros2 topic echo /face/joint_states`; confirm `lab_c3_face_ros.sh` and `face_main.py` are both running |
| Face features jump around | Extend calibration with `--calib-frames 60`, or hold still during the "Calibrating..." phase |
