#!/usr/bin/env bash
# Lab C3 — Drive the Face Robot via ROS 2 / RViz2
# ================================================
# Launches robot_state_publisher (remapped onto the /face/* topics) and
# RViz2. Run `python src/face_main.py` in a separate terminal to actually
# publish joint states from your facial expressions.
#
# Usage (from project root):
#   ./labs/lab_c3_face_ros.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

URDF_FILE="$PROJECT_ROOT/robot/face.urdf"
RVIZ_CONFIG="$PROJECT_ROOT/config/face_view.rviz"

if [ ! -f "$URDF_FILE" ]; then
    echo "ERROR: URDF not found at $URDF_FILE"
    exit 1
fi

if ! command -v ros2 &>/dev/null; then
    for SETUP in /opt/ros/humble/setup.bash /opt/ros/jazzy/setup.bash; do
        if [ -f "$SETUP" ]; then
            # shellcheck disable=SC1090
            source "$SETUP"
            break
        fi
    done
fi

if ! command -v ros2 &>/dev/null; then
    echo "ERROR: ros2 not found."
    echo "  Activate your conda env:  conda activate ros_env"
    echo "  Or source ROS 2:          source /opt/ros/humble/setup.bash"
    exit 1
fi

if ! command -v rviz2 &>/dev/null; then
    echo "ERROR: rviz2 not found."
    echo "  sudo apt install ros-humble-rviz2"
    exit 1
fi

echo "============================================"
echo "  Lab C3 — Face Robot over ROS 2"
echo "============================================"
echo "URDF   : $URDF_FILE"
echo "Config : $RVIZ_CONFIG"
echo ""

# robot_state_publisher: takes the URDF as a parameter, subscribes to
# /face/joint_states, computes forward kinematics, publishes /tf.
echo "[1/2] Launching robot_state_publisher (namespaced to /face/*)..."
URDF_TEXT=$(cat "$URDF_FILE")
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$URDF_TEXT" \
    -r joint_states:=/face/joint_states \
    -r robot_description:=/face/robot_description &
RSP_PID=$!
echo "      PID: $RSP_PID"

sleep 1

echo "[2/2] Launching RViz2..."
echo "      -> Run 'python src/face_main.py' in another terminal to drive it"
echo "      -> Press Ctrl-C to quit"
echo ""
rviz2 -d "$RVIZ_CONFIG"

kill $RSP_PID 2>/dev/null
echo "Stopped."
