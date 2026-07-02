#!/usr/bin/env bash
# Lab B — Visualize the Humanoid URDF in RViz2
# =============================================
# Run this from the project root on Ubuntu with ROS 2 Humble/Jazzy.
# Opens two terminals: one for robot_state_publisher, one for RViz2.
#
# Usage:
#   chmod +x labs/lab_b_rviz_urdf.sh
#   ./labs/lab_b_rviz_urdf.sh

set -e

ROS_SETUP="/opt/ros/humble/setup.bash"
if [ ! -f "$ROS_SETUP" ]; then
    ROS_SETUP="/opt/ros/jazzy/setup.bash"
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "ERROR: ROS 2 not found. Install Humble or Jazzy first."
    echo "  https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

URDF_FILE="$(pwd)/robot/humanoid.urdf"
RVIZ_CONFIG="$(pwd)/config/robot_view.rviz"

if [ ! -f "$URDF_FILE" ]; then
    echo "ERROR: URDF not found at $URDF_FILE"
    echo "  Run this script from the project root directory."
    exit 1
fi

echo "============================================"
echo "  Lab B — Hello RViz2"
echo "============================================"
echo "URDF   : $URDF_FILE"
echo "Config : $RVIZ_CONFIG"
echo ""

# Terminal 1: robot_state_publisher (publishes /tf from /joint_states + URDF)
echo "[1/2] Launching robot_state_publisher in background..."
source "$ROS_SETUP"
URDF_TEXT=$(cat "$URDF_FILE")

ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$URDF_TEXT" &
RSP_PID=$!
echo "      PID: $RSP_PID"

sleep 2

# Terminal 2: RViz2
echo "[2/2] Launching RViz2..."
echo "      → Add 'RobotModel' display if not already shown"
echo "      → Fixed Frame should be 'base_link'"
echo "      → Press Ctrl-C here to shut everything down"
echo ""
rviz2 -d "$RVIZ_CONFIG"

# Cleanup on exit
kill $RSP_PID 2>/dev/null
echo "Stopped robot_state_publisher."
