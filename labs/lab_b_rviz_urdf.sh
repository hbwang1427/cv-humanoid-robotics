#!/usr/bin/env bash
# Lab B — Visualize the Humanoid URDF in RViz2
# =============================================
# Works with ROS 2 installed via conda/mamba (ros_env) or system apt.
#
# Usage (from project root):
#   ./labs/lab_b_rviz_urdf.sh

set -e

# Resolve project root relative to this script — works regardless of where you call it from
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

URDF_FILE="$PROJECT_ROOT/robot/humanoid.urdf"
RVIZ_CONFIG="$PROJECT_ROOT/config/robot_view.rviz"

if [ ! -f "$URDF_FILE" ]; then
    echo "ERROR: URDF not found at $URDF_FILE"
    exit 1
fi

# Source system ROS 2 if ros2 is not already on PATH.
# Under conda/mamba ros_env it is already available — no sourcing needed.
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
echo "  Lab B — Hello RViz2"
echo "============================================"
echo "URDF   : $URDF_FILE"
echo "Config : $RVIZ_CONFIG"
echo ""

# Launch robot_state_publisher in background
echo "[1/3] Launching robot_state_publisher..."
URDF_TEXT=$(cat "$URDF_FILE")
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$URDF_TEXT" &
RSP_PID=$!
echo "      PID: $RSP_PID"

sleep 1

# Launch joint_state_publisher_gui — provides sliders to move each joint
echo "[2/3] Launching joint_state_publisher_gui..."
if command -v ros2 &>/dev/null && ros2 pkg list 2>/dev/null | grep -q joint_state_publisher_gui; then
    ros2 run joint_state_publisher_gui joint_state_publisher_gui &
    JSP_PID=$!
    echo "      PID: $JSP_PID  (slider window for moving joints)"
else
    echo "      WARNING: joint_state_publisher_gui not found."
    echo "      Install: sudo apt install ros-humble-joint-state-publisher-gui"
    JSP_PID=""
fi

sleep 1

# Launch RViz2 (foreground — Ctrl-C here shuts everything down)
echo "[3/3] Launching RViz2..."
echo "      -> Use the slider window to move robot joints"
echo "      -> Left-drag to orbit, scroll to zoom, middle-drag to pan"
echo "      -> Press Ctrl-C to quit"
echo ""
rviz2 -d "$RVIZ_CONFIG"

# Cleanup
kill $RSP_PID 2>/dev/null
[ -n "$JSP_PID" ] && kill $JSP_PID 2>/dev/null
echo "Stopped."
