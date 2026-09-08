#!/usr/bin/env bash
# Lab C — Mirror human motion onto a full humanoid in RViz2
# =========================================================
# Starts robot_state_publisher + RViz2 for the chosen robot. Run the pose
# publisher in a second terminal:
#
#   python src/main.py --robot h1
#
# Usage (from project root):
#   ./labs/lab_c_mirror.sh            # default: h1
#   ./labs/lab_c_mirror.sh g1
#
# Works with ROS 2 from RoboStack/conda (ros_env) or system apt. The RoboStack
# build cannot resolve `package://<pkg>` meshes from ROS_PACKAGE_PATH alone, so
# this script builds a throwaway ament prefix of symlinks for the Unitree
# description packages and exports AMENT_PREFIX_PATH.

set -e

ROBOT="${1:-h1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if ! command -v ros2 &>/dev/null; then
    for SETUP in /opt/ros/humble/setup.bash /opt/ros/jazzy/setup.bash; do
        [ -f "$SETUP" ] && { source "$SETUP"; break; }
    done
fi
command -v ros2  &>/dev/null || { echo "ERROR: ros2 not found (activate ros_env or source ROS 2)."; exit 1; }
command -v rviz2 &>/dev/null || { echo "ERROR: rviz2 not found."; exit 1; }

# --- Resolve URDF path + root link for the requested robot -------------------
# (plain $(...) command substitution — macOS bash 3.2 mis-parses a heredoc
#  nested inside process substitution `< <(... <<EOF ...)`)
INFO="$(python -c '
import sys, xml.etree.ElementTree as ET
sys.path.insert(0, "src")
from robot_registry import get_urdf_path
path = get_urdf_path(sys.argv[1])
root = ET.parse(path).getroot()
children = {j.find("child").get("link") for j in root.findall("joint")}
base = next(l.get("name") for l in root.findall("link") if l.get("name") not in children)
print(path, base)
' "$ROBOT")" || { echo "ERROR: could not resolve URDF for robot '$ROBOT'."; exit 1; }
URDF="${INFO%% *}"
FIXED_FRAME="${INFO##* }"

echo "============================================"
echo "  Lab C — motion mirroring: $ROBOT"
echo "============================================"
echo "URDF        : $URDF"
echo "Fixed frame : $FIXED_FRAME"
echo ""

# --- RoboStack package:// fix: throwaway ament prefix of symlinks ------------
AMENT_TMP="$(mktemp -d)"
mkdir -p "$AMENT_TMP/share/ament_index/resource_index/packages"
for d in "$ROOT"/robot/models/unitree_ros/robots/*_description; do
    [ -d "$d" ] || continue
    pkg="$(basename "$d")"
    ln -sfn "$d" "$AMENT_TMP/share/$pkg"
    : > "$AMENT_TMP/share/ament_index/resource_index/packages/$pkg"
done
export AMENT_PREFIX_PATH="$AMENT_TMP:${AMENT_PREFIX_PATH:-}"

# --- RViz config with the right fixed frame + latched description sub --------
RVIZ_CFG="$(mktemp -t robot_view.XXXXXX).rviz"
sed -e "s/Fixed Frame: base_link/Fixed Frame: $FIXED_FRAME/" \
    -e "s/Durability Policy: Volatile/Durability Policy: Transient Local/" \
    "$ROOT/config/robot_view.rviz" > "$RVIZ_CFG"

cleanup() {
    kill "$RSP_PID" 2>/dev/null || true
    rm -rf "$AMENT_TMP" "$RVIZ_CFG"
}
trap cleanup EXIT

# --- robot_state_publisher: URDF param + /joint_states -> /tf ----------------
echo "[1/2] robot_state_publisher..."
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat "$URDF")" &
RSP_PID=$!
sleep 1

echo "[2/2] RViz2..."
echo ""
echo "    >>> In another terminal:  python src/main.py --robot $ROBOT"
echo ""
rviz2 -d "$RVIZ_CFG"
