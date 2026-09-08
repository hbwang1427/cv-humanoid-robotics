"""
Publishes robot joint states and robot description to ROS 2 so RViz can
render the robot in real time.

Topics:
  /joint_states       (sensor_msgs/JointState)
  /robot_description  (std_msgs/String)  — latched
"""

import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

_MOVABLE = {"revolute", "continuous", "prismatic"}


class RobotPublisher(Node):
    def __init__(self, urdf_path: str):
        super().__init__("pose_mimic_publisher")

        # Latched QoS for robot_description (RViz reads it once on startup)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._desc_pub = self.create_publisher(String, "/robot_description", latched)
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)

        urdf_text = self._load_urdf(urdf_path)
        self._joint_order, self._limits = self._parse_joints(urdf_text)

        msg = String()
        msg.data = urdf_text
        self._desc_pub.publish(msg)
        self.get_logger().info(
            f"Published URDF from {urdf_path} ({len(self._joint_order)} movable joints)"
        )

    def _load_urdf(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def _parse_joints(self, urdf_text: str):
        """Return (ordered movable joint names, {name: (lower, upper)})."""
        root = ET.fromstring(urdf_text)
        order: list[str] = []
        limits: dict[str, tuple[float, float]] = {}
        for j in root.findall("joint"):
            if j.get("type") not in _MOVABLE:
                continue
            name = j.get("name")
            order.append(name)
            lim = j.find("limit")
            if lim is not None and lim.get("lower") is not None:
                limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
        return order, limits

    def publish_joints(self, joint_angles: dict[str, float]) -> None:
        """Publish a JointState covering *every* movable joint in the URDF.

        Joints present in ``joint_angles`` use that value (clamped to the URDF
        limit); every other joint is held at 0.0. robot_state_publisher only
        emits /tf for a joint once it has seen it in a JointState, so the full
        list must go out every frame or parts of the robot never render.
        """
        names = self._joint_order or list(joint_angles.keys())
        positions = []
        for name in names:
            value = float(joint_angles.get(name, 0.0))
            lo, hi = self._limits.get(name, (None, None))
            if lo is not None:
                value = max(lo, min(hi, value))
            positions.append(value)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self._js_pub.publish(msg)
