"""
Publishes robot joint states and robot description to ROS 2 so RViz can
render the robot in real time.

Topics:
  /joint_states       (sensor_msgs/JointState)
  /robot_description  (std_msgs/String)  — latched
"""

import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

class RobotPublisher(Node):
    def __init__(self, urdf_path: str):
        super().__init__("pose_mimic_publisher")

        # Latched QoS for robot_description (RViz reads it once on startup)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._desc_pub = self.create_publisher(String, "/robot_description", latched)
        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)

        urdf_text = self._load_urdf(urdf_path)
        msg = String()
        msg.data = urdf_text
        self._desc_pub.publish(msg)
        self.get_logger().info(f"Published URDF from {urdf_path}")

    def _load_urdf(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def publish_joints(self, joint_angles: dict[str, float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_angles.keys())
        msg.position = [float(v) for v in joint_angles.values()]
        self._js_pub.publish(msg)
