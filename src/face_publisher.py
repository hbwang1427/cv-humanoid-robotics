"""
Publishes face robot joint states + robot description to ROS 2. A
robot_state_publisher node (started via labs/lab_c3_face_ros or manually)
subscribes to /face/joint_states, runs forward kinematics against
robot/face.urdf, and publishes /tf — that's the "facial expression as
motion TF" link RViz2 renders.

Topics:
  /face/joint_states       (sensor_msgs/JointState)  -- continuous, drives TF
  /face/robot_description  (std_msgs/String)         -- latched
  /face/expression         (std_msgs/String)         -- discrete label
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from face_mapper import JOINT_LIMITS

FACE_JOINTS = list(JOINT_LIMITS.keys())


class FacePublisher(Node):
    def __init__(self, urdf_path: str, alpha: float = 0.3):
        super().__init__("face_expression_publisher")

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._desc_pub = self.create_publisher(String, "/face/robot_description", latched)
        self._js_pub = self.create_publisher(JointState, "/face/joint_states", 10)
        self._label_pub = self.create_publisher(String, "/face/expression", 10)

        urdf_text = self._load_urdf(urdf_path)
        msg = String()
        msg.data = urdf_text
        self._desc_pub.publish(msg)
        self.get_logger().info(f"Published face URDF from {urdf_path}")

        self.alpha = alpha  # EMA smoothing factor — the anti-flicker filter
        self._smoothed = {j: 0.0 for j in FACE_JOINTS}

    def _load_urdf(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def publish_face(self, joint_targets: dict[str, float], label: str) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for j in FACE_JOINTS:
            target = joint_targets.get(j, 0.0)
            self._smoothed[j] = (1 - self.alpha) * self._smoothed[j] + self.alpha * target
            msg.name.append(j)
            msg.position.append(self._smoothed[j])
        self._js_pub.publish(msg)
        self._label_pub.publish(String(data=label))
