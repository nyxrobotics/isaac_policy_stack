from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

@dataclass
class Ros2ControlTopicSink:
    node: Node
    topic: str

    def __post_init__(self):
        self.pub = self.node.create_publisher(Float64MultiArray, self.topic, 10)

    def publish_positions(self, q_target: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q_target.tolist()]
        self.pub.publish(msg)
