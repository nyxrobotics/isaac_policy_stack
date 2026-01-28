from __future__ import annotations
from typing import Any, Dict

def robot_interface_v1_skeleton(base_frame: str = "base") -> Dict[str, Any]:
    return {
        "version": 1,
        "frames": {
            "base_frame": base_frame,
        },
        "sources": {
            "imu": {"msg_type": "sensor_msgs/Imu", "topic": "/imu/data"},
            "joint_states": {"msg_type": "sensor_msgs/JointState", "topic": "/joint_states"},
            "cmd_vel": {"msg_type": "geometry_msgs/Twist", "topic": "/cmd_vel"},
        },
        "term_inputs": {
            # Fill with keys matching io_descriptor.json observation term names.
            # Example:
            # "base_lin_acc_sens": {
            #   "source": "imu",
            #   "vector_field": "linear_acceleration",
            #   "orientation_field": "orientation",
            #   "provides_specific_force": False,
            #   "gravity_mag": 9.81,
            # },
        },
        "control": {
            "sink": "ros2_control_topic",
            "mode": "position_targets",
            "topic": "/joint_group_pos_controller/commands",
            "rate_hz": 200,
            "decimation": 4,
            # Default: both action and joint-state observations are treated as "relative" (add/subtract offsets).
            "action_relative": True,
            "observation_relative": True,

            # Optional: apply controller-side offsets after mapping the policy output into controller joint order.
            # When enabled, offsets can be provided as either:
            # - offsets: [..]  # list aligned with the controller joint order
            # - offsets: {joint_name: value, ...}
            "rel": False,
            # "offsets": {},
        },
    }
