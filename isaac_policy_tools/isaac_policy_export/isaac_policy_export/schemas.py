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
        "joints": {
            "name_map": {},      # optional: ROS joint name -> policy joint name
            "default_pos": {},   # optional: policy joint defaults (for *_rel)
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
        },
    }
