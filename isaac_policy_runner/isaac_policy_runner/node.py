from __future__ import annotations

import os
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .io_descriptor import load_io_descriptors


class IsaacPolicyRunner(Node):
    def __init__(self) -> None:
        super().__init__("isaac_policy_runner")

        self.declare_parameter("model_dir", "")
        self.declare_parameter("use_internal_action_observation", True)
        self.declare_parameter("debug_print", True)
        self.declare_parameter("debug_every_n", 1)
        self.debug_print = self.get_parameter("debug_print").get_parameter_value().bool_value
        self.debug_every_n = int(self.get_parameter("debug_every_n").get_parameter_value().integer_value) or 1
        self._debug_count = 0

        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value
        self.use_internal_action_observation = (
            self.get_parameter("use_internal_action_observation").get_parameter_value().bool_value
        )

        if not model_dir:
            raise RuntimeError("Parameter 'model_dir' is required")

        io_path = os.path.join(model_dir, "IO_descriptors.yaml")
        onnx_path = os.path.join(model_dir, "policy.onnx")

        self.io = load_io_descriptors(io_path)
        self.obs_slices = self.io.observation.slices
        self.obs_size = int(self.io.observation.total_size)
        self.act_size = int(self.io.action_size)

        # Isaac Lab uses observation term name: 'last_action'
        self.last_action_slice = self.io.observation.slices.get("last_action", None)
        self.last_action = np.zeros((self.act_size,), dtype=np.float32)

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.sub = self.create_subscription(Float32MultiArray, "/policy/observations", self._cb_obs, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/policy/actions", 10)

        self.get_logger().info(
            f"Ready: obs_size={self.obs_size}, act_size={self.act_size}, "
            f"use_internal_action_observation={self.use_internal_action_observation}"
        )

    def _cb_obs(self, msg: Float32MultiArray) -> None:
        obs = np.asarray(msg.data, dtype=np.float32)
        if obs.size != self.obs_size:
            self.get_logger().warning(f"Observation size mismatch: got {obs.size}, expected {self.obs_size}")
            return

        if self.use_internal_action_observation and self.last_action_slice is not None:
            start, size = self.last_action_slice
            if size == self.act_size and (start + size) <= obs.size:
                obs = obs.copy()
                obs[start:start + size] = self.last_action
            else:
                self.get_logger().warning("last_action slice mismatch; cannot override")

        if self.debug_print:
            self._debug_count += 1
            if (self._debug_count % self.debug_every_n) == 0:
                def _sl(name: str):
                    if name not in self.obs_slices:
                        return None
                    s, n = self.obs_slices[name]
                    return obs[s:s+n]
                base_lin_vel = _sl('base_lin_vel')
                base_ang_vel = _sl('base_ang_vel')
                projected_gravity = _sl('projected_gravity')
                vel_cmd = _sl('generated_commands')
                joint_pos = _sl('joint_pos_rel')
                joint_vel = _sl('joint_vel_rel')
                if base_lin_vel is not None: self.get_logger().info(f"[policy] base_lin_vel: {base_lin_vel}")
                if base_ang_vel is not None: self.get_logger().info(f"[policy] base_ang_vel: {base_ang_vel}")
                if projected_gravity is not None: self.get_logger().info(f"[policy] projected_gravity: {projected_gravity}")
                if vel_cmd is not None: self.get_logger().info(f"[policy] velocity_commands: {vel_cmd}")
                if joint_pos is not None: self.get_logger().info(f"[policy] joint_pos: {joint_pos}")
                if joint_vel is not None: self.get_logger().info(f"[policy] joint_vel: {joint_vel}")

        inp = obs.reshape(1, -1)
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        act = np.asarray(out, dtype=np.float32).reshape(-1)

        if act.size != self.act_size:
            self.get_logger().warning(f"Action size mismatch: got {act.size}, expected {self.act_size}")
            return

        if self.debug_print and (self._debug_count % self.debug_every_n) == 0:
            self.get_logger().info(f"[policy] actions: {act}")
        self.last_action = act

        out_msg = Float32MultiArray()
        out_msg.data = act.tolist()
        self.pub.publish(out_msg)


def main() -> None:
    rclpy.init()
    node = IsaacPolicyRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
