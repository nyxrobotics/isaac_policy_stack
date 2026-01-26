# isaac_policy_tools (pure Python)

This directory contains **pure Python tools** that you run in your Isaac Lab environment
to *produce policy bundles* for ROS 2.

It is **not** an ament package and is intentionally not built by `colcon`.

- `isaac_policy_export/`: CLI for exporting policy bundles (ONNX + descriptors + wiring skeleton)

## Why pure Python here?

Keeping the exporter in the same repository makes versioning easy, while not mixing ROS build
dependencies (`ament`, `rclpy`) with Isaac Lab dependencies (`isaaclab`, training runners, etc.).
