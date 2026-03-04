# isaac_policy_runtime

A generic ROS 2 runtime that executes **Isaac Lab-exported policy bundles** (ONNX + descriptors) with a consistent
contract across robots and policies.

This package is intentionally **robot-agnostic**. All robot-specific wiring (topics, frames, mappings) lives inside
the bundle's `robot_interface.yaml`.

## What it does

1. Loads a *policy bundle* directory:
   - `policy.onnx`
   - `io_descriptor.json`
   - `action_config.json`
   - `robot_interface.yaml`
   - optional: `obs_normalization.json`, ``
2. Subscribes to ROS sources declared in `robot_interface.yaml` (IMU, joint states, cmd_vel, ...).
3. Builds the observation vector **in the exact order** specified by `io_descriptor.json`.
4. Runs ONNX inference (via `onnxruntime`).
5. Decodes the action vector using `action_config.json`.
6. Publishes commands to the configured control sink (v1: `ros2_control_topic` via `std_msgs/Float64MultiArray`).

## Install

This is an `ament_python` package. You still need ONNX Runtime installed in your Python environment:

```bash
pip install onnxruntime
```

(or `onnxruntime-gpu` if you know what you're doing.)

## Run

```bash
ros2 launch isaac_policy_runtime policy_runner.launch.py bundle_path:=/absolute/path/to/bundle
```

## Bundle contract (high level)

- **`action_config.json`** is the *single source of truth* for joint order and action scaling.
- **`robot_interface.yaml`** defines:
  - sources (topic + msg type)
  - per-term input wiring (`term_inputs`)
  - optional joint name mapping (`joints.name_map`)
  - control sink config

## Notes

- TF is optional. If `use_tf=true` and `tf2_ros` is available, IMU vectors can be transformed from `imu_frame` to `base_frame`.
- This runtime is designed to fail fast in `strict=true` mode:
  missing terms, missing joints, or missing source configs will throw an error on startup.
