# isaac_policy_stack (ROS 2 Humble)

This repository provides:

- **isaac_policy_runner**: Generic ROS 2 node that runs an Isaac Lab exported policy (ONNX) by subscribing to observations and publishing actions.
- **isaac_runner_generator**: ROS 2 CLI tool that generates a per-robot runner package (`<robot_name>_isaac_runner`) containing:
  - `<robot_name>_observations_bridge_node`
  - `<robot_name>_actions_bridge_node`
  - a sample launch file that starts both bridge nodes and `isaac_policy_runner`
  - a `model/` directory with placeholder model files:
    - `IO_descriptors.yaml`
    - `policy.onnx`
    - `policy.pt`

## Requirements

- ROS 2 Humble
- Python 3
- `onnxruntime` (CPU), `numpy`, `pyyaml`

Install python deps (example):

```bash
python3 -m pip install --user onnxruntime numpy pyyaml
```

## Build

```bash
cd <your_ws>/src
# place this repo here (isaac_policy_stack/)
cd ..
colcon build --symlink-install
source install/setup.bash
```

## Generate a robot runner package

```bash
ros2 run isaac_runner_generator create_robot_runner --robot <robot_name>
```

This creates:

```
<robot_name>_isaac_runner/
  launch/
  model/
  <robot_name>_isaac_runner/
```

Copy your trained model files into:

```
<robot_name>_isaac_runner/model/
  IO_descriptors.yaml
  policy.onnx
  policy.pt
```

## Run (sample launch)

```bash
ros2 launch <robot_name>_isaac_runner <robot_name>_isaac_runner.launch.py
```

### Topics

- `/policy/observations` (`std_msgs/Float32MultiArray`) -> input to policy
- `/policy/actions` (`std_msgs/Float32MultiArray`) -> output of policy
- `/joint_group_position_controller/commands` (`std_msgs/Float64MultiArray`) -> joint targets

### Controller

This stack assumes the robot uses:

```yaml
joint_group_position_controller:
  type: forward_command_controller/ForwardCommandController
```

## Notes

- The policy runner is callback-driven (no internal control loop).
- Generated observation bridge parameter `odom_twist_in_world_frame`:
  - If `false` (default), `/odom.twist.twist.linear` is forwarded as-is to `base_lin_vel`.
  - If `true`, the bridge interprets `/odom.twist.twist.linear` as world-frame linear velocity and rotates it into the body frame using the latest IMU orientation before publishing `base_lin_vel`.
  - If `/odom` arrives before `/imu`, the bridge temporarily forwards the odom twist as-is and logs a warning once until IMU data becomes available.
- Parameter `use_internal_action_observation`:
  - If `true`, the runner overwrites the `last_action` slice inside the incoming observation vector with the last action it published.
- Generated action bridge behavior:
  - Raw policy actions are clamped to `[-1, 1]` before scaling.
  - Final joint targets are clamped to `articulations.robot.default_joint_pos_limits` from `IO_descriptors.yaml` when limits are available.
  - Exported action `clip` is still respected before the joint-limit clamp.
