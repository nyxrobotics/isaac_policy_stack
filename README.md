# isaac_policy_stack

ROS2 runtime for executing policies trained in Isaac Lab / Isaac Gym on robots.

## Core Design

IO_descriptors.yaml is the single source of truth for the policy ABI.

Removed legacy files:
- observation_config.yaml
- action_config.yaml
- obs_normalization.yaml
- metadata.yaml

Observation and action layouts are derived directly from IO_descriptors.yaml.

## Policy Bundle Structure

policy_bundle/
 ├── policy.pt
 ├── policy.onnx
 ├── IO_descriptors.yaml
 └── robot_interface.yaml

Both policy.pt and policy.onnx exist in the bundle.

## Build

mkdir -p ws/src
cd ws/src
git clone <repo>
cd ..
colcon build
source install/setup.bash

## Export Policy Bundle

isaac-export-bundle \
  --checkpoint policy.pt \
  --io-descriptors IO_descriptors.yaml \
  --bundle_out my_policy_bundle

## Run Runtime

ros2 launch isaac_policy_runtime policy_runner.launch.py \
  bundle:=/path/to/policy_bundle

## IO_descriptors.yaml

Defines observation and action layout.

Example:

observations:
  - name: base_ang_vel
    dim: 3

actions:
  - name: joint_position
    joints: [hip_yaw, hip_roll, hip_pitch]

## Runtime Pipeline

ROS2 topics → observation terms → observation vector → policy → action vector → ROS2 commands
