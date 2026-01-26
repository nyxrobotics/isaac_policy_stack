# isaac_policy_stack (meta-package)

This is a **ROS 2 meta-package** that groups the core packages of this repository:

- `isaac_policy_runtime` — generic ONNX policy runner (rclpy).
- `isaac_policy_bringup` — launch files that start the runtime (and optionally simulation/hardware bringup).
- `isaac_policy_tools` — *pure Python* tooling (not built by colcon) for exporting policy bundles from Isaac Lab.

## Why a meta-package?

A meta-package makes it easy to install/track a whole “stack” as a single unit and
to discover the repository’s primary entry points. It does **not** build code itself.

## Build

From the workspace root:

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

You can now depend on `isaac_policy_stack` from downstream packages to pull in the whole stack.
