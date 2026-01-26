# isaac-policy-runner-pkggen

A small utility to generate a robot-specific ROS 2 `ament_python` package:

`<robot>_isaac_policy_runner`

The generated package contains:

- `launch/isaac_policy_runner.launch.py`
- `bundle/` (empty placeholder directory; policy bundles are managed by other tools, e.g. `isaac_policy_export`)

## Install (editable)

```bash
pip install -e .
```

## Generate a runner package

```bash
isaac-generate-runner-pkg \
  --ws_src ~/ros2_humble/src \
  --robot_name kuroko \
  --force
```

## Build and run

```bash
cd ~/ros2_humble
colcon build --packages-select kuroko_isaac_policy_runner
source install/setup.bash

ros2 launch kuroko_isaac_policy_runner isaac_policy_runner.launch.py policy_name:=kuroko_walk
```

`policy_name` selects a bundle directory under:

`share/kuroko_isaac_policy_runner/bundle/<policy_name>`
