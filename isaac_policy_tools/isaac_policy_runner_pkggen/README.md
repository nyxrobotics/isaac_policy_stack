# isaac-policy-runner-pkggen

A small utility to generate a robot-specific ROS 2 `ament_python` package:

`<robot>_isaac_policy_runner`

The generated package contains:

- `launch/isaac_policy_runner.launch.py`
- `bundle/` (a directory intended to contain one or more policy bundles)

## Why bundle is installed

This generator configures the generated runner package to **install all files under `bundle/**`**
into:

`share/<robot>_isaac_policy_runner/bundle/**`

This makes runtime lookup via `FindPackageShare(<robot>_isaac_policy_runner)/bundle/<policy_name>`
work reliably.

> Note: When you add/update bundles under `bundle/`, you must rebuild the runner package with `colcon build`.

## Install (editable)

```bash
pip install -e .
```

## Generate a runner package

```bash
isaac-generate-runner-pkg \
  --ws_src ~/ros2_humble/src/kuroko_ros2 \
  --robot_name kuroko \
  --force
```

## Add a bundle

Place a bundle under:

`<ws>/src/<robot>_isaac_policy_runner/bundle/<policy_name>/...`

Then rebuild:

```bash
cd ~/ros2_humble
colcon build --packages-select <robot>_isaac_policy_runner
source install/setup.bash
```

## Run

```bash
ros2 launch <robot>_isaac_policy_runner isaac_policy_runner.launch.py policy_name:=<policy_name>
```
