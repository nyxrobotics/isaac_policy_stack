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

### Copy bundles and auto-generate offsets yamls (recommended)

If your policy bundles contain `exported/IO_descriptors.yaml` with offsets,
this tool can copy bundles into the generated package and write `action_offsets.yaml` and
`joint_pos_offsets.yaml` into each bundle (when the offsets can be extracted).

Copy all bundles from a directory:

```bash
isaac-generate-runner-pkg \
  --ws_src ~/ros2_humble/src/kuroko_ros2 \
  --robot_name kuroko \
  --bundles_dir /path/to/bundles \
  --force
```

Or copy specific bundles:

```bash
isaac-generate-runner-pkg \
  --ws_src ~/ros2_humble/src/kuroko_ros2 \
  --robot_name kuroko \
  --bundle /path/to/bundles/kuroko_walk \
  --bundle /path/to/bundles/kuroko_trot \
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
