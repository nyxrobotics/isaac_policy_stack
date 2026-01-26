# isaac_policy_bringup

Bringup launch files that start `isaac_policy_runtime` with a selected policy bundle.

## Run

```bash
ros2 launch isaac_policy_bringup bringup.launch.py bundle_path:=/absolute/path/to/bundle
```

This package intentionally keeps logic minimal. The runtime node reads all topic and wiring
information from the bundle's `robot_interface.yaml`.
