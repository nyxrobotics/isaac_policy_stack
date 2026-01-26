# isaac_policy_export

A **pure Python** exporter that runs inside your Isaac Lab environment to produce *policy bundles*
consumed by `isaac_policy_runtime`.

This package is intentionally ROS-free: it does **not** import `rclpy` or build with `ament`.

## Install (in Isaac Lab Python env)

From this folder:

```bash
pip install -e .
```

## Usage

```bash
isaac-export-bundle \
  --bundle_out /path/to/policy_bundles/my_bundle \
  --robot_interface_version 1
```

This exporter is a scaffold: it writes the bundle file layout and a `robot_interface.yaml` skeleton.
You should integrate it with your Isaac Lab training/play code to:
- export `policy.onnx`
- generate `io_descriptor.json`
- generate `action_config.json`
- write optional `obs_normalization.json`

## Output contract

The exporter creates/updates these files:

- `robot_interface.yaml` (v1 schema)
- `metadata.yaml` (lightweight)

You can choose to overwrite or merge existing `robot_interface.yaml` depending on your workflow.
