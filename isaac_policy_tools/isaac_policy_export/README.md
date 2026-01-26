# isaac_policy_export

`isaac_policy_export` is a **pure Python utility package** for creating *policy bundles*
from trained Isaac Lab (RSL-RL) policies.

A **policy bundle** is a self-contained directory that can be consumed by a
ROS 2 runtime (e.g. `isaac_policy_runtime`) without depending on Isaac Lab.

This package is intentionally **not a ROS package** and has **no ROS dependencies**.

---

## What this tool does

`isaac-export-bundle` helps you:

- Create a **bundle directory skeleton**
- Generate a versioned `robot_interface.yaml`
- Copy exported policy files from Isaac Lab:
  - `policy.onnx` (**required**)
  - `policy.pt` (optional)
- Create placeholders for:
  - `io_descriptor.json`
  - `action_config.json`
  - `obs_normalization.json`

The goal is to clearly separate responsibilities:

| Component | Responsibility |
|---------|----------------|
| Isaac Lab | Training, simulation, exporting `policy.onnx` |
| isaac_policy_export | Converting Isaac Lab outputs into a portable bundle |
| ROS runtime | Loading and executing the bundle |

---

## Installation (editable, recommended)

Run this **inside the Isaac Lab Python environment**:

```bash
cd isaac_policy_tools/isaac_policy_export
pip install -e .
```

This installs the CLI entrypoint:

```bash
isaac-export-bundle
```

---

## Basic usage

### 1. Export policy from Isaac Lab

After training, run `play.py` to generate exported artifacts:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Kuroko-v0 \
  --headless \
  --num_envs 1 \
  --checkpoint model_XXXXX.pt
```

This creates:

```
logs/rsl_rl/<experiment>/<run>/exported/
  policy.onnx
  policy.pt
```

---

### 2. Create a bundle and copy policy files

```bash
isaac-export-bundle \
  --bundle_out /path/to/bundle/kuroko_walk \
  --exported_dir /path/to/logs/.../exported \
  --base_frame body_link \
  --force
```

Resulting directory:

```
kuroko_walk/
  policy.onnx
  policy.pt
  robot_interface.yaml
  metadata.yaml
  io_descriptor.json          (placeholder)
  action_config.json          (placeholder)
  obs_normalization.json      (placeholder)
```

---

## Command-line arguments

### Required

- `--bundle_out PATH`  
  Output directory for the bundle.

### Optional

- `--exported_dir PATH`  
  Path to Isaac Lab `exported/` directory containing `policy.onnx`.
  - If provided:
    - `policy.onnx` is **required**
    - `policy.pt` is copied if present
  - If omitted:
    - Empty placeholder files are created instead

- `--base_frame NAME`  
  Base frame name written into `robot_interface.yaml` (default: `base`)

- `--robot_interface_version v1`  
  Schema version for `robot_interface.yaml` (currently only `v1`)

- `--force`  
  Overwrite existing files in the bundle directory

---

## Generated files

### robot_interface.yaml

A **versioned skeleton** describing how a policy interfaces with ROS:

- Frames
- Sensor sources
- Term-to-input mappings
- Control output sink

You are expected to **edit this file** to match:
- ROS topic names
- Controller interfaces
- Sensor availability

---

### metadata.yaml

Contains bundle-level metadata:

- Creation timestamp
- Tool name
- Schema versions

This file is informational and safe to commit.

---

### Placeholder files

The following files are created empty and must be filled later:

- `io_descriptor.json`  
  (exported from Isaac Lab via `export_io_descriptors.py`)
- `action_config.json`  
  (joint order, scaling, action semantics)
- `obs_normalization.json` (optional)

---

## Design principles

- **Pure Python**
- **No ROS imports**
- **No Isaac Lab imports**
- Explicit inputs, no hidden side effects
- Stable, versioned output formats

This tool should remain usable even if:
- Isaac Lab versions change
- ROS distributions change
- The runtime implementation evolves

---

## What this tool intentionally does NOT do

- Run training or simulation
- Launch Isaac Lab
- Infer ROS topic names automatically
- Guess joint order or controller semantics
- Modify policies

Those responsibilities belong elsewhere.

---

## Typical workflow

1. Train policy in Isaac Lab
2. Run `play.py` → get `exported/policy.onnx`
3. Run `isaac-export-bundle`
4. Fill in remaining bundle files
5. Commit or distribute the bundle
6. Load it in ROS via `isaac_policy_runtime`

---

## License

BSD-3-Clause
