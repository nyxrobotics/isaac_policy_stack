#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
#
# isaac_policy_export/export_bundle.py
#
# Bundle exporter (redesigned):
# - Copies Isaac Lab artifacts into bundle/exported/:
#     - policy.onnx (required when --exported_dir is used)
#     - policy.pt   (optional)
#     - IO_descriptors.yaml (optional but recommended)
# - Auto-generates runtime ABI files from exported/IO_descriptors.yaml:
#     - io_descriptor.json
#     - action_config.json  (policy joint order only)
# - Writes robot_interface.yaml with controller_name only (NO joint order).
#   Controller joint order is resolved at runtime via ROS2 parameters:
#     ros2 param get <controller_name> joints
#
# Notes:
# - We intentionally do NOT parse any controller YAML here.
# - We intentionally do NOT assume /joint_states order; runtime uses JointState.name[].

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .schemas import robot_interface_v1_skeleton


# -----------------------------
# Utilities
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any, force: bool) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path} exists (use --force to overwrite)")
        return
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {path}")


def _write_yaml(path: Path, data: Any, force: bool) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path} exists (use --force to overwrite)")
        return
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Wrote {path}")


def _ensure_placeholder(path: Path, force: bool, note: str) -> None:
    if path.exists() and not force:
        print(f"[SKIP] {path} exists (use --force to overwrite)")
        return
    _ensure_dir(path.parent)
    path.write_bytes(b"")
    print(f"[NOTE] Created placeholder {path} (empty). {note}")


def _copy_file(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[SKIP] {dst} exists (use --force to overwrite)")
        return
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    print(f"[OK] Copied {src} -> {dst}")


def _require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{what} not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"{what} is not a file: {path}")


# -----------------------------
# CLI
# -----------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="isaac-export-bundle",
        description="Create a policy bundle (copy Isaac Lab exported/ artifacts + generate runtime ABI).",
    )

    p.add_argument("--bundle_out", required=True, type=str, help="Output bundle directory path.")
    p.add_argument("--base_frame", default="base", type=str, help="Base frame name in robot_interface.yaml.")
    p.add_argument(
        "--robot_interface_version",
        default="v1",
        choices=["v1"],
        help="Robot interface schema version (currently only v1).",
    )

    # Isaac Lab exported dir
    p.add_argument(
        "--exported_dir",
        default=None,
        type=str,
        help="Path to Isaac Lab exported/ directory containing policy.onnx, policy.pt, IO_descriptors.yaml.",
    )

    # Controller wiring (exporter writes names only; runtime resolves joint order via ROS params)
    p.add_argument(
        "--controller_name",
        default="/joint_group_position_controller",
        type=str,
        help="ROS2 controller node name (used to query param 'joints' at runtime).",
    )
    p.add_argument(
        "--controller_manager_name",
        default="/controller_manager",
        type=str,
        help="ROS2 controller_manager node name (optional; runtime may use it for validation/introspection).",
    )
    p.add_argument(
        "--command_topic",
        default=None,
        type=str,
        help="Command topic to publish to. Default: <controller_name>/commands",
    )
    p.add_argument(
        "--command_msg_type",
        default="std_msgs/msg/Float64MultiArray",
        type=str,
        help="Command message type (e.g. std_msgs/msg/Float64MultiArray).",
    )

    # Defaults for action config generation
    p.add_argument(
        "--action_scale",
        default=0.5,
        type=float,
        help="Default action scale written to action_config.json (if generated).",
    )
    p.add_argument(
        "--action_use_default_offset",
        action="store_true",
        default=True,
        help="Write use_default_offset=true to action_config.json (if generated).",
    )

    p.add_argument("--force", action="store_true", help="Overwrite existing files in bundle_out.")

    return p.parse_args(argv)


# -----------------------------
# Core
# -----------------------------

@dataclass(frozen=True)
class ExportConfig:
    bundle_out: Path
    base_frame: str
    robot_interface_version: str
    exported_dir: Optional[Path]
    force: bool

    controller_name: str
    controller_manager_name: str
    command_topic: str
    command_msg_type: str

    action_scale: float
    action_use_default_offset: bool


def _write_metadata(cfg: ExportConfig) -> None:
    metadata = {
        "bundle_format_version": 1,
        "created_at_utc": _utc_now_iso(),
        "tool": {"name": "isaac_policy_export", "entrypoint": "isaac-export-bundle"},
        "robot_interface_version": cfg.robot_interface_version,
        "layout": {
            "exported_dir": "exported/",
            "exported_artifacts": ["policy.onnx", "policy.pt", "IO_descriptors.yaml"],
            "runtime_abi_files": ["io_descriptor.json", "action_config.json", "obs_normalization.json"],
        },
        "runtime_assumptions": {
            "controller_joint_order_resolution": {
                "method": "ros2_param_get",
                "controller_param": "joints",
                "controller_node": cfg.controller_name,
            },
            "joint_states_resolution": {
                "method": "name_indexing",
                "note": "Use sensor_msgs/JointState.name[] to align positions/velocities.",
            },
        },
    }
    _write_yaml(cfg.bundle_out / "metadata.yaml", metadata, force=cfg.force)


def _write_robot_interface(cfg: ExportConfig) -> None:
    if cfg.robot_interface_version != "v1":
        raise RuntimeError(f"Unsupported robot_interface_version: {cfg.robot_interface_version}")

    data = robot_interface_v1_skeleton(base_frame=cfg.base_frame)

    # Ensure minimal sources exist (topics are runtime-configurable; keep generic defaults)
    data.setdefault("sources", {})
    data["sources"].setdefault("joint_states", {"topic": "/joint_states", "msg_type": "sensor_msgs/msg/JointState"})
    data["sources"].setdefault("imu", {"topic": "/imu/data", "msg_type": "sensor_msgs/msg/Imu"})
    data["sources"].setdefault("cmd_vel", {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist"})

    # Control wiring: controller name only; runtime must resolve joints via ROS parameters.
    data["control"] = {
        "type": "forward_command_controller",
        "controller_name": cfg.controller_name,
        "controller_manager_name": cfg.controller_manager_name,
        "command_topic": cfg.command_topic,
        "command_msg_type": cfg.command_msg_type,
        "rate_hz": 50,
        # Explicitly state: order is resolved at runtime via ROS params
        "command_joint_order": None,
        "command_joint_order_resolution": {
            "method": "ros_param",
            "node": cfg.controller_name,
            "param": "joints",
        },
        # Policy joint order comes from action_config.json (generated from IO_descriptors.yaml)
        "policy_joint_order_source": "action_config.json",
    }

    # term_inputs intentionally left skeletal; wiring is robot-specific and should be edited.
    data.setdefault("term_inputs", {})

    _write_yaml(cfg.bundle_out / "robot_interface.yaml", data, force=cfg.force)


def _copy_exported_artifacts(cfg: ExportConfig) -> None:
    if cfg.exported_dir is None:
        return

    src_dir = cfg.exported_dir
    if not src_dir.exists() or not src_dir.is_dir():
        raise RuntimeError(f"--exported_dir does not exist or is not a directory: {src_dir}")

    dst_dir = cfg.bundle_out / "exported"
    _ensure_dir(dst_dir)

    # policy.onnx required
    onnx_src = src_dir / "policy.onnx"
    _require_file(onnx_src, "policy.onnx")
    _copy_file(onnx_src, dst_dir / "policy.onnx", force=cfg.force)

    # policy.pt optional
    pt_src = src_dir / "policy.pt"
    if pt_src.exists() and pt_src.is_file():
        _copy_file(pt_src, dst_dir / "policy.pt", force=cfg.force)
    else:
        print(f"[WARN] {pt_src} not found, skipping")

    # IO_descriptors.yaml optional but recommended
    io_src = src_dir / "IO_descriptors.yaml"
    if io_src.exists() and io_src.is_file():
        _copy_file(io_src, dst_dir / "IO_descriptors.yaml", force=cfg.force)
    else:
        print(f"[WARN] {io_src} not found, skipping")


def _generate_runtime_abi_from_io_descriptors(cfg: ExportConfig) -> None:
    io_yaml = cfg.bundle_out / "exported" / "IO_descriptors.yaml"
    if not io_yaml.exists():
        print(f"[WARN] {io_yaml} not found. Cannot auto-generate io_descriptor.json/action_config.json.")
        return

    with io_yaml.open("r", encoding="utf-8") as f:
        io = yaml.safe_load(f)

    # ---- Observations (policy group)
    obs_terms = io.get("observations", {}).get("policy", None)
    if not isinstance(obs_terms, list) or len(obs_terms) == 0:
        raise RuntimeError("IO_descriptors.yaml does not contain a non-empty observations.policy list")

    # Isaac Lab term names -> runtime ABI term names
    name_map = {
        "generated_commands": "velocity_commands",
        "joint_pos_rel": "joint_pos",
        "joint_vel_rel": "joint_vel",
        "last_action": "actions",
    }

    out_obs: list[dict[str, Any]] = []
    total_dim = 0
    for term in obs_terms:
        src_name = term.get("name")
        shape = term.get("shape")
        if src_name is None or shape is None:
            raise RuntimeError(f"Invalid observation term entry: {term}")

        dst_name = name_map.get(str(src_name), str(src_name))

        if isinstance(shape, int):
            shape_list = [int(shape)]
        else:
            shape_list = [int(x) for x in list(shape)]

        dim = 1
        for s in shape_list:
            dim *= int(s)
        total_dim += dim

        out_obs.append({"name": dst_name, "shape": shape_list})

    io_desc = {
        "version": 1,
        "observation_group": "policy",
        "observations": out_obs,
        "total_dim": int(total_dim),
        "source": {"type": "isaaclab", "file": "exported/IO_descriptors.yaml"},
        "name_map": name_map,
    }
    _write_json(cfg.bundle_out / "io_descriptor.json", io_desc, force=cfg.force)

    # ---- Actions: policy joint order only (NOT controller order)
    actions = io.get("actions", None)
    joint_names: Optional[list[str]] = None
    if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
        j = actions[0].get("joint_names", None)
        if isinstance(j, list) and len(j) > 0:
            joint_names = [str(x) for x in j]

    if joint_names is None:
        print("[WARN] Could not find actions[0].joint_names in IO_descriptors.yaml; skipping action_config.json generation.")
        return

    action_cfg = {
        "type": "joint_position",
        # This is POLICY joint order (ABI). Runtime remaps to controller joints at runtime.
        "policy_joint_order": joint_names,
        "scale": float(cfg.action_scale),
        "use_default_offset": bool(cfg.action_use_default_offset),
        "clip": None,
        "source": {"type": "isaaclab", "file": "exported/IO_descriptors.yaml"},
    }
    _write_json(cfg.bundle_out / "action_config.json", action_cfg, force=cfg.force)


def _write_placeholders_if_needed(cfg: ExportConfig) -> None:
    _ensure_dir(cfg.bundle_out / "exported")

    if cfg.exported_dir is None:
        _ensure_placeholder(
            cfg.bundle_out / "exported" / "policy.onnx",
            force=cfg.force,
            note="Replace with real content (from Isaac Lab exported/).",
        )
        _ensure_placeholder(
            cfg.bundle_out / "exported" / "policy.pt",
            force=cfg.force,
            note="Optional. Replace with real content (from Isaac Lab exported/).",
        )
        _ensure_placeholder(
            cfg.bundle_out / "exported" / "IO_descriptors.yaml",
            force=cfg.force,
            note="Optional but recommended. Copy from Isaac Lab exported/.",
        )

    if not (cfg.bundle_out / "io_descriptor.json").exists():
        _ensure_placeholder(
            cfg.bundle_out / "io_descriptor.json",
            force=cfg.force,
            note="Replace or auto-generate from exported/IO_descriptors.yaml.",
        )
    if not (cfg.bundle_out / "action_config.json").exists():
        _ensure_placeholder(
            cfg.bundle_out / "action_config.json",
            force=cfg.force,
            note="Replace or auto-generate from exported/IO_descriptors.yaml.",
        )

    _ensure_placeholder(
        cfg.bundle_out / "obs_normalization.json",
        force=cfg.force,
        note="Optional. Replace with mean/std if you use observation normalization.",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    bundle_out = Path(args.bundle_out).expanduser().resolve()
    exported_dir = Path(args.exported_dir).expanduser().resolve() if args.exported_dir else None

    controller_name = str(args.controller_name)
    command_topic = str(args.command_topic) if args.command_topic else f"{controller_name}/commands"

    cfg = ExportConfig(
        bundle_out=bundle_out,
        base_frame=str(args.base_frame),
        robot_interface_version=str(args.robot_interface_version),
        exported_dir=exported_dir,
        force=bool(args.force),
        controller_name=controller_name,
        controller_manager_name=str(args.controller_manager_name),
        command_topic=command_topic,
        command_msg_type=str(args.command_msg_type),
        action_scale=float(args.action_scale),
        action_use_default_offset=bool(args.action_use_default_offset),
    )

    _ensure_dir(cfg.bundle_out)

    # 1) Always write skeleton files
    _write_robot_interface(cfg)
    _write_metadata(cfg)

    # 2) Copy Isaac Lab artifacts into bundle/exported/
    _copy_exported_artifacts(cfg)

    # 3) Generate runtime ABI from bundle/exported/IO_descriptors.yaml
    _generate_runtime_abi_from_io_descriptors(cfg)

    # 4) Fill any missing files with placeholders
    _write_placeholders_if_needed(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
