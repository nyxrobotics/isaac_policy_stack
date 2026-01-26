from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import ActionConfig, ObservationSpec, RobotInterface, SourceSpec


@dataclass(frozen=True)
class PolicyBundle:
    root: Path
    policy_onnx: Path
    io_descriptor: Dict[str, Any]
    observations: list[ObservationSpec]
    action_config: ActionConfig
    robot_interface: RobotInterface
    obs_normalization: Dict[str, Any] | None
    metadata: Dict[str, Any] | None


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def load_bundle(bundle_path: str) -> PolicyBundle:
    root = Path(bundle_path).expanduser().resolve()

    # Preferred layout: bundle/exported/policy.onnx
    # Backwards-compatible layout: bundle/policy.onnx
    policy_onnx = root / "exported" / "policy.onnx"
    if not policy_onnx.exists():
        policy_onnx = root / "policy.onnx"

    io_desc_path = root / "io_descriptor.json"
    action_cfg_path = root / "action_config.json"
    robot_if_path = root / "robot_interface.yaml"

    _require(policy_onnx)
    _require(io_desc_path)
    _require(action_cfg_path)
    _require(robot_if_path)

    io_desc = json.loads(io_desc_path.read_text(encoding="utf-8"))
    obs_specs: list[ObservationSpec] = []
    for o in io_desc.get("observations", []):
        obs_specs.append(
            ObservationSpec(
                name=o["name"],
                func=o.get("func", o["name"]),
                params=o.get("params"),
                clip=tuple(o["clip"]) if o.get("clip") else None,
            )
        )

    ac = json.loads(action_cfg_path.read_text(encoding="utf-8"))
    # Newer bundles may use 'policy_joint_order' to avoid ambiguity.
    joint_order = list(ac.get("policy_joint_order") or ac.get("joint_order") or [])
    action_config = ActionConfig(
        type=ac.get("type", "joint_position"),
        joint_order=joint_order,
        scale=ac.get("scale"),
        use_default_offset=ac.get("use_default_offset"),
        clip=tuple(ac.get("clip") or [-1.0, 1.0]),
    )

    rif = yaml.safe_load(robot_if_path.read_text(encoding="utf-8"))
    version = int(rif.get("version", 1))
    frames = dict(rif.get("frames", {}))
    sources: dict[str, SourceSpec] = {}
    for k, v in dict(rif.get("sources", {})).items():
        sources[str(k)] = SourceSpec(msg_type=str(v["msg_type"]), topic=str(v["topic"]))

    robot_interface = RobotInterface(
        version=version,
        frames=frames,
        sources=sources,
        joints=dict(rif.get("joints", {})),
        term_inputs=dict(rif.get("term_inputs", {})),
        control=dict(rif.get("control", {})),
    )

    obs_norm_path = root / "obs_normalization.json"
    obs_norm = None
    if obs_norm_path.exists():
        obs_norm = json.loads(obs_norm_path.read_text(encoding="utf-8"))

    meta_path = root / "metadata.yaml"
    meta = None
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    return PolicyBundle(
        root=root,
        policy_onnx=policy_onnx,
        io_descriptor=io_desc,
        observations=obs_specs,
        action_config=action_config,
        robot_interface=robot_interface,
        obs_normalization=obs_norm,
        metadata=meta,
    )
