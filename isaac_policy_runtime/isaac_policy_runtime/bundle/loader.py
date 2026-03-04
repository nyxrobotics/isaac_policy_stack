from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import ActionConfig, ObservationConfig, ObservationSpec, RobotInterface, SourceSpec


@dataclass(frozen=True)
class PolicyBundle:
    root: Path
    policy_onnx: Path
    io_descriptors: Dict[str, Any] | None
    observation_config: ObservationConfig
    action_config: ActionConfig
    robot_interface: RobotInterface
    obs_normalization: Dict[str, Any] | None
    metadata: Dict[str, Any] | None


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

def _prod(shape: list[int] | None) -> int | None:
    if not shape:
        return None
    d = 1
    for s in shape:
        d *= int(s)
    return int(d)



def _load_io_descriptors_yaml(root: Path) -> dict[str, Any] | None:
    """Load Isaac Lab IO_descriptors.yaml (preferred: io_descriptors.yaml copy).

    We accept multiple filenames for backwards-compatibility:
      - <bundle>/io_descriptors.yaml              (preferred, lowercase)
      - <bundle>/exported/IO_descriptors.yaml     (exported by isaac_policy_export)
      - <bundle>/IO_descriptors.yaml              (legacy)
    """
    candidates = [
        root / "io_descriptors.yaml",
        root / "exported" / "IO_descriptors.yaml",
        root / "IO_descriptors.yaml",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
    return None


def _find_obs_term(ioy: dict[str, Any], name: str) -> dict[str, Any] | None:
    obs = (ioy or {}).get("observations") or {}
    policy = obs.get("policy") if isinstance(obs, dict) else None
    if not isinstance(policy, list):
        return None
    for t in policy:
        if isinstance(t, dict) and str(t.get("name")) == name:
            return t
    return None


def _build_configs_from_io_descriptors(ioy: dict[str, Any], default_action_use_default_offset: bool = True) -> tuple[ActionConfig, ObservationConfig]:
    # ---- Actions
    actions_list = ioy.get("actions") if isinstance(ioy, dict) else None
    if not (isinstance(actions_list, list) and len(actions_list) > 0 and isinstance(actions_list[0], dict)):
        raise RuntimeError("io_descriptors.yaml is missing 'actions[0]' entry")

    a0 = actions_list[0]
    joint_order = [str(x) for x in list(a0.get("joint_names") or [])]
    if not joint_order:
        raise RuntimeError("io_descriptors.yaml actions[0].joint_names is empty")

    offset = None
    if isinstance(a0.get("offset"), list):
        offset = [float(x) for x in list(a0.get("offset") or [])]

    clip = a0.get("clip")
    clip_tuple = (-1.0, 1.0)
    if isinstance(clip, list) and len(clip) == 2:
        clip_tuple = (float(clip[0]), float(clip[1]))

    scale = None
    if a0.get("scale") is not None:
        try:
            scale = float(a0.get("scale"))
        except Exception:
            scale = None

    action_cfg = ActionConfig(
        type="joint_position",
        joint_order=joint_order,
        scale=scale,
        use_default_offset=bool(default_action_use_default_offset),
        offset=offset,
        relative=True,
        clip=clip_tuple,
    )

    # ---- Observations
    obs = (ioy.get("observations") or {}) if isinstance(ioy, dict) else {}
    policy = obs.get("policy") if isinstance(obs, dict) else None
    if not isinstance(policy, list) or len(policy) == 0:
        raise RuntimeError("io_descriptors.yaml observations.policy is empty")

    terms: list[ObservationSpec] = []
    total_dim = 0
    for t in policy:
        if not isinstance(t, dict):
            raise RuntimeError(f"Invalid observation term: {t}")
        name = str(t.get("name"))
        shape_raw = t.get("shape")
        if shape_raw is None:
            raise RuntimeError(f"Observation term '{name}' missing shape")
        if isinstance(shape_raw, int):
            shape = [int(shape_raw)]
        else:
            shape = [int(x) for x in list(shape_raw)]
        dim = _prod(shape) or 0
        total_dim += dim

        params: dict[str, Any] = {}
        if isinstance(t.get("joint_names"), list):
            params["joint_order"] = [str(x) for x in list(t.get("joint_names") or [])]
        if isinstance(t.get("joint_pos_offsets"), list):
            params["offsets"] = [float(x) for x in list(t.get("joint_pos_offsets") or [])]
        if isinstance(t.get("joint_vel_offsets"), list):
            params["offsets"] = [float(x) for x in list(t.get("joint_vel_offsets") or [])]
        if params:
            params.setdefault("relative", True)

        terms.append(
            ObservationSpec(
                name=name,
                func=name,
                params=params if params else None,
                clip=None,
                shape=shape,
                dim=int(dim),
            )
        )

    obs_cfg = ObservationConfig(terms=terms, total_dim=int(total_dim))
    return action_cfg, obs_cfg

def load_bundle(bundle_path: str) -> PolicyBundle:
    root = Path(bundle_path).expanduser().resolve()

    # Preferred layout: bundle/exported/policy.onnx
    # Backwards-compatible layout: bundle/policy.onnx
    policy_onnx = root / "exported" / "policy.onnx"
    if not policy_onnx.exists():
        policy_onnx = root / "policy.onnx"

    robot_if_path = root / "robot_interface.yaml"

    _require(policy_onnx)
    _require(robot_if_path)


    io_desc_yaml = _load_io_descriptors_yaml(root)

    # New design: IO_descriptors.yaml is the single source of truth for the ABI.
    if io_desc_yaml is None:
        raise FileNotFoundError(
            "Missing IO_descriptors.yaml in the bundle. Provide one of: io_descriptors.yaml, exported/IO_descriptors.yaml, or IO_descriptors.yaml."
        )

    action_config, observation_config = _build_configs_from_io_descriptors(io_desc_yaml)
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

    # Prefer YAML normalization; keep JSON as backwards-compatible input.
    obs_norm = None
    obs_norm_yaml = root / ""
    obs_norm_json = root / "obs_normalization.json"
    if obs_norm_yaml.exists():
        try:
            obs_norm = yaml.safe_load(obs_norm_yaml.read_text(encoding="utf-8"))
        except Exception:
            obs_norm = None
    elif obs_norm_json.exists():
        try:
            obs_norm = json.loads(obs_norm_json.read_text(encoding="utf-8"))
        except Exception:
            obs_norm = None

    meta_path = root / ""
    meta = None
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    return PolicyBundle(
        root=root,
        policy_onnx=policy_onnx,
        io_descriptors=io_desc_yaml,
        observation_config=observation_config,
        action_config=action_config,
        robot_interface=robot_interface,
        obs_normalization=obs_norm,
        metadata=meta,
    )
