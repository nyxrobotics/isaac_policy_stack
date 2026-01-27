from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import yaml


def _sanitize_pkg_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("robot_name is empty after sanitization")
    return s


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copytree(src: Path, dst: Path) -> None:
    """Copy a directory tree (Python <3.8 compatible behavior).

    shutil.copytree(..., dirs_exist_ok=True) is only available on newer Python,
    so we implement an always-overwrite copy here.
    """

    if not src.is_dir():
        raise NotADirectoryError(str(src))
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out_dir = dst / rel if rel != "." else dst
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (out_dir / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, out_dir / f)


def _extract_action_offsets_from_io_descriptors(io_path: Path) -> dict[str, float] | None:
    """Extract actions[0].offset aligned with actions[0].joint_names."""

    if not io_path.exists():
        return None

    with io_path.open("r", encoding="utf-8") as f:
        io = yaml.safe_load(f) or {}

    actions = io.get("actions")
    if not (isinstance(actions, list) and actions and isinstance(actions[0], dict)):
        return None
    a0 = actions[0]
    jn = a0.get("joint_names")
    off = a0.get("offset")
    if not (isinstance(jn, list) and isinstance(off, list) and len(jn) == len(off) and len(jn) > 0):
        return None
    return {str(n): float(v) for n, v in zip(jn, off)}


def _extract_joint_pos_offsets_from_io_descriptors(io_path: Path) -> dict[str, float] | None:
    """Extract observation joint_pos_offsets aligned with observation joint_names.

    Searches IO_descriptors.yaml observations/policy entries for the first item containing
    both 'joint_pos_offsets' and 'joint_names'.
    """

    if not io_path.exists():
        return None

    with io_path.open("r", encoding="utf-8") as f:
        io = yaml.safe_load(f) or {}

    observations = io.get("observations")
    if not isinstance(observations, dict):
        return None
    policy_list = observations.get("policy")
    if not isinstance(policy_list, list):
        return None

    for e in policy_list:
        if not isinstance(e, dict):
            continue
        jn = e.get("joint_names")
        off = e.get("joint_pos_offsets")
        if isinstance(jn, list) and isinstance(off, list) and len(jn) == len(off) and len(jn) > 0:
            return {str(n): float(v) for n, v in zip(jn, off)}
    return None


def _write_offsets_yamls(bundle_dir: Path) -> None:
    """Write action_offsets.yaml and joint_pos_offsets.yaml into a bundle when available."""

    io_path = bundle_dir / "exported" / "IO_descriptors.yaml"

    act = _extract_action_offsets_from_io_descriptors(io_path)
    if act:
        out = {
            "version": 1,
            "source": {"type": "isaaclab", "file": "exported/IO_descriptors.yaml"},
            "joints": act,
        }
        (bundle_dir / "action_offsets.yaml").write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    jpos = _extract_joint_pos_offsets_from_io_descriptors(io_path)
    if jpos:
        out = {
            "version": 1,
            "source": {"type": "isaaclab", "file": "exported/IO_descriptors.yaml"},
            "joints": jpos,
        }
        (bundle_dir / "joint_pos_offsets.yaml").write_text(
            yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
        )


def _load_required_observation_terms(bundle_dir: Path) -> list[str]:
    """Load required observation term names from io_descriptor.json (preferred).

    Returns an empty list if io_descriptor.json is missing or unreadable.
    """

    io_desc = bundle_dir / "io_descriptor.json"
    if not io_desc.exists():
        return []
    try:
        import json

        data = json.loads(io_desc.read_text(encoding="utf-8"))
        obs = list(data.get("observations", []) or [])
        names = [str(o.get("name")) for o in obs if isinstance(o, dict) and o.get("name")]
        # De-dup while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out
    except Exception:
        return []


def _ensure_term_inputs_template(bundle_dir: Path) -> None:
    """Ensure robot_interface.yaml contains a term_inputs template for required terms."""

    robot_if_path = bundle_dir / "robot_interface.yaml"
    if not robot_if_path.exists():
        return

    rif = yaml.safe_load(robot_if_path.read_text(encoding="utf-8")) or {}
    required_terms = _load_required_observation_terms(bundle_dir)
    if not required_terms:
        # Nothing to template.
        return

    term_inputs = dict(rif.get("term_inputs", {}) or {})

    # Typical defaults (can be edited by the user).
    typical = {
        "base_ang_vel": {
            "source": "imu",
            "vector_field": "angular_velocity",
        },
        "base_lin_acc_sens": {
            "source": "imu",
            "vector_field": "linear_acceleration",
        },
        "velocity_commands": {
            "source": "cmd_vel",
            "mapping": {
                "lin_x": "linear.x",
                "lin_y": "linear.y",
                "ang_z": "angular.z",
            },
        },
        "joint_pos": {
            "source": "joint_states",
            "vector_field": "position",
            "relative": True,
        },
        "joint_vel": {
            "source": "joint_states",
            "vector_field": "velocity",
        },
        "actions": {
            "source": "internal",
            "note": "Filled automatically by the runner (last_action).",
        },
    }

    for name in required_terms:
        if name in term_inputs:
            continue
        if name in typical:
            term_inputs[name] = typical[name]
        else:
            term_inputs[name] = {
                "source": "<write_your_topic>",
                "note": "TODO: Fill in fields for this term.",
            }

    rif["term_inputs"] = term_inputs

    # Ensure control.relative exists (default: false)
    ctrl = dict(rif.get("control", {}) or {})
    ctrl.setdefault("relative", False)
    rif["control"] = ctrl

    robot_if_path.write_text(yaml.safe_dump(rif, sort_keys=False), encoding="utf-8")


def _render_package_xml(pkg: str) -> str:
    return f'''<?xml version="1.0"?>
<package format="3">
  <name>{pkg}</name>
  <version>0.1.0</version>
  <description>Robot-specific wrapper package to launch isaac_policy_runtime with bundled policies.</description>

  <maintainer email="you@example.com">Your Name</maintainer>
  <license>BSD-3-Clause</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>isaac_policy_runtime</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
'''


def _render_setup_py(pkg: str) -> str:
    return f'''from setuptools import setup
import os

package_name = "{pkg}"


def _bundle_data_files():
    data_files = []
    src_root = os.path.join(os.path.dirname(__file__), "bundle")
    if not os.path.isdir(src_root):
        return data_files

    for root, _, files in os.walk(src_root):
        rel_dir = os.path.relpath(root, os.path.dirname(__file__))  # bundle/<policy_name>/...
        install_dir = os.path.join("share", package_name, rel_dir)
        paths = [os.path.join(root, f) for f in files]
        if paths:
            data_files.append((install_dir, paths))
    return data_files


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/isaac_policy_runner.launch.py"]),
        *_bundle_data_files(),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Robot-specific wrapper package to launch isaac_policy_runtime with bundled policies.",
    license="BSD-3-Clause",
)
'''


def _render_setup_cfg() -> str:
    return '''[develop]
script_dir=$base/lib
[install]
install_scripts=$base/lib
'''


def _render_launch_py(pkg: str) -> str:
    return f'''from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_name = LaunchConfiguration("policy_name")
    bundle_path = PathJoinSubstitution([
        FindPackageShare("{pkg}"),
        "bundle",
        policy_name,
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_name",
            description="Policy bundle name under share/{pkg}/bundle/<policy_name>.",
        ),
        Node(
            package="isaac_policy_runtime",
            executable="isaac_policy_runner",
            name="isaac_policy_runner",
            output="screen",
            parameters=[{{"bundle_path": bundle_path}}],
        ),
    ])
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="isaac-generate-runner-pkg")
    ap.add_argument("--ws_src", required=True, help="Workspace src/ directory (e.g., ~/ros2_humble/src)")
    ap.add_argument("--robot_name", required=True, help="Robot name (e.g., kuroko)")
    ap.add_argument(
        "--bundles_dir",
        default=None,
        help=(
            "Optional directory containing one or more policy bundles. "
            "Each direct child directory is copied into <pkg>/bundle/<policy_name>/ and "
            "action_offsets.yaml and joint_pos_offsets.yaml are generated from exported/IO_descriptors.yaml when possible."
        ),
    )
    ap.add_argument(
        "--bundle",
        action="append",
        default=[],
        help=(
            "Optional: path to a single policy bundle directory. Can be specified multiple times. "
            "Copied into <pkg>/bundle/<policy_name>/ and offsets yamls are generated when possible."
        ),
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing package directory")
    args = ap.parse_args(argv)

    ws_src = Path(args.ws_src).expanduser().resolve()
    if not ws_src.exists():
        raise FileNotFoundError(f"--ws_src not found: {ws_src}")

    robot = _sanitize_pkg_name(args.robot_name)
    pkg = f"{robot}_isaac_policy_runner"
    pkg_dir = ws_src / pkg

    if pkg_dir.exists():
        if not args.force:
            raise FileExistsError(f"{pkg_dir} exists (use --force)")
        shutil.rmtree(pkg_dir)

    (pkg_dir / pkg).mkdir(parents=True, exist_ok=True)
    (pkg_dir / "launch").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "bundle").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "resource").mkdir(parents=True, exist_ok=True)

    _write_text(pkg_dir / "package.xml", _render_package_xml(pkg))
    _write_text(pkg_dir / "setup.py", _render_setup_py(pkg))
    _write_text(pkg_dir / "setup.cfg", _render_setup_cfg())
    _write_text(pkg_dir / pkg / "__init__.py", "")
    _write_text(pkg_dir / "launch" / "isaac_policy_runner.launch.py", _render_launch_py(pkg))
    _write_text(pkg_dir / "resource" / pkg, "")

    # ---- Optional: copy bundles and generate offsets yamls
    bundle_dst_root = pkg_dir / "bundle"

    bundle_sources: list[Path] = []
    if args.bundles_dir is not None:
        bd = Path(args.bundles_dir).expanduser().resolve()
        if not bd.exists() or not bd.is_dir():
            raise FileNotFoundError(f"--bundles_dir not found or not a directory: {bd}")
        for child in sorted(bd.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                bundle_sources.append(child)

    for b in args.bundle or []:
        bp = Path(b).expanduser().resolve()
        if not bp.exists() or not bp.is_dir():
            raise FileNotFoundError(f"--bundle not found or not a directory: {bp}")
        bundle_sources.append(bp)

    # De-dup by resolved path
    uniq: dict[str, Path] = {}
    for p in bundle_sources:
        uniq[str(p)] = p

    for src_bundle in uniq.values():
        policy_name = src_bundle.name
        dst_bundle = bundle_dst_root / policy_name
        if dst_bundle.exists():
            shutil.rmtree(dst_bundle)
        _copytree(src_bundle, dst_bundle)
        _write_offsets_yamls(dst_bundle)
        _ensure_term_inputs_template(dst_bundle)

    print(f"[OK] Generated package: {pkg_dir}")
    print(f"Build: colcon build --packages-select {pkg}")
    print("Note: bundle/** is installed. After adding/updating bundles, rebuild the runner package.")
    if uniq:
        print(
            "Note: action_offsets.yaml and joint_pos_offsets.yaml were generated for bundles that contain exported/IO_descriptors.yaml offsets."
        )
    print(f"Run:   ros2 launch {pkg} isaac_policy_runner.launch.py policy_name:=<policy_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
