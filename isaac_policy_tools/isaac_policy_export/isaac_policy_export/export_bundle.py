from __future__ import annotations

import argparse
from pathlib import Path
import time
import yaml

from .schemas import robot_interface_v1_skeleton

def main():
    p = argparse.ArgumentParser(description="Create a ROS-consumable policy bundle skeleton.")
    p.add_argument("--bundle_out", required=True, help="Output bundle directory")
    p.add_argument("--base_frame", default="base", help="Base frame name for robot_interface.yaml skeleton")
    p.add_argument("--robot_interface_version", type=int, default=1, help="robot_interface.yaml schema version")
    p.add_argument("--force", action="store_true", help="Overwrite robot_interface.yaml if it exists")
    args = p.parse_args()

    out = Path(args.bundle_out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rif_path = out / "robot_interface.yaml"
    if rif_path.exists() and not args.force:
        print(f"[SKIP] {rif_path} exists. Use --force to overwrite.")
    else:
        if args.robot_interface_version != 1:
            raise ValueError("Only robot_interface version 1 is supported by this skeleton generator.")
        rif = robot_interface_v1_skeleton(base_frame=args.base_frame)
        rif_path.write_text(yaml.safe_dump(rif, sort_keys=False), encoding="utf-8")
        print(f"[OK] Wrote {rif_path}")

    meta_path = out / "metadata.yaml"
    meta = {
        "created_at_unix": int(time.time()),
        "note": "Fill policy.onnx / io_descriptor.json / action_config.json with your Isaac Lab exporter.",
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"[OK] Wrote {meta_path}")

    # Touch placeholders (optional)
    for fname in ["policy.onnx", "io_descriptor.json", "action_config.json", "obs_normalization.json"]:
        fpath = out / fname
        if not fpath.exists():
            fpath.write_text("", encoding="utf-8")
            print(f"[NOTE] Created placeholder {fpath} (empty). Replace with real content.")

if __name__ == "__main__":
    main()
