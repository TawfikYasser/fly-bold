import os
import shutil
from pathlib import Path
from typing import List, Optional

import yaml

import config


def ensure_client_split(data_root: str, keep_index: int) -> Path:
    """Ensure the desired client split exists locally, downloading if missing."""

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    target_dir = root / f"client_{keep_index}"
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    # Check for pre-partitioned data
    # The default location for pre-partitioned data
    pre_partitioned_root = Path(os.getenv("PRE_PARTITIONED_ROOT", "/app/datasets/coco_partitions"))
    pre_partitioned_dir = pre_partitioned_root / f"client_{keep_index}"
    
    if pre_partitioned_dir.exists():
        print(f"Found pre-partitioned data at {pre_partitioned_dir}")
        yaml_path = pre_partitioned_dir / "coco_client.yaml"
        if yaml_path.exists():
             return pre_partitioned_dir
        else:
             print(f"Warning: {pre_partitioned_dir} exists but missing coco_client.yaml")

    else:
        raise FileNotFoundError(
            f"Pre-partitioned data not found at {pre_partitioned_dir}. "
            "Please ensure the data is present on the VM."
        )


def main() -> None:
    ensure_client_split(config.DATA_ROOT, config.CLIENT_INDEX)


if __name__ == "__main__":
    main()
