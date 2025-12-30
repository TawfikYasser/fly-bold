"""Fetch and keep only the current client's split.

Downloads the relevant split from Hugging Face (token read from HF_TOKEN in
.env when present), places it under the configured DATA_ROOT, and removes other
client_* folders to keep disk usage minimal.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

import yaml

import config


def _unique_prefixes(preferred: Optional[str]) -> List[str]:
    prefixes: List[str] = []
    if preferred not in (None, "", "."):
        prefixes.append(preferred.strip("/"))
    prefixes.extend(["", "data"])

    seen = set()
    deduped: List[str] = []
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        deduped.append(prefix)
    return deduped

def _rewrite_dataset_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "coco_client.yaml"
    if not yaml_path.exists():
        return

    try:
        data = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return

    changed = False
    desired_path = str(dataset_dir.resolve())
    if data.get("path") != desired_path:
        data["path"] = desired_path
        changed = True

    for key in ("train", "val", "test"):
        if key not in data:
            continue
        val = data[key]
        try:
            p = Path(val)
        except TypeError:
            continue

        if p.is_absolute():
            try:
                rel = p.relative_to(dataset_dir)
                data[key] = str(rel)
                changed = True
            except ValueError:
                # Leave absolute path as-is if not under dataset_dir
                pass

    if changed:
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))


def ensure_client_split(data_root: str, keep_index: int) -> Path:
    """Ensure the desired client split exists locally, downloading if missing."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - only hit when dependency missing
        raise RuntimeError(
            "huggingface_hub is required to fetch the dataset; install it in the client environment."
        ) from exc

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    target_dir = root / f"client_{keep_index}"
    if target_dir.exists() and any(target_dir.iterdir()):
        _rewrite_dataset_yaml(target_dir)
        return target_dir

    token = config.HF_TOKEN or os.getenv("HF_TOKEN")
    if token in (None, ""):
        raise RuntimeError(
            "HF_TOKEN is required to download the private dataset anirudhsengar/coco."
        )

    prefixes = _unique_prefixes(config.HF_DATASET_SUBDIR)
    download_base = root / ".hf_download"
    download_base.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    def _find_client_dir(base: Path) -> Optional[Path]:
        for path in base.rglob(f"client_{keep_index}"):
            if path.is_dir():
                return path
        return None

    for prefix in prefixes:
        pattern = f"{prefix}/client_{keep_index}/**" if prefix else f"client_{keep_index}/**"
        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=config.HF_DATASET_REPO,
                    repo_type="dataset",
                    allow_patterns=[pattern],
                    token=token,
                    local_dir=download_base,
                    local_dir_use_symlinks=False,
                )
            )
        except Exception as exc:  # pragma: no cover - network/remote failures
            last_error = exc
            continue

        candidate = snapshot_path / prefix / f"client_{keep_index}" if prefix else snapshot_path / f"client_{keep_index}"
        if not candidate.exists():
            candidate = _find_client_dir(snapshot_path)

        if candidate and candidate.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            candidate.rename(target_dir)
            shutil.rmtree(download_base, ignore_errors=True)
            _rewrite_dataset_yaml(target_dir)
            return target_dir

        last_error = FileNotFoundError(f"Missing expected path {candidate}")

    raise RuntimeError(
        f"Failed to download client split client_{keep_index} from {config.HF_DATASET_REPO}"
    ) from last_error


def main() -> None:
    ensure_client_split(config.DATA_ROOT, config.CLIENT_INDEX)


if __name__ == "__main__":
    main()
