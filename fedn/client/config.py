import hashlib
import os
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
ENV_PATHS: List[Path] = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    BASE_DIR.parent / "client" / ".env",
]


def _load_env_file(env_path: Path) -> Dict[str, str]:
    """Load key=value pairs from a .env file, without overriding existing envs."""
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def _load_env_files(paths: List[Path]) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    for path in paths:
        loaded.update(_load_env_file(path))
    return loaded


_loaded_env = _load_env_files(ENV_PATHS)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(float(os.getenv(name, default)))


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _resolve_under_base(raw: str) -> str:
    """Resolve a path relative to BASE_DIR when not absolute."""

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return str(candidate.resolve())


# Tunables
IMG_SIZE = _get_int("YOLO_IMG", 512)
BATCH_SIZE = _get_int("YOLO_BATCH", 24)
LR = _get_float("YOLO_LR", 0.005)
EPOCHS = _get_int("YOLO_EPOCHS", 1)
YOLO_SIZE = os.getenv("YOLO_SIZE", "s")
YOLO_NC = _get_int("YOLO_NC", 80)

# Paths (normalize to absolute to avoid CWD issues)
_data_root_raw = os.getenv(
    "YOLO_SPLITS_TARGET",
    "/app/datasets/coco_partitions",
)
DATA_ROOT = _resolve_under_base(_data_root_raw)

_runs_dir_raw = os.getenv("YOLO_RUNS_DIR", str(BASE_DIR / "yolo_runs"))
RUNS_DIR = _resolve_under_base(_runs_dir_raw)

# Identity
CLIENT_ID = _get_int("FEDN_CLIENT_ID", 0)
YOLO_CLIENT_INDEX_ENV = os.getenv("YOLO_CLIENT_INDEX")
YOLO_NUM_CLIENTS = _get_int("YOLO_NUM_CLIENTS", 10)

def _client_index_from_fedn() -> int:
    fedn_raw = os.getenv("FEDN_CLIENT_ID")
    if not fedn_raw:
        return 0
    try:
        return int(fedn_raw)
    except ValueError:
        # Stable hash to spread UUID/string IDs across splits
        h = hashlib.sha1(fedn_raw.encode()).hexdigest()
        return int(h, 16) % max(1, YOLO_NUM_CLIENTS)


CLIENT_INDEX = (
    int(YOLO_CLIENT_INDEX_ENV)
    if YOLO_CLIENT_INDEX_ENV not in (None, "")
    else _client_index_from_fedn()
)
