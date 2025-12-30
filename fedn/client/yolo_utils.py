import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import yaml
from yolov5.models.yolo import Model

# Compatibility shim for Pillow >=10 where FreeTypeFont.getsize was removed.
# YOLOv5's plotting utilities call `font.getsize(label)`; newer Pillow
# provides `getbbox` instead. Provide `getsize` if missing so validation
# and plotting continue to work with newer Pillow versions (10+ / 12+).
try:
    from PIL import ImageFont

    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def _compat_getsize(self, text):
            # getbbox returns (x0, y0, x1, y1)
            try:
                bbox = self.getbbox(text)
                if bbox is None:
                    return (0, 0)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                # Fallback conservative estimate
                mask = self.getmask(text)
                return mask.size

        ImageFont.FreeTypeFont.getsize = _compat_getsize
except Exception:
    # If PIL is unavailable or shim fails, plotting may still break downstream;
    # leave behavior unchanged so errors surface normally.
    pass

YOLO_SIZE_TO_YAML = {
    "n": "yolov5n.yaml",
    "s": "yolov5s.yaml",
    "m": "yolov5m.yaml",
    "l": "yolov5l.yaml",
    "x": "yolov5x.yaml",
}


def resolve_yolo_yaml(yolo_size: str) -> Path:
    """Return path to the YOLO model yaml shipped with the yolov5 package."""
    yaml_name = YOLO_SIZE_TO_YAML.get(yolo_size, YOLO_SIZE_TO_YAML["s"])
    base = Path(__import__("yolov5").__file__).resolve().parent
    return base / "models" / yaml_name


def build_yolo_model(yolo_size: str = "s", nc: int = 80) -> Model:
    yaml_path = resolve_yolo_yaml(yolo_size)
    return Model(cfg=str(yaml_path), ch=3, nc=nc)


def state_dict_to_numpy_list(state_dict: Dict[str, torch.Tensor]) -> List:
    return [val.detach().cpu().numpy() for _, val in state_dict.items()]


def numpy_list_to_state_dict(model: Model, arrays: Iterable) -> Dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    arrays_list = list(arrays)
    if len(keys) != len(arrays_list):
        raise ValueError("Array list length does not match model state_dict")
    mapped = {k: torch.tensor(arrays_list[i]) for i, k in enumerate(keys)}
    return mapped


def save_state_dict_as_yolo_checkpoint(state_dict: Dict[str, torch.Tensor], yolo_size: str, save_path: str, nc: int = 80) -> str:
    model = build_yolo_model(yolo_size, nc)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading state_dict into YOLO model: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading state_dict into YOLO model: {unexpected}")
    # Ensure class metadata is propagated for downstream val that expects it on model and detect layer.
    try:
        model.nc = nc
    except Exception:
        pass
    model.yaml["nc"] = nc
    if not hasattr(model, "names") or model.names is None:
        model.names = [str(i) for i in range(nc)]
    else:
        # normalize to list[str]
        model.names = [str(n) for n in model.names]
    try:
        if hasattr(model, "model"):
            setattr(model.model, "nc", nc)
            for m in model.model:
                if hasattr(m, "nc"):
                    m.nc = nc
    except Exception:
        pass
    ckpt = {
        "epoch": -1,
        "model": model,
        "optimizer": None,
        "yaml": YOLO_SIZE_TO_YAML.get(yolo_size, YOLO_SIZE_TO_YAML["s"]),
        "nc": int(model.yaml.get("nc", nc)),
        "names": model.names if hasattr(model, "names") else None,
        "version": "fedn-yolo",
    }
    save_path = str(save_path)
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    return save_path


def load_yolo_checkpoint_as_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    yolo_model = ckpt.get("model")
    if yolo_model is None:
        raise RuntimeError("Checkpoint missing model")
    if isinstance(yolo_model, dict):
        return yolo_model
    return yolo_model.state_dict()


def count_images_from_yaml(data_yaml: str) -> Tuple[int, int]:
    """Count images for train/val entries, honoring the dataset `path` root.

    YOLO data.yaml files often express `train`/`val` as paths relative to a
    `path` key (or the yaml's parent directory). When we rewrite the yaml to
    use relative paths, simple `Path(value)` resolution points at the process
    CWD and yields zero. Normalize all paths against the declared dataset root
    so we count correctly in packaged/relocated environments.
    """

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_root = cfg.get("path")
    base_dir = Path(dataset_root) if dataset_root else Path(data_yaml).parent
    base_dir = base_dir.expanduser().resolve()

    train_path = cfg.get("train")
    val_path = cfg.get("val")

    def _count(path_value) -> int:
        if path_value is None:
            return 0
        if isinstance(path_value, list):
            return sum(_count(item) for item in path_value)

        # Resolve relative entries against dataset root to avoid CWD issues.
        p = Path(str(path_value))
        if not p.is_absolute():
            p = base_dir / p

        if p.is_dir():
            return len(list(p.glob("*.jpg"))) + len(list(p.glob("*.png")))
        if p.is_file():
            return len([line for line in p.read_text().splitlines() if line.strip()])
        return 0

    return _count(train_path), _count(val_path)
