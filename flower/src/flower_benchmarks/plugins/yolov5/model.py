# flower_benchmarks/plugins/yolov5/model.py
"""
Helpers to interop between Flower model state_dicts and YOLOv5 checkpoint files.
We keep things minimal: we create a YOLO-style checkpoint dict with a 'model' key
so YOLO's CLI/train.py can load it and continue training.
"""
import os
from yolov5.models.yolo import Model
import torch

YoloSizeToPretrained = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}

# def save_state_dict_as_yolo_checkpoint(state_dict: dict, out_path: str):
#     """
#     Wrap a PyTorch state_dict into a YOLO-friendly checkpoint dict and save.
#     YOLO typically expects ckpt['model'] or a full checkpoint; this creates ckpt['model'].
#     """
#     ckpt = {"model": state_dict}
#     torch.save(ckpt, out_path)

import torch
from pathlib import Path
from yolov5.models.yolo import Model

def save_state_dict_as_yolo_checkpoint(
    state_dict: dict,
    yolo_size: str,
    save_path: str,
    nc: int = None
):
    """
    Convert a Flower-style YOLO state_dict into a YOLO checkpoint file (.pt)
    identical to YOLOv5's best.pt / last.pt format.
    """
    YoloSizeToYaml = {
        "n": "yolov5n.yaml",
        "s": "yolov5s.yaml",
        "m": "yolov5m.yaml",
        "l": "yolov5l.yaml",
        "x": "yolov5x.yaml",
    }

    yaml_name = YoloSizeToYaml.get(yolo_size, "yolov5n.yaml")
    save_path = Path(save_path)
    
    if nc is None:
        nc = 80

    # FIX: Ensure state_dict values are proper tensors
    cleaned_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cleaned_state[k] = v.cpu()
        elif isinstance(v, np.ndarray):
            cleaned_state[k] = torch.from_numpy(v).cpu()
        else:
            cleaned_state[k] = v

    ckpt = {
        "epoch": -1,
        "best_fitness": None,
        "model": cleaned_state,  # Use cleaned state
        "optimizer": None,
        "ema": None,  # Add EMA field for compatibility
        "updates": None,
        "yaml": yaml_name,
        "nc": nc,
        "names": [str(i) for i in range(nc)],  # Add default names
    }

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"[model] YOLO checkpoint saved to: {save_path}")



def load_yolo_checkpoint_as_state_dict(ckpt_path: str) -> dict:
    """
    Load a YOLO .pt checkpoint (best.pt / last.pt) saved by YOLO train.py and
    return a state_dict suitable for ArrayRecord conversion.
    """
    # Load checkpoint normally
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    yolo_model = ckpt.get("model", None)
    if yolo_model is None:
        raise RuntimeError("No 'model' key in checkpoint")

    # If model key is already a state_dict (e.g. OrderedDict/dict), return it directly
    if isinstance(yolo_model, dict):
        return yolo_model

    # Otherwise assume it's a Model module and extract the state dict
    state_dict = yolo_model.state_dict()
    return state_dict
