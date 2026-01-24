import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import config
from fedn.utils.helpers.helpers import save_metrics

from model import load_parameters
from yolo_utils import count_images_from_yaml, save_state_dict_as_yolo_checkpoint


def parse_yolo_results(results):
    metrics = {}
    if isinstance(results, dict):
        metrics = results.get("metrics", results)
        if metrics is None:
            metrics = {}
    elif isinstance(results, (list, tuple)) and results:
        # Check for standard tuple: (mp, mr, map50, map, box_loss, obj_loss, cls_loss)
        first = results[0] if isinstance(results[0], (list, tuple)) else results
        
        if len(first) >= 4:
            metrics = {
                "mp": float(first[0]),
                "mr": float(first[1]),
                "mAP@0.5": float(first[2]),
                "mAP": float(first[3]),
            }
            # Extract loss if available (indices 4, 5, 6 for box, obj, cls)
            # YOLOv5 returns these if validation includes loss calculation
            if len(first) >= 7:
                 box = float(first[4])
                 obj = float(first[5])
                 cls = float(first[6])
                 metrics["box_loss"] = box
                 metrics["obj_loss"] = obj
                 metrics["cls_loss"] = cls
                 metrics["loss"] = box + obj + cls

    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            cleaned[k] = float(v)
    return cleaned


def run_yolo_val(weights_pt: str, data_yaml: str, img: int, task: str = "val"):
    try:
        os.environ.setdefault("PYTHONPATH", os.getcwd())
        yval = importlib.import_module("yolov5.val")
        # Run validation
        results = yval.run(weights=weights_pt, data=data_yaml, imgsz=img, task=task, verbose=False)
        return parse_yolo_results(results)
    except Exception as exc:  # fallback to subprocess
        print(f"In-process val failed ({exc}), falling back to subprocess")
        cmd = [
            sys.executable,
            "-m",
            "yolov5.val",
            "--weights",
            weights_pt,
            "--data",
            data_yaml,
            "--img",
            str(img),
            "--task",
            task,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        print(proc.stdout[-1000:])
        if proc.returncode != 0:
            print(proc.stderr[-1000:])
            raise RuntimeError("YOLO validation failed")
        return {}

def validate(in_model_path: str, out_json_path: str):
    """Validate a YOLOv5 model checkpoint with client data and emit metrics JSON."""

    client_id = config.CLIENT_INDEX
    data_root = config.DATA_ROOT
    yolo_size = config.YOLO_SIZE
    img = config.IMG_SIZE
    nc = config.YOLO_NC

    data_yaml = Path(data_root) / f"client_{client_id}" / "coco_client.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing client data yaml at {data_yaml}. Ensure the packaged dataset is available."
        )

    model = load_parameters(in_model_path, yolo_size=yolo_size, nc=nc)
    tmp_weights = Path(out_json_path).with_suffix(".pt")
    save_state_dict_as_yolo_checkpoint(model.state_dict(), yolo_size, tmp_weights, nc)

    save_state_dict_as_yolo_checkpoint(model.state_dict(), yolo_size, tmp_weights, nc)

    # 1. Evaluate on Validation Set
    val_metrics = run_yolo_val(str(tmp_weights), str(data_yaml), img, task="val")
    
    # Combine metrics
    final_report = {}
    
    # Add Eval metrics (standard keys like mAP, loss)
    for k, v in val_metrics.items():
        final_report[k] = v  # e.g. "loss", "mAP"

    train_images, val_images = count_images_from_yaml(str(data_yaml))
    final_report.setdefault("num_train_examples", train_images)
    final_report.setdefault("num_val_examples", val_images)
    final_report.setdefault("client_id", client_id)

    save_metrics(final_report, out_json_path)


def parse_args():
    parser = argparse.ArgumentParser(description="FEDn YOLOv5 validate entrypoint")
    parser.add_argument("in_model", help="Input model weights")
    parser.add_argument("out_json", help="Output metrics path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.in_model, args.out_json)