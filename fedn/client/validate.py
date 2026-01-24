import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import config
from data import resolve_client_yaml
from fedn.utils.helpers.helpers import save_metrics

from model import load_parameters
from yolo_utils import count_images_from_yaml, save_state_dict_as_yolo_checkpoint


def _find_latest_results_csv(val_root: Path) -> Path | None:
    if not val_root.exists():
        return None
    candidates = sorted(val_root.glob("exp*/results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def parse_yolo_eval_results_csv(results_csv: Path) -> dict:
    """Parse YOLOv5 validation metrics from results.csv and compute total loss."""
    import csv

    if not results_csv or not results_csv.exists():
        return {}

    try:
        lines = results_csv.read_text().splitlines()
        data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
        if len(data_lines) < 2:
            return {}
        reader = csv.DictReader(data_lines)
        rows = list(reader)
        if not rows:
            return {}

        last_row = {k.strip(): v.strip() for k, v in rows[-1].items()}

        box_loss = float(last_row.get('val/box_loss', last_row.get('box_loss', 0.0)))
        obj_loss = float(last_row.get('val/obj_loss', last_row.get('obj_loss', 0.0)))
        cls_loss = float(last_row.get('val/cls_loss', last_row.get('cls_loss', 0.0)))
        total_loss = box_loss + obj_loss + cls_loss

        metrics = {
            'loss': total_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
        }

        # Optional metrics if present in results.csv
        if 'metrics/precision' in last_row:
            metrics['mp'] = float(last_row.get('metrics/precision', 0.0))
        if 'metrics/recall' in last_row:
            metrics['mr'] = float(last_row.get('metrics/recall', 0.0))
        if 'metrics/mAP_0.5' in last_row:
            metrics['mAP@0.5'] = float(last_row.get('metrics/mAP_0.5', 0.0))
        if 'metrics/mAP_0.5:0.95' in last_row:
            metrics['mAP'] = float(last_row.get('metrics/mAP_0.5:0.95', 0.0))

        return metrics
    except Exception:
        return {}


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
    def _run_with_compute_loss():
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.dataloaders import create_dataloader
        from yolov5.utils.general import check_dataset, check_img_size, colorstr
        from yolov5.utils.torch_utils import select_device
        from yolov5.utils.loss import ComputeLoss
        import yaml
        import yolov5

        data = check_dataset(data_yaml)
        device = select_device("")
        model = DetectMultiBackend(weights_pt, device=device, data=data)
        if not getattr(model, "model", None):
            raise RuntimeError("ComputeLoss requires a PyTorch model")

        # Ensure model has hyp set (required for ComputeLoss)
        hyp_path = Path(yolov5.__file__).resolve().parent / "data" / "hyps" / "hyp.scratch-low.yaml"
        hyp = {}
        if hyp_path.exists():
            try:
                hyp = yaml.safe_load(hyp_path.read_text()) or {}
            except Exception:
                hyp = {}

        if not getattr(model.model, "hyp", None):
            model.model.hyp = hyp

        stride = model.stride
        imgsz = check_img_size(img, s=stride)
        batch_size = 16
        pad, rect = (0.5, model.pt)

        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls=False,
            pad=pad,
            rect=rect,
            workers=8,
            prefix=colorstr(f"{task}: "),
        )[0]

        compute_loss = ComputeLoss(model.model)
        yval = importlib.import_module("yolov5.val")
        results = yval.run(
            data=data,
            model=model,
            dataloader=dataloader,
            imgsz=imgsz,
            batch_size=batch_size,
            task=task,
            verbose=False,
            plots=False,
            save_json=False,
            save_txt=False,
            save_conf=False,
            compute_loss=compute_loss,
        )
        return parse_yolo_results(results)

    try:
        os.environ.setdefault("PYTHONPATH", os.getcwd())
        metrics = _run_with_compute_loss()
        if metrics:
            return metrics
    except Exception as exc:
        print(f"Compute-loss val failed ({exc}), falling back to standard val")

    try:
        os.environ.setdefault("PYTHONPATH", os.getcwd())
        yval = importlib.import_module("yolov5.val")
        # Run validation
        results = yval.run(weights=weights_pt, data=data_yaml, imgsz=img, task=task, verbose=False)

        metrics = parse_yolo_results(results)

        # Parse results.csv for validation loss and metrics if available
        save_dir = None
        if isinstance(results, dict):
            save_dir = results.get("save_dir") or results.get("save_dir")
        if save_dir:
            results_csv = Path(save_dir) / "results.csv"
        else:
            results_csv = _find_latest_results_csv(Path(os.getcwd()) / "runs" / "val")

        csv_metrics = parse_yolo_eval_results_csv(results_csv) if results_csv else {}
        metrics.update(csv_metrics)

        return metrics
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
        # Parse latest results.csv after subprocess run
        results_csv = _find_latest_results_csv(Path(os.getcwd()) / "runs" / "val")
        return parse_yolo_eval_results_csv(results_csv) if results_csv else {}

def validate(in_model_path: str, out_json_path: str):
    """Validate a YOLOv5 model checkpoint with client data and emit metrics JSON."""

    client_id = config.CLIENT_INDEX
    data_root = config.DATA_ROOT
    yolo_size = config.YOLO_SIZE
    img = config.IMG_SIZE
    nc = config.YOLO_NC

    data_yaml = resolve_client_yaml(data_root, client_id)

    model = load_parameters(in_model_path, yolo_size=yolo_size, nc=nc)
    tmp_weights = Path(out_json_path).with_suffix(".pt")
    save_state_dict_as_yolo_checkpoint(model.state_dict(), yolo_size, tmp_weights, nc)

    save_state_dict_as_yolo_checkpoint(model.state_dict(), yolo_size, tmp_weights, nc)

    # 1. Evaluate on Validation Set
    eval_start_time = time.perf_counter()
    val_metrics = run_yolo_val(str(tmp_weights), str(data_yaml), img, task="val")
    eval_time = time.perf_counter() - eval_start_time
    
    # Combine metrics
    final_report = {}
    
    # Add Eval metrics (standard keys like mAP, loss)
    for k, v in val_metrics.items():
        final_report[k] = v  # e.g. "loss", "mAP"

    train_images, val_images = count_images_from_yaml(str(data_yaml))
    final_report.setdefault("num_train_examples", train_images)
    final_report.setdefault("num_val_examples", val_images)
    final_report.setdefault("client_id", client_id)

    # Flower-style evaluation fields
    final_report["client_eval_time"] = float(eval_time)
    final_report["client_eval_loss"] = float(val_metrics.get("loss", 0.0))
    final_report["client_eval_acc_mr"] = float(val_metrics.get("mr", 0.0))
    final_report["client_eval_acc_mp"] = float(val_metrics.get("mp", 0.0))
    final_report["client_eval_acc_mAP@0.5"] = float(val_metrics.get("mAP@0.5", 0.0))
    final_report["client_eval_acc_mAP"] = float(val_metrics.get("mAP", 0.0))
    final_report["num-examples"] = float(val_images)

    save_metrics(final_report, out_json_path)


def parse_args():
    parser = argparse.ArgumentParser(description="FEDn YOLOv5 validate entrypoint")
    parser.add_argument("in_model", help="Input model weights")
    parser.add_argument("out_json", help="Output metrics path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate(args.in_model, args.out_json)