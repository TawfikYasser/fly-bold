import argparse
import os
import subprocess
import sys
import sys
from pathlib import Path
import csv

import config
from fedn.utils.helpers.helpers import save_metadata

from model import compile_model, load_parameters, save_parameters
from yolo_utils import count_images_from_yaml, load_yolo_checkpoint_as_state_dict, save_state_dict_as_yolo_checkpoint
import yaml
import json


def run_yolo_train(state_dict, data_yaml: str, yolo_size: str, epochs: int, img: int, batch: int, lr: float, runs_dir: str, run_name: str, mu: float = 0.0):
    if mu > 0:
        print(f"FedProx: mu parameter received ({mu}). Standard YOLOv5 training does not support proximal term. Please implement custom training loop or use modified YOLOv5.")

    tmp_weights = Path(runs_dir) / f"{run_name}_init.pt"
    save_state_dict_as_yolo_checkpoint(state_dict, yolo_size, tmp_weights)

    # Build a temp hyp file to override lr0 for older YOLOv5 CLIs that lack --lr0
    yolo_pkg = Path(__import__("yolov5").__file__).resolve().parent
    base_hyp = yolo_pkg / "data" / "hyps" / "hyp.scratch-low.yaml"
    tmp_hyp = Path(runs_dir) / f"{run_name}_hyp.yaml"
    try:
        hyp_cfg = yaml.safe_load(base_hyp.read_text()) if base_hyp.exists() else {}
    except Exception:
        hyp_cfg = {}
    hyp_cfg["lr0"] = lr
    tmp_hyp.parent.mkdir(parents=True, exist_ok=True)
    tmp_hyp.write_text(yaml.safe_dump(hyp_cfg))

    cmd = [
        sys.executable,
        "-m",
        "yolov5.train",
        "--img",
        str(img),
        "--batch-size",
        str(batch),
        "--epochs",
        str(epochs),
        "--hyp",
        str(tmp_hyp),
        "--data",
        data_yaml,
        "--weights",
        str(tmp_weights),
        "--project",
        runs_dir,
        "--name",
        run_name,
        "--exist-ok",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", os.getcwd())
    print("Running YOLOv5 training:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("YOLOv5 training stderr tail:")
        print(proc.stderr[-2000:])
        raise RuntimeError("YOLOv5 training failed")

    weights_dir = Path(runs_dir) / run_name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    chosen = best if best.exists() else last
    if not chosen.exists():
        raise FileNotFoundError(f"No trained weights found in {weights_dir}")

    new_state = load_yolo_checkpoint_as_state_dict(chosen)
    
    # Parse results.csv for metrics
    metrics = {}
    total_loss = float("nan")
    results_csv = Path(runs_dir) / run_name / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv, "r") as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    headers = [h.strip() for h in reader[0]]
                    values = [float(v) for v in reader[-1]] # Last epoch
                    
                    # Map standard YOLOv5 keys
                    if len(headers) == len(values):
                        row_dict = dict(zip(headers, values))
                        
                        # Train Loss
                        box = row_dict.get("train/box_loss", 0)
                        obj = row_dict.get("train/obj_loss", 0)
                        cls = row_dict.get("train/cls_loss", 0)
                        total_loss = box + obj + cls
                        
                        metrics["train_loss"] = total_loss
                        metrics["train_box_loss"] = box
                        metrics["train_obj_loss"] = obj
                        metrics["train_cls_loss"] = cls
                        
                        # Train/Validation Metrics logging
                        metrics["train_mAP50"] = row_dict.get("metrics/mAP_0.5", 0)
                        metrics["train_mAP"] = row_dict.get("metrics/mAP_0.5:0.95", 0)
                        metrics["train_precision"] = row_dict.get("metrics/precision", 0)
                        metrics["train_recall"] = row_dict.get("metrics/recall", 0)
                        
        except Exception as e:
            print(f"Failed to parse results.csv: {e}")

    return new_state, str(chosen), proc.stdout[-2000:], metrics


def train(in_model_path, out_model_path, client_index: int, data_root: str, yolo_size: str, epochs: int, img: int, batch_size: int, lr: float, runs_dir: str, nc: int):
    data_yaml = Path(data_root) / f"client_{client_index}" / "coco_client.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing client data yaml: {data_yaml}")

    model = load_parameters(in_model_path, yolo_size=yolo_size, nc=nc)
    state_dict = model.state_dict()

    # Check for FedProx 'mu' parameter in the input model's metadata
    mu = 0.0
    try:
        metadata_path = in_model_path + "-metadata"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                input_metadata = json.load(f)
                # Check for 'mu' or 'proximal_mu'
                mu = input_metadata.get("mu") or input_metadata.get("proximal_mu") or 0.0
                if mu:
                    mu = float(mu)
                    print(f"FedProx: mu parameter received ({mu}).")
    except Exception as e:
        print(f"FedProx: Failed to read input metadata: {e}")

    run_name = f"client_{client_index}"
    new_state, weights_path, log_tail, training_metrics = run_yolo_train(
        state_dict,
        str(data_yaml),
        yolo_size,
        epochs,
        img,
        batch_size,
        lr,
        runs_dir,
        run_name,
        mu=mu
    )

    updated_model = compile_model(yolo_size, nc)
    updated_model.load_state_dict(new_state, strict=False)
    save_parameters(updated_model, out_model_path)

    train_images, _ = count_images_from_yaml(str(data_yaml))
    metadata = {
        "num_examples": train_images,
        "client_id": client_index,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img,
        "lr": lr,
        "weights_path": weights_path,
        "yolo_size": yolo_size,
        "log_tail": log_tail,
        "metrics": training_metrics, # Add metrics to metadata
        "mu": mu # Propagate mu to output metadata if needed for tracking
    }
    save_metadata(metadata, out_model_path)


def proximal_loss(loss, model, server_model, mu):
    """
    Reference implementation of FedProx proximal term calculation.
    
    Args:
        loss (torch.Tensor): The original loss value.
        model (torch.nn.Module): The local model being trained.
        server_model (torch.nn.Module): The global model from the previous round.
        mu (float): The proximal term coefficient.
        
    Returns:
        torch.Tensor: The modified loss with the proximal term added.
    """
    if mu <= 0:
        return loss
        
    proximal_term = 0.0
    for w, w_t in zip(model.parameters(), server_model.parameters()):
        proximal_term += (w - w_t).norm(2)**2
        
    return loss + (mu / 2) * proximal_term



def parse_args():
    parser = argparse.ArgumentParser(description="FEDn YOLOv5 train entrypoint")
    parser.add_argument("in_model", help="Input model file from FEDn")
    parser.add_argument("out_model", help="Output model update path")
    parser.add_argument("--client-id", type=int, default=config.CLIENT_INDEX)
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--yolo-size", default=config.YOLO_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--img", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--runs-dir", default=config.RUNS_DIR)
    parser.add_argument("--nc", type=int, default=config.YOLO_NC)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.in_model,
        args.out_model,
        args.client_id,
        args.data_root,
        args.yolo_size,
        args.epochs,
        args.img,
        args.batch_size,
        args.lr,
        args.runs_dir,
        args.nc,
    )