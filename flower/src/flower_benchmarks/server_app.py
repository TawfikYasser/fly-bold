"""flower-benchmarks: A Flower / PyTorch app."""

import shutil
import torch
import os
import sys
import json
from typing import List, Tuple, Dict, Optional
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flower_benchmarks.task import Net

# Ensure the parent directory is in sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from flower_benchmarks.plugins.yolov5.model import load_yolo_checkpoint_as_state_dict, save_state_dict_as_yolo_checkpoint, YoloSizeToPretrained
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download

# REMOVED: All Prometheus imports and initialization (lines 14-31)

# Global variable to store round logs
ALL_ROUND_LOGS = []
CURRENT_ROUND = 0


def get_config(key: str, context: Context, default=None, type_converter=str):
    """Get configuration with precedence: env var > run_config > node_config > default"""
    env_key = key.upper().replace("-", "_")
    if env_key in os.environ:
        value = os.environ[env_key]
        try:
            return type_converter(value)
        except:
            return value
    
    if key in context.run_config:
        return context.run_config[key]
    
    if key in context.node_config:
        return context.node_config[key]
    
    return default

# Create ServerApp
app = ServerApp()


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def custom_train_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Collect per-client training metrics for this round and aggregate them."""
    global ALL_ROUND_LOGS, CURRENT_ROUND

    if not record_dicts:
        return MetricRecord({})

    clients_logs = []
    total_data_server_to_clients = 0.0
    total_data_clients_to_server = 0.0
    total_train_time = 0.0
    total_train_loss = 0.0
    total_examples = 0.0
    total_mr = 0.0
    total_mp = 0.0
    total_mAP50 = 0.0
    total_mAP = 0.0
    
    lr = 0.01

    for record_dict in record_dicts:
        if "metrics" not in record_dict:
            continue
        
        metrics = record_dict["metrics"]
        client_id = _safe_float(metrics.get("client_id", 0))
        num_examples = _safe_float(metrics.get("num-examples", 1.0))
        
        client_train_time = _safe_float(metrics.get("client_train_time", 0.0))
        client_train_loss = _safe_float(metrics.get("client_train_loss", 0.0))
        client_train_acc_mr = _safe_float(metrics.get("client_train_acc_mr", 0.0))
        client_train_acc_mp = _safe_float(metrics.get("client_train_acc_mp", 0.0))
        client_train_acc_mAP50 = _safe_float(metrics.get("client_train_acc_mAP@0.5", 0.0))
        client_train_acc_mAP = _safe_float(metrics.get("client_train_acc_mAP", 0.0))
        lr = metrics.get("lr", 0.01)
        
        data_received = _safe_float(metrics.get("data_received_from_server", 0.0))
        data_sent = _safe_float(metrics.get("data_sent_to_server", 0.0))
        total_data_server_to_clients += data_received
        total_data_clients_to_server += data_sent
        
        total_train_loss += client_train_loss * num_examples
        total_mr += client_train_acc_mr * num_examples
        total_mp += client_train_acc_mp * num_examples
        total_mAP50 += client_train_acc_mAP50 * num_examples
        total_mAP += client_train_acc_mAP * num_examples
        total_train_time += client_train_time
        total_examples += num_examples
        
        client_log = {
            "client_id": int(client_id),
            "client_train_acc": {
                "mr": client_train_acc_mr,
                "mp": client_train_acc_mp,
                "mAP@0.5": client_train_acc_mAP50,
                "mAP": client_train_acc_mAP,
                "aggregated": (client_train_acc_mr + client_train_acc_mp + client_train_acc_mAP50 + client_train_acc_mAP) / 4.0
            },
            "client_train_loss": client_train_loss,
            "client_train_time": client_train_time,
            "client_train_num_examples": int(num_examples)
        }
        clients_logs.append(client_log)

    round_train_loss = total_train_loss / total_examples if total_examples > 0 else 0.0
    round_train_acc_mr = total_mr / total_examples if total_examples > 0 else 0.0
    round_train_acc_mp = total_mp / total_examples if total_examples > 0 else 0.0
    round_train_acc_mAP50 = total_mAP50 / total_examples if total_examples > 0 else 0.0
    round_train_acc_mAP = total_mAP / total_examples if total_examples > 0 else 0.0
    
    round_train_acc_aggregated = (round_train_acc_mr + round_train_acc_mp + round_train_acc_mAP50 + round_train_acc_mAP) / 4.0

    total_round_data = total_data_server_to_clients + total_data_clients_to_server
    total_round_data_mb = round(total_round_data / (1024 ** 2), 4)

    CURRENT_ROUND = len(ALL_ROUND_LOGS)

    ALL_ROUND_LOGS.append({
        "round_id": CURRENT_ROUND,
        "round_duration": total_train_time,
        "training_num_examples": int(total_examples),
        "round_train_loss": round_train_loss,
        "lr": lr,
        "clients_logs": clients_logs,
        "round_training_acc": {
            "mr": round_train_acc_mr,
            "mp": round_train_acc_mp,
            "mAP@0.5": round_train_acc_mAP50,
            "mAP": round_train_acc_mAP,
            "aggregated": round_train_acc_aggregated
        }
    })

    # REMOVED: Prometheus metric setting (lines 158-161)

    return MetricRecord({})


def custom_eval_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Aggregate client evaluation metrics and append to current round log."""
    global ALL_ROUND_LOGS

    total_eval_loss = 0.0
    total_eval_mr = 0.0
    total_eval_mp = 0.0
    total_eval_mAP50 = 0.0
    total_eval_mAP = 0.0
    total_examples = 0.0
    total_eval_time = 0.0
    round_eval_acc_aggregated = 0.0

    if ALL_ROUND_LOGS:
        current_round = ALL_ROUND_LOGS[-1]
        clients_logs = current_round.get("clients_logs", [])
        client_logs_map = {int(cl["client_id"]): cl for cl in clients_logs}

        for record_dict in record_dicts:
            if "metrics" not in record_dict:
                continue
            
            metrics = record_dict["metrics"]
            client_id = int(_safe_float(metrics.get("client_id", 0)))
            num_examples = _safe_float(metrics.get("num-examples", 1.0))
            
            client_eval_loss = _safe_float(metrics.get("client_eval_loss", 0.0))
            client_eval_acc_mr = _safe_float(metrics.get("client_eval_acc_mr", 0.0))
            client_eval_acc_mp = _safe_float(metrics.get("client_eval_acc_mp", 0.0))
            client_eval_acc_mAP50 = _safe_float(metrics.get("client_eval_acc_mAP@0.5", 0.0))
            client_eval_acc_mAP = _safe_float(metrics.get("client_eval_acc_mAP", 0.0))
            client_eval_time = _safe_float(metrics.get("client_eval_time", 0.0))
            
            total_eval_loss += client_eval_loss * num_examples
            total_eval_mr += client_eval_acc_mr * num_examples
            total_eval_mp += client_eval_acc_mp * num_examples
            total_eval_mAP50 += client_eval_acc_mAP50 * num_examples
            total_eval_mAP += client_eval_acc_mAP * num_examples
            total_eval_time += client_eval_time
            total_examples += num_examples
            
            if client_id in client_logs_map:
                client_logs_map[client_id]["client_eval_acc"] = {
                    "mr": client_eval_acc_mr,
                    "mp": client_eval_acc_mp,
                    "mAP@0.5": client_eval_acc_mAP50,
                    "mAP": client_eval_acc_mAP,
                    "aggregated": (client_eval_acc_mr + client_eval_acc_mp + client_eval_acc_mAP50 + client_eval_acc_mAP) / 4.0
                }
                client_logs_map[client_id]["client_eval_loss"] = client_eval_loss
                client_logs_map[client_id]["client_eval_time"] = client_eval_time
                client_logs_map[client_id]["client_eval_num_example"] = int(num_examples)
        
        round_eval_loss = total_eval_loss / total_examples if total_examples > 0 else 0.0
        round_eval_acc_mr = total_eval_mr / total_examples if total_examples > 0 else 0.0
        round_eval_acc_mp = total_eval_mp / total_examples if total_examples > 0 else 0.0
        round_eval_acc_mAP50 = total_eval_mAP50 / total_examples if total_examples > 0 else 0.0
        round_eval_acc_mAP = total_eval_mAP / total_examples if total_examples > 0 else 0.0
        
        round_eval_acc_aggregated = (round_eval_acc_mr + round_eval_acc_mp + round_eval_acc_mAP50 + round_eval_acc_mAP) / 4.0
        
        current_round["round_eval_loss"] = round_eval_loss
        current_round["round_eval_acc"] = {
            "mr": round_eval_acc_mr,
            "mp": round_eval_acc_mp,
            "mAP@0.5": round_eval_acc_mAP50,
            "mAP": round_eval_acc_mAP,
            "aggregated": round_eval_acc_aggregated
        }

    # REMOVED: Prometheus metric setting (lines 237-239)

    return MetricRecord({"round_eval_acc": round_eval_acc_aggregated})


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for ServerApp."""

    fraction_train = get_config("fraction-train", context, default=1.0, type_converter=float)
    fraction_evaluate = get_config("fraction-evaluate", context, default=1.0, type_converter=float)
    num_rounds = get_config("num-server-rounds", context, default=5, type_converter=int)
    lr = get_config("lr", context, default=0.01, type_converter=float)
    task_type = get_config("task", context, default="classification")

    if task_type == "detection":
        yolo_size = get_config("yolo_size", context, default="n")
        experiment_name = get_config("experiment_name", context, default=f"EXP_YOLOv5_{yolo_size}_detection")
    else:
        experiment_name = get_config("experiment_name", context, default="EXP_CNN_fashion_mnist_dataset")

    run_id = get_config("run_id", context, default="1")

    # Load global model
    if task_type == "detection":
        yolo_size = get_config("yolo_size", context, default="n")
        weight_name = YoloSizeToPretrained.get(yolo_size, "yolov5n.pt")

        candidate_paths = [
            os.path.join(os.getcwd(), "yolov5", weight_name),
            os.path.join(os.getcwd(), weight_name),
            weight_name,
        ]
        print("Looking for YOLO weights in candidate paths:", candidate_paths)

        weight_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                weight_path = p
                print(f"Found YOLO weights at: {weight_path}")
                break

        if weight_path is None:
            try:
                attempt_download(weight_name)
                weight_path = weight_name
                print(f"Downloaded YOLO weights to: {weight_path}")
            except Exception:
                print("Failed to download YOLO weights. Using empty arrays.")
                weight_path = None

        if weight_path:
            try:
                print(f"Loading YOLO weights from: {weight_path}")
                state_dict = load_yolo_checkpoint_as_state_dict(weight_path)
                arrays = ArrayRecord(state_dict)
                print(f"YOLO initial arrays loaded successfully with {len(arrays)} layers.")
            except Exception as e:
                print(f"Failed to load YOLO weights: {e}")
                arrays = ArrayRecord({})
        else:
            arrays = ArrayRecord({})
    else:
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

    # FIXED: Removed initial_parameters argument (not supported in Flower 1.22.0+)
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=custom_train_metrics_aggregation,
        evaluate_metrics_aggr_fn=custom_eval_metrics_aggregation,
    )

    train_cfg = {"lr": lr, "num_rounds": num_rounds}
    try:
        if task_type == "detection":
            train_cfg["yolo_size"] = yolo_size
    except Exception:
        pass

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        num_rounds=num_rounds,
    )

    # Save final model
    state_dict = result.arrays.to_torch_state_dict()

    if task_type == "detection":
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        yolo_size = context.run_config.get("yolo_size", "n")
        try:
            save_state_dict_as_yolo_checkpoint(state_dict, yolo_size, out_path)
        except Exception:
            torch.save({"model": state_dict}, out_path)
    else:
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        torch.save(state_dict, out_path)

    # Save round logs
    logs_path = f"{experiment_name}_{run_id}_logs.json"
    try:
        with open(logs_path, "w") as f:
            json.dump(ALL_ROUND_LOGS, f, indent=2)
    except Exception:
        pass

    print(f"Run completed. Final model saved to {out_path}, logs saved to {logs_path}.")