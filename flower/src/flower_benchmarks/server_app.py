"""flower-benchmarks: Optimized Flower Server App with efficient aggregation."""

import shutil
import torch
import os
import sys
import json
from typing import List, Tuple, Dict, Optional
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flower_benchmarks.task import Net

# Ensure parent directory is in sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from flower_benchmarks.plugins.yolov5.model import (
    load_yolo_checkpoint_as_state_dict, 
    save_state_dict_as_yolo_checkpoint, 
    YoloSizeToPretrained
)
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download

# =====================================================================
# GLOBAL STATE (needed for Flower's aggregation callbacks)
# =====================================================================
ALL_ROUND_LOGS = []
CURRENT_ROUND = 0


def get_config(key: str, context: Context, default=None, type_converter=str):
    """Get configuration with precedence: env var > run_config > node_config > default."""
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
    """Safely convert to float."""
    try:
        return float(v)
    except Exception:
        return default


def custom_train_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """
    OPTIMIZED: Single-pass aggregation with efficient data extraction.
    âœ… FIXED: Returns properly structured MetricRecord instead of empty dict.
    """
    global ALL_ROUND_LOGS, CURRENT_ROUND

    if not record_dicts:
        print("[SERVER] No training results to aggregate")
        return MetricRecord({})

    print(f"[SERVER] Aggregating training metrics from {len(record_dicts)} clients")

    # OPTIMIZED: Extract all client data in single pass
    clients_data = []
    for i, record_dict in enumerate(record_dicts):
        if "metrics" not in record_dict:
            print(f"[SERVER] Warning: record_dict {i} has no metrics")
            continue
        
        metrics = record_dict["metrics"]
        
        # Debug: print first client's keys
        if i == 0:
            print(f"[SERVER] First client metrics keys: {sorted(metrics.keys())}")

        # Defensive defaults (server never trusts clients)
        num_examples = max(1, int(_safe_float(metrics.get("num-examples", 1))))
        
        # Extract all metrics at once with safe defaults
        client_data = {
            'id': int(_safe_float(metrics.get("client_id", 0))),
            'examples': num_examples,
            'train_time': _safe_float(metrics.get("client_train_time", 0.0)),
            'loss': _safe_float(metrics.get("client_train_loss", 0.0)),
            'mr': _safe_float(metrics.get("client_train_acc_mr", 0.0)),
            'mp': _safe_float(metrics.get("client_train_acc_mp", 0.0)),
            'mAP50': _safe_float(metrics.get("client_train_acc_mAP@0.5", 0.0)),
            'mAP': _safe_float(metrics.get("client_train_acc_mAP", 0.0)),
            'lr': _safe_float(metrics.get("lr", 0.01)),
            'data_received': _safe_float(metrics.get("data_received_from_server", 0.0)),
            'data_sent': _safe_float(metrics.get("data_sent_to_server", 0.0)),
            'round_duration': _safe_float(metrics.get("round_duration", 0.0)),
        }
        clients_data.append(client_data)
    
    if not clients_data:
        print("[SERVER] No valid client data extracted")
        return MetricRecord({})
    
    # OPTIMIZED: Vectorized aggregation
    total_examples = sum(c['examples'] for c in clients_data)
    
    # Weighted averages
    round_train_loss = sum(c['loss'] * c['examples'] for c in clients_data) / total_examples if total_examples > 0 else 0.0
    round_train_acc_mr = sum(c['mr'] * c['examples'] for c in clients_data) / total_examples if total_examples > 0 else 0.0
    round_train_acc_mp = sum(c['mp'] * c['examples'] for c in clients_data) / total_examples if total_examples > 0 else 0.0
    round_train_acc_mAP50 = sum(c['mAP50'] * c['examples'] for c in clients_data) / total_examples if total_examples > 0 else 0.0
    round_train_acc_mAP = sum(c['mAP'] * c['examples'] for c in clients_data) / total_examples if total_examples > 0 else 0.0
    
    # Aggregate metrics
    round_train_acc_aggregated = (round_train_acc_mr + round_train_acc_mp + 
                                   round_train_acc_mAP50 + round_train_acc_mAP) / 4.0
    max_train_time = max(c['train_time'] for c in clients_data) if clients_data else 0.0
    max_round_duration = max(c['round_duration'] for c in clients_data) if clients_data else 0.0
    total_data_transferred = sum(c['data_received'] + c['data_sent'] for c in clients_data)
    total_data_mb = round(total_data_transferred / (1024 ** 2), 4)
    
    # Get learning rate (should be same for all clients)
    lr = clients_data[0]['lr'] if clients_data else 0.01
    
    # Build client logs for output
    clients_logs = []
    for c in clients_data:
        client_log = {
            "client_id": c['id'],
            "client_train_acc": {
                "mr": c['mr'],
                "mp": c['mp'],
                "mAP@0.5": c['mAP50'],
                "mAP": c['mAP'],
                "aggregated": (c['mr'] + c['mp'] + c['mAP50'] + c['mAP']) / 4.0
            },
            "client_train_loss": c['loss'],
            "client_train_time": c['train_time'],
            "client_train_num_examples": int(c['examples'])
        }
        clients_logs.append(client_log)
    
    # Update global round logs
    CURRENT_ROUND = len(ALL_ROUND_LOGS)
    
    round_log = {
        "round_id": CURRENT_ROUND,
        "round_duration": max_round_duration,
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
        },
        "round_data_transferred_mb": total_data_mb,
        "round_data_transferred_bytes": int(total_data_transferred)
    }
    
    ALL_ROUND_LOGS.append(round_log)
    
    print(f"\n{'='*70}")
    print(f"ROUND {CURRENT_ROUND+1} TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Participating Clients: {len(clients_data)}")
    print(f"Training Loss:     {round_train_loss:.4f}")
    print(f"Training mAP@0.5:  {round_train_acc_mAP50:.4f}")
    print(f"Training mAP:      {round_train_acc_mAP:.4f}")
    print(f"Aggregated Score:  {round_train_acc_aggregated:.4f}")
    print(f"Round Duration:    {max_round_duration:.2f}s")
    print(f"Data Transferred:  {total_data_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    # FIXED: Return aggregated metrics for Flower (not empty dict)
    return MetricRecord({
        "train_loss": round_train_loss,
        "train_accuracy": round_train_acc_aggregated,
        "train_mAP": round_train_acc_mAP,
    })

def custom_eval_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """
    OPTIMIZED: Single-pass evaluation aggregation with efficient data extraction.
    âœ… FIXED: Returns properly structured MetricRecord.
    """
    global ALL_ROUND_LOGS

    if not record_dicts:
        print("[SERVER] No evaluation results to aggregate")
        return MetricRecord({})
    
    if not ALL_ROUND_LOGS:
        print("[SERVER] Warning: No round logs available yet")
        return MetricRecord({})
    
    print(f"[SERVER] Aggregating evaluation metrics from {len(record_dicts)} clients")
    
    # OPTIMIZED: Extract all evaluation data in single pass
    eval_data = []
    for i, record_dict in enumerate(record_dicts):
        if "metrics" not in record_dict:
            print(f"[SERVER] Warning: eval record_dict {i} has no metrics")
            continue
        
        metrics = record_dict["metrics"]
        
        # Debug: print first client's keys
        if i == 0:
            print(f"[SERVER] First client eval keys: {sorted(metrics.keys())}")
        
        client_eval = {
            'id': int(_safe_float(metrics.get("client_id", 0))),
            'examples': _safe_float(metrics.get("num-examples", 1.0)),
            'loss': _safe_float(metrics.get("client_eval_loss", 0.0)),
            'mr': _safe_float(metrics.get("client_eval_acc_mr", 0.0)),
            'mp': _safe_float(metrics.get("client_eval_acc_mp", 0.0)),
            'mAP50': _safe_float(metrics.get("client_eval_acc_mAP@0.5", 0.0)),
            'mAP': _safe_float(metrics.get("client_eval_acc_mAP", 0.0)),
            'eval_time': _safe_float(metrics.get("client_eval_time", 0.0)),
        }
        eval_data.append(client_eval)
    
    if not eval_data:
        print("[SERVER] No valid evaluation data extracted")
        return MetricRecord({})
    
    # OPTIMIZED: Vectorized aggregation
    total_examples = sum(c['examples'] for c in eval_data)
    
    round_eval_loss = sum(c['loss'] * c['examples'] for c in eval_data) / total_examples
    round_eval_acc_mr = sum(c['mr'] * c['examples'] for c in eval_data) / total_examples
    round_eval_acc_mp = sum(c['mp'] * c['examples'] for c in eval_data) / total_examples
    round_eval_acc_mAP50 = sum(c['mAP50'] * c['examples'] for c in eval_data) / total_examples
    round_eval_acc_mAP = sum(c['mAP'] * c['examples'] for c in eval_data) / total_examples
    
    round_eval_acc_aggregated = (round_eval_acc_mr + round_eval_acc_mp + 
                                  round_eval_acc_mAP50 + round_eval_acc_mAP) / 4.0
    max_eval_time = max(c['eval_time'] for c in eval_data)
    
    # Update current round with evaluation metrics
    current_round = ALL_ROUND_LOGS[-1]
    current_round["round_eval_time"] = max_eval_time
    current_round["round_eval_loss"] = round_eval_loss
    current_round["round_eval_acc"] = {
        "mr": round_eval_acc_mr,
        "mp": round_eval_acc_mp,
        "mAP@0.5": round_eval_acc_mAP50,
        "mAP": round_eval_acc_mAP,
        "aggregated": round_eval_acc_aggregated
    }
    
    # Add evaluation data to client logs
    clients_logs = current_round.get("clients_logs", [])
    client_logs_map = {cl["client_id"]: cl for cl in clients_logs}
    
    for c in eval_data:
        if c['id'] in client_logs_map:
            client_logs_map[c['id']]["client_eval_acc"] = {
                "mr": c['mr'],
                "mp": c['mp'],
                "mAP@0.5": c['mAP50'],
                "mAP": c['mAP'],
                "aggregated": (c['mr'] + c['mp'] + c['mAP50'] + c['mAP']) / 4.0
            }
            client_logs_map[c['id']]["client_eval_loss"] = c['loss']
            client_logs_map[c['id']]["client_eval_time"] = c['eval_time']
            client_logs_map[c['id']]["client_eval_num_examples"] = int(c['examples'])
    
    print(f"\n{'='*70}")
    print(f"ROUND {CURRENT_ROUND+1} EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Participating Clients: {len(eval_data)}")
    print(f"Validation Loss:   {round_eval_loss:.4f}")
    print(f"Validation mAP@0.5: {round_eval_acc_mAP50:.4f}")
    print(f"Validation mAP:    {round_eval_acc_mAP:.4f}")
    print(f"Aggregated Score:  {round_eval_acc_aggregated:.4f}")
    print(f"Eval Duration:     {max_eval_time:.2f}s")
    print(f"{'='*70}\n")
    
    # âœ… FIXED: Return aggregated metrics for Flower
    return MetricRecord({
        "eval_loss": round_eval_loss,
        "eval_accuracy": round_eval_acc_aggregated,
        "eval_mAP": round_eval_acc_mAP,
    })


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for ServerApp with optimized configuration."""

    # Get configuration
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

    print(f"\n{'='*70}")
    print(f"FEDERATED LEARNING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Experiment:     {experiment_name}")
    print(f"Run ID:         {run_id}")
    print(f"Task:           {task_type}")
    print(f"Rounds:         {num_rounds}")
    print(f"Fraction Train: {fraction_train}")
    print(f"Fraction Eval:  {fraction_evaluate}")
    print(f"Learning Rate:  {lr}")
    if task_type == "detection":
        print(f"YOLO Size:      {yolo_size}")
    print(f"{'='*70}\n")

    # Load global model
    if task_type == "detection":
        yolo_size = get_config("yolo_size", context, default="n")
        weight_name = YoloSizeToPretrained.get(yolo_size, "yolov5n.pt")

        candidate_paths = [
            os.path.join(os.getcwd(), "yolov5", weight_name),
            os.path.join(os.getcwd(), weight_name),
            weight_name,
        ]
        
        print(f"Loading YOLO weights: {weight_name}")
        
        weight_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                weight_path = p
                print(f"âœ… Found YOLO weights at: {weight_path}")
                break

        if weight_path is None:
            try:
                print(f"Downloading YOLO weights: {weight_name}")
                attempt_download(weight_name)
                weight_path = weight_name
                print(f"âœ… Downloaded to: {weight_path}")
            except Exception as e:
                print(f"âš ï¸  Failed to download YOLO weights: {e}")
                print("Using empty arrays.")
                weight_path = None

        if weight_path:
            try:
                state_dict = load_yolo_checkpoint_as_state_dict(weight_path)
                arrays = ArrayRecord(state_dict)
                print(f"âœ… YOLO initial arrays loaded: {len(arrays)} layers")
            except Exception as e:
                print(f"âŒ Failed to load YOLO weights: {e}")
                arrays = ArrayRecord({})
        else:
            arrays = ArrayRecord({})
    else:
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())
        print(f"âœ… Classification model initialized")

    # Create strategy with optimized aggregation
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=custom_train_metrics_aggregation,
        evaluate_metrics_aggr_fn=custom_eval_metrics_aggregation,
    )

    # Training configuration
    train_cfg = {"lr": lr, "num_rounds": num_rounds}
    if task_type == "detection":
        train_cfg["yolo_size"] = yolo_size

    print(f"\n{'='*70}")
    print(f"STARTING FEDERATED LEARNING")
    print(f"{'='*70}\n")

    # Start training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(train_cfg),
        num_rounds=num_rounds,
    )

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}\n")

    # Save final model
    state_dict = result.arrays.to_torch_state_dict()

    if task_type == "detection":
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        yolo_size = context.run_config.get("yolo_size", "n")
        try:
            save_state_dict_as_yolo_checkpoint(state_dict, yolo_size, out_path)
            print(f"âœ… Final YOLO model saved: {out_path}")
        except Exception as e:
            print(f"âš ï¸  Could not save YOLO checkpoint: {e}")
            torch.save({"model": state_dict}, out_path)
            print(f"âœ… Saved as PyTorch state dict: {out_path}")
    else:
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        torch.save(state_dict, out_path)
        print(f"âœ… Final model saved: {out_path}")

    # Save round logs
    logs_path = f"{experiment_name}_{run_id}_logs.json"
    try:
        with open(logs_path, "w") as f:
            json.dump(ALL_ROUND_LOGS, f, indent=2)
        print(f"âœ… Training logs saved: {logs_path}")
    except Exception as e:
        print(f"âš ï¸  Could not save logs: {e}")

    # Print summary statistics
    if ALL_ROUND_LOGS:
        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Rounds:      {len(ALL_ROUND_LOGS)}")
        
        final_round = ALL_ROUND_LOGS[-1]
        print(f"\nFinal Round Metrics:")
        print(f"  Training Loss:   {final_round.get('round_train_loss', 0):.4f}")
        print(f"  Training mAP:    {final_round.get('round_training_acc', {}).get('mAP', 0):.4f}")
        
        if 'round_eval_acc' in final_round:
            print(f"  Validation Loss: {final_round.get('round_eval_loss', 0):.4f}")
            print(f"  Validation mAP:  {final_round.get('round_eval_acc', {}).get('mAP', 0):.4f}")
        
        total_time = sum(r.get('round_duration', 0) for r in ALL_ROUND_LOGS)
        total_data_mb = sum(r.get('round_data_transferred_mb', 0) for r in ALL_ROUND_LOGS)
        
        print(f"\nTotal Training Time: {total_time/60:.2f} minutes")
        print(f"Total Data Transfer: {total_data_mb:.2f} MB")
        print(f"{'='*70}\n")

    print(f"Run completed successfully!")
    print(f"Final model: {out_path}")
    print(f"Training logs: {logs_path}")