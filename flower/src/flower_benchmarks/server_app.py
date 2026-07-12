"""flower-benchmarks: Optimized Flower Server App with efficient aggregation."""

import shutil
import torch
import os
import sys
import json
import time
from typing import List, Tuple, Dict, Optional
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam, FedYogi, FedProx
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
# from flower_benchmarks.task import ResourceMonitor  # COMMENTED: Resource monitoring disabled
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download
import numpy as np

# HPO Backend Selection: FLAML or Optuna
USE_FLAML = os.getenv("USE_FLAML", "false").lower() in ("true", "1")

if not USE_FLAML:
    import optuna
    from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
else:
    from flower_benchmarks.flaml_hpo import (
        create_flaml_study, lr_only_warm_start, TrialPruneSignal as FLAMLTrialPruneSignal
    )

# =====================================================================
# GLOBAL STATE (needed for Flower's aggregation callbacks)
# =====================================================================
ALL_ROUND_LOGS = []
CURRENT_ROUND = 0

# FIX 4: Global trial handle so the eval callback can report intermediate
# mAP values to Optuna and trigger early pruning without restructuring
# strategy.start() into a round-by-round loop.
CURRENT_OPTUNA_TRIAL = None


class _TrialPruneSignal(Exception):
    """Raised when a trial should be pruned (early stopped).
    For Optuna: caught by _run_trial() and re-raised as optuna.exceptions.TrialPruned
    For FLAML: caught by _run_trial() and re-raised as FLAMLTrialPruneSignal
    """
    pass


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
    OPTIMIZED: Single-pass aggregation with efficient data extraction and resource monitoring.
    âœ… FIXED: Returns properly structured MetricRecord instead of empty dict.
    """
    global ALL_ROUND_LOGS, CURRENT_ROUND
    
    # ✅ START SERVER AGGREGATION MONITORING
    # aggregation_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # aggregation_monitor.start()  # COMMENTED: Resource monitoring disabled
    # aggregation_start_time = time.perf_counter()  # COMMENTED: Resource monitoring disabled

    if not record_dicts:
        print("[SERVER] No training results to aggregate")
        return MetricRecord({})

    print(f"[SERVER] Aggregating training metrics from {len(record_dicts)} clients")
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] TRAINING PHASE - CLIENT RESULTS")
    print(f"[SERVER] {'='*80}")

    # OPTIMIZED: Extract all client data in single pass
    clients_data = []
    train_success_count = 0
    
    for i, record_dict in enumerate(record_dicts):
        if "metrics" not in record_dict:
            print(f"[SERVER] ⚠️ [CLIENT {i}] Warning: record_dict has no metrics")
            continue
        
        metrics = record_dict["metrics"]
    

        # Defensive defaults (server never trusts clients)
        num_examples = max(1, int(_safe_float(metrics.get("num-examples", 1))))
        client_id = int(_safe_float(metrics.get("client_id", i)))
        
        # Extract all metrics at once with safe defaults
        client_data = {
            'id': client_id,
            'examples': num_examples,
            'train_time': _safe_float(metrics.get("client_train_time", 0.0)),
            'loss': _safe_float(metrics.get("client_train_loss", 0.0)),
            'mr': _safe_float(metrics.get("client_train_acc_mr", 0.0)),
            'mp': _safe_float(metrics.get("client_train_acc_mp", 0.0)),
            'mAP50': _safe_float(metrics.get("client_train_acc_mAP@0.5", 0.0)),
            'mAP': _safe_float(metrics.get("client_train_acc_mAP", 0.0)),
            'lr': _safe_float(metrics.get("lr", 0.01)),
            # Nested-HPO: this client's own inner-search winners for this round
            # (present only when client_hpo_enabled=true on the client).
            'hpo_epochs': _safe_float(metrics.get("client_hpo_best_local_epochs", -1.0)),
            'hpo_batch':  _safe_float(metrics.get("client_hpo_best_batch_size", -1.0)),
            'hpo_train_map50': _safe_float(metrics.get("client_hpo_best_train_mAP@0.5", -1.0)),
            'data_received': _safe_float(metrics.get("data_received_from_server", 0.0)),
            'data_sent': _safe_float(metrics.get("data_sent_to_server", 0.0)),
            'round_duration': _safe_float(metrics.get("round_duration", 0.0)),
            # COMMENTED: Resource monitoring disabled
            # 'train_cpu_peak': _safe_float(metrics.get("train_resources_per_process_cpu_peak", 0.0)),
            # 'train_cpu_avg': _safe_float(metrics.get("train_resources_per_process_cpu_avg", 0.0)),
            # 'train_ram_peak_mb': _safe_float(metrics.get("train_resources_per_process_ram_peak_mb", 0.0)),
            # 'train_ram_avg_mb': _safe_float(metrics.get("train_resources_per_process_ram_avg_mb", 0.0)),
            # 'train_ram_peak_pct': _safe_float(metrics.get("train_resources_per_process_ram_peak_pct", 0.0)),
            # 'train_ram_avg_pct': _safe_float(metrics.get("train_resources_per_process_ram_avg_pct", 0.0)),
            # COMMENTED: Resource monitoring disabled
            # 'train_sys_cpu_peak': _safe_float(metrics.get("train_resources_system_cpu_peak", 0.0)),
            # 'train_sys_cpu_avg': _safe_float(metrics.get("train_resources_system_cpu_avg", 0.0)),
            # 'train_sys_ram_peak_mb': _safe_float(metrics.get("train_resources_system_ram_peak_mb", 0.0)),
            # 'train_sys_ram_avg_mb': _safe_float(metrics.get("train_resources_system_ram_avg_mb", 0.0)),
            # 'train_sys_ram_peak_pct': _safe_float(metrics.get("train_resources_system_ram_peak_pct", 0.0)),
            # 'train_sys_ram_avg_pct': _safe_float(metrics.get("train_resources_system_ram_avg_pct", 0.0)),
        }
        clients_data.append(client_data)
        train_success_count += 1
        
        # Print per-client result with metrics
        print(f"[SERVER] ✅ [CLIENT {client_id}] TRAIN COMPLETE - Loss: {client_data['loss']:.4f}, mAP@0.5: {client_data['mAP50']:.4f}, mAP: {client_data['mAP']:.4f}, Time: {client_data['train_time']:.2f}s")
        # COMMENTED: Resource monitoring disabled
        # print(f"[SERVER]    Resources -      CPU: {client_data['train_cpu_peak']:.1f}% peak / {client_data['train_cpu_avg']:.1f}% avg, RAM: {client_data['train_ram_peak_mb']:.1f} MB peak / {client_data['train_ram_avg_mb']:.1f} MB avg")
    
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
    
    # COMMENTED: Resource monitoring disabled
    # ✅ AGGREGATE CLIENT RESOURCE METRICS
    # avg_train_cpu_peak = sum(c['train_cpu_peak'] for c in clients_data) / len(clients_data)
    # avg_train_cpu_avg = sum(c['train_cpu_avg'] for c in clients_data) / len(clients_data)
    # max_train_ram_peak_mb = max(c['train_ram_peak_mb'] for c in clients_data)
    # avg_train_ram_avg_mb = sum(c['train_ram_avg_mb'] for c in clients_data) / len(clients_data)
    # max_train_ram_peak_pct = max(c['train_ram_peak_pct'] for c in clients_data)
    # avg_train_ram_avg_pct = sum(c['train_ram_avg_pct'] for c in clients_data) / len(clients_data)
    # 
    # avg_train_sys_cpu_peak = sum(c['train_sys_cpu_peak'] for c in clients_data) / len(clients_data)
    # avg_train_sys_cpu_avg = sum(c['train_sys_cpu_avg'] for c in clients_data) / len(clients_data)
    # max_train_sys_ram_peak_mb = max(c['train_sys_ram_peak_mb'] for c in clients_data)
    # avg_train_sys_ram_avg_mb = sum(c['train_sys_ram_avg_mb'] for c in clients_data) / len(clients_data)
    # max_train_sys_ram_peak_pct = max(c['train_sys_ram_peak_pct'] for c in clients_data)
    # avg_train_sys_ram_avg_pct = sum(c['train_sys_ram_avg_pct'] for c in clients_data) / len(clients_data)
    
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
            "client_train_num_examples": int(c['examples']),
            # Nested-HPO: -1 means client_hpo_enabled was off this round.
            "client_hpo_epochs": int(c['hpo_epochs']) if c['hpo_epochs'] >= 0 else None,
            "client_hpo_batch":  int(c['hpo_batch'])  if c['hpo_batch']  >= 0 else None,
            "client_hpo_train_map50": c['hpo_train_map50'] if c['hpo_train_map50'] >= 0 else None,
            # COMMENTED: Resource monitoring disabled
            # ✅ Per-client resource metrics
            # "client_train_resources": {
            #     "per_process": {
            #         "cpu_percent_peak": c['train_cpu_peak'],
            #         "cpu_percent_avg": c['train_cpu_avg'],
            #         "ram_mb_peak": c['train_ram_peak_mb'],
            #         "ram_mb_avg": c['train_ram_avg_mb'],
            #         "ram_percent_peak": c['train_ram_peak_pct'],
            #         "ram_percent_avg": c['train_ram_avg_pct'],
            #     },
            #     "system_wide": {
            #         "cpu_percent_peak": c['train_sys_cpu_peak'],
            #         "cpu_percent_avg": c['train_sys_cpu_avg'],
            #         "ram_mb_peak": c['train_sys_ram_peak_mb'],
            #         "ram_mb_avg": c['train_sys_ram_avg_mb'],
            #         "ram_percent_peak": c['train_sys_ram_peak_pct'],
            #         "ram_percent_avg": c['train_sys_ram_avg_pct'],
            #     }
            # }
        }
        clients_logs.append(client_log)
    
    # ✅ STOP SERVER AGGREGATION MONITORING
    # aggregation_resources = aggregation_monitor.stop()  # COMMENTED: Resource monitoring disabled
    # print(f"[SERVER] Aggregation resources: CPU peak {aggregation_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {aggregation_resources['per_process']['memory_mb']['peak']:.1f} MB")  # COMMENTED: Resource monitoring disabled
    
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
        "round_data_transferred_bytes": int(total_data_transferred),
        # COMMENTED: Resource monitoring disabled
        # ✅ SERVER RESOURCE METRICS
        # "server_aggregation_resources": {
        #     "per_process": {
        #         "cpu_percent_peak": aggregation_resources['per_process']['cpu_percent']['peak'],
        #         "cpu_percent_avg": aggregation_resources['per_process']['cpu_percent']['avg'],
        #         "ram_mb_peak": aggregation_resources['per_process']['memory_mb']['peak'],
        #         "ram_mb_avg": aggregation_resources['per_process']['memory_mb']['avg'],
        #         "ram_percent_peak": aggregation_resources['per_process']['memory_percent']['peak'],
        #         "ram_percent_avg": aggregation_resources['per_process']['memory_percent']['avg'],
        #     },
        #     "system_wide": {
        #         "cpu_percent_peak": aggregation_resources['system_wide']['cpu_percent']['peak'],
        #         "cpu_percent_avg": aggregation_resources['system_wide']['cpu_percent']['avg'],
        #         "ram_mb_peak": aggregation_resources['system_wide']['memory_mb']['peak'],
        #         "ram_mb_avg": aggregation_resources['system_wide']['memory_mb']['avg'],
        #         "ram_percent_peak": aggregation_resources['system_wide']['memory_percent']['peak'],
        #         "ram_percent_avg": aggregation_resources['system_wide']['memory_percent']['avg'],
        #     }
        # },
        # ✅ AGGREGATED CLIENT TRAINING RESOURCES
        # "aggregated_client_training_resources": {
        #     "per_process": {
        #         "cpu_percent_peak_avg": avg_train_cpu_peak,
        #         "cpu_percent_avg": avg_train_cpu_avg,
        #         "ram_mb_peak_max": max_train_ram_peak_mb,
        #         "ram_mb_avg": avg_train_ram_avg_mb,
        #         "ram_percent_peak_max": max_train_ram_peak_pct,
        #         "ram_percent_avg": avg_train_ram_avg_pct,
        #     },
        #     "system_wide": {
        #         "cpu_percent_peak_avg": avg_train_sys_cpu_peak,
        #         "cpu_percent_avg": avg_train_sys_cpu_avg,
        #         "ram_mb_peak_max": max_train_sys_ram_peak_mb,
        #         "ram_mb_avg": avg_train_sys_ram_avg_mb,
        #         "ram_percent_peak_max": max_train_sys_ram_peak_pct,
        #         "ram_percent_avg": avg_train_sys_ram_avg_pct,
        #     }
        # }
    }
    
    ALL_ROUND_LOGS.append(round_log)
    
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] ROUND {(CURRENT_ROUND+1)} TRAINING SUMMARY")
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] Participating Clients: {len(clients_data)} | Success: {train_success_count}")
    print(f"[SERVER] Training Loss:     {round_train_loss:.4f}")
    print(f"[SERVER] Training mAP@0.5:  {round_train_acc_mAP50:.4f}")
    print(f"[SERVER] Training mAP:      {round_train_acc_mAP:.4f}")
    print(f"[SERVER] Aggregated Score:  {round_train_acc_aggregated:.4f}")
    print(f"[SERVER] Round Duration:    {max_round_duration:.2f}s")
    print(f"[SERVER] Data Transferred:  {total_data_mb:.2f} MB")
    # COMMENTED: Resource monitoring disabled
    # print(f"[SERVER] Avg Client CPU: {avg_train_cpu_peak:.1f}% (peak), RAM: {max_train_ram_peak_mb:.1f} MB (peak)")
    # print(f"[SERVER] Server CPU: {aggregation_resources['per_process']['cpu_percent']['peak']:.1f}% (peak), RAM: {aggregation_resources['per_process']['memory_mb']['peak']:.1f} MB (peak)")
    print(f"[SERVER] {'='*80}\n")
    
    # FIXED: Return aggregated metrics for Flower (not empty dict)
    return MetricRecord({
        "train_loss": round_train_loss,
        "train_accuracy": round_train_acc_aggregated,
        "train_mAP50": round_train_acc_mAP50,
    })

def custom_eval_metrics_aggregation(record_dicts: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """
    OPTIMIZED: Single-pass evaluation aggregation with efficient data extraction and resource monitoring.
    âœ… FIXED: Returns properly structured MetricRecord.
    """
    global ALL_ROUND_LOGS
    
    # COMMENTED: Resource monitoring disabled
    # START SERVER AGGREGATION MONITORING FOR EVALUATION
    # aggregation_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # aggregation_monitor.start()  # COMMENTED: Resource monitoring disabled
    # aggregation_start_time = time.perf_counter()  # COMMENTED: Resource monitoring disabled

    if not record_dicts:
        print("[SERVER] No evaluation results to aggregate")
        return MetricRecord({})
    
    # if not ALL_ROUND_LOGS:
    #     print("[SERVER] Warning: No round logs available yet")
    #     return MetricRecord({})
    
    print(f"[SERVER] Aggregating evaluation metrics from {len(record_dicts)} clients")
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] EVALUATION PHASE - CLIENT RESULTS")
    print(f"[SERVER] {'='*80}")
    
    # OPTIMIZED: Extract all evaluation data in single pass
    eval_data = []
    eval_success_count = 0
    
    for i, record_dict in enumerate(record_dicts):
        if "metrics" not in record_dict:
            print(f"[SERVER] Warning: eval record_dict {i} has no metrics")
            continue
        
        metrics = record_dict["metrics"]
    
        
        client_id = int(_safe_float(metrics.get("client_id", 0)))
        eval_success_count += 1  # Metrics received = client completed evaluation
        
        client_eval = {
            'id': client_id,
            'examples': _safe_float(metrics.get("num-examples", 1.0)),
            'loss': _safe_float(metrics.get("client_eval_loss", 0.0)),
            'mr': _safe_float(metrics.get("client_eval_acc_mr", 0.0)),
            'mp': _safe_float(metrics.get("client_eval_acc_mp", 0.0)),
            'mAP50': _safe_float(metrics.get("client_eval_acc_mAP@0.5", 0.0)),
            'mAP': _safe_float(metrics.get("client_eval_acc_mAP", 0.0)),
            'eval_time': _safe_float(metrics.get("client_eval_time", 0.0)),
            # COMMENTED: Resource monitoring disabled
            # Evaluation phase resource metrics (per-process)
            # 'eval_cpu_peak': _safe_float(metrics.get("eval_resources_per_process_cpu_peak", 0.0)),
            # 'eval_cpu_avg': _safe_float(metrics.get("eval_resources_per_process_cpu_avg", 0.0)),
            # 'eval_ram_peak_mb': _safe_float(metrics.get("eval_resources_per_process_ram_peak_mb", 0.0)),
            # 'eval_ram_avg_mb': _safe_float(metrics.get("eval_resources_per_process_ram_avg_mb", 0.0)),
            # 'eval_ram_peak_pct': _safe_float(metrics.get("eval_resources_per_process_ram_peak_pct", 0.0)),
            # 'eval_ram_avg_pct': _safe_float(metrics.get("eval_resources_per_process_ram_avg_pct", 0.0)),
            # Evaluation phase resource metrics (system-wide)
            # 'eval_sys_cpu_peak': _safe_float(metrics.get("eval_resources_system_cpu_peak", 0.0)),
            # 'eval_sys_cpu_avg': _safe_float(metrics.get("eval_resources_system_cpu_avg", 0.0)),
            # 'eval_sys_ram_peak_mb': _safe_float(metrics.get("eval_resources_system_ram_peak_mb", 0.0)),
            # 'eval_sys_ram_avg_mb': _safe_float(metrics.get("eval_resources_system_ram_avg_mb", 0.0)),
            # 'eval_sys_ram_peak_pct': _safe_float(metrics.get("eval_resources_system_ram_peak_pct", 0.0)),
            # 'eval_sys_ram_avg_pct': _safe_float(metrics.get("eval_resources_system_ram_avg_pct", 0.0)),
        }
        eval_data.append(client_eval)
        
        # Print per-client evaluation results
        print(f"[SERVER] ✅ [CLIENT {client_id}] EVAL COMPLETE - Loss: {client_eval['loss']:.4f}, mAP@0.5: {client_eval['mAP50']:.4f}, mAP: {client_eval['mAP']:.4f}, Time: {client_eval['eval_time']:.2f}s")
        # COMMENTED: Resource monitoring disabled
        # print(f"[SERVER]    Resources -     CPU: {client_eval['eval_cpu_peak']:.1f}% peak / {client_eval['eval_cpu_avg']:.1f}% avg, RAM: {client_eval['eval_ram_peak_mb']:.1f} MB peak / {client_eval['eval_ram_avg_mb']:.1f} MB avg")
    
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
    
    # COMMENTED: Resource monitoring disabled
    # AGGREGATE CLIENT EVALUATION RESOURCE METRICS
    # avg_eval_cpu_peak = sum(c['eval_cpu_peak'] for c in eval_data) / len(eval_data)
    # avg_eval_cpu_avg = sum(c['eval_cpu_avg'] for c in eval_data) / len(eval_data)
    # max_eval_ram_peak_mb = max(c['eval_ram_peak_mb'] for c in eval_data)
    # avg_eval_ram_avg_mb = sum(c['eval_ram_avg_mb'] for c in eval_data) / len(eval_data)
    # max_eval_ram_peak_pct = max(c['eval_ram_peak_pct'] for c in eval_data)
    # avg_eval_ram_avg_pct = sum(c['eval_ram_avg_pct'] for c in eval_data) / len(eval_data)
    # 
    # avg_eval_sys_cpu_peak = sum(c['eval_sys_cpu_peak'] for c in eval_data) / len(eval_data)
    # avg_eval_sys_cpu_avg = sum(c['eval_sys_cpu_avg'] for c in eval_data) / len(eval_data)
    # max_eval_sys_ram_peak_mb = max(c['eval_sys_ram_peak_mb'] for c in eval_data)
    # avg_eval_sys_ram_avg_mb = sum(c['eval_sys_ram_avg_mb'] for c in eval_data) / len(eval_data)
    # max_eval_sys_ram_peak_pct = max(c['eval_sys_ram_peak_pct'] for c in eval_data)
    # avg_eval_sys_ram_avg_pct = sum(c['eval_sys_ram_avg_pct'] for c in eval_data) / len(eval_data)
    
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
            # COMMENTED: Resource monitoring disabled
            # Per-client evaluation resource metrics
            # client_logs_map[c['id']]["client_eval_resources"] = {
            #     "per_process": {
            #         "cpu_percent_peak": c['eval_cpu_peak'],
            #         "cpu_percent_avg": c['eval_cpu_avg'],
            #         "ram_mb_peak": c['eval_ram_peak_mb'],
            #         "ram_mb_avg": c['eval_ram_avg_mb'],
            #         "ram_percent_peak": c['eval_ram_peak_pct'],
            #         "ram_percent_avg": c['eval_ram_avg_pct'],
            #     },
            #     "system_wide": {
            #         "cpu_percent_peak": c['eval_sys_cpu_peak'],
            #         "cpu_percent_avg": c['eval_sys_cpu_avg'],
            #         "ram_mb_peak": c['eval_sys_ram_peak_mb'],
            #         "ram_mb_avg": c['eval_sys_ram_avg_mb'],
            #         "ram_percent_peak": c['eval_sys_ram_peak_pct'],
            #         "ram_percent_avg": c['eval_sys_ram_avg_pct'],
            #     }
            # }
    
    # COMMENTED: Resource monitoring disabled
    # STOP SERVER AGGREGATION MONITORING
    # aggregation_resources = aggregation_monitor.stop()  # COMMENTED: Resource monitoring disabled
    # print(f"[SERVER] Eval aggregation resources: CPU peak {aggregation_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {aggregation_resources['per_process']['memory_mb']['peak']:.1f} MB")  # COMMENTED: Resource monitoring disabled
    
    # ADD AGGREGATED EVALUATION RESOURCES TO ROUND LOG
    # current_round["server_eval_resources"] = {
    #     "per_process": {
    #         "cpu_percent_peak": aggregation_resources['per_process']['cpu_percent']['peak'],
    #         "cpu_percent_avg": aggregation_resources['per_process']['cpu_percent']['avg'],
    #         "ram_mb_peak": aggregation_resources['per_process']['memory_mb']['peak'],
    #         "ram_mb_avg": aggregation_resources['per_process']['memory_mb']['avg'],
    #         "ram_percent_peak": aggregation_resources['per_process']['memory_percent']['peak'],
    #         "ram_percent_avg": aggregation_resources['per_process']['memory_percent']['avg'],
    #     },
    #     "system_wide": {
    #         "cpu_percent_peak": aggregation_resources['system_wide']['cpu_percent']['peak'],
    #         "cpu_percent_avg": aggregation_resources['system_wide']['cpu_percent']['avg'],
    #         "ram_mb_peak": aggregation_resources['system_wide']['memory_mb']['peak'],
    #         "ram_mb_avg": aggregation_resources['system_wide']['memory_mb']['avg'],
    #         "ram_percent_peak": aggregation_resources['system_wide']['memory_percent']['peak'],
    #         "ram_percent_avg": aggregation_resources['system_wide']['memory_percent']['avg'],
    #     }
    # }
    
    # COMMENTED: Resource monitoring disabled
    # ADD AGGREGATED CLIENT EVALUATION RESOURCES TO ROUND LOG
    # current_round["aggregated_client_eval_resources"] = {
    #     "per_process": {
    #         "cpu_percent_peak_avg": avg_eval_cpu_peak,
    #         "cpu_percent_avg": avg_eval_cpu_avg,
    #         "ram_mb_peak_max": max_eval_ram_peak_mb,
    #         "ram_mb_avg": avg_eval_ram_avg_mb,
    #         "ram_percent_peak_max": max_eval_ram_peak_pct,
    #         "ram_percent_avg": avg_eval_ram_avg_pct,
    #     },
    #     "system_wide": {
    #         "cpu_percent_peak_avg": avg_eval_sys_cpu_peak,
    #         "cpu_percent_avg": avg_eval_sys_cpu_avg,
    #         "ram_mb_peak_max": max_eval_sys_ram_peak_mb,
    #         "ram_mb_avg": avg_eval_sys_ram_avg_mb,
    #         "ram_percent_peak_max": max_eval_sys_ram_peak_pct,
    #         "ram_percent_avg": avg_eval_sys_ram_avg_pct,
    #     }
    # }
    
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] ROUND {(CURRENT_ROUND+1)} EVALUATION SUMMARY")
    print(f"[SERVER] {'='*80}")
    print(f"[SERVER] Participating Clients: {len(eval_data)} | Success: {eval_success_count}")
    print(f"[SERVER] Validation Loss:   {round_eval_loss:.4f}")
    print(f"[SERVER] Validation mAP@0.5: {round_eval_acc_mAP50:.4f}")
    print(f"[SERVER] Validation mAP:    {round_eval_acc_mAP:.4f}")
    print(f"[SERVER] Aggregated Score:  {round_eval_acc_aggregated:.4f}")
    print(f"[SERVER] Eval Duration:     {max_eval_time:.2f}s")
    # COMMENTED: Resource monitoring disabled
    # print(f"[SERVER] Avg Client CPU: {avg_eval_cpu_peak:.1f}% (peak), RAM: {max_eval_ram_peak_mb:.1f} MB (peak)")
    # print(f"[SERVER] Server CPU: {aggregation_resources['per_process']['cpu_percent']['peak']:.1f}% (peak), RAM: {aggregation_resources['per_process']['memory_mb']['peak']:.1f} MB (peak)")
    print(f"[SERVER] {'='*80}\n")
    
    # FIX 4: Report intermediate result to Optuna after every eval round so
    # MedianPruner can kill diverging trials early (e.g. FedAdam + high LR
    # is detectable by round 2).  step = 0-indexed round number relative to
    # this trial.  Only active during HPO; CURRENT_OPTUNA_TRIAL is None for
    # the final full run, so this block is a no-op there.
    if CURRENT_OPTUNA_TRIAL is not None:
        step = len(ALL_ROUND_LOGS) - 1  # 0-indexed: just appended above
        CURRENT_OPTUNA_TRIAL.report(round_eval_acc_mAP50, step=step)
        if CURRENT_OPTUNA_TRIAL.should_prune():
            print(f"[OPTUNA] Pruning trial at step {(step+1)} "
                  f"(mAP@0.5={round_eval_acc_mAP50:.4f})")
            raise _TrialPruneSignal()

    # ✅ FIXED: Return aggregated metrics for Flower
    return MetricRecord({
        "eval_loss": round_eval_loss,
        "eval_accuracy": round_eval_acc_aggregated,
        "eval_mAP50": round_eval_acc_mAP50,
    })

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for ServerApp with optimized configuration."""

    # ── FedAdam / FedYogi Array-compatibility patch ───────────────────────
    # Flower's FedOpt strategies call Array(v) after momentum math, but the
    # result can be a plain Python/dict value that Array() rejects.  We swap
    # the Array symbol in both module namespaces with a thin wrapper that
    # coerces any non-tensor/ndarray value through np.asarray() first.
    import flwr.serverapp.strategy.fedadam as _fedadam_mod
    import flwr.serverapp.strategy.fedyogi as _fedyogi_mod
    from flwr.common.record.array import Array as _OrigArray

    def _NpSafeArray(v, *args, **kwargs):
        if not isinstance(v, (np.ndarray, torch.Tensor)):
            v = np.asarray(v)
        return _OrigArray(v, *args, **kwargs)

    _fedadam_mod.Array = _NpSafeArray
    _fedyogi_mod.Array = _NpSafeArray
    # ─────────────────────────────────────────────────────────────────────

    # Get configuration
    fraction_train = get_config("fraction-train", context, default=1.0, type_converter=float)
    fraction_evaluate = get_config("fraction-evaluate", context, default=1.0, type_converter=float)
    num_rounds = get_config("num-server-rounds", context, default=5, type_converter=int)
    lr = get_config("lr", context, default=0.01, type_converter=float)
    task_type = get_config("task", context, default="classification")
    dataset_number = get_config("dataset", context, default=100, type_converter=int)

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
        pretrained_checkpoint = get_config("pretrained_checkpoint", context, default="").strip()
        print(f"Using the pretrained checkpoint path: '{pretrained_checkpoint}'")  # Debug print
        # =====================================================================
        # CHECKPOINT RESUME LOGIC
        # =====================================================================
        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            # Load from checkpoint (resume from previous run)
            print(f"Loading from pretrained checkpoint: {pretrained_checkpoint}")
            try:
                state_dict = load_yolo_checkpoint_as_state_dict(pretrained_checkpoint)
                arrays = ArrayRecord(state_dict)
                print(f"âœ… YOLO checkpoint loaded: {len(arrays)} layers")
            except Exception as e:
                print(f"âŒ Failed to load checkpoint: {e}")
                print("Falling back to pretrained weights...")
                # Fall back to pretrained weights
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
        elif pretrained_checkpoint:
            # Checkpoint path specified but file not found
            print(f"âš ï¸  Checkpoint path specified but not found: {pretrained_checkpoint}")
            print("Falling back to pretrained weights...")
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
            # No checkpoint specified, use standard pretrained weights
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


    strategy_id     = get_config("strategy",        context, default=1,  type_converter=int)
    # Fixed epochs/batch used INSIDE the server's HPO trial loop (nested-HPO
    # mode: server only searches lr; these are just the values forwarded to
    # clients as a starting point -- if client_hpo_enabled=true, each client
    # overrides them with its own inner search result every round anyway).
    local_epochs_cfg = get_config("local-epochs", context, default=3,  type_converter=int)
    batch_size_cfg   = get_config("batch_size",    context, default=16, type_converter=int)
    # Try new hpo_trials first, fall back to old n_optuna_trials for compatibility
    hpo_trials      = get_config("hpo_trials",      context, default=None, type_converter=int)
    if hpo_trials is None:
        hpo_trials = get_config("n_optuna_trials", context, default=0,  type_converter=int)
    hpo_rounds      = get_config("hpo_rounds",      context, default=3,  type_converter=int)
    hpo_mode        = get_config("hpo_mode",        context, default="hpo_then_train", type_converter=str)
    
    # Read FLAML config parameters
    use_flaml       = USE_FLAML  # Use global setting from environment/imports
    flaml_time_budget = get_config("flaml_time_budget", context, default=3600, type_converter=int)
    flaml_metric      = get_config("flaml_metric",      context, default="mAP50", type_converter=str)
    flaml_estimator   = get_config("flaml_estimator",   context, default="lgb", type_converter=str)
    flaml_sample_size = get_config("flaml_sample_size", context, default=32, type_converter=int)

    # Client-side local HPO flags -- forwarded to clients via train_cfg below.
    # Stored as bool/int here; env-var overrides (CLIENT_HPO_ENABLED=true/1) are
    # coerced the same way USE_FLAML is, since TOML/run_config booleans don't
    # round-trip cleanly through string env vars.
    _client_hpo_raw = get_config("client_hpo_enabled", context, default=False, type_converter=str)
    client_hpo_enabled = (
        _client_hpo_raw if isinstance(_client_hpo_raw, bool)
        else str(_client_hpo_raw).strip().lower() in ("1", "true", "yes")
    )
    client_hpo_trials  = get_config("client_hpo_trials", context, default=3, type_converter=int)

    # Forwarded to clients every round so they know whether to run their own
    # ABS (adaptive batch/epoch) loop -- server doesn't control ABS decisions
    # itself, it just needs to pass the flag through in train_cfg.
    _adaptive_batch_raw = get_config("adaptive_batch_enabled", context, default=False, type_converter=str)
    adaptive_batch_enabled_cfg = (
        _adaptive_batch_raw if isinstance(_adaptive_batch_raw, bool)
        else str(_adaptive_batch_raw).strip().lower() in ("1", "true", "yes")
    )

    # Adaptive Global LR tuning (server-side), used only for the FINAL run
    # (not inside the Optuna/FLAML HPO trial loop).
    _adaptive_lr_raw = get_config("adaptive_lr_enabled", context, default=False, type_converter=str)
    adaptive_lr_enabled = (
        _adaptive_lr_raw if isinstance(_adaptive_lr_raw, bool)
        else str(_adaptive_lr_raw).strip().lower() in ("1", "true", "yes")
    )
    adaptive_lr_min            = get_config("adaptive_lr_min", context, default=0.0001, type_converter=float)
    adaptive_lr_max            = get_config("adaptive_lr_max", context, default=0.01, type_converter=float)
    adaptive_lr_max_reductions = get_config("adaptive_lr_max_reductions", context, default=3, type_converter=int)
    adaptive_lr_growth_factor  = get_config("adaptive_lr_growth_factor", context, default=1.2, type_converter=float)
    adaptive_lr_backoff_factor = get_config("adaptive_lr_backoff_factor", context, default=0.5, type_converter=float)

    # Backend forwarded to clients every round so a single USE_FLAML switch
    # controls BOTH the server's outer lr-search and every client's inner
    # epochs/batch-search -- no separate client-side env var to keep in sync.
    client_hpo_backend = "flaml" if USE_FLAML else "optuna"

    if client_hpo_enabled and hpo_trials and hpo_trials > 0:
        print(
            f"[INFO] NESTED HPO MODE: server searches lr only ({hpo_trials} trials, "
            f"backend={client_hpo_backend}); each client independently searches "
            f"local_epochs/batch_size ({client_hpo_trials} trials/round, same backend) "
            f"conditioned on that round's lr. mAP is reported back up to the server's "
            f"study as a joint function of all three."
        )
    elif client_hpo_enabled and (not hpo_trials or hpo_trials == 0):
        print(
            f"[WARN] client_hpo_enabled=true but hpo_trials=0 -- the server will use "
            f"a single fixed lr for the whole run while clients still search "
            f"epochs/batch each round. That's valid (client-only tuning), but if you "
            f"intended nested joint search, set hpo_trials > 0."
        )

    # -- Snapshot initial weights ONCE so every trial/run starts identically.
    # ArrayRecord is consumed by strategy.start(), so we keep the raw state_dict
    # and rebuild a fresh ArrayRecord before each strategy.start() call.
    initial_state_dict = arrays.to_torch_state_dict()

    def _make_strategy(sid, eta=None, eta_l=None, beta_1=None, beta_2=None,
                       tau=None, proximal_mu=None):
        """Always construct a fresh strategy instance.
        Stateful strategies (FedAdam/FedYogi) carry momentum buffers; reusing the
        same instance across trials would corrupt the server optimizer state for
        trial N+1 with gradients from trial N.

        Strategy-specific kwargs are forwarded only when explicitly provided so
        that Flower's documented defaults are preserved whenever a param is None.
        """
        kwargs = dict(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            train_metrics_aggr_fn=custom_train_metrics_aggregation,
            evaluate_metrics_aggr_fn=custom_eval_metrics_aggregation,
        )
        if sid == 2:
            print("Using strategy: FedYogi")
            # FedYogi defaults: eta=1e-2, eta_l=0.0316, beta_1=0.9, beta_2=0.99, tau=1e-3
            if eta         is not None: kwargs["eta"]    = eta
            if eta_l       is not None: kwargs["eta_l"]  = eta_l
            if beta_1      is not None: kwargs["beta_1"] = beta_1
            if beta_2      is not None: kwargs["beta_2"] = beta_2
            if tau         is not None: kwargs["tau"]    = tau
            return FedYogi(**kwargs)
        elif sid == 3:
            print("Using strategy: FedAdam")
            # FedAdam defaults: eta=1e-1, eta_l=1e-1, beta_1=0.9, beta_2=0.99, tau=1e-3
            if eta         is not None: kwargs["eta"]    = eta
            if eta_l       is not None: kwargs["eta_l"]  = eta_l
            if beta_1      is not None: kwargs["beta_1"] = beta_1
            if beta_2      is not None: kwargs["beta_2"] = beta_2
            if tau         is not None: kwargs["tau"]    = tau
            return FedAdam(**kwargs)
        elif sid == 4:
            print("Using strategy: FedProx")
            # FedProx default: proximal_mu=0.0 (0.0 == FedAvg; higher = more regularisation)
            if proximal_mu is not None: kwargs["proximal_mu"] = proximal_mu
            return FedProx(**kwargs)
        else:
            print("Using strategy: FedAvg")
            return FedAvg(**kwargs)

    def _run_trial(
        trial_lr, n_rounds, trial_epochs, trial_batch, trial_tag,
        trial_strategy=None, optuna_trial=None,
        # FedYogi / FedAdam params (None → use Flower's documented defaults)
        eta=None, eta_l=None, beta_1=None, beta_2=None, tau=None,
        # FedProx param
        proximal_mu=None,
    ):
        """Execute one complete FL session and return (result, logs_snapshot).

        Resets global round state before every call so round IDs and metric
        extraction are relative to THIS session, never polluted by a prior trial.

        trial_strategy: if provided, overrides the strategy_id read from config.
        This lets Optuna suggest a strategy per trial without touching the outer
        strategy_id variable.

        optuna_trial: the live Optuna Trial object (or None for the final run).
        When provided, each eval round reports its mAP@0.5 so MedianPruner can
        kill diverging trials before they waste their remaining proxy rounds.

        Strategy-specific params are forwarded to _make_strategy and then into
        Flower's strategy constructor only when not None, preserving defaults.
        """
        global ALL_ROUND_LOGS, CURRENT_ROUND, CURRENT_OPTUNA_TRIAL
        ALL_ROUND_LOGS = []
        CURRENT_ROUND  = 0
        CURRENT_OPTUNA_TRIAL = optuna_trial  # FIX 4: expose to eval callback

        # Fresh weights: scientifically mandatory -- every trial must start from
        # identical weights or HP comparison is meaningless.
        fresh_arrays  = ArrayRecord(torch_state_dict=initial_state_dict, keep_input=True)
        effective_sid = trial_strategy if trial_strategy is not None else strategy_id
        strategy      = _make_strategy(
            sid=effective_sid,
            eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau,
            proximal_mu=proximal_mu,
        )

        train_cfg = {
            "lr":           trial_lr,
            "num_rounds":   n_rounds,
            "local-epochs": trial_epochs,   # client reads from msg.content["config"]
            "batch_size":   trial_batch,    # client reads from msg.content["config"]
            # Client-side local HPO (independent of this function's own Optuna/FLAML
            # trial loop). Forwarded every round so clients can self-tune epochs/batch
            # when enabled. Static for the whole _run_trial call -- not searched here.
            "client_hpo_enabled": client_hpo_enabled,
            "client_hpo_trials":  client_hpo_trials,
            "client_hpo_backend": client_hpo_backend,
            "adaptive_batch_enabled": adaptive_batch_enabled_cfg,
        }
        if task_type == "detection":
            train_cfg["yolo_size"] = yolo_size

        print(f"\n{'='*70}")
        print(f"STARTING FL TRIAL  [{trial_tag}]")
        print(f"  lr={trial_lr:.6f}  rounds={n_rounds} strategy={effective_sid} local_epochs={trial_epochs} batch_size={trial_batch}")
        print(f"{'='*70}\n")

        try:
            result = strategy.start(
                grid=grid,
                initial_arrays=fresh_arrays,
                train_config=ConfigRecord(train_cfg),
                num_rounds=n_rounds,
                timeout=18000,
            )
        except _TrialPruneSignal:
            # Eval callback raised the prune signal -- re-raise as appropriate exception
            # so the HPO framework records this trial as pruned cleanly.
            CURRENT_OPTUNA_TRIAL = None
            if USE_FLAML:
                raise FLAMLTrialPruneSignal()
            else:
                raise optuna.exceptions.TrialPruned()
        finally:
            CURRENT_OPTUNA_TRIAL = None  # always clear after the trial ends

        # Return a copy so the global can be safely reset next trial
        return result, list(ALL_ROUND_LOGS)

    def _run_adaptive_lr_final_run(
        n_rounds, trial_epochs, trial_batch, trial_tag,
        lr_min, lr_max, max_reductions, growth_factor, backoff_factor,
        trial_strategy=None,
        eta=None, eta_l=None, beta_1=None, beta_2=None, tau=None, proximal_mu=None,
    ):
        """
        Global learning-rate tuning ("ALR"), run ONCE for the final training
        run (not inside the Optuna/FLAML HPO trial loop).

        Unlike _run_trial (which must build a fresh strategy + fresh arrays
        every call to keep HPO trials scientifically isolated), this function
        builds ONE strategy instance and calls strategy.start(num_rounds=1)
        repeatedly, threading the previous round's resulting arrays back in
        as `initial_arrays` for the next round. This is required so that
        FedAdam/FedYogi's momentum buffers persist correctly across rounds --
        exactly like a normal multi-round run, just with an lr that changes
        round-to-round based on validation-loss trend.

        Algorithm:
          - lr starts at lr_min. lr_max_dynamic starts at lr_max.
          - After each round, compare this round's eval loss to the previous:
              - improved  -> lr = min(lr * growth_factor, lr_max_dynamic)  (explore higher)
              - regressed -> lr_max_dynamic = lr (cap shrinks to the value that
                              caused the regression), lr = max(lr_min, lr * backoff_factor)
                              (fall back to a lower value), reductions_used += 1
          - Training terminates early if reductions_used exceeds max_reductions.

        Returns: (result, logs_snapshot) -- same shape as _run_trial's return,
        so the caller (the FINAL RUN block) doesn't need to change downstream code.
        """
        global ALL_ROUND_LOGS, CURRENT_ROUND
        ALL_ROUND_LOGS = []
        CURRENT_ROUND  = 0

        effective_sid = trial_strategy if trial_strategy is not None else strategy_id
        strategy = _make_strategy(
            sid=effective_sid,
            eta=eta, eta_l=eta_l, beta_1=beta_1, beta_2=beta_2, tau=tau,
            proximal_mu=proximal_mu,
        )

        lr = lr_min
        lr_max_dynamic = lr_max
        reductions_used = 0
        prev_val_loss = None
        current_arrays = ArrayRecord(torch_state_dict=initial_state_dict, keep_input=True)

        print(f"\n{'='*70}")
        print(f"STARTING ADAPTIVE-LR FINAL RUN  [{trial_tag}]")
        print(f"  lr_min={lr_min:.6f}  lr_max={lr_max:.6f}  max_reductions={max_reductions}  "
              f"growth={growth_factor}  backoff={backoff_factor}  rounds={n_rounds}")
        print(f"{'='*70}\n")

        result = None
        for round_num in range(1, n_rounds + 1):
            train_cfg = {
                "lr":           lr,
                "num_rounds":   n_rounds,
                "local-epochs": trial_epochs,
                "batch_size":   trial_batch,
                "client_hpo_enabled": client_hpo_enabled,
                "client_hpo_trials":  client_hpo_trials,
                "adaptive_batch_enabled": adaptive_batch_enabled_cfg,
            }
            if task_type == "detection":
                train_cfg["yolo_size"] = yolo_size

            print(f"[ALR] Round {round_num}/{n_rounds} -- lr={lr:.6f} "
                  f"(cap={lr_max_dynamic:.6f}, reductions_used={reductions_used}/{max_reductions})")

            result = strategy.start(
                grid=grid,
                initial_arrays=current_arrays,
                train_config=ConfigRecord(train_cfg),
                num_rounds=1,
                timeout=18000,
            )

            eval_loss = None
            if ALL_ROUND_LOGS:
                eval_loss = ALL_ROUND_LOGS[-1].get("round_eval_loss")

            if eval_loss is not None and prev_val_loss is not None:
                if eval_loss < prev_val_loss:
                    lr = min(lr * growth_factor, lr_max_dynamic)
                    print(f"[ALR] Validation loss improved ({prev_val_loss:.4f} -> {eval_loss:.4f}) "
                          f"-- raising lr to {lr:.6f}")
                else:
                    lr_max_dynamic = lr
                    lr = max(lr_min, lr * backoff_factor)
                    reductions_used += 1
                    print(f"[ALR] Validation loss regressed ({prev_val_loss:.4f} -> {eval_loss:.4f}) "
                          f"-- capping lr at {lr_max_dynamic:.6f}, falling back to {lr:.6f} "
                          f"(reduction {reductions_used}/{max_reductions})")
                    if reductions_used > max_reductions:
                        print(f"[ALR] Max LR reductions exceeded -- terminating early "
                              f"after round {round_num}/{n_rounds}.")
                        break

            if eval_loss is not None:
                prev_val_loss = eval_loss
            current_arrays = result.arrays

        return result, list(ALL_ROUND_LOGS)

    # ================================================================
    # HPO BLOCK -- skipped entirely when hpo_trials == 0
    # Supports both Optuna and FLAML backends based on USE_FLAML setting
    # ================================================================
    if hpo_trials > 0:
        if USE_FLAML:
            print(f"\n{'='*70}")
            print(f"FLAML HPO: {hpo_trials} trials x {hpo_rounds} proxy rounds each")
            print(f"HPO Mode: {hpo_mode}")
            print(f"{'='*70}\n")

            _flaml_prev_lr, _flaml_prev_scores = lr_only_warm_start(dataset_number)
            study = create_flaml_study(
                study_name=f"{experiment_name}_{run_id}_hpo",
                direction="maximize",
                time_budget=flaml_time_budget,
                metric_name=flaml_metric,
                seed=42,
                points_to_evaluate=_flaml_prev_lr,
                evaluated_rewards=_flaml_prev_scores,
                # search_space omitted -> FLAMLStudy defaults to lr-only (see flaml_hpo.py)
            )

            def objective_flaml(trial):
                # NESTED HPO: server suggests ONLY lr. epochs/batch are fixed
                # here (each client searches its own inner value instead --
                # see client_hpo_enabled/client_hpo_backend forwarded in
                # train_cfg). strategy is fixed to the configured strategy_id,
                # not searched, so trials stay comparable on lr alone.
                trial_lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)

                print(f"\n[FLAML-OBJECTIVE] Starting trial {(trial.number)+1} with lr={trial_lr:.6f} "
                      f"(strategy={strategy_id} fixed, epochs/batch delegated to clients)")

                _, trial_logs = _run_trial(
                    trial_lr, n_rounds=hpo_rounds,
                    trial_epochs=local_epochs_cfg, trial_batch=batch_size_cfg,
                    trial_tag=f"flaml_trial_{(trial.number)+1}",
                    trial_strategy=strategy_id, optuna_trial=trial,
                )

                if not trial_logs:
                    raise FLAMLTrialPruneSignal()

                eval_maps = [r.get("round_eval_acc", {}).get("mAP@0.5", 0.0) for r in trial_logs]
                mAP = max(eval_maps) if eval_maps else 0.0

                # Joint record: pull each client's own (epochs, batch) winner
                # per round out of trial_logs so the full (lr, epochs, batch)
                # -> mAP triple is visible, not just lr -> mAP.
                per_client_configs = []
                for round_log in trial_logs:
                    for cl in round_log.get("clients_logs", []):
                        if cl.get("client_hpo_epochs") is not None:
                            per_client_configs.append({
                                "round": round_log["round_id"], "client": cl["client_id"],
                                "epochs": cl["client_hpo_epochs"], "batch": cl["client_hpo_batch"],
                                "client_train_map50": cl.get("client_hpo_train_map50"),
                            })
                trial.set_user_attr("per_client_configs", per_client_configs)
                if per_client_configs:
                    import statistics as _stats
                    epochs_mode = _stats.mode([c["epochs"] for c in per_client_configs])
                    batch_mode  = _stats.mode([c["batch"] for c in per_client_configs])
                    trial.set_user_attr("epochs_mode", epochs_mode)
                    trial.set_user_attr("batch_mode", batch_mode)
                    print(f"[FLAML] Trial {(trial.number+1):>3} client-side winners -- "
                          f"most common epochs={epochs_mode}, batch={batch_mode} "
                          f"(across {len(per_client_configs)} client-rounds)")

                try:
                    trial_logs_path = f"{experiment_name}_{run_id}_hpo_trial_{trial.number}_logs.json"
                    with open(trial_logs_path, "w") as f:
                        json.dump({"round_logs": trial_logs, "per_client_configs": per_client_configs}, f, indent=2)
                except Exception as e:
                    print(f"[FLAML] Warning: {e}")

                print(f"[FLAML] Trial {(trial.number+1):>3} -> mAP@0.5={mAP:.4f}  (lr={trial_lr:.5f})")
                return mAP

            study.optimize(objective_flaml, n_trials=hpo_trials)
            best = study.best_params
            print(f"\n{'='*70}")
            print(f"FLAML COMPLETE -- Best trial #{(study.best_trial.number)+1}")
            print(f"  mAP@0.5  = {study.best_value:.4f}")
            print(f"  lr       = {best['lr']:.6f} | strategy = {strategy_id} (fixed)")
            print(f"  client-side winners: epochs={study.best_trial.user_attrs.get('epochs_mode')}, "
                  f"batch={study.best_trial.user_attrs.get('batch_mode')}")
            print(f"{'='*70}\n")

        else:
            # OPTUNA BACKEND
            print(f"\n{'='*70}")
            print(f"OPTUNA HPO: {hpo_trials} trials x {hpo_rounds} proxy rounds each")
            print(f"HPO Mode: {hpo_mode}")
            print(f"{'='*70}\n")

            optuna.logging.set_verbosity(optuna.logging.INFO)

            study = optuna.create_study(
                study_name=f"{experiment_name}_{run_id}_hpo",
                direction="maximize",
                storage=f"sqlite:///{experiment_name}_{run_id}_hpo.db",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=hpo_rounds, reduction_factor=3),
            )

            def objective(trial):
                # NESTED HPO: server suggests ONLY lr. epochs/batch are fixed
                # placeholders here -- each client overrides them every round
                # with its own inner Optuna/FLAML search (client_hpo_enabled),
                # conditioned on this trial's lr. strategy is fixed to the
                # configured strategy_id (not searched), so trials are
                # comparable on lr alone.
                trial_lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)

                print(f"\n[OPTUNA-OBJECTIVE] Starting trial {(trial.number)+1} with lr={trial_lr:.6f} "
                      f"(strategy={strategy_id} fixed, epochs/batch delegated to clients)")

                _, trial_logs = _run_trial(
                    trial_lr,
                    n_rounds=hpo_rounds,
                    trial_epochs=local_epochs_cfg,
                    trial_batch=batch_size_cfg,
                    trial_tag=f"optuna_trial_{(trial.number)+1}",
                    trial_strategy=strategy_id,
                    optuna_trial=trial,
                )

                if not trial_logs:
                    raise optuna.exceptions.TrialPruned()

                eval_maps = [r.get("round_eval_acc", {}).get("mAP@0.5", 0.0) for r in trial_logs]
                mAP = max(eval_maps) if eval_maps else 0.0

                # Joint record: pull each client's own (epochs, batch) winner
                # per round out of trial_logs, so the full (lr, epochs, batch)
                # -> mAP triple this trial actually produced is visible and
                # persisted -- not just the lr -> mAP scalar Optuna sees.
                per_client_configs = []
                for round_log in trial_logs:
                    for cl in round_log.get("clients_logs", []):
                        if cl.get("client_hpo_epochs") is not None:
                            per_client_configs.append({
                                "round": round_log["round_id"], "client": cl["client_id"],
                                "epochs": cl["client_hpo_epochs"], "batch": cl["client_hpo_batch"],
                                "client_train_map50": cl.get("client_hpo_train_map50"),
                            })
                trial.set_user_attr("per_client_configs", per_client_configs)
                if per_client_configs:
                    import statistics as _stats
                    epochs_mode = _stats.mode([c["epochs"] for c in per_client_configs])
                    batch_mode  = _stats.mode([c["batch"] for c in per_client_configs])
                    trial.set_user_attr("epochs_mode", epochs_mode)
                    trial.set_user_attr("batch_mode", batch_mode)
                    print(f"[OPTUNA] Trial {(trial.number+1):>3} client-side winners -- "
                          f"most common epochs={epochs_mode}, batch={batch_mode} "
                          f"(across {len(per_client_configs)} client-rounds)")

                try:
                    trial_logs_path = f"{experiment_name}_{run_id}_hpo_trial_{trial.number}_logs.json"
                    with open(trial_logs_path, "w") as f:
                        json.dump({"round_logs": trial_logs, "per_client_configs": per_client_configs}, f, indent=2)
                except Exception as e:
                    print(f"[OPTUNA] Warning: could not save trial logs: {e}")

                print(f"[OPTUNA] Trial {(trial.number+1):>3} -> mAP@0.5={mAP:.4f}  (lr={trial_lr:.5f})")
                return mAP

            # Seed with the known best baseline before letting Optuna explore.
            # This becomes trial 0 — TPE builds its probability model around it
            # immediately instead of starting blind.  load_if_exists=True means
            # this is skipped automatically if the study already ran it (resume).
            if len(study.trials) == 0:
                study.enqueue_trial({"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1})
        
            # ── lr-only warm start ──────────────────────────────────────────
            # Old full (lr+epochs+batch+strategy+meta-param) history is
            # reduced to just its lr values -- distinct lr's, best-known score
            # each -- matching the trimmed 1-D search space above.
            lr_distribution = {"lr": optuna.distributions.FloatDistribution(0.0001, 0.01, log=True)}
            try:
                from flower_benchmarks.flaml_hpo import lr_only_warm_start as _lr_warm
                warm_lrs, warm_scores = _lr_warm(dataset_number)
            except Exception:
                warm_lrs, warm_scores = None, None
            if warm_lrs:
                for cfg, score in zip(warm_lrs, warm_scores):
                    study.add_trial(optuna.trial.create_trial(
                        params=cfg, distributions=lr_distribution,
                        value=score, state=optuna.trial.TrialState.COMPLETE,
                    ))
                print(f"[OPTUNA] Warm-started with {len(warm_lrs)} historical lr point(s): "
                      f"{[round(l['lr'], 6) for l in warm_lrs]}")

            study.optimize(objective, n_trials=hpo_trials)

            best = study.best_params
            best_epochs_mode = study.best_trial.user_attrs.get("epochs_mode", local_epochs_cfg)
            best_batch_mode  = study.best_trial.user_attrs.get("batch_mode", batch_size_cfg)
            print(f"\n{'='*70}")
            print(f"OPTUNA COMPLETE -- Best trial #{(study.best_trial.number)+1}")
            print(f"  mAP@0.5  = {study.best_value:.4f}")
            print(f"  lr       = {best['lr']:.6f}")
            print(f"  strategy = {strategy_id} (fixed)")
            print(f"  client-side winners: epochs={best_epochs_mode}, batch={best_batch_mode}")
            print(f"{'='*70}\n")

        # Promote HPO winner as the value for the final full run.
        # epochs/batch are NOT promoted from the server search (it never
        # searched them) -- they stay whatever client_hpo_enabled/
        # adaptive_batch_enabled resolve to each round of the final run.
        # strategy is fixed throughout (never searched).
        lr             = best["lr"]
        final_epochs   = local_epochs_cfg
        final_batch    = batch_size_cfg
        final_strategy = strategy_id
        final_strategy_kwargs = {}  # fixed strategy -> Flower's own defaults apply
    else:
        # No HPO -- fall through using values already read from config
        final_epochs   = int(get_config("local-epochs", context, default=3, type_converter=int))
        final_batch    = int(get_config("batch_size",   context, default=16, type_converter=int))
        final_strategy = strategy_id  # use whatever was set in config, unchanged
        final_strategy_kwargs = {}    # no HPO-tuned params; _make_strategy uses Flower defaults
        
        # Apply strategy-specific parameters if provided in code
        if final_strategy == 2:  # FedYogi
            final_strategy_kwargs = dict(
                eta    = 0.01,        # FedYogi default
                eta_l  = 0.0316,
                beta_1 = 0.9,
                beta_2 = 0.99,
                tau    = 0.001,
            )
        elif final_strategy == 3:  # FedAdam
            # ✅ CUSTOMIZE FedAdam PARAMETERS HERE
            final_strategy_kwargs = dict(
                eta    = 0.00125693,         # FedAdam server-side learning rate
                eta_l  = 0.0777663,         # FedAdam client-side learning rate
                beta_1 = 0.829967,         # Exponential decay for 1st moment
                beta_2 = 0.92004,        # Exponential decay for 2nd moment
                tau    = 0.000134188,       # Regularization coefficient
            )
        elif final_strategy == 4:  # FedProx
            final_strategy_kwargs = dict(
                proximal_mu = 0.0,    # FedProx regularization (0.0 == FedAvg)
            )

    # ================================================================
    # FINAL (or sole) FULL RUN
    # num_rounds is the TOTAL budget.  HPO already consumed
    # hpo_trials * hpo_rounds of it, so the final run only
    # gets the remainder.  When HPO is off, hpo_consumed = 0 and
    # final_rounds == num_rounds, preserving the original behaviour.
    # When hpo_mode == "hpo_only", skip final training entirely.
    # ================================================================
    hpo_consumed = hpo_trials * hpo_rounds if hpo_trials > 0 else 0
    final_rounds = num_rounds - hpo_consumed
    
    # Check if we should skip final training based on hpo_mode
    skip_final_training = (hpo_mode == "hpo_only" and hpo_trials > 0)
 
    if final_rounds <= 0:
        print(f"[WARN] HPO consumed all {num_rounds} rounds "
              f"({hpo_trials} trials x {hpo_rounds} hpo_rounds). "
              f"No rounds left for the final run. "
              f"Increase num-server-rounds or reduce hpo_trials/hpo_rounds.")
    elif skip_final_training:
        print(f"\n[INFO] HPO Mode is '{hpo_mode}' - skipping final training run")
        print(f"Best HPO result will be used for model export")
    else:
        print(f"\n[INFO] Round budget: {num_rounds} total -- "
              f"{hpo_consumed} HPO -- {final_rounds} final run")
        if adaptive_lr_enabled:
            result, _ = _run_adaptive_lr_final_run(
                n_rounds=final_rounds,
                trial_epochs=final_epochs,
                trial_batch=final_batch,
                trial_tag=f"final_run_{run_id}",
                lr_min=adaptive_lr_min,
                lr_max=adaptive_lr_max,
                max_reductions=adaptive_lr_max_reductions,
                growth_factor=adaptive_lr_growth_factor,
                backoff_factor=adaptive_lr_backoff_factor,
                trial_strategy=final_strategy,
                **final_strategy_kwargs,
            )
        else:
            result, _ = _run_trial(
                lr,
                n_rounds=final_rounds,
                trial_epochs=final_epochs,
                trial_batch=final_batch,
                trial_tag=f"final_run_{run_id}",
                trial_strategy=final_strategy,
                **final_strategy_kwargs,
            )
 
    if final_rounds <= 0 or skip_final_training:
        print(f"\nFinal run skipped or no budget available")
        if use_flaml and hpo_trials > 0:
            print(f"[INFO] Best HPO params from FLAML will be used")
        elif hpo_trials > 0:
            print(f"[INFO] Best HPO params from Optuna will be used")
        return  # budget exhausted or hpo_only mode

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
            print(f"[OK] Final YOLO model saved: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save YOLO checkpoint: {e}")
            torch.save({"model": state_dict}, out_path)
            print(f"[OK] Saved as PyTorch state dict: {out_path}")
    else:
        out_path = f"{experiment_name}_{run_id}_final_model.pt"
        torch.save(state_dict, out_path)
        print(f"[OK] Final model saved: {out_path}")

    # Save round logs (ALL_ROUND_LOGS now holds only the final run's rounds)
    logs_path = f"{experiment_name}_{run_id}_logs.json"
    try:
        with open(logs_path, "w") as f:
            json.dump(ALL_ROUND_LOGS, f, indent=2)
        print(f"[OK] Training logs saved: {logs_path}")
    except Exception as e:
        print(f"[WARN] Could not save logs: {e}")

    # Print summary statistics
    if ALL_ROUND_LOGS:
        print(f"\n{'='*70}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*70}")
        print(f"Total Rounds:      {len(ALL_ROUND_LOGS)}")

        final_round = ALL_ROUND_LOGS[-1]
        print(f"\nFinal Round Metrics:")
        print(f"  Training Loss:   {final_round.get('round_train_loss', 0):.4f}")
        print(f"  Training mAP@0.5:    {final_round.get('round_training_mAP@0.5', {}).get('mAP@0.5', 0):.4f}")

        if 'round_eval_acc' in final_round:
            print(f"  Validation Loss: {final_round.get('round_eval_loss', 0):.4f}")
            print(f"  Validation mAP@0.5:  {final_round.get('round_eval_mAP@0.5', {}).get('mAP@0.5', 0):.4f}")

        total_time    = sum(r.get("round_duration", 0)            for r in ALL_ROUND_LOGS)
        total_data_mb = sum(r.get("round_data_transferred_mb", 0) for r in ALL_ROUND_LOGS)

        print(f"\nTotal Training Time: {total_time/60:.2f} minutes")
        print(f"Total Data Transfer: {total_data_mb:.2f} MB")
        print(f"{'='*70}\n")

    print(f"Run completed successfully!")
    print(f"Final model: {out_path}")
    print(f"Training logs: {logs_path}")