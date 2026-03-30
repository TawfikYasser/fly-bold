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
import optuna
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

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
    """Raised inside the eval aggregation callback when Optuna decides to prune.
    Caught by _run_trial() and re-raised as optuna.exceptions.TrialPruned so
    strategy.start() is interrupted cleanly without corrupting the study DB.
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


    strategy_id     = get_config("strategy",        context, default=1,  type_converter=int)
    n_optuna_trials = get_config("n_optuna_trials", context, default=0,  type_converter=int)
    hpo_rounds      = get_config("hpo_rounds",      context, default=3,  type_converter=int)

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
            # FIX 4: eval callback raised the prune signal -- re-raise as the
            # Optuna exception so the study records this trial as pruned cleanly.
            CURRENT_OPTUNA_TRIAL = None
            raise optuna.exceptions.TrialPruned()
        finally:
            CURRENT_OPTUNA_TRIAL = None  # always clear after the trial ends

        # Return a copy so the global can be safely reset next trial
        return result, list(ALL_ROUND_LOGS)

    # ================================================================
    # OPTUNA HPO BLOCK -- skipped entirely when n_optuna_trials == 0
    # ================================================================
    if n_optuna_trials > 0:
        print(f"\n{'='*70}")
        print(f"OPTUNA HPO: {n_optuna_trials} trials x {hpo_rounds} proxy rounds each")
        print(f"{'='*70}\n")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # SQLite storage: completed trials survive VM preemption / crashes.
        # load_if_exists=True means a restarted run resumes from where it stopped.
        study = optuna.create_study(
            study_name=f"{experiment_name}_{run_id}_hpo",
            direction="maximize",
            storage=f"sqlite:///{experiment_name}_{run_id}_hpo.db",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            # FIX 4: MedianPruner kills a trial at step S if its reported mAP
            # is below the median of all completed trials at the same step.
            # n_startup_trials=3: don't prune before we have enough baseline data.
            # n_warmup_steps=1: never prune on the very first round (too noisy).
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        )

        def objective(trial):
            trial_lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
            trial_epochs   = trial.suggest_int("local_epochs",   1, 5)
            trial_batch    = trial.suggest_categorical("batch_size", [8, 16, 32])
            trial_strategy = trial.suggest_categorical("strategy", [1, 2, 3, 4])

            print(f"\n[OPTUNA-OBJECTIVE] Starting trial {(trial.number)+1} with lr={trial_lr:.6f}, "
                  f"strategy={trial_strategy}, local_epochs={trial_epochs}, batch_size={trial_batch}")

            if trial_strategy == 1: # FedAvg

                _, trial_logs = _run_trial(
                    trial_lr,
                    n_rounds=hpo_rounds,
                    trial_epochs=trial_epochs,
                    trial_batch=trial_batch,
                    trial_tag=f"optuna_trial_{(trial.number)+1}",
                    trial_strategy=trial_strategy,
                    optuna_trial=trial,
                )

            elif trial_strategy == 2: # FedYogi
                # Suggest around FedYogi defaults: eta=1e-2, eta_l=0.0316, tau=1e-3
                t_eta    = trial.suggest_float("yogi_eta",    1e-4, 1e-1, log=True)
                t_eta_l  = trial.suggest_float("yogi_eta_l",  1e-3, 1e-1, log=True)
                t_beta_1 = trial.suggest_float("yogi_beta_1", 0.8,  0.99)
                t_beta_2 = trial.suggest_float("yogi_beta_2", 0.9,  0.999)
                t_tau    = trial.suggest_float("yogi_tau",    1e-4, 1e-2, log=True)

                _, trial_logs = _run_trial(
                    trial_lr,
                    n_rounds=hpo_rounds,
                    trial_epochs=trial_epochs,
                    trial_batch=trial_batch,
                    trial_tag=f"optuna_trial_{(trial.number)+1}",
                    trial_strategy=trial_strategy,
                    optuna_trial=trial,
                    eta=t_eta, eta_l=t_eta_l, beta_1=t_beta_1,
                    beta_2=t_beta_2, tau=t_tau,
                )

            elif trial_strategy == 3: # FedAdam
                # Suggest around FedAdam defaults: eta=1e-1, eta_l=1e-1, tau=1e-3
                t_eta    = trial.suggest_float("adam_eta",    1e-3, 5e-1, log=True)
                t_eta_l  = trial.suggest_float("adam_eta_l",  1e-3, 5e-1, log=True)
                t_beta_1 = trial.suggest_float("adam_beta_1", 0.8,  0.99)
                t_beta_2 = trial.suggest_float("adam_beta_2", 0.9,  0.999)
                t_tau    = trial.suggest_float("adam_tau",    1e-4, 1e-2, log=True)

                _, trial_logs = _run_trial(
                    trial_lr,
                    n_rounds=hpo_rounds,
                    trial_epochs=trial_epochs,
                    trial_batch=trial_batch,
                    trial_tag=f"optuna_trial_{(trial.number)+1}",
                    trial_strategy=trial_strategy,
                    optuna_trial=trial,
                    eta=t_eta, eta_l=t_eta_l, beta_1=t_beta_1,
                    beta_2=t_beta_2, tau=t_tau,
                )

            elif trial_strategy == 4: # FedProx
                # proximal_mu=0.0 == FedAvg; useful range is [0.001, 10.0]
                t_proximal_mu = trial.suggest_float("proximal_mu", 0.001, 10.0, log=True)

                _, trial_logs = _run_trial(
                    trial_lr,
                    n_rounds=hpo_rounds,
                    trial_epochs=trial_epochs,
                    trial_batch=trial_batch,
                    trial_tag=f"optuna_trial_{(trial.number)+1}",
                    trial_strategy=trial_strategy,
                    optuna_trial=trial,
                    proximal_mu=t_proximal_mu,
                )

            # Empty logs means all clients failed -- treat as a pruned trial
            if not trial_logs:
                raise optuna.exceptions.TrialPruned()

            # FIX 2: Use max mAP@0.5 across all proxy rounds, not just the last.
            # A single dip at round N (FL non-IID variance is high) would unfairly
            # kill a good config if we only read the final round.  max() is more
            # robust; mean-of-last-2 is an alternative if you prefer smoothing.
            eval_maps = [r.get("round_eval_acc", {}).get("mAP@0.5", 0.0) for r in trial_logs]
            mAP = max(eval_maps) if eval_maps else 0.0

            # Persist per-trial logs for post-analysis (non-fatal if it fails)
            try:
                trial_logs_path = f"{experiment_name}_{run_id}_hpo_trial_{trial.number}_logs.json"
                with open(trial_logs_path, "w") as f:
                    json.dump(trial_logs, f, indent=2)
            except Exception as e:
                print(f"[OPTUNA] Warning: could not save trial logs: {e}")

            print(f"[OPTUNA] Trial {(trial.number+1):>3} -> mAP@0.5={mAP:.4f}  "
                  f"(lr={trial_lr:.5f}, strategy={trial_strategy}, epochs={trial_epochs}, batch={trial_batch})")
            return mAP

        # Seed with the known best baseline before letting Optuna explore.
        # This becomes trial 0 — TPE builds its probability model around it
        # immediately instead of starting blind.  load_if_exists=True means
        # this is skipped automatically if the study already ran it (resume).
        if len(study.trials) == 0:
            study.enqueue_trial({"lr": 0.001, "local_epochs": 3, "batch_size": 16, "strategy": 1})
        
        distributions = {
            "fedavg":{ "lr": optuna.distributions.FloatDistribution(0.0001, 0.01, log=True),
            "strategy": optuna.distributions.CategoricalDistribution([1, 2, 3, 4]),
            "local_epochs": optuna.distributions.IntDistribution(1, 5),
            "batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32]),},
            
            "fedyogi":{ "lr": optuna.distributions.FloatDistribution(0.0001, 0.01, log=True),
            "strategy": optuna.distributions.CategoricalDistribution([1, 2, 3, 4]),
            "local_epochs": optuna.distributions.IntDistribution(1, 5),
            "batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32]),
                "yogi_eta": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
                "yogi_eta_l": optuna.distributions.FloatDistribution(1e-3, 1e-1, log=True),
                "yogi_beta_1": optuna.distributions.FloatDistribution(0.8, 0.99),
                "yogi_beta_2": optuna.distributions.FloatDistribution(0.9, 0.999),
                "yogi_tau": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),},
            
            "fedadam":{ "lr": optuna.distributions.FloatDistribution(0.0001, 0.01, log=True),
            "strategy": optuna.distributions.CategoricalDistribution([1, 2, 3, 4]),
            "local_epochs": optuna.distributions.IntDistribution(1, 5),
            "batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32]),
                "adam_eta": optuna.distributions.FloatDistribution(1e-3, 5e-1, log=True),
                "adam_eta_l": optuna.distributions.FloatDistribution(1e-3, 5e-1, log=True),
                "adam_beta_1": optuna.distributions.FloatDistribution(0.8, 0.99),
                "adam_beta_2": optuna.distributions.FloatDistribution(0.9, 0.999),
                "adam_tau": optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True),},

            "fedprox":{ "lr": optuna.distributions.FloatDistribution(0.0001, 0.01, log=True),
            "strategy": optuna.distributions.CategoricalDistribution([1, 2, 3, 4]),
            "local_epochs": optuna.distributions.IntDistribution(1, 5),
            "batch_size": optuna.distributions.CategoricalDistribution([8, 16, 32]),
                "proximal_mu": optuna.distributions.FloatDistribution(0.001, 10.0, log=True),},
        }
        best_prev_params = {
        1:{"lr": 0.001,
            "local_epochs": 3,
            "batch_size": 16,
            "strategy": 1,},
        2:{"lr": 0.000561,
            "local_epochs": 5,
            "batch_size": 8,
            "strategy": 3,
            "adam_eta": 0.0814829,
            "adam_eta_l": 0.00113647,
            "adam_beta_1": 0.984283,
            "adam_beta_2": 0.982412,
            "adam_tau": 0.000265875,},
        3:{"lr": 0.000231,
            "local_epochs": 1,
            "batch_size": 16,
            "strategy": 2,
            "yogi_eta": 0.00125628,
            "yogi_eta_l": 0.00816846,
            "yogi_beta_1": 0.949183,
            "yogi_beta_2": 0.919768,
            "yogi_tau": 0.00106775,},
        4:{"lr": 0.001530,
            "local_epochs": 1,
            "batch_size": 8,
            "strategy": 2,
            "yogi_eta": 0.000196343,
            "yogi_eta_l": 0.0233596,
            "yogi_beta_1": 0.883629,
            "yogi_beta_2": 0.912082,
            "yogi_tau": 0.000978034,},
        5:{"lr": 0.000117,
            "local_epochs": 5,
            "batch_size": 16,
            "strategy": 4,
            "proximal_mu": 1.2604664585649468,},
        }
        study.add_trial(optuna.trial.create_trial(
            params=best_prev_params[1],
            distributions=distributions["fedavg"],
            value=0.5309,
            state=optuna.trial.TrialState.COMPLETE,
        ))
        study.add_trial(optuna.trial.create_trial(
            params=best_prev_params[1],
            distributions=distributions["fedavg"],
            value=0.5211,
            state=optuna.trial.TrialState.COMPLETE,
        ))
        study.add_trial(optuna.trial.create_trial(
            params=best_prev_params[1],
            distributions=distributions["fedavg"],
            value=0.5200,
            state=optuna.trial.TrialState.COMPLETE,
        ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[1],
        #     distributions=distributions["fedavg"],
        #     value=0.5218,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[1],
        #     distributions=distributions["fedavg"],
        #     value=0.5181,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[1],
        #     distributions=distributions["fedavg"],
        #     value=0.5162,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[1],
        #     distributions=distributions["fedavg"],
        #     value=0.5048,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[2],
        #     distributions=distributions["fedadam"],
        #     value=0.5001,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[3],
        #     distributions=distributions["fedyogi"],
        #     value=0.5001,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[4],
        #     distributions=distributions["fedyogi"],
        #     value=0.4448,
        #     state=optuna.trial.TrialState.PRUNED,
        # ))
        # study.add_trial(optuna.trial.create_trial(
        #     params=best_prev_params[5],
        #     distributions=distributions["fedprox"],
        #     value=0.5039,
        #     state=optuna.trial.TrialState.COMPLETE,
        # ))

        study.optimize(objective, n_trials=n_optuna_trials)

        best = study.best_params
        print(f"\n{'='*70}")
        print(f"OPTUNA COMPLETE -- Best trial #{(study.best_trial.number)+1}")
        print(f"  mAP@0.5  = {study.best_value:.4f}")
        print(f"  lr       = {best['lr']:.6f}")
        print(f"  epochs   = {best['local_epochs']}")
        print(f"  batch    = {best['batch_size']}")
        print(f"  strategy = {best['strategy']} "
              f"({'FedAvg' if best['strategy']==1 else 'FedYogi' if best['strategy']==2 else 'FedAdam' if best['strategy']==3 else 'FedProx'})")
        print(f"{'='*70}\n")

        # Promote HPO winners as values for the final full run
        lr             = best["lr"]
        final_epochs   = best["local_epochs"]
        final_batch    = best["batch_size"]
        final_strategy = best["strategy"]

        # Retrieve the best strategy-specific params (may not exist if
        # the winning strategy has no tunable params beyond lr, e.g. FedAvg)
        final_strategy_kwargs = {}
        if final_strategy == 2:  # FedYogi
            final_strategy_kwargs = dict(
                eta    = best.get("yogi_eta"),
                eta_l  = best.get("yogi_eta_l"),
                beta_1 = best.get("yogi_beta_1"),
                beta_2 = best.get("yogi_beta_2"),
                tau    = best.get("yogi_tau"),
            )
        elif final_strategy == 3:  # FedAdam
            final_strategy_kwargs = dict(
                eta    = best.get("adam_eta"),
                eta_l  = best.get("adam_eta_l"),
                beta_1 = best.get("adam_beta_1"),
                beta_2 = best.get("adam_beta_2"),
                tau    = best.get("adam_tau"),
            )
        elif final_strategy == 4:  # FedProx
            final_strategy_kwargs = dict(
                proximal_mu = best.get("proximal_mu"),
            )
    else:
        # No HPO -- fall through using values already read from config
        final_epochs   = int(get_config("local-epochs", context, default=3, type_converter=int))
        final_batch    = int(get_config("batch_size",   context, default=16, type_converter=int))
        final_strategy = strategy_id  # use whatever was set in config, unchanged
        final_strategy_kwargs = {}    # no HPO-tuned params; _make_strategy uses Flower defaults

    # ================================================================
    # FINAL (or sole) FULL RUN
    # num_rounds is the TOTAL budget.  HPO already consumed
    # n_optuna_trials * hpo_rounds of it, so the final run only
    # gets the remainder.  When HPO is off, hpo_consumed = 0 and
    # final_rounds == num_rounds, preserving the original behaviour.
    # ================================================================
    hpo_consumed = n_optuna_trials * hpo_rounds if n_optuna_trials > 0 else 0
    final_rounds = num_rounds - hpo_consumed
 
    if final_rounds <= 0:
        print(f"[WARN] HPO consumed all {num_rounds} rounds "
              f"({n_optuna_trials} trials x {hpo_rounds} hpo_rounds). "
              f"No rounds left for the final run. "
              f"Increase num-server-rounds or reduce n_optuna_trials/hpo_rounds.")
    else:
        print(f"\n[INFO] Round budget: {num_rounds} total -- "
              f"{hpo_consumed} HPO -- {final_rounds} final run")
        result, _ = _run_trial(
            lr,
            n_rounds=final_rounds,
            trial_epochs=final_epochs,
            trial_batch=final_batch,
            trial_tag=f"final_run_{run_id}",
            trial_strategy=final_strategy,
            **final_strategy_kwargs,
        )
 
    if final_rounds <= 0:
        return  # budget exhausted by HPO, nothing to save

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