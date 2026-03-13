"""flower-benchmarks: Optimized Flower Client App with YOLOv5."""

import os
# Force optimal thread usage BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
import time
import torch
import numpy as np
import sys
import pickle
import yaml
from pathlib import Path
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flower_benchmarks.task import Net, load_data, test as test_fn, train as train_fn
from flower_benchmarks.plugins.yolov5.model import (
    load_yolo_checkpoint_as_state_dict, 
    save_state_dict_as_yolo_checkpoint
)
from flower_benchmarks.task import (
    yolo_train_from_state_and_return_state_dict, 
    yolo_evaluate_weights_and_parse_map,
    ResourceMonitor
)

# =====================================================================
# GLOBAL CACHES (significantly improves performance)
# =====================================================================
_DATASET_CACHE = {}  # Cache dataset paths to avoid repeated file system ops
_YOLO_MODELS_SETUP = False  # Track if YOLOv5 models are ready

def ensure_yolo_models_available():
    """Copy yolov5 models directory - run once per container lifecycle."""
    global _YOLO_MODELS_SETUP
    
    if _YOLO_MODELS_SETUP:
        return
    
    import shutil
    import glob
    
    # 1. Copy to working directory (for subprocess mode)
    target_models_dir = os.path.join(os.getcwd(), "yolov5", "models")
    
    if not os.path.exists(target_models_dir):
        possible_sources = [
            "/app/yolov5/models",
            os.path.join(os.path.dirname(__file__), "..", "..", "yolov5", "models"),
        ]
        
        for src in possible_sources:
            if os.path.exists(src):
                print(f"[setup] Copying yolov5 models from {src} to {target_models_dir}")
                os.makedirs(os.path.dirname(target_models_dir), exist_ok=True)
                shutil.copytree(src, target_models_dir)
                break
    
    # 2. Copy to Flower's installed app directory (for in-process mode)
    flower_app_dirs = glob.glob("/root/.flwr/apps/tawfik.flower_benchmarks.*/")
    
    for app_dir in flower_app_dirs:
        flower_models_dir = os.path.join(app_dir, "yolov5", "models")
        
        if not os.path.exists(flower_models_dir) or not list(Path(flower_models_dir).glob("*.yaml")):
            print(f"[setup] Copying yolov5 models to Flower app dir: {flower_models_dir}")
            
            source_dir = "/app/yolov5/models"
            if os.path.exists(source_dir):
                os.makedirs(flower_models_dir, exist_ok=True)
                
                for yaml_file in Path(source_dir).glob("*.yaml"):
                    dest_file = os.path.join(flower_models_dir, yaml_file.name)
                    if not os.path.exists(dest_file):
                        shutil.copy2(yaml_file, dest_file)
                
                source_hub = os.path.join(source_dir, "hub")
                dest_hub = os.path.join(flower_models_dir, "hub")
                if os.path.exists(source_hub) and not os.path.exists(dest_hub):
                    shutil.copytree(source_hub, dest_hub)
    
    _YOLO_MODELS_SETUP = True

def get_config(key: str, context: Context, default=None, type_converter=str):
    """Get configuration with clear precedence: env > run_config > node_config > default."""
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

# Flower ClientApp
app = ClientApp()

def _safe_float(v, default=0.0):
    """Safely convert to float."""
    try:
        return float(v)
    except Exception:
        return default

def calculate_message_size_fast(msg: Message) -> int:
    """
    OPTIMIZED: Fast size calculation without expensive conversions.
    Ensures result is never zero to satisfy Flower's validation.
    """
    total_size = 0
    
    # Calculate arrays size efficiently
    if "arrays" in msg.content:
        arrays_record = msg.content["arrays"]
        
        # Try to get size from internal data structure without full conversion
        if hasattr(arrays_record, '_data'):
            for value in arrays_record._data.values():
                if hasattr(value, 'nbytes'):
                    total_size += value.nbytes
                elif isinstance(value, torch.Tensor):
                    total_size += value.nelement() * value.element_size()
                else:
                    total_size += sys.getsizeof(value)
        else:
            # Fallback: minimal conversion
            state_dict = arrays_record.to_torch_state_dict()
            for param in state_dict.values():
                total_size += param.nelement() * param.element_size()
    
    # Metrics and config are negligible (~few KB)
    if "metrics" in msg.content:
        total_size += sys.getsizeof(pickle.dumps(dict(msg.content["metrics"])))
    
    if "config" in msg.content:
        total_size += sys.getsizeof(pickle.dumps(dict(msg.content["config"])))
    
    # âœ… CRITICAL: Ensure minimum size to avoid Flower validation error
    return max(total_size, 1)

def prepare_client_yolo_dataset_prepartitioned(client_id: int):
    """
    OPTIMIZED: Use pre-partitioned dataset with caching.
    Previous version verified file system every round (expensive with 10k+ images).
    Now verifies once and caches the result.
    """
    cache_key = f"client_{client_id}"
    
    # Return cached result if available
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    
    dataset_number = int(get_config("dataset", context=None, default=1, type_converter=int))
    dataset_str = str(dataset_number).zfill(3)

    print(f"[dataset] Preparing pre-partitioned dataset for client {client_id} (dataset choice {dataset_str})")
    
    # First-time setup
    partition_root = f"/app/datasets_{dataset_str}/coco_partitions/client_{client_id}"
    data_yaml = os.path.join(partition_root, f"coco_client_dataset_{dataset_str}.yaml")
    
    # Verify partition exists (only once)
    if not os.path.exists(partition_root):
        raise FileNotFoundError(
            f"âŒ Pre-partitioned data not found for client {client_id}!\n"
            f"Expected location: {partition_root}\n\n"
            f"Please run 03b-partition-dataset.sh BEFORE starting training."
        )
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"âŒ YAML config missing for client {client_id}!\n"
            f"Expected: {data_yaml}\n\n"
            f"The partition may be incomplete. Please re-run 03b-partition-dataset.sh"
        )
    
    # Quick validation (check if directories exist, don't count files)
    train_dir = os.path.join(partition_root, 'images', 'train2017')
    val_dir = os.path.join(partition_root, 'images', 'val2017')
    
    if not os.path.exists(train_dir):
        raise RuntimeError(
            f"âŒ No training directory found for client {client_id}!\n"
            f"Location: {train_dir}\n\n"
            f"Please re-run 03b-partition-dataset.sh"
        )
    
    # Cache the result
    _DATASET_CACHE[cache_key] = (data_yaml, partition_root)
    print(f"âœ… [dataset] Cached dataset info for client {client_id}")
    print(f"   Location: {partition_root}")
    
    return _DATASET_CACHE[cache_key]

@app.train()
def train(msg: Message, context: Context):
    """Training function with guaranteed consistent metric structure and resource monitoring."""
    Path("/app/.healthy").touch()

    ensure_yolo_models_available()

    server_round = int(msg.content["config"].get("server-round", 0))
    client_id = int(get_config("partition-id", context, default=0))
    run_id = str(get_config("run_id", context, default="1"))

    print(f"\n[CLIENT {client_id}] Starting training for round {server_round}")

    # ✅ START OVERALL ROUND MONITORING
    round_monitor = ResourceMonitor(sample_interval=0.5)
    round_monitor.start()
    round_start_time = time.perf_counter()

    received_state = msg.content["arrays"].to_torch_state_dict()
    data_yaml, client_dataset_root = prepare_client_yolo_dataset_prepartitioned(client_id)

    model_size = get_config("yolo_size", context, default="n")
    epochs = int(get_config("local-epochs", context, default=1))
    img = int(get_config("img_size", context, default=640))
    batch = int(get_config("batch_size", context, default=16))

    train_start = time.perf_counter()
    train_status = "FAILED"
    train_error_msg = ""
    round_log = {}
    
    # ✅ START TRAINING PHASE MONITORING
    train_monitor = ResourceMonitor(sample_interval=0.5)
    train_monitor.start()

    try:
        print(f"[CLIENT {client_id}] Calling yolo_train_from_state_and_return_state_dict...")
        new_state, round_log = yolo_train_from_state_and_return_state_dict(
            received_state,
            model_size=model_size,
            client_dataset_yaml=data_yaml,
            epochs=epochs,
            img=img,
            batch=batch,
            run_dir=context.run_config.get("yolo_runs_dir"),
            client_tag=f"client{client_id}",
            round_idx=server_round,
            run_id=run_id,
        )
        print(f"[CLIENT {client_id}] yolo_train returned round_log keys: {sorted(round_log.keys())}")
        print(f"[CLIENT {client_id}] Train metrics - Loss: {round_log.get('loss', 0.0):.4f}, mAP@0.5: {round_log.get('mAP@0.5', 0.0):.4f}, mAP: {round_log.get('mAP', 0.0):.4f}")
        train_status = "SUCCESS"
    except Exception as e:
        print(f"[CLIENT {client_id}] TRAINING FAILED with error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        train_error_msg = f"{type(e).__name__}: {str(e)}"
        new_state = received_state
    
    # ✅ STOP TRAINING PHASE MONITORING
    train_resources = train_monitor.stop()
    print(f"[CLIENT {client_id}] Training resources: CPU peak {train_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {train_resources['per_process']['memory_mb']['peak']:.1f} MB")

    train_time = time.perf_counter() - train_start

    model_record = ArrayRecord(torch_state_dict=new_state, keep_input=True)

    # âœ… CRITICAL: Ensure sent size is never zero
    sent_size = sum(p.nelement() * p.element_size() for p in new_state.values())
    sent_size = max(sent_size, 1)  # Minimum 1 byte
    
    received_size = calculate_message_size_fast(msg)

    # Cache num examples once
    cache_key = f"client_{client_id}_num_examples"
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = max(
            1, len(list(Path(client_dataset_root).glob("images/train2017/*.jpg")))
        )
    num_examples = _DATASET_CACHE[cache_key]

    # âœ… FIXED: Build metrics with explicit float conversion - GUARANTEED CONSISTENT
    # Using direct assignment instead of dict comprehension for maximum reliability
    # NOTE: Only numeric types allowed in MetricRecord per Flower framework requirements
    metrics = {
        "num-examples": float(num_examples),
        "client_id": float(client_id),
        "lr": float(msg.content["config"].get("lr", 0.01)),
        "client_train_time": float(train_time),
        "client_train_loss": float(round_log.get("loss", 0.0)),
        "client_train_acc_mr": float(round_log.get("mr", 0.0)),
        "client_train_acc_mp": float(round_log.get("mp", 0.0)),
        "client_train_acc_mAP@0.5": float(round_log.get("mAP@0.5", 0.0)),
        "client_train_acc_mAP": float(round_log.get("mAP", 0.0)),
        "data_received_from_server": float(received_size),
        "data_sent_to_server": float(sent_size),
        "num_rounds": float(msg.content["config"].get("num_rounds", 0)),
        "server_round_number": float(server_round),
        "round_duration": float(round_log.get("round_duration", 0.0)),
        "round_start_time": float(round_log.get("round_start_time", 0.0)),
        "round_end_time": float(round_log.get("round_end_time", 0.0)),
        # ✅ Training phase resource metrics
        "train_resources_per_process_cpu_peak": float(train_resources['per_process']['cpu_percent']['peak']),
        "train_resources_per_process_cpu_avg": float(train_resources['per_process']['cpu_percent']['avg']),
        "train_resources_per_process_ram_peak_mb": float(train_resources['per_process']['memory_mb']['peak']),
        "train_resources_per_process_ram_avg_mb": float(train_resources['per_process']['memory_mb']['avg']),
        "train_resources_per_process_ram_peak_pct": float(train_resources['per_process']['memory_percent']['peak']),
        "train_resources_per_process_ram_avg_pct": float(train_resources['per_process']['memory_percent']['avg']),
        "train_resources_system_cpu_peak": float(train_resources['system_wide']['cpu_percent']['peak']),
        "train_resources_system_cpu_avg": float(train_resources['system_wide']['cpu_percent']['avg']),
        "train_resources_system_ram_peak_mb": float(train_resources['system_wide']['memory_mb']['peak']),
        "train_resources_system_ram_avg_mb": float(train_resources['system_wide']['memory_mb']['avg']),
        "train_resources_system_ram_peak_pct": float(train_resources['system_wide']['memory_percent']['peak']),
        "train_resources_system_ram_avg_pct": float(train_resources['system_wide']['memory_percent']['avg']),
    }

    status_icon = "✅" if train_status == "SUCCESS" else "❌"
    print(f"[CLIENT {client_id}] {status_icon} Training {train_status}")
    print(f"[CLIENT {client_id}] Sent: {sent_size} bytes, Received: {received_size} bytes")

    # ✅ Memory cleanup after training to prevent accumulation across rounds
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print(f"[CLIENT {client_id}] Memory cleanup completed after training")

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local validation data with guaranteed consistent metrics and resource monitoring."""
    partition_id = context.node_config.get("partition-id", int(get_config("partition-id", context, default=0)))
    client_id = partition_id
    
    print(f"\n[CLIENT {client_id}] Starting evaluation")
    
    # ✅ START OVERALL ROUND MONITORING
    round_monitor = ResourceMonitor(sample_interval=0.5)
    round_monitor.start()
    round_start_time = time.perf_counter()
    
    eval_status = "FAILED"
    eval_error_msg = ""
    val_metrics = {}
    eval_time = 0.0
    checkpoint_path = None
    
    # ✅ START EVALUATION PHASE MONITORING
    eval_monitor = ResourceMonitor(sample_interval=0.5)
    eval_monitor.start()
    eval_phase_start = time.perf_counter()
    
    try:
        # Use the same dataset preparation as train function
        data_yaml, partition_root = prepare_client_yolo_dataset_prepartitioned(client_id)
        
        server_round = msg.content["config"].get("server-round", 0)
        
        # ✅ FIX: Look for actual checkpoint files (best.pt or last.pt)
        weights_dir = os.path.join(
            context.run_config.get("yolo_runs_dir", "runs/train"),
            f"client{client_id}_r{server_round}", "weights"
        )
        
        # Try best.pt first, then last.pt, then any .pt file
        checkpoint_path = None
        candidate_files = [
            os.path.join(weights_dir, "best.pt"),
            os.path.join(weights_dir, "last.pt"),
        ]
        
        for candidate in candidate_files:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                print(f"[CLIENT {client_id}] Found checkpoint: {checkpoint_path}")
                break
        
        if checkpoint_path is None:
            # Fallback: find any .pt file in weights directory
            if os.path.exists(weights_dir):
                pt_files = list(Path(weights_dir).glob("*.pt"))
                if pt_files:
                    checkpoint_path = str(pt_files[0])
                    print(f"[CLIENT {client_id}] Using fallback checkpoint: {checkpoint_path}")
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint found for client {client_id} round {server_round}\n"
                f"Searched in: {weights_dir}\n"
                f"Expected files: best.pt or last.pt"
            )
        
        print(f"[CLIENT {client_id}] Checkpoint exists and is valid: {checkpoint_path}")
        print(f"[CLIENT {client_id}] Calling yolo_evaluate_weights_and_parse_map...")
        
        eval_start_time = time.perf_counter()
        val_metrics = yolo_evaluate_weights_and_parse_map(
            checkpoint_path, data_yaml, 
            img=get_config("img_size", context, default=640),
            run_dir=context.run_config.get("yolo_runs_dir", "runs/train"),
            client_tag=f"client{client_id}",
            round_idx=server_round
        )
        eval_time = time.perf_counter() - eval_start_time
        
        print(f"[CLIENT {client_id}] yolo_evaluate returned metrics: {val_metrics}")
        print(f"[CLIENT {client_id}] Eval metrics - Loss: {val_metrics.get('loss', 0.0):.4f}, mAP@0.5: {val_metrics.get('mAP@0.5', 0.0):.4f}, mAP: {val_metrics.get('mAP', 0.0):.4f}")
        eval_status = "SUCCESS"
    except Exception as e:
        print(f"[CLIENT {client_id}] EVALUATION FAILED with error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        eval_error_msg = f"{type(e).__name__}: {str(e)}"
    
    # ✅ STOP EVALUATION PHASE MONITORING
    eval_resources = eval_monitor.stop()
    print(f"[CLIENT {client_id}] Evaluation resources: CPU peak {eval_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {eval_resources['per_process']['memory_mb']['peak']:.1f} MB")
    
    print(f"[CLIENT {client_id}] Validation metrics: {val_metrics}")
    
    val_cache_key = f"client_{client_id}_num_val_examples"
    if val_cache_key not in _DATASET_CACHE:
        num_val_examples = len(list(Path(partition_root).glob('images/val2017/*.jpg')))
        _DATASET_CACHE[val_cache_key] = num_val_examples
    else:
        num_val_examples = _DATASET_CACHE[val_cache_key]
    
    # ✅ FIXED: Consistent eval metrics structure - explicit dict literal
    # NOTE: Only numeric types allowed in MetricRecord per Flower framework requirements
    metrics = {
        "num-examples": float(max(1, int(num_val_examples))),
        "client_id": float(int(client_id)),
        "client_eval_acc_mr": float(val_metrics.get("mr", 0.0)),
        "client_eval_acc_mp": float(val_metrics.get("mp", 0.0)),
        "client_eval_acc_mAP@0.5": float(val_metrics.get("mAP@0.5", 0.0)),
        "client_eval_acc_mAP": float(val_metrics.get("mAP", 0.0)),
        "client_eval_loss": float(val_metrics.get("loss", 0.0)),
        "client_eval_time": float(eval_time),
        # ✅ Evaluation phase resource metrics
        "eval_resources_per_process_cpu_peak": float(eval_resources['per_process']['cpu_percent']['peak']),
        "eval_resources_per_process_cpu_avg": float(eval_resources['per_process']['cpu_percent']['avg']),
        "eval_resources_per_process_ram_peak_mb": float(eval_resources['per_process']['memory_mb']['peak']),
        "eval_resources_per_process_ram_avg_mb": float(eval_resources['per_process']['memory_mb']['avg']),
        "eval_resources_per_process_ram_peak_pct": float(eval_resources['per_process']['memory_percent']['peak']),
        "eval_resources_per_process_ram_avg_pct": float(eval_resources['per_process']['memory_percent']['avg']),
        "eval_resources_system_cpu_peak": float(eval_resources['system_wide']['cpu_percent']['peak']),
        "eval_resources_system_cpu_avg": float(eval_resources['system_wide']['cpu_percent']['avg']),
        "eval_resources_system_ram_peak_mb": float(eval_resources['system_wide']['memory_mb']['peak']),
        "eval_resources_system_ram_avg_mb": float(eval_resources['system_wide']['memory_mb']['avg']),
        "eval_resources_system_ram_peak_pct": float(eval_resources['system_wide']['memory_percent']['peak']),
        "eval_resources_system_ram_avg_pct": float(eval_resources['system_wide']['memory_percent']['avg']),
    }
    
    status_icon = "✅" if eval_status == "SUCCESS" else "❌"
    print(f"[CLIENT {client_id}] {status_icon} Evaluation {eval_status}")
    
    # ✅ Memory cleanup after evaluation to prevent accumulation across rounds
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print(f"[CLIENT {client_id}] Memory cleanup completed after evaluation")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    Path("/app/.healthy").touch()
    return Message(content=content, reply_to=msg)