"""flower-benchmarks: A Flower / PyTorch app."""

import os
import time
import torch
import numpy as np
import sys
import pickle
import yaml
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flower_benchmarks.task import Net, load_data, test as test_fn, train as train_fn

# Resolve yolov5 imports
import os
import sys

from flower_benchmarks.plugins.yolov5.model import load_yolo_checkpoint_as_state_dict, save_state_dict_as_yolo_checkpoint
from flower_benchmarks.task import yolo_train_from_state_and_return_state_dict, prepare_client_yolo_dataset, yolo_evaluate_weights_and_parse_map
from pathlib import Path

# REMOVED: All Prometheus imports and initialization (lines 16-82)

def get_config(key: str, context: Context, default=None, type_converter=str):
    """
    Get configuration with clear precedence:
    1. Environment variable (highest priority)
    2. Context run_config
    3. Context node_config
    4. Default value (lowest priority)
    """
    # Check environment first
    env_key = key.upper().replace("-", "_")
    if env_key in os.environ:
        value = os.environ[env_key]
        try:
            return type_converter(value)
        except:
            return value
    
    # Check run_config (from server)
    if key in context.run_config:
        return context.run_config[key]
    
    # Check node_config (from federation settings)
    if key in context.node_config:
        return context.node_config[key]
    
    # Return default
    return default

# Flower ClientApp
app = ClientApp()

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def calculate_message_size(msg: Message) -> dict:
    """Calculate size of different components in a message."""
    sizes = {}
    
    # Calculate arrays size (model parameters)
    if "arrays" in msg.content:
        arrays_size = 0
        state_dict = msg.content["arrays"].to_torch_state_dict()
        for param in state_dict.values():
            arrays_size += param.nelement() * param.element_size()
        sizes["arrays_bytes"] = arrays_size
    
    # Calculate metrics size
    if "metrics" in msg.content:
        metrics_size = sys.getsizeof(pickle.dumps(dict(msg.content["metrics"])))
        sizes["metrics_bytes"] = metrics_size
    
    # Calculate config size
    if "config" in msg.content:
        config_size = sys.getsizeof(pickle.dumps(dict(msg.content["config"])))
        sizes["config_bytes"] = config_size
    
    sizes["total_bytes"] = sum(sizes.values())
    return sizes["total_bytes"]

def extract_yolov5_weights_as_arrays(state_dict: dict):
    """Convert a YOLOv5 state_dict (PyTorch tensors) into a dictionary of NumPy arrays."""
    weights_dict = {}

    for key, val in state_dict.items():
        if isinstance(val, torch.Tensor):
            weights_dict[key] = val.cpu().detach().numpy()
        elif isinstance(val, np.ndarray):
            weights_dict[key] = val
        else:
            weights_dict[key] = np.array(val)

    return weights_dict

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data. Supports classification and detection."""
    try:
        Path("/app/.healthy").touch()
    except Exception:
        pass

    received_sizes = calculate_message_size(msg)
    task_type = get_config("task", context, default="classification")

    if task_type == "detection":
        # YOLO detection flow
        num_rounds = msg.content["config"].get("num_rounds", 0)
        server_round = msg.content["config"].get("server-round", 0)
        client_id = get_config("partition-id", context, default=0, type_converter=int)

        print(f"[client_train] Starting train round {server_round} for client {client_id}")
        
        received_state = msg.content["arrays"].to_torch_state_dict()
        
        # CHANGED: Use /app/datasets/coco instead of ./datasets/coco
        coco_root = os.path.abspath(get_config("coco_root", context, default="/app/datasets/coco"))
        tmp_clients_base = os.path.abspath(get_config("yolo_tmp_dir", context, default="./yolov5/tmp_coco_clients"))
        num_clients = get_config("num_clients", context, default=10, type_converter=int)
        
        print(f"[client_train] coco_root: {coco_root}, tmp_clients_base: {tmp_clients_base}, num_clients: {num_clients}")

        print(f"[client_train] Preparing dataset for client {client_id}...")
        data_yaml, client_dataset_root = prepare_client_yolo_dataset(
            coco_root, tmp_clients_base, client_id, num_clients,
            alpha=float(os.environ.get("DIRICHLET_ALPHA", context.run_config.get("dirichlet_alpha", 0.7))),
            seed=context.run_config.get("dirichlet_seed", 42)
        )
        print(f"[client_train] Dataset prepared. data_yaml: {data_yaml}")

        model_size = msg.content.get("config", {}).get("yolo_size") or os.environ.get("YOLO_SIZE", context.run_config.get("yolo_size", "n"))
        epochs = int(os.environ.get("LOCAL_EPOCHS", context.run_config.get("local-epochs", 1)))
        img = int(os.environ.get("IMG_SIZE", context.run_config.get("img_size", 640)))
        batch = int(os.environ.get("BATCH_SIZE", context.run_config.get("batch_size", 16)))
        run_id = str(os.environ.get("RUN_ID", context.run_config.get("run_id", "1")))

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
            run_id=run_id
        )

        # Align returned state to server-sent shapes
        try:
            for k in received_state.keys():
                trained_val = new_state.get(k)
                if trained_val is not None and isinstance(trained_val, torch.Tensor):
                    if trained_val.shape != received_state[k].shape:
                        print(f"[client_train] WARNING: Shape mismatch for {k}: "
                            f"received {received_state[k].shape}, trained {trained_val.shape}")
            final_state = new_state
        except Exception as e:
            print(f"[client_train] Error checking state dict shapes: {e}")
            final_state = new_state

        round_log["num_rounds"] = msg.content["config"]["num_rounds"]
        round_log["server_round_number"] = msg.content["config"]["server-round"]
        round_log["data_received_from_server"] = received_sizes
        
        # Save local trained model
        try:
            tmp_out_ckpt = os.path.join(
                context.run_config.get("yolo_runs_dir", "runs/train"),
                f"client{client_id}_r{server_round}", "weights", 
                f"client{client_id}_r{server_round}_val.pt"
            )
            save_state_dict_as_yolo_checkpoint(final_state, model_size, tmp_out_ckpt)
            load_yolo_checkpoint_as_state_dict(tmp_out_ckpt)
        except Exception as e:
            print(f"Error saving/loading YOLO checkpoint: {e}")

        # Evaluate on training data
        train_eval_metrics = {}
        try:
            train_data_yaml = data_yaml.replace("coco_client.yaml", "coco_client_train_only.yaml")
            import yaml
            if os.path.exists(data_yaml):
                with open(data_yaml, 'r') as f:
                    data_cfg = yaml.safe_load(f)
                data_cfg['val'] = data_cfg.get('train', '')
                with open(train_data_yaml, 'w') as f:
                    yaml.dump(data_cfg, f)
            
            print(f"[client_train] Evaluating training metrics from: {tmp_out_ckpt}")
            train_eval_metrics = yolo_evaluate_weights_and_parse_map(
                tmp_out_ckpt, train_data_yaml, 
                img=context.run_config.get("img_size", 640)
            )
            print(f"[client_train] Training evaluation metrics: {train_eval_metrics}")
        except Exception as e:
            print(f"Warning: Could not evaluate training metrics: {e}")
            train_eval_metrics = {"mr": 0.0, "mp": 0.0, "mAP@0.5": 0.0, "mAP": 0.0, "loss": 0.0}

        # Construct reply
        final_weights = extract_yolov5_weights_as_arrays(new_state)
        import torch as _torch
        torch_state_dict = {k: _torch.tensor(v) if not isinstance(v, _torch.Tensor) else v for k, v in final_weights.items()}
        model_record = ArrayRecord(torch_state_dict=torch_state_dict, keep_input=True)
        
        client_train_time = round_log.get("round_duration", 0.0)
        
        round_log["client_id"] = client_id
        round_log["lr"] = msg.content["config"].get("lr")
        round_log["client_train_time"] = client_train_time
        round_log["client_train_acc_mr"] = _safe_float(train_eval_metrics.get("mr", 0.0))
        round_log["client_train_acc_mp"] = _safe_float(train_eval_metrics.get("mp", 0.0))
        round_log["client_train_acc_mAP@0.5"] = _safe_float(train_eval_metrics.get("mAP@0.5", 0.0))
        round_log["client_train_acc_mAP"] = _safe_float(train_eval_metrics.get("mAP", 0.0))
        round_log["client_train_loss"] = _safe_float(train_eval_metrics.get("loss", 0.0))
        
        metrics = {
            "num-examples": len(list(Path(client_dataset_root).glob('images/train2017/*.jpg'))),
            **round_log
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply_msg = Message(content=content, reply_to=msg)
        sent_sizes = calculate_message_size(reply_msg)
        metrics["data_sent_to_server"] = sent_sizes

        # REMOVED: Prometheus metric setting (lines 479-481)
        
        Path("/app/.healthy").touch()
        return reply_msg
    
    else:
        # Classification flow (unchanged)
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cpu")
        model.to(device)
        partition_id = context.node_config.get("partition-id", int(os.environ.get("PARTITION_ID", os.environ.get("CLIENT_ID", 0))))
        num_partitions = context.node_config.get("num-partitions", int(os.environ.get("NUM_CLIENTS", 1)))
        trainloader, _ = load_data(partition_id, num_partitions, server_round=msg.content["config"].get("server-round"))
        print(f"Client {partition_id}: starting training with {len(trainloader.dataset)} samples.")
        train_loss, round_log = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            partition_id,
            device,
        )
        round_log["num_rounds"] = msg.content["config"]["num_rounds"]
        round_log["server_round_number"] = msg.content["config"]["server-round"]
        round_log["data_received_from_server"] = received_sizes
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
            **round_log
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        reply_msg = Message(content=content, reply_to=msg)
        sent_sizes = calculate_message_size(reply_msg)
        metrics["data_sent_to_server"] = sent_sizes
        return reply_msg


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    task_type = context.run_config.get("task", "classification")

    if task_type == "detection":
        partition_id = context.node_config.get("partition-id", int(get_config("partition-id", context, default=0)))
        num_partitions = get_config("num_clients", context, default=10, type_converter=int)
        tmp_clients_base = os.path.abspath(get_config("yolo_tmp_dir", context, default="./yolov5/tmp_coco_clients"))
        client_id = partition_id
        client_dataset_root = os.path.join(tmp_clients_base, f"client_{client_id}")
        val_yaml = os.path.abspath(os.path.join(client_dataset_root, "coco_client.yaml"))
        current_round = get_config("server-round", context, default=0)

        tmp_out_ckpt = os.path.join(
            get_config("yolo_runs_dir", context, default="runs/train"),
            f"client{client_id}_r{current_round}",
            "weights", f"client{client_id}_r{current_round}_val.pt"
        )

        eval_start_time = time.perf_counter()
        val_metrics = yolo_evaluate_weights_and_parse_map(
            tmp_out_ckpt, val_yaml, 
            img=get_config("img_size", context, default=640)
        )
        eval_end_time = time.perf_counter()
        client_eval_time = eval_end_time - eval_start_time

        print(f"Client {client_id} eval metrics: {val_metrics}")
        
        client_eval_acc_mr = _safe_float(val_metrics.get("mr", 0.0))
        client_eval_acc_mp = _safe_float(val_metrics.get("mp", 0.0))
        client_eval_acc_mAP50 = _safe_float(val_metrics.get("mAP@0.5", 0.0))
        client_eval_acc_mAP = _safe_float(val_metrics.get("mAP", 0.0))
        client_eval_loss = _safe_float(val_metrics.get("loss", 0.0))
        
        metrics = {
            "client_id": client_id,
            "client_eval_acc_mr": client_eval_acc_mr,
            "client_eval_acc_mp": client_eval_acc_mp,
            "client_eval_acc_mAP@0.5": client_eval_acc_mAP50,
            "client_eval_acc_mAP": client_eval_acc_mAP,
            "client_eval_loss": client_eval_loss,
            "client_eval_time": client_eval_time,
            "num-examples": len(list(Path(client_dataset_root).glob('images/val2017/*.jpg'))),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

    else:
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        _, valloader = load_data(partition_id, num_partitions, server_round=msg.content["config"].get("server-round"))

        eval_loss, eval_acc = test_fn(model, valloader, device)

        metrics = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        Path("/app/.healthy").touch()
        return Message(content=content, reply_to=msg)