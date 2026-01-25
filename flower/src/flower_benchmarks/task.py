# task.py - YOLO and Classification Training/Evaluation
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import time
import os
os.environ["YOLOv5_AUTOINSTALL"] = "false"
import json
import subprocess
import sys
import importlib
from pathlib import Path

def parse_yolo_training_results(run_dir: str, name: str):
    """
    Parse YOLOv5 training metrics from results.csv file.
    Returns dict with loss, precision, recall, mAP@0.5, mAP@0.5:0.95
    """
    import csv
    
    results_csv = Path(run_dir) / name / "results.csv"
    
    if not results_csv.exists():
        print(f"[WARNING] results.csv not found at {results_csv}")
        return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
    
    try:
        # Read the CSV file
        with open(results_csv, 'r') as f:
            # Skip the header comment line if it starts with #
            lines = f.readlines()
            # Filter out comment lines
            data_lines = [line for line in lines if not line.strip().startswith('#')]
            
            if len(data_lines) < 2:  # Need header + at least one data row
                print(f"[WARNING] results.csv has insufficient data")
                return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
            
            # Parse CSV
            reader = csv.DictReader(data_lines)
            rows = list(reader)
            
            if not rows:
                print(f"[WARNING] results.csv is empty")
                return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
        
        # Get the last epoch (final training metrics)
        last_row = rows[-1]
        
        # Strip whitespace from keys
        last_row = {k.strip(): v.strip() for k, v in last_row.items()}
        
        # YOLOv5 results.csv typical columns:
        # epoch, train/box_loss, train/obj_loss, train/cls_loss, 
        # metrics/precision, metrics/recall, metrics/mAP_0.5, metrics/mAP_0.5:0.95
        
        precision = float(last_row.get('metrics/precision', last_row.get('precision', 0.0)))
        recall = float(last_row.get('metrics/recall', last_row.get('recall', 0.0)))
        map50 = float(last_row.get('metrics/mAP_0.5', last_row.get('mAP_0.5', 0.0)))
        map_5095 = float(last_row.get('metrics/mAP_0.5:0.95', last_row.get('mAP_0.5:0.95', 0.0)))
        
        # Calculate total loss (sum of box, obj, cls losses)
        box_loss = float(last_row.get('train/box_loss', last_row.get('box_loss', 0.0)))
        obj_loss = float(last_row.get('train/obj_loss', last_row.get('obj_loss', 0.0)))
        cls_loss = float(last_row.get('train/cls_loss', last_row.get('cls_loss', 0.0)))
        total_loss = box_loss + obj_loss + cls_loss
        
        metrics = {
            "loss": total_loss,
            "mp": precision,
            "mr": recall,
            "mAP@0.5": map50,
            "mAP": map_5095
        }
        
        print(f"[yolo_train] Parsed training metrics from results.csv:")
        print(f"             loss={total_loss:.4f}, P={precision:.4f}, R={recall:.4f}, mAP@0.5={map50:.4f}, mAP={map_5095:.4f}")
        
        return metrics
    except Exception as e:
        print(f"[ERROR] Failed to parse results.csv: {e}")
        import traceback
        traceback.print_exc()
        return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}

def parse_yolo_evaluation_results(run_dir: str, name: str):
    """
    Parse YOLOv5 evaluation metrics from results.csv file.
    Returns dict with loss, precision, recall, mAP@0.5, mAP@0.5:0.95
    """
    import csv
    
    results_csv = Path(run_dir) / name / "results.csv"
    
    if not results_csv.exists():
        print(f"[WARNING] results.csv not found at {results_csv}")
        return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
    
    try:
        # Read the CSV file
        with open(results_csv, 'r') as f:
            # Skip the header comment line if it starts with #
            lines = f.readlines()
            # Filter out comment lines
            data_lines = [line for line in lines if not line.strip().startswith('#')]
            
            if len(data_lines) < 2:  # Need header + at least one data row
                print(f"[WARNING] results.csv has insufficient data")
                return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
            
            # Parse CSV
            reader = csv.DictReader(data_lines)
            rows = list(reader)
            
            if not rows:
                print(f"[WARNING] results.csv is empty")
                return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}
        
        # Get the last epoch (final training metrics)
        last_row = rows[-1]
        
        # Strip whitespace from keys
        last_row = {k.strip(): v.strip() for k, v in last_row.items()}
        
        # Calculate total loss (sum of box, obj, cls losses)
        box_loss = float(last_row.get('val/box_loss', last_row.get('box_loss', 0.0)))
        obj_loss = float(last_row.get('val/obj_loss', last_row.get('obj_loss', 0.0)))
        cls_loss = float(last_row.get('val/cls_loss', last_row.get('cls_loss', 0.0)))
        total_loss = box_loss + obj_loss + cls_loss
        
        metrics = {
            "loss": total_loss
        }
        
        print(f"[yolo_train] Parsed training metrics from results.csv:")
        
        return metrics

    except Exception as e:
        print(f"[ERROR] Failed to parse results.csv: {e}")
        import traceback
        traceback.print_exc()
        return {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}

# keep your existing small CNN for non-detection tasks
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------
# Existing classification transforms/data loaders (unchanged)
# -------------------
fds = None  # Cache FederatedDataset
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, ), (0.5, ))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def load_data(partition_id: int, num_partitions: int, server_round: int):
    """Load partition CIFAR10-like data (existing behavior)."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                           partition_by="label",
                                           seed=server_round, # Change seed each round for more realistic simulation
                                           alpha=0.7)
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    fds = None
    return trainloader, testloader

def train(net, trainloader, epochs, lr, partition_id, device):
    """Train the existing small CNN."""
    round_log = {}
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    start = time.perf_counter()
    round_log["round_start_time"] = start
    for _ in range(epochs):
        round_log["client_id"] = partition_id
        round_log["epoch"] = _ + 1
        round_log["lr"] = lr
        start = time.perf_counter()
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    end = time.perf_counter()
    round_log["round_end_time"] = end
    round_log["round_duration"] = end - start
    round_log["round_loss"] = avg_trainloss
    return avg_trainloss, round_log

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

# -------------------
# YOLOv5 detection helpers (NEW)
# -------------------
from flower_benchmarks.plugins.yolov5.model import save_state_dict_as_yolo_checkpoint, load_yolo_checkpoint_as_state_dict, YoloSizeToPretrained
from flower_benchmarks.plugins.yolov5.dataset import partition_coco128_dir, write_client_dataset_yolo_layout, write_data_yaml

def prepare_client_yolo_dataset(global_coco_root: str, tmp_client_root_base: str, client_id: int,
                                num_clients: int, alpha: float = 0.7, seed: int = 42):
    """
    Partition the COCO128 dataset (if not already done) and prepare the client-specific
    YOLO folder (images + labels) and data yaml. Returns path to data yaml.
    """
    os.makedirs(tmp_client_root_base, exist_ok=True)
    
    # FIX: Create a mapping from arbitrary client_id to sequential partition index
    # Use a deterministic hash to ensure consistency across restarts
    import hashlib
    
    # Get all possible client IDs from environment or use a large range
    max_possible_clients = 150  # VM1: 0-49, VM2: 50-99, VM3: 100-149
    
    # Create deterministic mapping: sort all possible IDs and assign sequential indices
    all_client_ids = list(range(max_possible_clients))
    
    # Find position of current client_id in sorted list
    try:
        partition_idx = all_client_ids.index(int(client_id)) % num_clients
    except ValueError:
        # Fallback for IDs outside range
        partition_idx = int(client_id) % num_clients
    
    print(f"[dataset] Client ID {client_id} mapped to partition index {partition_idx} (total partitions: {num_clients})")
    
    partitions = partition_coco128_dir(global_coco_root, num_clients, alpha=alpha, seed=seed)
    client_partition = partitions[partition_idx]
    
    client_dataset_root = write_client_dataset_yolo_layout(global_coco_root, tmp_client_root_base, client_id, client_partition)
    # create data yaml
    data_yaml = os.path.join(tmp_client_root_base, f"client_{client_id}", "coco_client.yaml")
    write_data_yaml(client_dataset_root, data_yaml)
    return data_yaml, client_dataset_root

def yolo_train_from_state_and_return_state_dict(received_state_dict: dict,
                                                model_size: str,
                                                client_dataset_yaml: str,
                                                epochs: int,
                                                img: int = 640,
                                                batch: int = 16,
                                                run_dir: str = "runs/train",
                                                client_tag: str = "client",
                                                round_idx: int = 0,
                                                run_id: str = "1"):
    """
    1. Save the received_state_dict into a YOLO checkpoint file (received_weights.pt).
    2. Run YOLOv5 train.py as a subprocess using that checkpoint as --weights
    3. After finishing, locate the best/last weights and load them, returning a torch state_dict suitable for ArrayRecord.
    """

    round_log = {}
    round_log["round_start_time"] = time.perf_counter()

    name = f"{client_tag}_r{round_idx}"
    tmp_weights = f"received_weights_{client_tag}_r{round_idx}.pt"
    # Create absolute paths to avoid resolution issues in YOLOv5
    run_dir_abs = os.path.abspath(run_dir)
    full_path = os.path.join(run_dir_abs, name, "weights", tmp_weights)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    tmp_weights = full_path
    
    # save received state dict as YOLO checkpoint
    print(f"[yolo_train] Saving received state_dict to: {tmp_weights}")
    print(f"[yolo_train] State dict has {len(received_state_dict)} keys")
    save_state_dict_as_yolo_checkpoint(received_state_dict, model_size, tmp_weights)
    print(f"[yolo_train] Checkpoint saved. File exists: {os.path.exists(tmp_weights)}")
    if os.path.exists(tmp_weights):
        print(f"[yolo_train] Checkpoint file size: {os.path.getsize(tmp_weights)} bytes")

    cfg = "yolov5s.yaml" if model_size == "s" else "yolov5n.yaml" if model_size == "n" else \
          "yolov5m.yaml" if model_size == "m" else "yolov5l.yaml" if model_size == "l" else \
          "yolov5x.yaml" if model_size == "x" else "yolov5n.yaml"

    # Find absolute path to the cfg file
    possible_cfg_paths = [
        os.path.abspath(cfg),
        os.path.join(os.getcwd(), cfg),
        os.path.join(os.getcwd(), "yolov5", "models", cfg),
        # In the installed package case (sibling directory)
        os.path.join(os.path.dirname(__file__), "..", "yolov5", "models", cfg),
        # Fixed absolute path in Docker image
        os.path.join("/app", "yolov5", "models", cfg)
    ]
    chosen_cfg = None
    for p in possible_cfg_paths:
        print(f"[yolo_train] Checking cfg path: {p}")
        if os.path.exists(p):
            chosen_cfg = p
            break
    if chosen_cfg:
        cfg = chosen_cfg
        print(f"[yolo_train] Using absolute cfg path: {cfg}")
    else:
        print(f"[yolo_train] Warning: Could not find cfg file {cfg} in any of {possible_cfg_paths}")

    # Ensure project root / yolov5 folder is available for imports at runtime
    os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
    run_in_process = False  # default to subprocess to avoid memory spikes
    # Allow env run setting / config 'YOLO_INPROCESS' to enable in-process training if desired
    if os.environ.get('YOLO_INPROCESS', '').lower() in ('1', 'true', 'yes'):
        run_in_process = True

    if run_in_process:
        # Try in-process import & run
        try:
            # disable wandb by default for in-process runs
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_SILENT", "true")
            
            # CRITICAL FIX: Only ensure workspace root is in sys.path
            # Do NOT add yolov5 subdirectory
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)

            # Patch torch.load to disable weights_only for YOLO model loading
            orig_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return orig_load(*args, **kwargs)
            torch.load = patched_load

            ytrain = importlib.import_module("yolov5.train")
            print(f"[yolo_train] Starting in-process YOLOv5 training...")
            try:
                ytrain.run(
                    data=client_dataset_yaml,
                    imgsz=img,
                    batch_size=batch,
                    epochs=epochs,
                    cfg=cfg,
                    weights=tmp_weights,
                    project=str(run_dir),
                    name=name,
                    exist_ok=True,
                    disable_wandb=True,
                    cache="ram"
                )
                print(f"[yolo_train] In-process yolov5.train.run completed successfully.")
            except Exception as e:
                print(f"[yolo_train] In-process yolov5.train.run raised an exception: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

        except Exception as import_exc:
            print(f"[yolo_train] Could not run in-process: {type(import_exc).__name__}: {import_exc}")
            print("[yolo_train] Falling back to subprocess call of `python -m yolov5.train`")
            run_in_process = False

    if not run_in_process:
        # Subprocess training with FIXED PYTHONPATH
        cmd = [
            "python", "-m", "yolov5.train",
            "--img", str(img),
            "--batch-size", str(batch),
            "--epochs", str(epochs),
            "--data", client_dataset_yaml,
            "--weights", tmp_weights,
            "--project", run_dir_abs,
            "--name", name,
            "--exist-ok",
            "--cfg", cfg,
            "--cache", "ram"
        ]
        print(f"[yolo_train] Running subprocess training with command:")
        print(f"[yolo_train] {' '.join(cmd)}")
        
        # FIX: Consistent PYTHONPATH - only parent directory
        env = os.environ.copy()
        cwd = os.getcwd()
        if cwd.endswith('yolov5'):
            cwd = os.path.dirname(cwd)
        env["PYTHONPATH"] = cwd
        env["WANDB_MODE"] = "disabled"  # Changed from offline to disabled
        env["WANDB_SILENT"] = "true"
        env["WANDB_DISABLED"] = "true"  # Extra safety
        env["OMP_NUM_THREADS"] = "8"  # Limit OpenMP threads to avoid resource issues

        # Stream output directly to stdout/stderr so logs are visible in real-time
        proc = subprocess.run(cmd, check=False, env=env)
        print(f"[yolo_train] Subprocess returned with code: {proc.returncode}")
        
        if proc.returncode != 0:
            print("YOLOv5 subprocess training failed.")
        else:
            print(f"[yolo_train] YOLOv5 subprocess training completed successfully.")

    # try to find best.pt under runs/train/<name>/weights/best.pt or last.pt
    out_dir = Path(run_dir) / name / "weights"
    print(f"[yolo_train] Looking for trained weights in: {out_dir}")
    print(f"[yolo_train] Directory exists: {out_dir.exists()}")
    
    if out_dir.exists():
        print(f"[yolo_train] Directory contents: {list(out_dir.iterdir())}")
    
    candidate_best = out_dir / "best.pt"
    candidate_last = out_dir / "last.pt"
    
    print(f"[yolo_train] Checking for best.pt: {candidate_best.exists()}")
    print(f"[yolo_train] Checking for last.pt: {candidate_last.exists()}")
    
    if candidate_best.exists():
        result_ckpt = str(candidate_best)
        print(f"[yolo_train] Using best.pt: {result_ckpt}")
    elif candidate_last.exists():
        result_ckpt = str(candidate_last)
        print(f"[yolo_train] Using last.pt: {result_ckpt}")
    else:
        # fallback: pick any .pt in weights subdir
        if out_dir.exists():
            pt_files = list(out_dir.glob("*.pt"))
            print(f"[yolo_train] PT files found: {pt_files}")
            if pt_files:
                result_ckpt = str(pt_files[-1])
                print(f"[yolo_train] Using fallback pt file: {result_ckpt}")
            else:
                raise FileNotFoundError(f"No trained weights found in {out_dir}")
        else:
            raise FileNotFoundError(f"Training output directory not found: {out_dir}")

    final_state = load_yolo_checkpoint_as_state_dict(result_ckpt)
    round_log["round_end_time"] = time.perf_counter()
    round_log["round_duration"] = round_log["round_end_time"] - round_log["round_start_time"]

    print(f"[yolo_train] Adding training metrics to round_log...")
    training_metrics = parse_yolo_training_results(run_dir_abs, name)
    
    round_log["loss"] = training_metrics["loss"]
    round_log["mp"] = training_metrics["mp"]
    round_log["mr"] = training_metrics["mr"]
    round_log["mAP@0.5"] = training_metrics["mAP@0.5"]
    round_log["mAP"] = training_metrics["mAP"]

    print(f"[yolo_train] round_log now contains: {list(round_log.keys())}")

    try:
        # Remove the received_weights file
        if os.path.exists(tmp_weights):
            os.remove(tmp_weights)
            print(f"Cleaned up temporary checkpoint: {tmp_weights}")
        
        # Optionally keep only best.pt and remove intermediate epochs
        weights_dir = Path(run_dir) / name / "weights"
        if weights_dir.exists():
            for pt_file in weights_dir.glob("epoch*.pt"):
                os.remove(pt_file)
                print(f"Cleaned up intermediate checkpoint: {pt_file}")
                
    except Exception as e:
        print(f"Could not clean up temporary files: {e}")

    return final_state, round_log

def yolo_evaluate_weights_and_parse_map(weights_pt: str, data_yaml: str, img: int = 640, run_dir: str = "runs/train", client_tag: str = "client", round_idx: int = 0):
    """
    Evaluate YOLOv5 weights using in-process `yolov5.val.run` when possible.
    Falls back to subprocess if imports fail.
    Returns a dict containing parsed mAP metrics.
    """
    metrics = {}

    # Ensure PYTHONPATH includes current working directory (parent of yolov5)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    img = int(img)

    print(f"[yolo_eval] Evaluating weights: {weights_pt}")
    print(f"[yolo_eval] Weights file exists: {os.path.exists(weights_pt)}")
    
    try:
        # Patch torch.load to disable weights_only for YOLO model loading
        orig_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return orig_load(*args, **kwargs)
        torch.load = patched_load

        # Import YOLOv5 val module
        print("[yolo_eval] Attempting in-process evaluation...")
        yval = importlib.import_module("yolov5.val")

        # Run in-process validation
        results = yval.run(
            weights=weights_pt,
            data=data_yaml,
            imgsz=img,
            task='val',
            verbose=True,
            workers=3,
            half=False
        )
        
        print(f"[yolo_eval] In-process val.run returned results type: {type(results)}")
        
        # Extract metrics from results
        if isinstance(results, dict):
            print(f"[yolo_eval] Results is dict with keys: {results.keys()}")
            if "metrics" in results:
                metrics_dict = results["metrics"]
                if isinstance(metrics_dict, dict):
                    metrics = {
                        "mp": float(metrics_dict.get('precision', 0.0)),
                        "mr": float(metrics_dict.get('recall', 0.0)),
                        "mAP@0.5": float(metrics_dict.get('mAP_0.5', metrics_dict.get('mAP@0.5', 0.0))),
                        "mAP": float(metrics_dict.get('mAP', metrics_dict.get('mAP_0.5:0.95', 0.0))),
                    }
            else:
                for k, v in results.items():
                    if isinstance(v, (float, int)):
                        metrics[k] = float(v)
                        
        elif isinstance(results, (tuple, list)) and len(results) >= 1:
            print(f"[yolo_eval] Results is tuple/list with length {len(results)}")
            try:
                r = results[0]
                print(f"[yolo_eval] First element type: {type(r)}, value: {r}")
                if isinstance(r, (tuple, list)) and len(r) >= 4:
                    mp = float(r[0])
                    mr = float(r[1])
                    map50 = float(r[2])
                    amap = float(r[3])
                    metrics = {"mp": mp, "mr": mr, "mAP@0.5": map50, "mAP": amap}
                    print(f"[yolo_eval] Extracted metrics from tuple: {metrics}")
                else:
                    for item in results:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, (float, int)):
                                    metrics[k] = float(v)
            except Exception as e_tup:
                print(f"[yolo_eval] Could not parse tuple results: {e_tup}")
                import traceback
                traceback.print_exc()

        name = f"{client_tag}_r{round_idx}"
        loss_metrics = parse_yolo_evaluation_results(run_dir, name)
        metrics["loss"] = loss_metrics.get("loss", 0.0)
        print(f"[yolo_eval] In-process evaluation completed. Metrics: {metrics}")


        return metrics

    except Exception as e:
        print(f"[yolo_eval] In-process validation failed: {type(e).__name__}: {e}")
        print("[yolo_eval] Falling back to subprocess mode...")
        import traceback
        traceback.print_exc()

        # Fallback subprocess call with FIXED PYTHONPATH
        cmd = [
            "python", "-m", "yolov5.val",
            "--weights", weights_pt,
            "--data", data_yaml,
            "--img", str(img),
            "--verbose",
            "--workers", "3"
        ]
        print(f"[yolo_eval] Running subprocess: {' '.join(cmd)}")

        # FIX: Consistent PYTHONPATH
        env = os.environ.copy()
        cwd = os.getcwd()
        if cwd.endswith('yolov5'):
            cwd = os.path.dirname(cwd)
        env["PYTHONPATH"] = cwd
        env["WANDB_MODE"] = "offline"
        env["WANDB_SILENT"] = "true"

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

        print(f"[yolo_eval] Subprocess returned code: {proc.returncode}")
        stdout = proc.stdout
        stderr = proc.stderr
        
        if proc.returncode != 0:
            print("YOLOv5 subprocess evaluation failed.")
            print("=== STDOUT ===")
            print(stdout[-2000:] if len(stdout) > 2000 else stdout)
            print("=== STDERR ===")
            print(stderr[-2000:] if len(stderr) > 2000 else stderr)
        else:
            print("=== Subprocess Output (last 1000 chars) ===")
            print(stdout[-1000:] if len(stdout) > 1000 else stdout)

        # Parse mAP lines from stdout
        metrics = {"mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0, "loss": 0.0}
        
        for line in stdout.splitlines():
            line = line.strip()
            if "all " in line and any(char.isdigit() for char in line):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        metrics["mp"] = float(parts[1])
                        metrics["mr"] = float(parts[2])
                        metrics["mAP@0.5"] = float(parts[3])
                        metrics["mAP"] = float(parts[4])
                        print(f"[yolo_eval] Parsed metrics from line: {metrics}")
                        break
                    except Exception as e_parse:
                        print(f"[yolo_eval] Could not parse metrics from line: {line}, error: {e_parse}")
                        continue

        print(f"[yolo_eval] Subprocess evaluation completed. Final metrics: {metrics}")
        return metrics