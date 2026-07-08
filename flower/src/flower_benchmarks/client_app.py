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
    # ResourceMonitor  # COMMENTED: Resource monitoring disabled
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
    
    # Pre-download Arial.ttf using requests to handle 308 redirects that urllib fails on
    font_path = os.path.expanduser("~/.config/Ultralytics/Arial.ttf")
    if not os.path.exists(font_path):
        try:
            os.makedirs(os.path.dirname(font_path), exist_ok=True)
            import requests
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf"
            response = requests.get(url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(response.content)
                print(f"[setup] Pre-downloaded Arial.ttf to {font_path}")
        except Exception as e:
            print(f"[setup] Warning: could not pre-download Arial.ttf: {e}")
            
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


def get_bool_config(key: str, context: Context, default: bool = False) -> bool:
    """Like get_config, but coerces TOML bools / env-var strings to a real bool."""
    raw = get_config(key, context, default=default, type_converter=str)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes")


def run_client_local_hpo(
    received_state,
    model_size: str,
    data_yaml: str,
    img: int,
    client_id: int,
    run_id: str,
    server_round: int,
    n_trials: int,
    epoch_range=(1, 5),
    batch_choices=(8, 16, 32),
):
    """
    Client-side Optuna search for the best (local-epochs, batch_size) pair,
    using each trial's TRAIN mAP@0.5 as the objective (maximize).

    Each trial is a real, disposable YOLO training call on this client's own
    data -- there is no cheaper proxy here, so cost scales linearly with
    n_trials. The winning weights are NOT reused: only the winning
    hyperparameters are returned, and `train()` re-trains for real afterwards
    starting from `received_state` again, so the trials never pollute the
    weights that get sent back to the server.

    Returns: (best_epochs, best_batch, best_map, trials_info)
    """
    import optuna
    import time
    import gc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tag = f"[CLIENT {client_id}][ROUND {server_round}][HPO]"
    print(f"\n{'='*70}")
    print(f"{tag} STARTING LOCAL HPO -- {n_trials} trial(s)")
    print(f"{tag} search space: local_epochs in {list(range(epoch_range[0], epoch_range[1]+1))}, "
          f"batch_size in {list(batch_choices)}")
    print(f"{tag} objective: maximize TRAIN mAP@0.5")
    print(f"{'='*70}\n")

    trials_info = []
    running_best = {"map50": float("-inf"), "epochs": None, "batch": None, "trial": None}
    hpo_start = time.time()

    def objective(trial: "optuna.Trial") -> float:
        trial_epochs = trial.suggest_int("local_epochs", epoch_range[0], epoch_range[1])
        trial_batch = trial.suggest_categorical("batch_size", list(batch_choices))

        print(
            f"{tag} -- Trial {trial.number + 1}/{n_trials} START -- "
            f"local_epochs={trial_epochs}, batch_size={trial_batch}"
        )
        t0 = time.time()

        try:
            _, trial_log = yolo_train_from_state_and_return_state_dict(
                received_state,
                model_size=model_size,
                client_dataset_yaml=data_yaml,
                epochs=trial_epochs,
                img=img,
                batch=trial_batch,
                run_dir="/tmp/client_hpo_runs",
                client_tag=f"client{client_id}_hpo_t{trial.number}",
                round_idx=server_round,
                run_id=run_id,
            )
        except Exception as e:
            print(f"{tag} -- Trial {trial.number + 1}/{n_trials} FAILED ({time.time()-t0:.1f}s): {e}")
            trials_info.append({
                "trial": trial.number, "epochs": trial_epochs, "batch": trial_batch,
                "map50": None, "status": "FAILED", "duration_s": round(time.time() - t0, 1),
            })
            raise optuna.exceptions.TrialPruned()

        trial_map = float(trial_log.get("mAP@0.5", 0.0))
        trial_loss = trial_log.get("train_loss", trial_log.get("loss", None))
        duration = time.time() - t0

        trials_info.append({
            "trial": trial.number, "epochs": trial_epochs, "batch": trial_batch,
            "map50": trial_map, "status": "OK", "duration_s": round(duration, 1),
        })

        is_new_best = trial_map > running_best["map50"]
        if is_new_best:
            running_best.update(map50=trial_map, epochs=trial_epochs, batch=trial_batch, trial=trial.number)

        loss_str = f", train_loss={trial_loss:.4f}" if trial_loss is not None else ""
        print(
            f"{tag} -- Trial {trial.number + 1}/{n_trials} DONE  ({duration:.1f}s) -- "
            f"train_mAP@0.5={trial_map:.4f}{loss_str}"
            f"{'  <-- NEW BEST' if is_new_best else ''}"
        )
        print(
            f"{tag} -- running best so far: epochs={running_best['epochs']}, "
            f"batch={running_best['batch']}, mAP@0.5={running_best['map50']:.4f} "
            f"(trial {running_best['trial']})\n"
        )

        # Memory cleanup between disposable trial runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return trial_map

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=client_id),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_epochs = study.best_params["local_epochs"]
    best_batch = study.best_params["batch_size"]
    best_map = study.best_value
    total_time = time.time() - hpo_start

    # ── Summary table, sorted best-first ────────────────────────────────────
    print(f"{'='*70}")
    print(f"{tag} SEARCH COMPLETE -- {len(trials_info)} trial(s) in {total_time:.1f}s")
    print(f"{'='*70}")
    print(f"  {'Trial':>5}  {'Epochs':>6}  {'Batch':>5}  {'mAP@0.5':>8}  {'Time(s)':>8}  {'Status':>7}")
    sorted_trials = sorted(
        trials_info,
        key=lambda t: (t["map50"] is None, -(t["map50"] or 0)),
    )
    for t in sorted_trials:
        map_str = f"{t['map50']:.4f}" if t["map50"] is not None else "   --  "
        marker = "  <-- BEST" if (t["status"] == "OK" and t["epochs"] == best_epochs
                                   and t["batch"] == best_batch and t["map50"] == best_map) else ""
        print(
            f"  {t['trial']+1:>5}  {t['epochs']:>6}  {t['batch']:>5}  "
            f"{map_str:>8}  {t['duration_s']:>8}  {t['status']:>7}{marker}"
        )
    print(f"{'-'*70}")
    print(f"{tag} BEST -> local_epochs={best_epochs}, batch_size={best_batch}, "
          f"train_mAP@0.5={best_map:.4f}")
    print(f"{'='*70}\n")

    return best_epochs, best_batch, best_map, trials_info

# Flower ClientApp
app = ClientApp()

def run_adaptive_batch_epoch_training(
    received_state,
    model_size: str,
    data_yaml: str,
    img: int,
    client_id: int,
    run_id: str,
    server_round: int,
    batch_min: int,
    batch_max: int,
    max_increases: int,
    rmd_threshold: float,
    rmd_patience: int,
    growth_factor: float,
    max_epochs: int,
):
    """
    Adaptive local batch-size / epoch-count tuning ("ABS"), per the paper's
    per-client algorithm:

      - Start each round at batch_min.
      - Train ONE epoch at a time (yolo_train_from_state_and_return_state_dict
        is called repeatedly with epochs=1, feeding the returned state dict
        back in as the input checkpoint for the next epoch -- it already
        supports this since it's a pure state_dict -> state_dict + round_log
        call).
      - After every epoch, compute the local Relative Model Deviation (RMD):
            rho = ||W_t - W_{t-1}|| / ||W_{t-1}||
        i.e. the global (all-parameters-flattened) relative L2 change in
        weights across that epoch. Low rho => learning has stalled.
      - Bump the batch size when either:
          (a) rho stays below rmd_threshold for `rmd_patience` consecutive
              epochs (stalled learning -- noisy small-batch gradients), or
          (b) this epoch's training loss increased vs. the previous epoch
              (noisy/poor gradient signal).
      - Batch grows multiplicatively (by growth_factor, capped at batch_max)
        each time it's bumped, up to `max_increases` bumps total.
      - If a bump condition fires again after batch is already at batch_max
        (or increases are exhausted), training for this round stops --
        this is what "adjusts the number of epochs" means in practice: the
        epoch count is an OUTPUT of this loop, not a fixed input.
      - `max_epochs` is a hard safety ceiling so a pathological case can
        never hang a round indefinitely.

    Returns: (final_state_dict, final_round_log, epochs_run, final_batch, increases_used)
    """
    def _flat_norm_and_delta(state_a, state_b):
        """Relative L2 change between two state dicts, flattened & concatenated."""
        total_sq_delta = 0.0
        total_sq_norm = 0.0
        for k, v_b in state_b.items():
            v_a = state_a.get(k)
            if v_a is None or not torch.is_tensor(v_b) or not torch.is_tensor(v_a):
                continue
            v_a_f = v_a.detach().float()
            v_b_f = v_b.detach().float()
            total_sq_delta += torch.sum((v_b_f - v_a_f) ** 2).item()
            total_sq_norm += torch.sum(v_a_f ** 2).item()
        denom = total_sq_norm ** 0.5
        return (total_sq_delta ** 0.5) / denom if denom > 0 else 0.0

    batch = int(batch_min)
    increases_used = 0
    stalled_streak = 0
    prev_loss = None
    current_state = received_state
    final_round_log = {}
    epochs_run = 0

    tag = f"[CLIENT {client_id}] [ABS]"
    print(f"{tag} Starting adaptive batch/epoch training: batch_min={batch_min}, "
          f"batch_max={batch_max}, max_increases={max_increases}, "
          f"rmd_threshold={rmd_threshold}, rmd_patience={rmd_patience}")

    for epoch_num in range(1, max_epochs + 1):
        epoch_tag = f"client{client_id}_abs"
        new_state, round_log = yolo_train_from_state_and_return_state_dict(
            current_state,
            model_size=model_size,
            client_dataset_yaml=data_yaml,
            epochs=1,
            img=img,
            batch=batch,
            run_dir=f"./yolov5/runs/train_abs",
            client_tag=epoch_tag,
            round_idx=f"{server_round}_e{epoch_num}",
            run_id=run_id,
        )

        loss = round_log.get("loss", 0.0)
        rho = _flat_norm_and_delta(current_state, new_state)

        print(f"{tag} epoch {epoch_num}: batch={batch}, loss={loss:.4f}, rho={rho:.6f} "
              f"(increases_used={increases_used}/{max_increases})")

        current_state = new_state
        final_round_log = round_log
        epochs_run = epoch_num

        loss_regressed = (prev_loss is not None) and (loss > prev_loss)
        if rho < rmd_threshold:
            stalled_streak += 1
        else:
            stalled_streak = 0
        stalled = stalled_streak >= rmd_patience

        should_bump = loss_regressed or stalled

        if should_bump:
            if increases_used >= max_increases or batch >= batch_max:
                print(f"{tag} Stall/regression detected but batch already at cap "
                      f"or increases exhausted -- ending round after {epochs_run} epoch(s).")
                prev_loss = loss
                break
            new_batch = min(batch_max, max(batch + 1, int(round(batch * growth_factor))))
            print(f"{tag} {'Loss regressed' if loss_regressed else 'Learning stalled'} "
                  f"-- growing batch {batch} -> {new_batch}")
            batch = new_batch
            increases_used += 1
            stalled_streak = 0

        prev_loss = loss

    print(f"{tag} Finished: epochs_run={epochs_run}, final_batch={batch}, "
          f"increases_used={increases_used}")
    return current_state, final_round_log, epochs_run, batch, increases_used


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

def prepare_client_yolo_dataset_prepartitioned(client_id: int, context: Context = None):
    """
    OPTIMIZED: Use pre-partitioned dataset with caching.
    Previous version verified file system every round (expensive with 10k+ images).
    Now verifies once and caches the result.
    """
    cache_key = f"client_{client_id}"
    
    # Return cached result if available
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]
    
    dataset_number = int(get_config("dataset", context=context, default=1, type_converter=int))
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

    # COMMENTED: Resource monitoring disabled
    # ✅ START OVERALL ROUND MONITORING
    # round_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # round_monitor.start()  # COMMENTED: Resource monitoring disabled
    # round_start_time = time.perf_counter()  # COMMENTED: Resource monitoring disabled

    received_state = msg.content["arrays"].to_torch_state_dict()
    data_yaml, client_dataset_root = prepare_client_yolo_dataset_prepartitioned(client_id, context=context)

    model_size = get_config("yolo_size", context, default="n")
    # Read local-epochs and batch_size from msg.content["config"] first so the
    # server can override them per-trial during Optuna HPO without restarting
    # the run.  Falls back to run_config / node_config / default as before.
    # epochs = int(get_config("local-epochs", context, default=1))
    # batch = int(get_config("batch_size", context, default=16))
    epochs = int(msg.content["config"].get(
        "local-epochs", get_config("local-epochs", context, default=3)))
    img = int(get_config("img_size", context, default=640))
    batch = int(msg.content["config"].get(
        "batch_size",   get_config("batch_size",   context, default=16)))

    # =================================================================
    # CLIENT-SIDE LOCAL HPO (optional, independent of server-side HPO)
    # When enabled, this client runs its own small Optuna study every round
    # to pick (epochs, batch) that maximize ITS OWN train mAP@0.5, instead
    # of just using the epochs/batch the server sent. The search trials are
    # disposable -- they never touch `received_state` in place -- so the
    # real training call right after still starts from the fresh weights
    # the server sent this round.
    # =================================================================
    client_hpo_enabled = get_bool_config("client_hpo_enabled", context, default=False)
    if "client_hpo_enabled" in msg.content["config"]:
        client_hpo_enabled = bool(msg.content["config"].get("client_hpo_enabled"))

    client_hpo_trials = int(msg.content["config"].get(
        "client_hpo_trials", get_config("client_hpo_trials", context, default=3)))

    # =================================================================
    # ADAPTIVE BATCH/EPOCH TUNING (ABS) -- mutually exclusive with the
    # Optuna-based client_hpo above. If both are enabled, ABS wins and
    # client-side HPO is skipped for this round (a warning is printed).
    # =================================================================
    adaptive_batch_enabled = get_bool_config("adaptive_batch_enabled", context, default=False)
    if "adaptive_batch_enabled" in msg.content["config"]:
        adaptive_batch_enabled = bool(msg.content["config"].get("adaptive_batch_enabled"))

    adaptive_batch_epochs_run = 0
    adaptive_batch_increases_used = 0

    if adaptive_batch_enabled and client_hpo_enabled:
        print(f"[CLIENT {client_id}] [WARN] Both adaptive_batch_enabled and "
              f"client_hpo_enabled are true -- adaptive batch/epoch tuning "
              f"takes priority; client-side HPO is skipped this round.")
        client_hpo_enabled = False

    client_hpo_best_map = None
    client_hpo_trials_run = 0

    if adaptive_batch_enabled:
        batch_min     = int(get_config("adaptive_batch_min", context, default=8))
        batch_max     = int(get_config("adaptive_batch_max", context, default=64))
        max_increases = int(get_config("adaptive_batch_max_increases", context, default=4))
        rmd_threshold = float(get_config("adaptive_batch_rmd_threshold", context, default=0.01))
        rmd_patience  = int(get_config("adaptive_batch_rmd_patience", context, default=2))
        growth_factor = float(get_config("adaptive_batch_growth_factor", context, default=2.0))
        max_epochs    = int(get_config("adaptive_batch_max_epochs", context, default=10))

        (received_state, round_log_abs, adaptive_batch_epochs_run,
         batch, adaptive_batch_increases_used) = run_adaptive_batch_epoch_training(
            received_state,
            model_size=model_size,
            data_yaml=data_yaml,
            img=img,
            client_id=client_id,
            run_id=run_id,
            server_round=server_round,
            batch_min=batch_min,
            batch_max=batch_max,
            max_increases=max_increases,
            rmd_threshold=rmd_threshold,
            rmd_patience=rmd_patience,
            growth_factor=growth_factor,
            max_epochs=max_epochs,
        )
        # `received_state` now holds the fully-trained weights for this round
        # (already run epoch-by-epoch above), and `epochs`/`batch` reflect what
        # was actually used -- feed these into the normal metrics/response path
        # below instead of the single multi-epoch training call.
        epochs = adaptive_batch_epochs_run
        round_log = round_log_abs

    if client_hpo_enabled and client_hpo_trials > 0:
        print(f"[CLIENT {client_id}] [HPO] Client-side HPO enabled "
              f"({client_hpo_trials} trials) -- searching local-epochs/batch_size")
        try:
            hpo_epochs, hpo_batch, client_hpo_best_map, _trials_info = run_client_local_hpo(
                received_state,
                model_size=model_size,
                data_yaml=data_yaml,
                img=img,
                client_id=client_id,
                run_id=run_id,
                server_round=server_round,
                n_trials=client_hpo_trials,
            )
            epochs, batch = hpo_epochs, hpo_batch
            client_hpo_trials_run = client_hpo_trials
        except Exception as e:
            print(f"[CLIENT {client_id}] [HPO] Client-side HPO failed, "
                  f"falling back to server-provided epochs/batch: {e}")

    # Patch hyp.scratch-low.yaml so YOLO actually trains with the chosen lr.
    # Without this patch, YOLO reads lr0 from the YAML that was baked at
    # deploy time and silently ignores whatever lr Optuna suggested.
    lr = float(msg.content["config"].get("lr", get_config("lr", context, default=0.001)))
    hyp_path = os.path.join(os.getcwd(), "yolov5", "data", "hyps", "hyp.scratch-low.yaml")
    if os.path.exists(hyp_path):
        try:
            import re
            with open(hyp_path, "r") as _f:
                _hyp = _f.read()
            _hyp = re.sub(r'(lr0:\s*)[0-9.eE+-]+', rf'\g<1>{lr}', _hyp)
            with open(hyp_path, "w") as _f:
                _f.write(_hyp)
            print(f"[CLIENT {client_id}] Patched hyp.scratch-low.yaml: lr0={lr}")
        except Exception as _e:
            print(f"[CLIENT {client_id}] Warning: could not patch hyp YAML: {_e}")

    train_start = time.perf_counter()
    train_status = "FAILED"
    train_error_msg = ""
    if not adaptive_batch_enabled:
        round_log = {}
    
    # COMMENTED: Resource monitoring disabled
    # ✅ START TRAINING PHASE MONITORING
    # train_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # train_monitor.start()  # COMMENTED: Resource monitoring disabled

    try:
        if adaptive_batch_enabled:
            # ABS already ran the full epoch-by-epoch loop above; `received_state`
            # holds the final trained weights and `round_log` the last epoch's log.
            new_state = received_state
            print(f"[CLIENT {client_id}] ABS training already complete "
                  f"(epochs_run={epochs}, final_batch={batch}) -- skipping normal train call.")
            print(f"[CLIENT {client_id}] Train metrics - Loss: {round_log.get('loss', 0.0):.4f}, mAP@0.5: {round_log.get('mAP@0.5', 0.0):.4f}, mAP: {round_log.get('mAP', 0.0):.4f}")

            # ---------------------------------------------------------------
            # FIX: ABS trains into its own per-epoch run dirs
            # (./yolov5/runs/train_abs/client{id}_abs_r{round}_e{n}/weights),
            # but evaluate() unconditionally looks for a checkpoint at
            # {yolo_runs_dir}/client{id}_r{round}/weights/{best,last}.pt --
            # the same path the non-ABS path writes to. Without this, ABS
            # rounds always fail evaluation with a FileNotFoundError.
            # Write the final ABS weights to that exact expected location.
            # ---------------------------------------------------------------
            eval_run_dir = context.run_config.get("yolo_runs_dir", "runs/train")
            eval_weights_dir = os.path.join(eval_run_dir, f"client{client_id}_r{server_round}", "weights")
            os.makedirs(eval_weights_dir, exist_ok=True)
            eval_checkpoint_path = os.path.join(eval_weights_dir, "best.pt")
            try:
                save_state_dict_as_yolo_checkpoint(new_state, model_size, eval_checkpoint_path)
                print(f"[CLIENT {client_id}] [ABS] Saved final checkpoint for evaluate() at: {eval_checkpoint_path}")
            except Exception as _save_exc:
                print(f"[CLIENT {client_id}] [ABS] WARNING: could not save eval checkpoint "
                      f"at {eval_checkpoint_path}: {_save_exc}")
        else:
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
        raise e
    
    # COMMENTED: Resource monitoring disabled
    # ✅ STOP TRAINING PHASE MONITORING
    # train_resources = train_monitor.stop()  # COMMENTED: Resource monitoring disabled
    # print(f"[CLIENT {client_id}] Training resources: CPU peak {train_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {train_resources['per_process']['memory_mb']['peak']:.1f} MB")  # COMMENTED: Resource monitoring disabled

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
        # Client-side local HPO visibility (0/-1 when disabled for this round)
        "client_hpo_enabled": float(1.0 if client_hpo_enabled else 0.0),
        "client_hpo_trials_run": float(client_hpo_trials_run),
        "client_hpo_best_local_epochs": float(epochs),
        "client_hpo_best_batch_size": float(batch),
        "client_hpo_best_train_mAP@0.5": float(client_hpo_best_map if client_hpo_best_map is not None else -1.0),
        # Adaptive batch/epoch (ABS) visibility (0 when disabled for this round)
        "adaptive_batch_enabled": float(1.0 if adaptive_batch_enabled else 0.0),
        "adaptive_batch_epochs_run": float(adaptive_batch_epochs_run),
        "adaptive_batch_final_batch_size": float(batch if adaptive_batch_enabled else -1.0),
        "adaptive_batch_increases_used": float(adaptive_batch_increases_used),
        # COMMENTED: Resource monitoring disabled
        # Training phase resource metrics
        # "train_resources_per_process_cpu_peak": float(train_resources['per_process']['cpu_percent']['peak']),
        # "train_resources_per_process_cpu_avg": float(train_resources['per_process']['cpu_percent']['avg']),
        # "train_resources_per_process_ram_peak_mb": float(train_resources['per_process']['memory_mb']['peak']),
        # "train_resources_per_process_ram_avg_mb": float(train_resources['per_process']['memory_mb']['avg']),
        # "train_resources_per_process_ram_peak_pct": float(train_resources['per_process']['memory_percent']['peak']),
        # "train_resources_per_process_ram_avg_pct": float(train_resources['per_process']['memory_percent']['avg']),
        # "train_resources_system_cpu_peak": float(train_resources['system_wide']['cpu_percent']['peak']),
        # "train_resources_system_cpu_avg": float(train_resources['system_wide']['cpu_percent']['avg']),
        # "train_resources_system_ram_peak_mb": float(train_resources['system_wide']['memory_mb']['peak']),
        # "train_resources_system_ram_avg_mb": float(train_resources['system_wide']['memory_mb']['avg']),
        # "train_resources_system_ram_peak_pct": float(train_resources['system_wide']['memory_percent']['peak']),
        # "train_resources_system_ram_avg_pct": float(train_resources['system_wide']['memory_percent']['avg']),
    }

    status_icon = "✅" if train_status == "SUCCESS" else "❌"
    print(f"[CLIENT {client_id}] {status_icon} Training {train_status}")
    print(f"[CLIENT {client_id}] Sent: {sent_size} bytes, Received: {received_size} bytes")

    # Memory cleanup after training to prevent accumulation across rounds
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    del received_state, new_state
    gc.collect()
    # COMMENTED: Resource monitoring disabled
    # round_monitor.stop()  # COMMENTED: Resource monitoring disabled
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

    run_dir = context.run_config.get("yolo_runs_dir", "runs/train")

    # COMMENTED: Resource monitoring disabled
    # ✅ START OVERALL ROUND MONITORING
    # round_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # round_monitor.start()  # COMMENTED: Resource monitoring disabled
    # round_start_time = time.perf_counter()  # COMMENTED: Resource monitoring disabled
    
    eval_status = "FAILED"
    eval_error_msg = ""
    val_metrics = {}
    eval_time = 0.0
    checkpoint_path = None
    
    # COMMENTED: Resource monitoring disabled
    # ✅ START EVALUATION PHASE MONITORING
    # eval_monitor = ResourceMonitor(sample_interval=0.5)  # COMMENTED: Resource monitoring disabled
    # eval_monitor.start()  # COMMENTED: Resource monitoring disabled
    # eval_phase_start = time.perf_counter()  # COMMENTED: Resource monitoring disabled
    
    try:
        # Use the same dataset preparation as train function
        data_yaml, partition_root = prepare_client_yolo_dataset_prepartitioned(client_id, context=context)
        
        server_round = msg.content["config"].get("server-round", 0)
        
        # ✅ FIX: Look for actual checkpoint files (best.pt or last.pt)
        weights_dir = os.path.join(
            run_dir,
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
            run_dir=run_dir,
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
        raise e
    
    if val_metrics is None:
        print(f"[CLIENT {client_id}] WARNING: val_metrics is None, defaulting to zeros")
        val_metrics = {"loss": 0.0, "mp": 0.0, "mr": 0.0, "mAP@0.5": 0.0, "mAP": 0.0}

    
    # COMMENTED: Resource monitoring disabled
    # ✅ STOP EVALUATION PHASE MONITORING
    # eval_resources = eval_monitor.stop()  # COMMENTED: Resource monitoring disabled
    # print(f"[CLIENT {client_id}] Evaluation resources: CPU peak {eval_resources['per_process']['cpu_percent']['peak']:.1f}%, RAM peak {eval_resources['per_process']['memory_mb']['peak']:.1f} MB")  # COMMENTED: Resource monitoring disabled
    
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
        # COMMENTED: Resource monitoring disabled
        # Evaluation phase resource metrics
        # "eval_resources_per_process_cpu_peak": float(eval_resources['per_process']['cpu_percent']['peak']),
        # "eval_resources_per_process_cpu_avg": float(eval_resources['per_process']['cpu_percent']['avg']),
        # "eval_resources_per_process_ram_peak_mb": float(eval_resources['per_process']['memory_mb']['peak']),
        # "eval_resources_per_process_ram_avg_mb": float(eval_resources['per_process']['memory_mb']['avg']),
        # "eval_resources_per_process_ram_peak_pct": float(eval_resources['per_process']['memory_percent']['peak']),
        # "eval_resources_per_process_ram_avg_pct": float(eval_resources['per_process']['memory_percent']['avg']),
        # "eval_resources_system_cpu_peak": float(eval_resources['system_wide']['cpu_percent']['peak']),
        # "eval_resources_system_cpu_avg": float(eval_resources['system_wide']['cpu_percent']['avg']),
        # "eval_resources_system_ram_peak_mb": float(eval_resources['system_wide']['memory_mb']['peak']),
        # "eval_resources_system_ram_avg_mb": float(eval_resources['system_wide']['memory_mb']['avg']),
        # "eval_resources_system_ram_peak_pct": float(eval_resources['system_wide']['memory_percent']['peak']),
        # "eval_resources_system_ram_avg_pct": float(eval_resources['system_wide']['memory_percent']['avg']),
    }
    
    status_icon = "✅" if eval_status == "SUCCESS" else "❌"
    print(f"[CLIENT {client_id}] {status_icon} Evaluation {eval_status}")
    
    # Memory cleanup after evaluation to prevent accumulation across rounds
    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # COMMENTED: Resource monitoring disabled
    # round_monitor.stop()  # COMMENTED: Resource monitoring disabled
    import shutil
    prev_run = Path(run_dir) / f"client{client_id}_r{server_round - 1}"
    if prev_run.exists():
        shutil.rmtree(prev_run)
    gc.collect()
    print(f"[CLIENT {client_id}] Memory cleanup completed after evaluation")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    Path("/app/.healthy").touch()
    return Message(content=content, reply_to=msg)