#!/usr/bin/env python3
"""
Generate deterministic Dirichlet partitions for federated COCO dataset.
Supports generic dataset configurations with varying IID/non-IID distributions.
Each client gets random train size and random val size (independently).
Uses Flower Datasets partitioning logic on pre-downloaded GCS data.
"""

import sys
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import subprocess

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Skipping plots.")

# COCO 80 class names (in order 0-79)
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_alpha_for_client(client_id: int, seed: int, iid_clients: List[int], dataset_id_hash: int = 0) -> float:
    """
    Determine alpha based on whether client is in IID list.
    
    Args:
        client_id: Client ID
        seed: Random seed
        iid_clients: List of client IDs that should be IID (uniform distribution)
        dataset_id_hash: Hash of dataset_id for seed variation (default 0)
    
    Returns:
        alpha: 1e6 for IID clients, random [0.5, 1.5] for non-IID clients
    """
    # Use deterministic seed per client to ensure reproducibility
    # Ensure seed is within NumPy's valid range (0 to 2^32 - 1)
    safe_seed = (seed + dataset_id_hash * 1000 + client_id) % (2**32)
    np.random.seed(safe_seed)
    
    if client_id in iid_clients:
        return 1e6  # IID
    else:
        return float(np.random.uniform(0.5, 1.5))  # Non-IID


def get_gcs_image_list(bucket_name: str, split: str) -> List[str]:
    """Get list of image filenames from GCS bucket. Uses cache if available."""
    cache_file = Path(f"/app/coco_image_list_{split}.txt")
    
    # Check if cache exists and is recent (less than 1 hour old)
    if cache_file.exists():
        import time
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < 3600:  # 1 hour
            print(f"✓ Using cached image list for {split} (age: {int(file_age/60)} minutes)")
            with open(cache_file, 'r') as f:
                images = [line.strip() for line in f if line.strip()]
            print(f"Found {len(images)} images in {split} (from cache)")
            return sorted(images)
    
    gcs_path = f"gs://{bucket_name}/coco/images/{split}/"
    print(f"Fetching image list from {gcs_path}...")
    
    try:
        result = subprocess.run(
            ["gsutil", "ls", gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract filenames from full GCS paths
        images = []
        for line in result.stdout.strip().split('\n'):
            if line.endswith('.jpg'):
                filename = line.split('/')[-1]
                images.append(filename)
        
        # Save to cache
        with open(cache_file, 'w') as f:
            for img in images:
                f.write(f"{img}\n")
        
        print(f"Found {len(images)} images in {split}")
        return sorted(images)
    
    except subprocess.CalledProcessError as e:
        print(f"Error fetching from GCS: {e}")
        print(f"STDERR: {e.stderr}")
        sys.exit(1)


def get_gcs_label_mapping(bucket_name: str, images: List[str], split: str) -> Dict[str, List[int]]:
    """
    Download label files from GCS and create mapping: image -> [class_ids].
    Only returns images that have corresponding labels.
    Reuses existing labels if already downloaded.
    """
    print(f"Preparing labels for {len(images)} images from {split}...")

    temp_dir = Path(f"/app/coco_labels_{split}")
    labels_dir = temp_dir / split
    gcs_labels_path = f"gs://{bucket_name}/coco/labels/{split}/"

    # Download labels if directory doesn't exist
    if not labels_dir.exists():
        print(f"Downloading labels from {gcs_labels_path} ...")
        temp_dir.mkdir(exist_ok=True, parents=True)
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", "-n", gcs_labels_path, str(temp_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error downloading labels: {e}")
            sys.exit(1)
    else:
        print(f"✓ Labels directory exists. Skipping download.")

    # Get list of available label files
    available_label_files = list(labels_dir.glob("*.txt"))
    available_labels = {label_file.stem for label_file in available_label_files}
    
    print(f"Found {len(available_labels)} label files in {split}")

    # Filter images to only those with labels
    images_with_labels = [img for img in images if img.replace(".jpg", "") in available_labels]
    
    images_without_labels = len(images) - len(images_with_labels)
    if images_without_labels > 0:
        print(f"⚠️  Filtered out {images_without_labels} images without labels")
    print(f"✓ Using {len(images_with_labels)} images with labels for partitioning")

    # Parse labels only for images with labels
    image_to_classes = {}
    for img_name in images_with_labels:
        label_name = img_name.replace(".jpg", ".txt")
        label_path = labels_dir / label_name

        classes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            classes.append(int(float(parts[0])))
                        except ValueError:
                            pass

        image_to_classes[img_name] = classes

    print(f"Parsed labels for {len(image_to_classes)} images")
    return image_to_classes


def generate_client_configs(num_clients: int, seed: int, dataset_id_hash: int,
                            min_train: int, max_train: int,
                            min_val: int, max_val: int,
                            iid_clients: List[int]) -> List[Dict]:
    """
    Generate configuration for each client based on IID client list.
    Train and val sizes are independently randomized.
    
    Args:
        num_clients: Total number of clients
        seed: Random seed
        dataset_id_hash: Hash of dataset_id for seed variation
        min_train: Minimum training images per client
        max_train: Maximum training images per client
        min_val: Minimum validation images per client
        max_val: Maximum validation images per client
        iid_clients: List of client IDs that should be IID
    
    Returns:
        List of client configurations
    """
    # Ensure seed is within NumPy's valid range (0 to 2^32 - 1)
    safe_seed = (seed + dataset_id_hash * 10000) % (2**32)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    configs = []
    
    for client_id in range(num_clients):
        # Random train size
        n_train = np.random.randint(min_train, max_train + 1)
        # Random val size (independent)
        n_val = np.random.randint(min_val, max_val + 1)
        
        # Get alpha based on whether client is IID
        alpha = get_alpha_for_client(client_id, seed, iid_clients, dataset_id_hash)
        
        configs.append({
            'client_id': client_id,
            'alpha': alpha,
            'n_train': n_train,
            'n_val': n_val
        })
    
    return configs

def partition_images_dirichlet_balanced(
    images: List[str],
    image_to_classes: Dict[str, List[int]],
    client_configs: List[Dict],
    seed: int,
    source_split: str,
    target_key: str,  # 'n_train' or 'n_val'
    min_classes_per_iid_client: int = 65
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Partition dataset using client-wise target sizes and true Dirichlet class propensities.
    Guarantees target sizes are respected up to global split capacity constraints.
    """
    print(f"\n[BALANCED PARTITION] Starting balanced partitioning for {source_split} (target: {target_key})...")
    np.random.seed(seed)
    random.seed(seed)
    
    # Build class -> images mapping
    class_to_images = defaultdict(list)
    for img, classes in image_to_classes.items():
        if len(classes) == 0:
            continue
        primary_class = classes[0]
        class_to_images[primary_class].append(img)
    
    all_classes = sorted(class_to_images.keys())
    num_clients = len(client_configs)
    
    print(f"[{source_split}] Found {len(all_classes)} classes")
    print(f"[{source_split}] Total images with labels: {sum(len(imgs) for imgs in class_to_images.values())}")
    print(f"[{source_split}] Images per class - Min: {min(len(imgs) for imgs in class_to_images.values()) if class_to_images else 0}, Max: {max(len(imgs) for imgs in class_to_images.values()) if class_to_images else 0}")
    
    # Identify IID and Non-IID clients
    iid_clients = []
    non_iid_clients = []
    
    for cfg in client_configs:
        cid = cfg['client_id']
        alpha = cfg['alpha']
        
        if alpha is None:
            print(f"ERROR: Client {cid} has None alpha!")
            sys.exit(1)
            
        if alpha > 1000:
            iid_clients.append(cid)
        else:
            non_iid_clients.append(cid)
            
    print(f"[{source_split}] IID Clients: {iid_clients}")
    print(f"[{source_split}] Non-IID Clients: {non_iid_clients}")
    print(f"[{source_split}] PHASE 1: Ensuring minimum class coverage ({min_classes_per_iid_client} classes for IID)...")
    
    # Initialize assignments
    client_pools = {i: [] for i in range(num_clients)}
    client_current_sizes = {i: 0 for i in range(num_clients)}
    client_targets = {cfg['client_id']: cfg[target_key] for cfg in client_configs}
    
    # PHASE 1: Minimum coverage for first N classes (Only applies to IID clients if present)
    classes_to_cover = all_classes[:min_classes_per_iid_client]
    for cls in classes_to_cover:
        imgs = list(set(class_to_images[cls]))
        random.shuffle(imgs)
        
        if len(imgs) == 0:
            continue
        
        for i, cid in enumerate(iid_clients):
            if i < len(imgs):
                img = imgs[i]
                client_pools[cid].append((img, source_split))
                client_current_sizes[cid] += 1
    
    print(f"[{source_split}] Phase 1 complete. Average size: {sum(client_current_sizes.values()) / len(client_current_sizes) if client_current_sizes else 0:.1f} images")
    print(f"[{source_split}] Phase 1 client sizes: {client_current_sizes}")
    print(f"[{source_split}] PHASE 2: Distributing remaining images via Dirichlet propensities...")
    
    # Generate true Dirichlet propensities per class across ALL clients
    alphas = [cfg['alpha'] for cfg in client_configs]
    class_propensities = {}
    for cls in all_classes:
        class_propensities[cls] = np.random.dirichlet(alphas)
        
    # Gather all unassigned images from Phase 1
    assigned_images = set()
    for img_list in client_pools.values():
        for img, _ in img_list:
            assigned_images.add(img)
            
    all_remaining_items = []
    all_items = []  # Store ALL items for potential replacement reuse
    for cls in all_classes:
        for img in class_to_images[cls]:
            all_items.append((img, cls))
            if img not in assigned_images:
                all_remaining_items.append((img, cls))
                
    # Globally shuffle items to distribute classes fairly and avoid order bias
    random.shuffle(all_remaining_items)
    
    client_assigned_images = {i: set() for i in range(num_clients)}
    for cid, items in client_pools.items():
        for img, _ in items:
            client_assigned_images[cid].add(img)

    # Dynamic allocation loop matching propensities against client remaining budgets
    item_index = 0
    consecutive_failures = 0
    while True:
        # Determine which clients still need data
        active_clients = [cid for cid in range(num_clients) if client_current_sizes[cid] < client_targets[cid]]
        if not active_clients:
            break  # Every single client has perfectly reached its target profile size!
            
        if item_index >= len(all_remaining_items):
            if source_split == "val2017":
                all_remaining_items = all_items.copy()
                random.shuffle(all_remaining_items)
                item_index = 0
                if consecutive_failures > len(all_remaining_items) * 2:
                    print(f"[{source_split}] WARNING: Could not fulfill all targets even with sharing (all clients have all images). Breaking.")
                    break
            else:
                break
                
        img, cls = all_remaining_items[item_index]
        item_index += 1
        
        # Filter active clients to those who DON'T already have this image
        eligible_clients = [cid for cid in active_clients if img not in client_assigned_images[cid]]
        
        if not eligible_clients:
            # No active client can take this image (they all have it)
            consecutive_failures += 1
            continue
            
        # Extract and isolate class propensities for eligible clients only
        propensities = class_propensities[cls]
        active_weights = [propensities[cid] for cid in eligible_clients]
        
        sum_weights = sum(active_weights)
        if sum_weights > 1e-9:
            weights = [w / sum_weights for w in active_weights]
        else:
            # Fallback if remaining active propensities are near-zero
            n_active = len(eligible_clients)
            weights = [1.0 / n_active] * n_active
            
        # Sample client using optimized random.choices
        sampled_cid = random.choices(eligible_clients, weights=weights, k=1)[0]
        
        client_pools[sampled_cid].append((img, source_split))
        client_current_sizes[sampled_cid] += 1
        client_assigned_images[sampled_cid].add(img)
        consecutive_failures = 0

    # Debug summary checking final assignment states
    print(f"[{source_split}] Phase 2 status check:")
    for cid in range(num_clients):
        target = client_targets[cid]
        current = client_current_sizes[cid]
        status = "✓" if current >= target else "✗"
        print(f"  {status} Client {cid}: {current}/{target}")
        
    print(f"[{source_split}] Phase 2 complete. Final avg size: {sum(client_current_sizes.values()) / len(client_current_sizes) if client_current_sizes else 0:.1f} images")
    
    return client_pools

def generate_manifest(bucket_name: str, num_clients: int, 
                     min_train: int, max_train: int,
                     min_val: int, max_val: int,
                     seed: int, dataset_id: str,
                     same_seed_for_val: bool,
                     iid_clients: List[int]) -> dict:
    """Main function to generate partition manifest."""
    
    # Generate dataset description from IID client list
    if len(iid_clients) == 0:
        dataset_description = "All clients non-IID (α=0.5-1.5)"
    elif len(iid_clients) == num_clients:
        dataset_description = "All clients IID (α=1e6)"
    else:
        iid_str = ",".join(map(str, sorted(iid_clients)))
        dataset_description = f"Clients {iid_str} IID (α=1e6), others non-IID (α=0.5-1.5)"
    
    # Create numeric hash of dataset_id for seed variation
    # Ensure hash is positive and within reasonable range
    dataset_id_hash = abs(hash(dataset_id)) % 1000000
    
    print("\n" + "="*70)
    print(f"COCO Dataset Partitioning - Dataset {dataset_id}")
    print("="*70)
    print(f"Configuration: {dataset_description}")
    print(f"Bucket: {bucket_name}")
    print(f"Clients: {num_clients}")
    print(f"Train size range: [{min_train}, {max_train}]")
    print(f"Val size range: [{min_val}, {max_val}]")
    print(f"Seed: {seed}")
    print(f"Same seed for val: {same_seed_for_val}")
    print("="*70 + "\n")
    
    # Generate client configurations
    client_configs = generate_client_configs(
        num_clients, seed, dataset_id_hash, min_train, max_train, min_val, max_val, iid_clients
    )
    
    # Verify all configs have valid alphas
    for cfg in client_configs:
        if cfg['alpha'] is None:
            print(f"ERROR: Client {cfg['client_id']} has None alpha!")
            sys.exit(1)
    
    # Get training images and labels
    print("\n--- Processing Training Split ---")
    train_images = get_gcs_image_list(bucket_name, "train2017")
    print(f"Total train2017 images: {len(train_images)}")
    train_labels = get_gcs_label_mapping(bucket_name, train_images, "train2017")
    print(f"Total train2017 labels: {len(train_labels)}")
    
    # Get validation images and labels
    print("\n--- Processing Validation Split ---")
    val_images = get_gcs_image_list(bucket_name, "val2017")
    print(f"Total val2017 images: {len(val_images)}")
    val_labels = get_gcs_label_mapping(bucket_name, val_images, "val2017")
    print(f"Total val2017 labels: {len(val_labels)}")

    # Partition train2017 data FOR TRAINING ONLY
    print("\n--- Partitioning Training Data (from train2017 split) ---")
    train_pools = partition_images_dirichlet_balanced(
        train_images, train_labels, client_configs, seed, "train2017",
        target_key='n_train',  # Use n_train for training data
        min_classes_per_iid_client=65
    )
    print(f"Partitioned train2017 data into {num_clients} clients for TRAINING.")
    
    # Partition val2017 data FOR VALIDATION ONLY
    # Use very few classes for val since there are few val images (only 5000)
    # Phase 1 uses very little of the data, leaving plenty for Phase 2 to distribute evenly
    print("\n--- Partitioning Validation Data (from val2017 split) ---")
    val_seed = seed if same_seed_for_val else seed + 1000
    val_pools = partition_images_dirichlet_balanced(
        val_images, val_labels, client_configs, val_seed, "val2017",
        target_key='n_val',  # FIX: Use n_val for validation data
        min_classes_per_iid_client=5  # Very low for val to preserve images for Phase 2
    )
    print(f"Partitioned val2017 data into {num_clients} clients for VALIDATION.")

    # Keep train and val separate - NO MIXING
    # Train images come ONLY from train2017, val images come ONLY from val2017
    # Apply target sizes to each partition independently
    print("\n--- Preparing Client Data (No Split Mixing) ---")
    train_assignments = {}
    val_assignments = {}
    all_labels = {**train_labels, **val_labels}
    
    for client_id in range(num_clients):
        # Training images come ONLY from train2017 partition
        train_pool = train_pools[client_id]
        # Validation images come ONLY from val2017 partition
        val_pool = val_pools[client_id]
        
        # Get target sizes from config
        target_train = client_configs[client_id]['n_train']
        target_val = client_configs[client_id]['n_val']
        
        # Trim pools to target sizes (avoid mixing splits!)
        n_train = min(target_train, len(train_pool))
        n_val = min(target_val, len(val_pool))
        
        # Take first n_train from train partition, first n_val from val partition
        train_part = train_pool[:n_train]
        val_part = val_pool[:n_val]
        
        # Update config to reflect reality (may be less than target if not enough data)
        client_configs[client_id]['n_train'] = n_train
        client_configs[client_id]['n_val'] = n_val
        
        train_assignments[client_id] = train_part
        val_assignments[client_id] = val_part
        
        status_train = "✓" if n_train == target_train else "✗"
        status_val = "✓" if n_val == target_val else "✗"
        print(f"  {status_train} Client {client_id}: {n_train} train (target: {target_train}), {status_val} {n_val} val (target: {target_val})")    
        print("\n--- Computing Class Distributions ---")
    
    # Compute class distributions (for statistics and plots)
    def get_class_dist(assignments, labels):
        result = {}
        for cid, img_list in assignments.items():
            counts = defaultdict(int)
            for img_name, _ in img_list:
                classes = labels.get(img_name, [])
                if classes:
                    counts[classes[0]] += 1
            result[cid] = dict(counts)
        return result
    
    train_class_dist = get_class_dist(train_assignments, all_labels)
    val_class_dist = get_class_dist(val_assignments, all_labels)
    
    # Generate plots
    print("\n--- Generating Client Plots ---")
    plots_dir = Path(f"client_plots_dataset_{dataset_id}")
    plots_dir.mkdir(exist_ok=True)
    
    for client_id in range(num_clients):
        cfg = client_configs[client_id]
        plot_path = plots_dir / f"client_{client_id}_distribution.png"
        
        plot_client_distribution(
            client_id=client_id,
            train_dist=train_class_dist.get(client_id, {}),
            val_dist=val_class_dist.get(client_id, {}),
            alpha=cfg['alpha'],
            n_train=len(train_assignments[client_id]),
            n_val=len(val_assignments[client_id]),
            output_path=str(plot_path),
            dataset_id=dataset_id
        )
    
    # Build manifest
    manifest = {
        "metadata": {
            "dataset_id": dataset_id,
            "dataset_description": dataset_description,
            "iid_clients": sorted(iid_clients),
            "non_iid_clients": sorted([i for i in range(num_clients) if i not in iid_clients]),
            "seed": seed,
            "num_clients": num_clients,
            "min_train_per_client": min_train,
            "max_train_per_client": max_train,
            "min_val_per_client": min_val,
            "max_val_per_client": max_val,
            "same_seed_for_val": same_seed_for_val,
            "bucket_name": bucket_name
        },
        "partitions": {}
    }
    
    # Add client partitions
    for client_id in range(num_clients):
        cfg = client_configs[client_id]
        
        # Extract image names by source split
        # Training images come ONLY from train2017
        train_images_list = [img for img, src in train_assignments[client_id] if src == "train2017"]
        # Validation images come ONLY from val2017
        val_images_list = [img for img, src in val_assignments[client_id] if src == "val2017"]
        
        # Calculate training class statistics
        train_dist = train_class_dist.get(client_id, {})
        train_class_stats = {
            "total_unique_classes": len(train_dist),
            "classes_with_1_image": sum(1 for c in train_dist.values() if c == 1),
            "classes_with_2_10_images": sum(1 for c in train_dist.values() if 2 <= c <= 10),
            "classes_with_11_50_images": sum(1 for c in train_dist.values() if 11 <= c <= 50),
            "classes_with_51_100_images": sum(1 for c in train_dist.values() if 51 <= c <= 100),
            "classes_with_100_plus_images": sum(1 for c in train_dist.values() if c > 100),
            "min_class_count": min(train_dist.values()) if train_dist else 0,
            "max_class_count": max(train_dist.values()) if train_dist else 0,
            "median_class_count": sorted(train_dist.values())[len(train_dist)//2] if train_dist else 0,
            "mean_class_count": sum(train_dist.values()) / len(train_dist) if train_dist else 0,
            "class_coverage_ratio": len(train_dist) / 80.0
        }
        
        # Calculate validation class statistics
        val_dist = val_class_dist.get(client_id, {})
        val_class_stats = {
            "total_unique_classes": len(val_dist),
            "classes_with_1_image": sum(1 for c in val_dist.values() if c == 1),
            "classes_with_2_10_images": sum(1 for c in val_dist.values() if 2 <= c <= 10),
            "classes_with_11_50_images": sum(1 for c in val_dist.values() if 11 <= c <= 50),
            "classes_with_51_100_images": sum(1 for c in val_dist.values() if 51 <= c <= 100),
            "classes_with_100_plus_images": sum(1 for c in val_dist.values() if c > 100),
            "min_class_count": min(val_dist.values()) if val_dist else 0,
            "max_class_count": max(val_dist.values()) if val_dist else 0,
            "median_class_count": sorted(val_dist.values())[len(val_dist)//2] if val_dist else 0,
            "mean_class_count": sum(val_dist.values()) / len(val_dist) if val_dist else 0,
            "class_coverage_ratio": len(val_dist) / 80.0
        }
        
        manifest["partitions"][f"client_{client_id}"] = {
            "client_id": client_id,
            "alpha": cfg['alpha'],
            "n_train_target": cfg['n_train'],
            "n_val_target": cfg['n_val'],
            # Training images (all from train2017 split)
            "train_images": train_images_list,
            "train_split": "train2017",
            "train_count": len(train_images_list),
            "train_class_distribution": train_dist,
            "train_class_stats": train_class_stats,
            # Validation images (all from val2017 split)
            "val_images": val_images_list,
            "val_split": "val2017",
            "val_count": len(val_images_list),
            "val_class_distribution": val_dist,
            "val_class_stats": val_class_stats
        }
    
    # Print statistics
    print("\n" + "="*70)
    print(f"PARTITION STATISTICS - Dataset {dataset_id}")
    print("="*70)
    for client_id in range(num_clients):
        partition = manifest["partitions"][f"client_{client_id}"]
        print(f"\nClient {client_id} (α={partition['alpha']:.4f}):")
        print(f"  Train images: {partition['train_count']} (target: {partition['n_train_target']})")
        print(f"  Val images: {partition['val_count']} (target: {partition['n_val_target']})")
        print(f"  Train classes: {len(partition['train_class_distribution'])}")
        print(f"  Val classes: {len(partition['val_class_distribution'])}")
    
    total_train = sum(p["train_count"] for p in manifest["partitions"].values())
    total_val = sum(p["val_count"] for p in manifest["partitions"].values())
    
    print("\n" + "="*70)
    print("TOTALS:")
    print(f"  Total train images: {total_train}")
    print(f"  Total val images: {total_val}")
    print("="*70 + "\n")
    
    return manifest


def plot_client_distribution(client_id: int, 
                             train_dist: Dict[int, int],
                             val_dist: Dict[int, int],
                             alpha: float,
                             n_train: int,
                             n_val: int,
                             output_path: str,
                             dataset_id: str):
    """Generate individual plot for one client showing train/val distribution."""
    if not PLOTTING_AVAILABLE:
        return
    
    # Get all classes present in train or val
    all_classes = sorted(set(list(train_dist.keys()) + list(val_dist.keys())))
    if not all_classes:
        return
        
    # Prepare data
    class_names = [COCO_NAMES[cls] for cls in all_classes]
    train_counts = [train_dist.get(cls, 0) for cls in all_classes]
    val_counts = [val_dist.get(cls, 0) for cls in all_classes]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Bar positions
    x = np.arange(len(class_names))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_counts, width, label='Val', color='coral', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Class Name', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=14, fontweight='bold')
    ax.set_title(f'Dataset {dataset_id} - Client {client_id} Distribution (α={alpha:.2f}, Train={n_train}, Val={n_val})', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate COCO partitions for FL")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID (can be any string, e.g., 'dataset_5', 'experiment_1')")
    parser.add_argument("--iid-clients", required=True, 
                       help="Comma-separated list of IID client IDs (e.g., '0,2,5' or 'none' for all non-IID)")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--min-train-images", type=int, required=True, help="Min training images per client")
    parser.add_argument("--max-train-images", type=int, required=True, help="Max training images per client")
    parser.add_argument("--min-val-images", type=int, required=True, help="Min validation images per client")
    parser.add_argument("--max-val-images", type=int, required=True, help="Max validation images per client")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--same-seed-val", action="store_true", 
                       help="Use same seed for validation (identical distribution)")
    parser.add_argument("--output", default="partition_manifest.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Parse IID clients list
    if args.iid_clients.lower() == 'none':
        iid_clients = []
    else:
        try:
            iid_clients = [int(x.strip()) for x in args.iid_clients.split(',')]
            # Validate client IDs
            for cid in iid_clients:
                if cid < 0 or cid >= args.num_clients:
                    raise ValueError(f"Client ID {cid} out of range [0, {args.num_clients-1}]")
        except ValueError as e:
            print(f"Error parsing --iid-clients: {e}")
            sys.exit(1)
    
    # Generate manifest
    manifest = generate_manifest(
        args.bucket, args.num_clients, 
        args.min_train_images, args.max_train_images,
        args.min_val_images, args.max_val_images,
        args.seed, args.dataset_id, args.same_seed_val,
        iid_clients
    )
    
    # Save manifest with dataset-specific name
    output_file = f"partition_manifest_dataset_{args.dataset_id}.json"
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Partition manifest saved to: {output_file}")
    print(f"✅ Client plots saved to: client_plots_dataset_{args.dataset_id}/")


if __name__ == "__main__":
    main()