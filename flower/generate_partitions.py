#!/usr/bin/env python3
"""
Generate deterministic Dirichlet partitions for federated COCO dataset.
Supports 4 dataset configurations with varying IID/non-IID distributions.
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


def get_alpha_for_client(dataset_id: int, client_id: int, seed: int) -> float:
    """
    Determine alpha based on dataset type and client ID.
    
    Dataset 1: All clients IID (alpha=1e6)
    Dataset 2: Clients 0-4 IID, Clients 5-9 non-IID (0.5-1.5)
    Dataset 3: Clients 0-1 IID, Clients 2-9 non-IID (0.5-1.5)
    Dataset 4: All clients non-IID (0.5-1.5)
    """
    # Use deterministic seed per client to ensure reproducibility
    np.random.seed(seed + dataset_id * 1000 + client_id)
    
    if dataset_id == 1:
        return 1e6  # All IID
    elif dataset_id == 2:
        return 1e6 if client_id <= 4 else np.random.uniform(0.5, 1.5)
    elif dataset_id == 3:
        return 1e6 if client_id <= 1 else np.random.uniform(0.5, 1.5)
    elif dataset_id == 4:
        return np.random.uniform(0.5, 1.5)  # All non-IID
    else:
        raise ValueError(f"Invalid dataset_id: {dataset_id}. Must be 1-4.")


def get_gcs_image_list(bucket_name: str, split: str) -> List[str]:
    """Get list of image filenames from GCS bucket. Uses cache if available."""
    cache_file = Path(f"/tmp/coco_image_list_{split}.txt")
    
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
    Reuses existing labels if already downloaded.
    If labels are slightly incomplete, proceed without re-downloading.
    """
    print(f"Preparing labels for {len(images)} images from {split}...")

    temp_dir = Path(f"/tmp/coco_labels_{split}")
    labels_dir = temp_dir / split
    gcs_labels_path = f"gs://{bucket_name}/coco/labels/{split}/"

    # Accept incomplete labels if coverage is high enough
    MIN_LABEL_COVERAGE = 0.98  # 98% is "good enough" for partitioning
    expected = len(images)

    if labels_dir.exists():
        existing_labels = list(labels_dir.glob("*.txt"))
        existing_count = len(existing_labels)
        coverage = existing_count / expected if expected > 0 else 1.0

        if existing_count >= expected:
            print(f"✓ Labels directory exists with sufficient files ({existing_count} >= {expected}). Skipping download.")
            # Parse and return immediately
            image_to_classes = {}
            for img_name in images:
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
            
            print(f"Parsed labels for {len(image_to_classes)} images (from cache)")
            return image_to_classes
            
        elif coverage >= MIN_LABEL_COVERAGE:
            missing = expected - existing_count
            print(
                f"⚠ Labels directory incomplete ({existing_count} / {expected}). "
                f"Missing {missing} (~{(1.0-coverage)*100:.2f}%). "
                f"Coverage >= {MIN_LABEL_COVERAGE*100:.0f}%, proceeding without re-download."
            )
            # Use existing labels without re-downloading
        else:
            print(
                f"⚠ Labels directory too incomplete ({existing_count} / {expected}). "
                f"Coverage {coverage*100:.2f}% < {MIN_LABEL_COVERAGE*100:.0f}%, re-downloading..."
            )
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Deleted incomplete labels directory: {temp_dir}")

    # Download only if needed (directory missing, or we deleted it because it was too incomplete)
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

    # Parse labels (missing label file => empty classes)
    image_to_classes = {}
    for img_name in images:
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


def generate_client_configs(num_clients: int, seed: int, dataset_id: int,
                            min_train: int, max_train: int,
                            min_val: int, max_val: int) -> List[Dict]:
    """
    Generate configuration for each client based on dataset type.
    Train and val sizes are independently randomized.
    """
    np.random.seed(seed + dataset_id * 10000)
    random.seed(seed + dataset_id * 10000)
    
    configs = []
    
    for client_id in range(num_clients):
        # Random train size
        n_train = np.random.randint(min_train, max_train + 1)
        # Random val size (independent)
        n_val = np.random.randint(min_val, max_val + 1)
        
        # Get alpha based on dataset type
        alpha = get_alpha_for_client(dataset_id, client_id, seed)
        
        configs.append({
            'client_id': client_id,
            'alpha': alpha,
            'n_train': n_train,
            'n_val': n_val
        })
    
    return configs


def partition_images_dirichlet(
    images: List[str],
    image_to_classes: Dict[str, List[int]],
    client_configs: List[Dict],
    seed: int,
    source_split: str  # 'train2017' or 'val2017'
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Partition images using Dirichlet distribution.
    - IID Clients (alpha > 1000): Get a fixed 1/N share of every class (Uniform).
    - Non-IID Clients: Share the remaining images via Dirichlet distribution.
    
    Returns dict mapping client_id -> [(image_name, source_split), ...]
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Build class -> images mapping
    class_to_images = defaultdict(list)
    for img, classes in image_to_classes.items():
        if len(classes) == 0:
            continue
        # Use primary class
        primary_class = classes[0]
        class_to_images[primary_class].append(img)
    
    all_classes = sorted(class_to_images.keys())
    num_clients = len(client_configs)
    
    print(f"[{source_split}] Found {len(all_classes)} classes")
    print(f"[{source_split}] Total images with labels: {sum(len(imgs) for imgs in class_to_images.values())}")
    
    # Identify IID and Non-IID clients
    iid_clients = []
    non_iid_clients = []
    non_iid_alphas = []
    
    for cfg in client_configs:
        cid = cfg['client_id']
        alpha = cfg['alpha']
        if alpha > 1000:
            iid_clients.append(cid)
        else:
            non_iid_clients.append(cid)
            non_iid_alphas.append(alpha)
            
    print(f"[{source_split}] IID Clients: {iid_clients}")
    print(f"[{source_split}] Non-IID Clients: {non_iid_clients} (alphas: {non_iid_alphas})")
    
    # Initialize assignments
    client_pools = {i: [] for i in range(num_clients)}
    
    # Partition each class
    for cls in all_classes:
        imgs = list(set(class_to_images[cls]))
        random.shuffle(imgs)
        n_imgs = len(imgs)
        
        if n_imgs == 0:
            continue
            
        idx = 0
        
        # 1. Assign fixed share to IID clients
        share_per_client = n_imgs // num_clients
        
        for cid in iid_clients:
            end = idx + share_per_client
            if idx < n_imgs:
                subset = imgs[idx:end]
                for img in subset:
                    client_pools[cid].append((img, source_split))
                idx = end
        
        # 2. Assign remainder to Non-IID clients
        remaining_imgs = imgs[idx:]
        n_remaining = len(remaining_imgs)
        
        if n_remaining > 0 and len(non_iid_clients) > 0:
            # Dirichlet proportions for non-iid clients
            proportions = np.random.dirichlet(np.array(non_iid_alphas))
            
            # Calculate counts
            counts = (proportions * n_remaining).astype(int)
            
            # Distribute leftovers
            leftover = n_remaining - counts.sum()
            if leftover > 0:
                for i in np.argsort(proportions)[-leftover:]:
                    counts[i] += 1
            
            # Assign
            rem_idx = 0
            for i, cid in enumerate(non_iid_clients):
                count = counts[i]
                for _ in range(count):
                    if rem_idx >= n_remaining:
                        break
                    client_pools[cid].append((remaining_imgs[rem_idx], source_split))
                    rem_idx += 1
    
    return client_pools


def generate_manifest(bucket_name: str, num_clients: int, 
                     min_train: int, max_train: int,
                     min_val: int, max_val: int,
                     seed: int, dataset_id: int,
                     same_seed_for_val: bool) -> dict:
    """Main function to generate partition manifest."""
    
    # Dataset description
    dataset_descriptions = {
        1: "All clients IID (α=1e6)",
        2: "Clients 0-4 IID, Clients 5-9 non-IID (α=0.5-1.5)",
        3: "Clients 0-1 IID, Clients 2-9 non-IID (α=0.5-1.5)",
        4: "All clients non-IID (α=0.5-1.5)"
    }
    
    print("\n" + "="*70)
    print(f"COCO Dataset Partitioning - Dataset {dataset_id}")
    print("="*70)
    print(f"Configuration: {dataset_descriptions[dataset_id]}")
    print(f"Bucket: {bucket_name}")
    print(f"Clients: {num_clients}")
    print(f"Train size range: [{min_train}, {max_train}]")
    print(f"Val size range: [{min_val}, {max_val}]")
    print(f"Seed: {seed}")
    print(f"Same seed for val: {same_seed_for_val}")
    print("="*70 + "\n")
    
    # Generate client configurations
    client_configs = generate_client_configs(
        num_clients, seed, dataset_id, min_train, max_train, min_val, max_val
    )
    
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

    # Partition train2017 data
    print("\n--- Partitioning Training Data ---")
    train_pools = partition_images_dirichlet(
        train_images, train_labels, client_configs, seed, "train2017"
    )
    print(f"Partitioned train2017 data into {num_clients} clients.")
    
    # Partition val2017 data (with same or different seed)
    print("\n--- Partitioning Validation Data ---")
    val_seed = seed if same_seed_for_val else seed + 1000
    val_pools = partition_images_dirichlet(
        val_images, val_labels, client_configs, val_seed, "val2017"
    )
    print(f"Partitioned val2017 data into {num_clients} clients.")

    # Combine pools and split into train/val for each client
    print("\n--- Splitting Client Data into Train/Val ---")
    train_assignments = {}
    val_assignments = {}
    all_labels = {**train_labels, **val_labels}
    
    for client_id in range(num_clients):
        # Combine train2017 and val2017 pools for this client
        combined_pool = train_pools[client_id] + val_pools[client_id]
        random.shuffle(combined_pool)
        
        total_received = len(combined_pool)
        
        # Use target sizes from config
        n_train = min(client_configs[client_id]['n_train'], total_received)
        n_val = min(client_configs[client_id]['n_val'], total_received - n_train)
        
        # Take first n_train for training
        train_part = combined_pool[:n_train]
        # Take next n_val for validation
        val_part = combined_pool[n_train:n_train + n_val]
        
        # Update config to reflect reality
        client_configs[client_id]['n_train'] = n_train
        client_configs[client_id]['n_val'] = n_val
        
        train_assignments[client_id] = train_part
        val_assignments[client_id] = val_part
        
        print(f"  Client {client_id}: {len(train_part)} train, {len(val_part)} val "
              f"(from pool of {len(combined_pool)})")
    
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
            "dataset_description": dataset_descriptions[dataset_id],
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
        train_from_train = [img for img, src in train_assignments[client_id] if src == "train2017"]
        train_from_val = [img for img, src in train_assignments[client_id] if src == "val2017"]
        val_from_train = [img for img, src in val_assignments[client_id] if src == "train2017"]
        val_from_val = [img for img, src in val_assignments[client_id] if src == "val2017"]
        
        manifest["partitions"][f"client_{client_id}"] = {
            "client_id": client_id,
            "alpha": cfg['alpha'],
            "n_train_target": cfg['n_train'],
            "n_val_target": cfg['n_val'],
            # Images for training (with source split info)
            "train_images_from_train2017": train_from_train,
            "train_images_from_val2017": train_from_val,
            # Images for validation (with source split info)
            "val_images_from_train2017": val_from_train,
            "val_images_from_val2017": val_from_val,
            # Totals
            "train_count": len(train_assignments[client_id]),
            "val_count": len(val_assignments[client_id]),
            "train_class_distribution": train_class_dist.get(client_id, {}),
            "val_class_distribution": val_class_dist.get(client_id, {})
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
                             dataset_id: int):
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
    parser.add_argument("--dataset-id", type=int, required=True, choices=[1, 2, 3, 4],
                       help="Dataset ID (1-4) for different IID/non-IID configurations")
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
    
    # Generate manifest
    manifest = generate_manifest(
        args.bucket, args.num_clients, 
        args.min_train_images, args.max_train_images,
        args.min_val_images, args.max_val_images,
        args.seed, args.dataset_id, args.same_seed_val
    )
    
    # Save manifest with dataset-specific name
    output_file = f"partition_manifest_dataset_{args.dataset_id}.json"
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Partition manifest saved to: {output_file}")
    print(f"✅ Client plots saved to: client_plots_dataset_{args.dataset_id}/")
    print("\nNext step: Continue with partition-dataset.sh for remaining datasets")


if __name__ == "__main__":
    main()