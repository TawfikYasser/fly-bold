#!/usr/bin/env python3
"""
Generate deterministic Dirichlet partitions for federated COCO dataset.
Client 0 gets uniform distribution (alpha=1e6), others get varied alphas.
Each client gets random train size (1000-5000), val size is 50% of train.
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


def get_gcs_image_list(bucket_name: str, split: str) -> List[str]:
    """Get list of image filenames from GCS bucket."""
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
            print(f"✓ Labels directory exists ({existing_count} files). Skipping download.")
        elif coverage >= MIN_LABEL_COVERAGE:
            missing = expected - existing_count
            print(
                f"⚠ Labels directory incomplete ({existing_count} / {expected}). "
                f"Missing {missing} (~{(1.0-coverage)*100:.2f}%). "
                f"Coverage >= {MIN_LABEL_COVERAGE*100:.0f}%, proceeding without re-download."
            )
            # Do NOT delete and re-download
        else:
            print(
                f"⚠ Labels directory too incomplete ({existing_count} / {expected}). "
                f"Coverage {coverage*100:.2f}% < {MIN_LABEL_COVERAGE*100:.0f}%, re-downloading..."
            )
            import shutil
            shutil.rmtree(temp_dir)

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

def generate_client_configs(num_clients: int, seed: int, 
                            min_train: int = 1000, max_train: int = 5000,
                            alpha_min: float = 0.1, alpha_max: float = 1.5) -> List[Dict]:
    """
    Generate configuration for each client:
    - Client 0: alpha=1e6 (uniform), random train size
    - Clients 1-9: varied alphas, random train sizes
    - Val size = 50% of train size
    """
    np.random.seed(seed)
    random.seed(seed)
    
    configs = []
    
    for client_id in range(num_clients):
        # Random train size between min_train and max_train
        n_train = np.random.randint(min_train, max_train + 1)
        n_val = n_train // 2  # 50% of train
        
        # Client 0 gets uniform distribution
        if client_id == 0:
            alpha = 1e6
        else:
            alpha = np.random.uniform(alpha_min, alpha_max)
        
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
        # We aim for inclusive fairness: every client deserves 1/N of the data if possible.
        # IID clients get their 1/N slice of THIS class guaranteed.
        
        # Calculate share per client
        share_per_client = n_imgs // num_clients
        
        # If share is 0 (rare class), we skip IID assignment or just give nothing.
        # Let's give IID clients their share first.
        for cid in iid_clients:
            end = idx + share_per_client
            # Assign slice
            if idx < n_imgs:
                # If we run out of images (shouldn't happen with math above, but good to be safe)
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
                     seed: int, alpha_min: float, alpha_max: float,
                     same_seed_for_val: bool) -> dict:
    """Main function to generate partition manifest."""
    
    print("\n" + "="*70)
    print("COCO Dataset Partitioning for Federated Learning")
    print("="*70)
    print(f"Bucket: {bucket_name}")
    print(f"Clients: {num_clients}")
    print(f"Train size range: [{min_train}, {max_train}]")
    print(f"Val size: 50% of train")
    print(f"Seed: {seed}")
    print(f"Alpha range: [{alpha_min}, {alpha_max}]")
    print(f"Client 0: Uniform (α=1e6)")
    print(f"Same seed for val: {same_seed_for_val}")
    print("="*70 + "\n")
    
    # Generate client configurations
    client_configs = generate_client_configs(
        num_clients, seed, min_train, max_train, alpha_min, alpha_max
    )
    
    # OUTPUT SYNC FIX: Removed initial config print to avoid confusion with final partition results.
    # The final statistics at the end of the script will show the authoritative configuration.
    
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
        random.shuffle(combined_pool)  # Mix them up
        
        total_received = len(combined_pool)
        
        # 1. Calculate Potential Train: 2/3rds of what we received
        potential_train = int(total_received * (2/3))
        
        # 2. Apply Random Cap: Use the random target generated earlier as the maximum limit
        target_cap = client_configs[client_id]['n_train']
        n_train = min(potential_train, target_cap)
        
        # 3. Strait Validation Enforcement: Val is EXACTLY 50% of the final Train size
        n_val = int(n_train * 0.5)
        
        # Ensure we don't exceed total_received (rare edge case with rounding, but good safety)
        if n_train + n_val > total_received:
            # If we somehow don't have enough for the strict 2:1 split of the cap,
            # we fall back to the potential split which is naturally safe.
            n_train = potential_train
            n_val = int(potential_train * 0.5)
        
        # Take first n_train for training
        train_part = combined_pool[:n_train]
        # Take next n_val for validation
        val_part = combined_pool[n_train:n_train + n_val]
        
        # Update config to reflect reality so stats match
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
    plots_dir = Path("client_plots")
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
            output_path=str(plot_path)
        )
    
    # Build manifest
    manifest = {
        "metadata": {
            "seed": seed,
            "num_clients": num_clients,
            "min_train_per_client": min_train,
            "max_train_per_client": max_train,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
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
    print("PARTITION STATISTICS")
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
                             output_path: str):
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
    ax.set_title(f'Client {client_id} Data Distribution (α={alpha:.2f}, Train={n_train}, Val={n_val})', 
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
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--min-train", type=int, default=1000, help="Min training images per client")
    parser.add_argument("--max-train", type=int, default=5000, help="Max training images per client")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha-min", type=float, default=0.1, help="Min Dirichlet alpha")
    parser.add_argument("--alpha-max", type=float, default=1.5, help="Max Dirichlet alpha")
    parser.add_argument("--same-seed-val", action="store_true", 
                       help="Use same seed for validation (identical distribution)")
    parser.add_argument("--output", default="partition_manifest.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Generate manifest
    manifest = generate_manifest(
        args.bucket, args.num_clients, args.min_train, args.max_train,
        args.seed, args.alpha_min, args.alpha_max, args.same_seed_val
    )
    
    # Save manifest
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Partition manifest saved to: {args.output}")
    print(f"✅ Client plots saved to: client_plots/")
    print("\nNext step: Run partition-dataset.sh to distribute data to VMs")


if __name__ == "__main__":
    main()