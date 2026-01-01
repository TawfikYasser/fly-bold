#!/usr/bin/env python3
"""
Generate deterministic Dirichlet partitions for federated COCO dataset.
Each client gets a different alpha value to simulate real-world heterogeneity.
"""

import sys
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
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
    Reuses existing labels if already downloaded and complete.
    """
    print(f"Preparing labels for {len(images)} images from {split}...")

    temp_dir = Path(f"/tmp/coco_labels_{split}")
    labels_dir = temp_dir / split
    gcs_labels_path = f"gs://{bucket_name}/coco/labels/{split}/"

    # ---------- NEW: reuse labels if already present ----------
    if labels_dir.exists():
        existing_labels = list(labels_dir.glob("*.txt"))
        if len(existing_labels) == len(images):
            print(f"✔ Labels already exist locally ({len(existing_labels)} files). Skipping download.")
        else:
            print(
                f"⚠ Labels directory exists but incomplete "
                f"({len(existing_labels)} / {len(images)}). Re-downloading..."
            )
            import shutil
            shutil.rmtree(temp_dir)

    # ---------- Download only if needed ----------
    if not labels_dir.exists():
        print(f"Downloading labels from {gcs_labels_path} ...")
        temp_dir.mkdir(exist_ok=True)
        try:
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", gcs_labels_path, str(temp_dir)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error downloading labels: {e}")
            sys.exit(1)

    # ---------- Parse labels ----------
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

def generate_client_alphas(num_clients: int, seed: int, alpha_min: float = 0.1, 
                          alpha_max: float = 1.5) -> List[float]:
    """Generate deterministic random alpha values for each client."""
    np.random.seed(seed)
    alphas = np.random.uniform(alpha_min, alpha_max, size=num_clients)
    return alphas.tolist()


def partition_images_dirichlet(
    images: List[str],
    image_to_classes: Dict[str, List[int]],
    num_clients: int,
    client_alphas: List[float],
    seed: int
) -> Dict[int, List[str]]:
    """
    Partition images using Dirichlet distribution with different alpha per client.
    Each client gets images from a non-IID distribution based on their alpha.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Build class -> images mapping
    class_to_images = defaultdict(list)
    for img, classes in image_to_classes.items():
        if len(classes) == 0:
            continue  # Skip images without labels
        for cls in classes:
            class_to_images[cls].append(img)
    
    all_classes = sorted(class_to_images.keys())
    print(f"Found {len(all_classes)} classes in dataset")
    
    # Initialize client assignments
    client_assignments = {i: [] for i in range(num_clients)}
    
    # For each class, distribute images using Dirichlet
    for cls in all_classes:
        imgs = list(set(class_to_images[cls]))  # Unique images
        n = len(imgs)
        if n == 0:
            continue
        
        # Use mean of all client alphas for this class
        # This ensures overall heterogeneity while maintaining consistency
        alpha_mean = np.mean(client_alphas)
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha_mean)
        
        # Compute counts per client
        counts = (proportions * n).astype(int)
        
        # Adjust for rounding
        leftover = n - counts.sum()
        for i in np.argsort(proportions)[-leftover:]:
            counts[i] += 1
        
        # Shuffle and assign
        random.shuffle(imgs)
        idx = 0
        for client_id in range(num_clients):
            cnt = counts[client_id]
            for _ in range(cnt):
                if idx >= len(imgs):
                    break
                img_path = imgs[idx]
                if img_path not in client_assignments[client_id]:
                    client_assignments[client_id].append(img_path)
                idx += 1
        
        # Handle any remaining (rare)
        while idx < len(imgs):
            for client_id in range(num_clients):
                if idx >= len(imgs):
                    break
                img_path = imgs[idx]
                if img_path not in client_assignments[client_id]:
                    client_assignments[client_id].append(img_path)
                idx += 1
    
    return client_assignments


def apply_size_limit(assignments: Dict[int, List[str]], 
                     limit_per_client: int) -> Dict[int, List[str]]:
    """Randomly sample images to meet N_TRAIN or N_VAL limit per client."""
    limited = {}
    for client_id, images in assignments.items():
        if len(images) > limit_per_client:
            random.shuffle(images)
            limited[client_id] = images[:limit_per_client]
        else:
            limited[client_id] = images
    return limited


def compute_class_distribution(client_assignments: Dict[int, List[str]],
                               image_to_classes: Dict[str, List[int]]) -> Dict[int, Dict[int, int]]:
    """Compute class distribution for each client."""
    client_class_counts = {}
    
    for client_id, images in client_assignments.items():
        class_counts = defaultdict(int)
        for img in images:
            classes = image_to_classes.get(img, [])
            for cls in classes:
                class_counts[cls] += 1
        client_class_counts[client_id] = dict(class_counts)
    
    return client_class_counts


def plot_class_distributions(client_class_counts: Dict[int, Dict[int, int]],
                             client_alphas: List[float],
                             output_path: str,
                             split_name: str):
    """Generate visualization of class distributions across clients."""
    if not PLOTTING_AVAILABLE:
        return
    
    num_clients = len(client_class_counts)
    all_classes = set()
    for counts in client_class_counts.values():
        all_classes.update(counts.keys())
    all_classes = sorted(all_classes)
    
    # Create matrix: clients x classes
    matrix = np.zeros((num_clients, len(all_classes)))
    for client_id, counts in client_class_counts.items():
        for cls_idx, cls in enumerate(all_classes):
            matrix[client_id, cls_idx] = counts.get(cls, 0)
    
    # Normalize by row (per client)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, where=row_sums != 0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Heatmap of class distribution
    ax1 = axes[0]
    im = ax1.imshow(matrix_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xlabel('Class ID', fontsize=12)
    ax1.set_ylabel('Client ID', fontsize=12)
    ax1.set_title(f'Class Distribution Across Clients ({split_name})\n'
                  f'Normalized by Client (Each row sums to 1.0)', fontsize=14)
    ax1.set_yticks(range(num_clients))
    ax1.set_yticklabels([f'C{i}\n(α={client_alphas[i]:.2f})' for i in range(num_clients)])
    
    # Show only every 5th class on x-axis
    x_ticks = list(range(0, len(all_classes), 5))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([all_classes[i] for i in x_ticks])
    
    plt.colorbar(im, ax=ax1, label='Proportion')
    
    # Plot 2: Total samples per client
    ax2 = axes[1]
    # samples_per_client = [len(images) for images in client_class_counts.keys()]
    total_samples = [sum(counts.values()) for counts in client_class_counts.values()]
    
    x = range(num_clients)
    bars = ax2.bar(x, total_samples, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Client ID', fontsize=12)
    ax2.set_ylabel('Total Samples', fontsize=12)
    ax2.set_title(f'Sample Count per Client ({split_name})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{i}\n(α={client_alphas[i]:.2f})' for i in range(num_clients)])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, total_samples)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved class distribution plot: {output_path}")


def generate_manifest(bucket_name: str, num_clients: int, n_train: int, n_val: int,
                     seed: int, alpha_min: float, alpha_max: float) -> dict:
    """Main function to generate partition manifest."""
    
    print("\n" + "="*70)
    print("COCO Dataset Partitioning for Federated Learning")
    print("="*70)
    print(f"Bucket: {bucket_name}")
    print(f"Clients: {num_clients}")
    print(f"N_TRAIN per client: {n_train}")
    print(f"N_VAL per client: {n_val}")
    print(f"Seed: {seed}")
    print(f"Alpha range: [{alpha_min}, {alpha_max}]")
    print("="*70 + "\n")
    
    # Generate client alphas
    client_alphas = generate_client_alphas(num_clients, seed, alpha_min, alpha_max)
    print("Client Alpha values:")
    for i, alpha in enumerate(client_alphas):
        print(f"  Client {i}: α = {alpha:.4f}")
    print()
    
    # Get training images
    print("\n--- Processing Training Data ---")
    train_images = get_gcs_image_list(bucket_name, "train2017")
    train_image_to_classes = get_gcs_label_mapping(bucket_name, train_images, "train2017")
    
    # Partition training data
    train_assignments = partition_images_dirichlet(
        train_images, train_image_to_classes, num_clients, client_alphas, seed
    )
    
    # Apply N_TRAIN limit
    train_assignments = apply_size_limit(train_assignments, n_train)
    
    # Compute class distributions
    train_class_dist = compute_class_distribution(train_assignments, train_image_to_classes)

    # Get validation images
    print("\n--- Processing Validation Data ---")
    val_images = get_gcs_image_list(bucket_name, "val2017")
    val_image_to_classes = get_gcs_label_mapping(bucket_name, val_images, "val2017")
    
    # Partition validation data (non-IID)
    val_assignments = partition_images_dirichlet(
        val_images, val_image_to_classes, num_clients, client_alphas, seed + 1000
    )
    
    # Apply N_VAL limit
    val_assignments = apply_size_limit(val_assignments, n_val)
    
    # Compute validation class distributions
    val_class_dist = compute_class_distribution(val_assignments, val_image_to_classes)
    
    # Build manifest
    manifest = {
        "metadata": {
            "seed": seed,
            "num_clients": num_clients,
            "n_train_per_client": n_train,
            "n_val_per_client": n_val,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "dirichlet_alphas": client_alphas,
            "total_train_images": sum(len(imgs) for imgs in train_assignments.values()),
            "total_val_images": sum(len(imgs) for imgs in val_assignments.values()),
            "bucket_name": bucket_name
        },
        "partitions": {}
    }
    
    # Add client partitions
    for client_id in range(num_clients):
        manifest["partitions"][f"client_{client_id}"] = {
            "client_id": client_id,
            "alpha": client_alphas[client_id],
            "train_images": train_assignments[client_id],
            "val_images": val_assignments[client_id],
            "train_count": len(train_assignments[client_id]),
            "val_count": len(val_assignments[client_id]),
            "train_class_distribution": train_class_dist[client_id],
            "val_class_distribution": val_class_dist[client_id]
        }
    
    # Print statistics
    print("\n" + "="*70)
    print("PARTITION STATISTICS")
    print("="*70)
    for client_id in range(num_clients):
        partition = manifest["partitions"][f"client_{client_id}"]
        print(f"\nClient {client_id} (α={partition['alpha']:.4f}):")
        print(f"  Train images: {partition['train_count']}")
        print(f"  Val images: {partition['val_count']}")
        print(f"  Train classes: {len(partition['train_class_distribution'])}")
        print(f"  Val classes: {len(partition['val_class_distribution'])}")
    
    print("\n" + "="*70)
    print("TOTALS:")
    print(f"  Total train images: {manifest['metadata']['total_train_images']}")
    print(f"  Total val images: {manifest['metadata']['total_val_images']}")
    print("="*70 + "\n")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate COCO partitions for FL")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--n-train", type=int, default=10000, help="Training images per client")
    parser.add_argument("--n-val", type=int, default=5000, help="Validation images per client")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha-min", type=float, default=0.1, help="Min Dirichlet alpha")
    parser.add_argument("--alpha-max", type=float, default=1.5, help="Max Dirichlet alpha")
    parser.add_argument("--output", default="partition_manifest.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Generate manifest
    manifest = generate_manifest(
        args.bucket, args.num_clients, args.n_train, args.n_val,
        args.seed, args.alpha_min, args.alpha_max
    )
    
    # Save manifest
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Partition manifest saved to: {args.output}")
    print("\nNext step: Run partition-dataset.sh to distribute data to VMs")


if __name__ == "__main__":
    main()