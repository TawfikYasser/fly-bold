import os
import glob
import random
import shutil
from collections import defaultdict
import numpy as np
from pathlib import Path

global_coco_root = "/home/tawfik/SSDData/DeFeC3/coding/flower-benchmarks/datasets/coco128"
num_clients = 5
tmp_client_root_base = "./tmp_coco_clients"
client_id = 0  # Change as needed for different clients

def _read_labels_for_image(label_file):
    # Read classes present in a YOLO txt file
    if not os.path.exists(label_file):
        return []
    with open(label_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    classes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 1:
            try:
                classes.append(int(float(parts[0])))
            except Exception:
                pass
    return classes

def partition_coco128_dir(root_coco128: str, num_clients: int, alpha: float = 0.7, seed: int = 42):
    """
    Partition COCO128 images into `num_clients` disjoint image sets using a
    Dirichlet distribution over classes.

    Returns: dict client_id -> dict with keys 'train_images' and 'val_images' listing image absolute paths.
    """

    random.seed(seed)
    np.random.seed(seed)

    root = Path(root_coco128)
    print(f"Partitioning COCO128 dataset in {root}")
    train_imgs = sorted((root / "images" / "train2017").glob("*.jpg"))
    val_imgs = sorted((root / "images" / "val2017").glob("*.jpg"))

    print(f"Found {len(train_imgs)} training images and {len(val_imgs)} validation images.")

    print("Building image class mapping...")

    # Build mapping image -> classes (from labels)
    def build_map(img_list, split_name):
        mapping = {}
        for img in img_list:
            label_file = root / "labels" / split_name / (img.stem + ".txt")
            mapping[str(img)] = _read_labels_for_image(str(label_file))
        return mapping

    train_map = build_map(train_imgs, "train2017")
    print(train_map)
    print(f"Built training image map with {len(train_map)} entries.")
    val_map = build_map(val_imgs, "val2017")
    print(val_map)
    print(f"Built validation image map with {len(val_map)} entries.")


    # Get all classes present
    all_classes = set()
    for classes in train_map.values():
        all_classes.update(classes)
    all_classes = sorted(list(all_classes))
    print(f"Found classes in COCO128: {all_classes}")

    if not all_classes:
        raise RuntimeError("No classes found in COCO128 labels. Did you run get_coco128.sh?")

    # For reproducibility: list of train images per class
    class_to_images = defaultdict(list)
    for img, classes in train_map.items():
        if len(classes) == 0:
            # treat images with no labels as background class - skip them for now
            continue
        for c in classes:
            class_to_images[c].append(img)
    
    print("Classes to images mapping:")
    for c, imgs in class_to_images.items():
        print(f"  Class {c}: {len(imgs)} images")

    # Dirichlet partition across classes:
    client_assignments = {i: [] for i in range(num_clients)}
    # We'll assign each class's images across clients according to Dirichlet for that class
    for c, imgs in class_to_images.items():
        imgs = list(set(imgs))
        n = len(imgs)
        if n == 0:
            continue
        # draw proportions
        proportions = np.random.dirichlet(alpha=np.ones(num_clients) * alpha)
        # compute counts per client (must be integers; keep at least zero)
        counts = (proportions * n).astype(int)
        # adjust to match n by distributing remainders
        leftover = n - counts.sum()
        for i in np.argsort(proportions)[-leftover:]:
            counts[i] += 1
        # shuffle imgs and split
        random.shuffle(imgs)
        idx = 0
        for client_id in range(num_clients):
            cnt = counts[client_id]
            for _ in range(cnt):
                if idx >= len(imgs): break
                img_path = imgs[idx]
                # avoid duplicates: only add if not already assigned
                if img_path not in client_assignments[client_id]:
                    client_assignments[client_id].append(img_path)
                idx += 1
        # if some images unassigned (rare), give them round-robin
        while idx < len(imgs):
            for client_id in range(num_clients):
                if idx >= len(imgs): break
                img_path = imgs[idx]
                if img_path not in client_assignments[client_id]:
                    client_assignments[client_id].append(img_path)
                idx += 1

    print("Client assignments (training):")
    for i in range(num_clients):
        print(f"  Client {i}: {len(client_assignments[i])} images")

    # For validation split we use a simple round-robin or random split to clients
    val_list = list(val_map.keys())
    random.shuffle(val_list)
    val_assignments = {i: [] for i in range(num_clients)}
    for i, img in enumerate(val_list):
        val_assignments[i % num_clients].append(img)

    print("Client assignments (validation):")
    for i in range(num_clients):
        print(f"  Client {i}: {len(val_assignments[i])} images")

    # Convert image paths to canonical lists
    partitions = {}
    for i in range(num_clients):
        train_images = client_assignments[i]
        val_images = val_assignments[i]
        partitions[i] = {"train_images": train_images, "val_images": val_images}

    print("Print partitions:")
    for client_id, partition in partitions.items():
        print(f"  Client {client_id}:")
        print(f"    Train images: {len(partition['train_images'])}")
        print(f"    Val images: {len(partition['val_images'])}")

    print("Partitioning complete.")

    return partitions

def write_client_dataset_yolo_layout(root_coco128: str, out_base: str, client_id: int, partition: dict):
    """
    Create a client-specific dataset directory with YOLO layout (images + labels).
    We will create directory:
      <out_base>/client_{client_id}/images/train2017
                         /images/val2017
                         /labels/train2017
                         /labels/val2017
    and symlink/copy the required image and label files into it.
    Returns the path to the dataset root for that client.
    """
    client_root = Path(out_base) / f"client_{client_id}"
    images_train_dir = client_root / "images" / "train2017"
    images_val_dir = client_root / "images" / "val2017"
    labels_train_dir = client_root / "labels" / "train2017"
    labels_val_dir = client_root / "labels" / "val2017"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {d}")

    root = Path(root_coco128)
    print(f"Writing client dataset to {client_root}")
    for src_img in partition["train_images"]:
        src_img_path = Path(src_img)
        print(f"  Processing train image: {src_img_path}")
        dest_img = images_train_dir / src_img_path.name
        print(f"  Destination: {dest_img}")
        try:
            os.symlink(src_img_path, dest_img)
        except Exception:
            shutil.copy(src_img_path, dest_img)
        # label
        src_label = root / "labels" / "train2017" / (src_img_path.stem + ".txt")
        print(f"  Source label: {src_label}")
        dest_label = labels_train_dir / (src_img_path.stem + ".txt")
        print(f"  Destination label: {dest_label}")
        if src_label.exists():
            try:
                os.symlink(src_label, dest_label)
            except Exception:
                shutil.copy(src_label, dest_label)

    for src_img in partition["val_images"]:
        src_img_path = Path(src_img)
        print(f"  Processing val image: {src_img_path}")
        dest_img = images_val_dir / src_img_path.name
        print(f"  Destination: {dest_img}")
        try:
            os.symlink(src_img_path, dest_img)
        except Exception:
            shutil.copy(src_img_path, dest_img)
        # label
        src_label = root / "labels" / "val2017" / (src_img_path.stem + ".txt")
        print(f"  Source label: {src_label}")
        dest_label = labels_val_dir / (src_img_path.stem + ".txt")
        print(f"  Destination label: {dest_label}")
        if src_label.exists():
            try:
                os.symlink(src_label, dest_label)
            except Exception:
                shutil.copy(src_label, dest_label)
    print("Client dataset preparation complete. in ", client_root)
    return str(client_root)


def write_data_yaml(client_dataset_root: str, yaml_path: str, names=None):
    """
    Create a YOLO dataset yaml file referencing the client dataset paths.
    """
    import yaml
    d = {
        "train": os.path.join(client_dataset_root, "images", "train2017"),
        "val": os.path.join(client_dataset_root, "images", "val2017"),
        "nc": len(names) if names else 80,
        "names": names if names else [str(i) for i in range(80)]
    }
    print(f"Writing data yaml to {yaml_path}:")
    print(d)
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    print(f"Created directory for yaml: {os.path.dirname(yaml_path)}")
    with open(yaml_path, "w") as f:
        yaml.dump(d, f)
    return yaml_path


# print("Starting partitioning...")
# print(f"Using global COCO128 root: {global_coco_root}")
# print(f"Preparing dataset for client {client_id} out of {num_clients} clients in base dir {tmp_client_root_base}")
# os.makedirs(tmp_client_root_base, exist_ok=True)


# partitions = partition_coco128_dir(global_coco_root, num_clients, alpha=0.7, seed=42)
# client_partition = partitions[client_id]

# print("Partition for this client:")
# print(f"  Train images: {len(client_partition['train_images'])}")
# print(f"  Val images: {len(client_partition['val_images'])}")

# print("Writing client dataset in YOLO layout...")

# client_dataset_root = write_client_dataset_yolo_layout(global_coco_root, tmp_client_root_base, client_id, client_partition)

# data_yaml = os.path.join(tmp_client_root_base, f"client_{client_id}", "coco128_client.yaml")

# print(f"Writing data yaml to {data_yaml}...")

# print(write_data_yaml(client_dataset_root, data_yaml))

# import yaml
# d = {
#     "train": os.path.join(client_dataset_root, "images", "train2017"),
#     "val": os.path.join(client_dataset_root, "images", "val2017"),
#     "nc": len(names) if names else 80,
#     "names": names if names else [str(i) for i in range(80)]
# }
# os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
# with open(yaml_path, "w") as f:
#     yaml.dump(d, f)
# return yaml_path
# print([str(i) for i in range(80)])

import torch

# Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Default: yolov5s

# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
img = "https://ultralytics.com/images/zidane.jpg"  # Example image

# Perform inference (handles batching, resizing, normalization automatically)
results = model(img)

# Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
results.print()  # Print results to console
results.show()  # Display results in a window
results.save()  # Save results to runs/detect/exp