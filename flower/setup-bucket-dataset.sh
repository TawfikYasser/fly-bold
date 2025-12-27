#!/bin/bash

# ============================================================
# Setup GCS Bucket and Prepare COCO Dataset (YOLO format)
# ============================================================

set -e

PROJECT_ID="inf022"
BUCKET_NAME="flybold-coco-${PROJECT_ID}"
REGION="us-central1"

TEMP_VM_NAME="coco-downloader-temp"
TEMP_VM_ZONE="${REGION}-a"

# -------------------------------
# Logging helpers
# -------------------------------
echo_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

echo_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
    exit 1
}

# ============================================================
# GCP Setup
# ============================================================

echo_info "Setting GCP project"
gcloud config set project "${PROJECT_ID}"

# -------------------------------
# Create GCS bucket (idempotent)
# -------------------------------
echo_info "Ensuring GCS bucket exists: gs://${BUCKET_NAME}"
if gsutil ls -b "gs://${BUCKET_NAME}" &>/dev/null; then
    echo "Bucket already exists"
else
    gsutil mb -p "${PROJECT_ID}" -c STANDARD -l "${REGION}" "gs://${BUCKET_NAME}/"
    echo_success "Bucket created"
fi

# -------------------------------
# Initialize run_id
# -------------------------------
echo_info "Ensuring run_id counter exists"
if gsutil ls "gs://${BUCKET_NAME}/run_id.txt" &>/dev/null; then
    echo "run_id.txt already exists"
else
    echo "1" | gsutil cp - "gs://${BUCKET_NAME}/run_id.txt"
    echo_success "run_id initialized"
fi

# -------------------------------
# Skip everything if COCO exists
# -------------------------------
echo_info "Checking if COCO already exists in GCS"
if gsutil ls "gs://${BUCKET_NAME}/coco/images/train2017/" &>/dev/null && \
   gsutil ls "gs://${BUCKET_NAME}/coco/images/val2017/" &>/dev/null; then
    echo_success "COCO dataset already exists in GCS — nothing to do"
    exit 0
fi

# ============================================================
# Temporary VM
# ============================================================

echo_info "Ensuring temporary VM exists"
if gcloud compute instances describe "${TEMP_VM_NAME}" \
        --zone="${TEMP_VM_ZONE}" &>/dev/null; then
    echo "Temp VM already exists"
else
    gcloud compute instances create "${TEMP_VM_NAME}" \
        --zone="${TEMP_VM_ZONE}" \
        --machine-type=n1-standard-4 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=100GB \
        --scopes=storage-rw \
        --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y wget unzip python3
'
    echo "Waiting for VM to be ready..."
    sleep 60
fi

# ============================================================
# Download + Convert on VM
# ============================================================

echo_info "Preparing COCO dataset on VM"

gcloud compute ssh "${TEMP_VM_NAME}" \
    --zone="${TEMP_VM_ZONE}" \
    --command='
set -e

COCO_DIR=/tmp/coco
IMAGES_DIR=$COCO_DIR/images
ANN_DIR=$COCO_DIR/annotations
LABELS_DIR=$COCO_DIR/labels

mkdir -p $IMAGES_DIR $ANN_DIR $LABELS_DIR
cd $COCO_DIR

# -------------------------------
# Skip download if already present
# -------------------------------
if [ -d "$IMAGES_DIR/train2017" ] && \
   [ -d "$IMAGES_DIR/val2017" ] && \
   [ -f "$ANN_DIR/instances_train2017.json" ]; then
    echo "COCO already downloaded on VM"
else
    echo "Downloading COCO dataset..."

    wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
    wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
    wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    unzip -q train2017.zip -d images/
    unzip -q val2017.zip -d images/
    unzip -q annotations_trainval2017.zip -d annotations/

    echo "Download complete"
fi

# -------------------------------
# Convert COCO → YOLO
# -------------------------------
echo "Converting annotations to YOLO format..."

python3 << PYEOF
import json
import os

def coco_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return (
        (x + bw / 2) / w,
        (y + bh / 2) / h,
        bw / w,
        bh / h,
    )

base = "/tmp/coco"
ann_dir = os.path.join(base, "annotations")
labels_dir = os.path.join(base, "labels")

for split in ["train2017", "val2017"]:
    ann_file = os.path.join(ann_dir, f"instances_{split}.json")

    with open(ann_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: i for i, c in enumerate(data["categories"])}

    out_dir = os.path.join(labels_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    grouped = {}
    for a in data["annotations"]:
        grouped.setdefault(a["image_id"], []).append(a)

    for img_id, annos in grouped.items():
        img = images[img_id]
        label_path = os.path.join(
            out_dir,
            img["file_name"].replace(".jpg", ".txt")
        )

        with open(label_path, "w") as f:
            for a in annos:
                cls = categories[a["category_id"]]
                x, y, w, h = coco_to_yolo(
                    a["bbox"],
                    img["width"],
                    img["height"]
                )
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("YOLO conversion complete")
PYEOF

echo "COCO ready"
'

# ============================================================
# Upload to GCS
# ============================================================

echo_info "Uploading COCO dataset to GCS"

gcloud compute ssh "${TEMP_VM_NAME}" \
    --zone="${TEMP_VM_ZONE}" \
    --command="
gsutil -m cp -r /tmp/coco/images gs://${BUCKET_NAME}/coco/
gsutil -m cp -r /tmp/coco/labels gs://${BUCKET_NAME}/coco/
gsutil -m cp -r /tmp/coco/annotations gs://${BUCKET_NAME}/coco/
"

echo_success "COCO dataset uploaded to gs://${BUCKET_NAME}/coco"

# ============================================================
# Cleanup
# ============================================================

echo_info "Deleting temporary VM"
gcloud compute instances delete "${TEMP_VM_NAME}" \
    --zone="${TEMP_VM_ZONE}" \
    --quiet

echo_success "Setup complete"
echo "Bucket: gs://${BUCKET_NAME}"
echo "Dataset path: gs://${BUCKET_NAME}/coco/"
