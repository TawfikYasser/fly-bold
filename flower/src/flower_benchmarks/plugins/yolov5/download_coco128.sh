#!/usr/bin/env bash
# flower_benchmarks/plugins/yolov5/download_coco128.sh
# Download COCO128 zip from Ultralytics release and ensure YOLOv5 layout:
# datasets/coco128/images/train2017, images/val2017, labels/train2017, labels/val2017
#
# Usage:
#   cd <project-root>
#   chmod +x flower_benchmarks/plugins/yolov5/download_coco128.sh
#   ./flower_benchmarks/plugins/yolov5/download_coco128.sh
set -euo pipefail

# Config
COCO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
TMPDIR="$(mktemp -d)"
SCRIPT_DIR="$(pwd)"
DEST_BASE="${SCRIPT_DIR}/datasets"
DEST_DIR="${DEST_BASE}/coco128"
VAL_RATIO=0.2   # fraction of images to reserve for validation if val2017 missing (adjustable)
SHUFFLE_CMD="shuf"  # fallback to python if shuf not available

echo
echo "=== Download & prepare COCO128 (v0.0.0 asset) ==="
echo "Working in temp dir: $TMPDIR"
echo

cd "$TMPDIR"

# 1) download
if command -v wget >/dev/null 2>&1; then
  echo "Downloading coco128.zip with wget..."
  wget -q -O coco128.zip "$COCO_URL"
elif command -v curl >/dev/null 2>&1; then
  echo "Downloading coco128.zip with curl..."
  curl -L -s -o coco128.zip "$COCO_URL"
else
  echo "Error: neither wget nor curl is installed. Please install one and retry." >&2
  exit 1
fi

# 2) extract
EXTRACT_DIR="$TMPDIR/extracted"
mkdir -p "$EXTRACT_DIR"
if command -v unzip >/dev/null 2>&1; then
  echo "Extracting coco128.zip with unzip..."
  unzip -q coco128.zip -d "$EXTRACT_DIR"
else
  echo "unzip not found; using python zipfile to extract..."
  python - <<'PY'
import zipfile
zf = zipfile.ZipFile('coco128.zip','r')
zf.extractall('extracted')
print("Extracted with python zipfile")
PY
fi

# 3) locate candidate dataset dir
FOUND=""
if [ -d "${EXTRACT_DIR}/coco128" ]; then
  FOUND="${EXTRACT_DIR}/coco128"
else
  # scan for directories containing images/ and labels/
  echo "Searching for directory with YOLO layout (images/ + labels/)..."
  while IFS= read -r -d '' d; do
    if [ -d "$d/images" ] || [ -d "$d/labels" ]; then
      FOUND="$d"
      break
    fi
  done < <(find "$EXTRACT_DIR" -maxdepth 3 -type d -print0)
fi

if [ -z "$FOUND" ]; then
  echo "Could not locate coco128 dataset folder after extraction. Debug listing:"
  find "$EXTRACT_DIR" -maxdepth 3 -print
  echo "Aborting." >&2
  exit 1
fi

echo "Found candidate dataset dir: $FOUND"

# 4) prepare destination
mkdir -p "$DEST_BASE"
if [ -d "$DEST_DIR" ]; then
  echo "Existing datasets/coco128 found — removing it to replace with fresh copy..."
  rm -rf "$DEST_DIR"
fi
mkdir -p "$DEST_DIR"

# 5) copy or move contents
echo "Copying dataset content to $DEST_DIR ..."
# use rsync if available to preserve structure, else cp -r
if command -v rsync >/dev/null 2>&1; then
  rsync -a "$FOUND"/ "$DEST_DIR"/
else
  cp -r "$FOUND"/. "$DEST_DIR"/
fi

# 6) normalize layout:
# We want: images/train2017, images/val2017, labels/train2017, labels/val2017
IMAGES_DIR="${DEST_DIR}/images"
LABELS_DIR="${DEST_DIR}/labels"

# create directories if missing
mkdir -p "$IMAGES_DIR" "$LABELS_DIR"

# Helper to move flat images/labels into train2017/val2017 if needed
ensure_subdirs() {
  local base="$1"   # images or labels
  local want_train="$2"
  # If base/train2017 exists, OK
  if [ -d "${DEST_DIR}/${base}/train2017" ]; then
    return 0
  fi
  # If files are directly under base (no train2017), create train2017 and move them in
  shopt -s nullglob
  files=("${DEST_DIR}/${base}"/*)
  shopt -u nullglob
  if [ ${#files[@]} -gt 0 ]; then
    echo "Creating ${base}/train2017 and moving files into it..."
    mkdir -p "${DEST_DIR}/${base}/train2017"
    # Move only files (skip directories like 'val2017' if they exist)
    for f in "${DEST_DIR}/${base}"/*; do
      if [ -f "$f" ]; then
        mv "$f" "${DEST_DIR}/${base}/train2017/"
      fi
    done
  else
    # No files directly under base; maybe base already has train2017/val2017 (handled earlier)
    :
  fi
}

# Normalize images
ensure_subdirs "images" "train2017"
ensure_subdirs "labels" "train2017"

# Now check for val2017 presence; if missing, we create it by splitting train2017
IMG_TRAIN="${DEST_DIR}/images/train2017"
IMG_VAL="${DEST_DIR}/images/val2017"
LBL_TRAIN="${DEST_DIR}/labels/train2017"
LBL_VAL="${DEST_DIR}/labels/val2017"

# If val is missing but there are images in train, perform split
if [ ! -d "$IMG_VAL" ]; then
  echo "val2017 directory not found. Creating val2017 by splitting train2017 with ratio $VAL_RATIO ..."
  mkdir -p "$IMG_VAL"
  mkdir -p "$LBL_VAL"

  # Gather all image files (jpg/png)
  mapfile -t ALL_IMAGES < <(find "$IMG_TRAIN" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \))
  TOTAL=${#ALL_IMAGES[@]}
  if [ "$TOTAL" -eq 0 ]; then
    echo "No images found in $IMG_TRAIN — aborting." >&2
    exit 1
  fi

  # Number to move
  # Use python for robust rounding/shuffling if shuf not present
  VAL_COUNT=$(python - <<PY
import math, json, sys, random
total = ${TOTAL}
val_ratio = ${VAL_RATIO}
val = int(round(total * val_ratio))
# ensure at least 1 validation image if possible but not more than total-1
val = max(1, min(val, total-1))
print(val)
PY
)

  echo "Found $TOTAL images in train2017; moving $VAL_COUNT to val2017."

  # Select VAL_COUNT images randomly
  if command -v shuf >/dev/null 2>&1; then
    # shuf is available: use it
    mapfile -t SELECTED < <(printf "%s\n" "${ALL_IMAGES[@]}" | shuf -n "$VAL_COUNT")
  else
    # fallback to python sampling
    mapfile -t SELECTED < <(python - <<PY
import random, sys, json
imgs = json.loads('''$(python - <<'PY2'
import json,sys
imgs = []
import os
p='''"$IMG_TRAIN"'''
for f in sorted([x for x in __import__('glob').glob(p+'/*') if __import__('os').path.isfile(x)]):
    imgs.append(f)
print(json.dumps(imgs))
PY2
)''')
random.shuffle(imgs)
for i in imgs[:${VAL_COUNT}]:
    print(i)
PY
)
  fi

  # Move selected images and associated labels
  for img in "${SELECTED[@]}"; do
    basename=$(basename "$img")
    stem="${basename%.*}"
    # Move image
    mv "$img" "$IMG_VAL/"
    # Move corresponding label if exists, else create empty
    src_label="${LBL_TRAIN}/${stem}.txt"
    dest_label="${LBL_VAL}/${stem}.txt"
    if [ -f "$src_label" ]; then
      mv "$src_label" "$dest_label"
    else
      # create empty label file to avoid YOLO complaining
      echo -n "" > "$dest_label"
    fi
  done
else
  echo "val2017 already exists — leaving as-is."
  # Ensure label val dir exists
  mkdir -p "$LBL_VAL"
  # If val images exist but val labels missing, try to copy labels by matching stems
  if [ -d "$IMG_VAL" ] && [ ! -d "$LBL_VAL" ]; then
    mkdir -p "$LBL_VAL"
  fi
fi

# Ensure label files exist for remaining train images (create empty if missing)
for img in "${DEST_DIR}/images/train2017/"*; do
  if [ -f "$img" ]; then
    stem="$(basename "$img")"; stem="${stem%.*}"
    lbl="${DEST_DIR}/labels/train2017/${stem}.txt"
    [ -f "$lbl" ] || echo -n "" > "$lbl"
  fi
done
for img in "${DEST_DIR}/images/val2017/"*; do
  if [ -f "$img" ]; then
    stem="$(basename "$img")"; stem="${stem%.*}"
    lbl="${DEST_DIR}/labels/val2017/${stem}.txt"
    [ -f "$lbl" ] || echo -n "" > "$lbl"
  fi
done

# 7) sanity report
NUM_TRAIN_IMG=$(find "${DEST_DIR}/images/train2017" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l || true)
NUM_VAL_IMG=$(find "${DEST_DIR}/images/val2017" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l || true)
NUM_TRAIN_LBL=$(find "${DEST_DIR}/labels/train2017" -type f -iname '*.txt' | wc -l || true)
NUM_VAL_LBL=$(find "${DEST_DIR}/labels/val2017" -type f -iname '*.txt' | wc -l || true)

echo
echo "COCO128 prepared at: $DEST_DIR"
echo "Train images: $NUM_TRAIN_IMG   Train labels: $NUM_TRAIN_LBL"
echo "Val   images: $NUM_VAL_IMG    Val   labels: $NUM_VAL_LBL"
echo

echo "Cleaning up tempdir $TMPDIR"
rm -rf "$TMPDIR" || true

echo "Done. You can now use: $DEST_DIR"
