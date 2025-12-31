#!/bin/bash
set -e

# =========================
# Config
# =========================
PROJECT_ID="inf022"
ZONE="us-central1-a"
VM_NAME="partition-tmp-vm"

MACHINE_TYPE="n1-standard-8"
DISK_SIZE="50GB"

# Files
PY_SCRIPT="generate_partitions.py"
SH_SCRIPT="partition-dataset.sh"
ENV_FILE=".env"
VM_INFO_FILE="vm-info.txt"

# Local output dir
OUTPUT_DIR="partition_outputs"

# =========================
# Safety checks
# =========================
[ -f "$PY_SCRIPT" ] || { echo "Missing $PY_SCRIPT"; exit 1; }
[ -f "$SH_SCRIPT" ] || { echo "Missing $SH_SCRIPT"; exit 1; }

gcloud config set project "$PROJECT_ID" >/dev/null

# =========================
# 1. Create VM
# =========================
echo "[INFO] Creating temporary VM"

if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "[INFO] Temp VM already exists, using it..."
else
    gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size="$DISK_SIZE" \
    --scopes=cloud-platform \
    --metadata=startup-script='#!/bin/bash
    set -e

    apt-get update -y
    apt-get install -y python3 python3-pip curl ca-certificates gnupg

    # Install Google Cloud SDK (for gsutil)
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt-get update -y
    apt-get install -y google-cloud-sdk

    # Python deps required by generate_partitions.py
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy matplotlib seaborn
    '

    echo "[INFO] Waiting for VM to initialize..."
    sleep 40
fi

# =========================
# 2. Copy scripts
# =========================
echo "[INFO] Copying scripts to VM"

gcloud compute scp "$PY_SCRIPT" "$VM_NAME:/tmp/" --zone="$ZONE"
gcloud compute scp "$SH_SCRIPT" "$VM_NAME:/tmp/" --zone="$ZONE"
gcloud compute scp "$ENV_FILE" "$VM_NAME:/tmp/" --zone="$ZONE"
gcloud compute scp "$VM_INFO_FILE" "$VM_NAME:/tmp/" --zone="$ZONE"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="chmod +x /tmp/$SH_SCRIPT"

# =========================
# 3. Run ONLY partition-dataset.sh
# =========================
echo "[INFO] Running partition-dataset.sh on VM"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd /tmp
./$SH_SCRIPT
"

# =========================
# 4. Copy outputs back
# =========================
echo "[INFO] Copying outputs back"

mkdir -p "$OUTPUT_DIR"

gcloud compute scp \
  "$VM_NAME:/tmp/*.json" \
  "$OUTPUT_DIR/" \
  --zone="$ZONE" || true

echo "[INFO] Done. Outputs saved to $OUTPUT_DIR/"