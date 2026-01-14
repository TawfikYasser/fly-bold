#!/bin/bash
set -e

# =========================
# Config
# =========================
PROJECT_ID="inf022"
ZONE="us-central1-a"
VM_NAME="partition-tmp-vm"

MACHINE_TYPE="n1-standard-16"  # Increased RAM for better performance
DISK_SIZE="100GB"  # Increased for label caching

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
echo "[INFO] Creating temporary VM for partitioning"

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
    apt-get install -y python3 python3-pip python3-venv curl ca-certificates gnupg

    # Install Google Cloud SDK (for gsutil)
    echo "deb http://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt-get update -y
    apt-get install -y google-cloud-sdk

    # Create virtual environment for Python dependencies
    python3 -m venv /opt/partition_env
    source /opt/partition_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install required packages
    pip install numpy
    pip install matplotlib
    pip install seaborn
    pip install flwr-datasets

    echo "VM setup complete"
    '

    echo "[INFO] Waiting for VM to initialize (60 seconds)..."
    sleep 60
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
# 3. Run partition-dataset.sh
# =========================
echo "[INFO] Running partition-dataset.sh on VM"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd /tmp

# Ensure virtual environment exists
# Ensure virtual environment exists with correct permissions
sudo mkdir -p /opt/partition_env
sudo chown -R $USER:$USER /opt/partition_env

if [ ! -f /opt/partition_env/bin/activate ]; then
  python3 -m venv /opt/partition_env
fi
source /opt/partition_env/bin/activate

# Install/refresh required Python deps (handles reused VMs)
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet numpy matplotlib seaborn 'flwr-datasets>=0.5.0'

# Verify packages
echo '[VM] Verifying installed packages:'
python3 -c 'import numpy; print(f\"  numpy: {numpy.__version__}\")'
python3 -c 'import matplotlib; print(f\"  matplotlib: {matplotlib.__version__}\")'
python3 -c 'import seaborn; print(f\"  seaborn: {seaborn.__version__}\")'
python3 -c 'import flwr_datasets; print(f\"  flwr-datasets: {flwr_datasets.__version__}\")'

# Run partitioning script
./$SH_SCRIPT
"

# =========================
# 4. Copy outputs back
# =========================
echo "[INFO] Copying outputs back to local machine"

mkdir -p "$OUTPUT_DIR"

# Copy manifest
gcloud compute scp \
  "$VM_NAME:/tmp/*.json" \
  "$OUTPUT_DIR/" \
  --zone="$ZONE" || true

# Copy plots directory
gcloud compute scp \
  --recurse \
  "$VM_NAME:/tmp/client_plots" \
  "$OUTPUT_DIR/" \
  --zone="$ZONE" || true

echo "[SUCCESS] Done! Outputs saved to $OUTPUT_DIR/"
echo ""
echo "Files downloaded:"
ls -lh "$OUTPUT_DIR/"

if [ -d "$OUTPUT_DIR/client_plots" ]; then
    echo ""
    echo "Client plots:"
    ls -1 "$OUTPUT_DIR/client_plots/"
fi

# # =========================
# # 5. Ask about VM cleanup
# # =========================
# echo ""
# # Delete VM automatically
# echo "[INFO] Deleting temporary VM ($VM_NAME)..."
# gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
# echo "[SUCCESS] Temporary VM deleted."
# echo ""
# echo "[SUCCESS] ✅ Partitioning complete!"
# echo ""
