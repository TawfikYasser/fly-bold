#!/bin/bash
set -e

PROJECT_ID="inf022"
ZONE="us-central1-a"
VM_NAME="partition-tmp-vm"
MACHINE_TYPE="n1-standard-16"  # 16 vCPUs, 60GB RAM
DISK_SIZE="100GB"

# Required files
PY_SCRIPT="generate_partitions.py"
SH_SCRIPT="partition-dataset.sh"
ENV_FILE=".env"
VM_INFO_FILE="vm-info.txt"

################################################################################
# PHASE 1: CREATE AND PREPARE TEMPORARY VM
################################################################################

echo ""
echo "=========================================="
echo "PHASE 1: Setting up temporary VM"
echo "=========================================="

# Validate required files exist locally
for file in "$PY_SCRIPT" "$SH_SCRIPT" "$ENV_FILE" "$VM_INFO_FILE"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file missing: $file"
        exit 1
    fi
done

gcloud config set project "$PROJECT_ID" --quiet

# Create VM if it doesn't exist
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "[INFO] Temp VM already exists: $VM_NAME"
else
    echo "[INFO] Creating temp VM: $VM_NAME"
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size="$DISK_SIZE" \
        --scopes=cloud-platform \
        --quiet
fi

# Wait for SSH
echo "[INFO] Waiting for SSH access..."
MAX_RETRIES=30
for i in $(seq 1 $MAX_RETRIES); do
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ready" --quiet &>/dev/null; then
        echo "[SUCCESS] SSH ready!"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: SSH timeout after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "  Waiting... ($i/$MAX_RETRIES)"
    sleep 10
done

# Create /app directory on VM with proper permissions
echo "[INFO] Creating /app directory on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="sudo mkdir -p /app && sudo chown -R $USER:$USER /app"

# Copy files to VM
echo "[INFO] Copying files to VM..."
gcloud compute scp "$PY_SCRIPT" "$SH_SCRIPT" "$ENV_FILE" "$VM_INFO_FILE" \
    "$VM_NAME:/app/" --zone="$ZONE" --quiet
################################################################################
# PHASE 1.5: GET USER INPUT FOR DATASET CONFIGURATION
################################################################################

echo ""
echo "=========================================="
echo "Dataset Configuration"
echo "=========================================="
echo ""

# Get dataset ID
while true; do
    read -p "Enter dataset ID (e.g., dataset_5, exp_heterog_v2): " DATASET_ID
    if [ -n "$DATASET_ID" ]; then
        echo "[INFO] Dataset ID: $DATASET_ID"
        break
    else
        echo "[ERROR] Dataset ID cannot be empty"
    fi
done

# Get IID clients
while true; do
    read -p "Enter IID client IDs (comma-separated, e.g., 0,3,5) or 'none' for all non-IID: " IID_CLIENTS
    if [ -n "$IID_CLIENTS" ]; then
        echo "[INFO] IID Clients: $IID_CLIENTS"
        break
    else
        echo "[ERROR] Input cannot be empty (use 'none' for all non-IID)"
    fi
done

# Display configuration summary
echo ""
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "  Dataset ID: $DATASET_ID"
echo "  IID Clients: $IID_CLIENTS"
echo "  Train range: Will be read from .env"
echo "  Val range: Will be read from .env"
echo "=========================================="
echo ""

read -p "Proceed with this configuration? (y/n) [y]: " confirm
confirm=${confirm:-y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "[INFO] Aborted by user"
    exit 0
fi

# Export variables to pass to VM
export DATASET_ID
export IID_CLIENTS

################################################################################
# PHASE 2: INSTALL DEPENDENCIES AND GENERATE MANIFESTS
################################################################################

echo ""
echo "=========================================="
echo "PHASE 2: Installing dependencies"
echo "=========================================="

# Create setup script to run on VM
cat > /tmp/vm_setup.sh << 'SETUP_SCRIPT'
#!/bin/bash
set -e

echo "[VM] Installing Python dependencies..."

# Create /opt/partition_env with sudo, then immediately hand ownership
# to the SSH user BEFORE creating the venv so all venv files are user-owned
sudo mkdir -p /opt/partition_env
sudo chown "$USER:$USER" /opt/partition_env

# Create venv as the SSH user (not root) so pip install works later without sudo
if [ ! -f /opt/partition_env/bin/activate ]; then
    python3 -m venv /opt/partition_env
fi

source /opt/partition_env/bin/activate

# Install packages
pip install --quiet --upgrade pip
pip install --quiet numpy matplotlib seaborn

echo "[VM] Installed packages:"
python3 -c 'import numpy; print(f"  ✓ numpy {numpy.__version__}")'
python3 -c 'import matplotlib; print(f"  ✓ matplotlib {matplotlib.__version__}")'
python3 -c 'import seaborn; print(f"  ✓ seaborn {seaborn.__version__}")'

# Configure gcloud
gcloud config set project inf022 --quiet
echo "[VM] Setup complete!"
SETUP_SCRIPT

# Copy and run setup script — run as the SSH user (not sudo) so the venv is user-owned
gcloud compute scp /tmp/vm_setup.sh "$VM_NAME:/app/" --zone="$ZONE" --quiet
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="bash /app/vm_setup.sh"

################################################################################
# PHASE 3: RUN PARTITION WORKFLOW
################################################################################

echo ""
echo "=========================================="
echo "PHASE 3: Running partition workflow"
echo "=========================================="
echo ""

# Create wrapper script that runs the partition workflow
cat > /tmp/run_workflow.sh << 'WORKFLOW_SCRIPT'
#!/bin/bash
set -e
set -o pipefail

# Ensure output is not buffered
export PYTHONUNBUFFERED=1

cd /app

echo "[WORKFLOW] Starting partition workflow..."

# Activate Python environment
if [ -f /opt/partition_env/bin/activate ]; then
    source /opt/partition_env/bin/activate
    echo "[WORKFLOW] Python env activated"
else
    echo "[WORKFLOW] ERROR: Python env not found"
    exit 1
fi

# Make partition script executable
chmod +x partition-dataset.sh
echo "[WORKFLOW] partition-dataset.sh is executable"

# Export user inputs for partition-dataset.sh
export DATASET_ID="$DATASET_ID"
export IID_CLIENTS="$IID_CLIENTS"

# Run the partition workflow
echo "[WORKFLOW] Running partition-dataset.sh with:"
echo "[WORKFLOW]   DATASET_ID=$DATASET_ID"
echo "[WORKFLOW]   IID_CLIENTS=$IID_CLIENTS"
./partition-dataset.sh

echo ""
echo "[WORKFLOW] ✅ Partition workflow complete!"
WORKFLOW_SCRIPT

# Copy and execute workflow
gcloud compute scp /tmp/run_workflow.sh "$VM_NAME:/app/" --zone="$ZONE" --quiet

echo "[INFO] Executing partition workflow on VM..."
echo "[INFO] (Streaming live output from VM)"
echo ""

# Execute with proper TTY allocation for live streaming
# Use --ssh-flag to force pseudo-terminal allocation
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --ssh-flag="-t" --command="
export DATASET_ID='$DATASET_ID'
export IID_CLIENTS='$IID_CLIENTS'
bash -l /app/run_workflow.sh
"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Partition workflow failed on VM"
    echo "To debug, SSH to VM: gcloud compute ssh $VM_NAME --zone=$ZONE"
    exit 1
fi

################################################################################
# PHASE 4: COPY RESULTS BACK AND CLEANUP
################################################################################

echo ""
echo "=========================================="
echo "PHASE 4: Copying results back"
echo "=========================================="

mkdir -p partition_outputs

# Copy manifests
echo "[INFO] Downloading manifests..."
gcloud compute scp \
    "$VM_NAME:/app/partition_manifest_dataset_${DATASET_ID}.json" \
    "partition_outputs/" \
    --zone="$ZONE" --quiet 2>/dev/null || \
    echo "  [WARNING] Dataset ${DATASET_ID} manifest not found"

# Copy plots
echo "[INFO] Downloading plots..."
gcloud compute scp --recurse \
    "$VM_NAME:/app/client_plots_dataset_${DATASET_ID}" \
    "partition_outputs/" \
    --zone="$ZONE" --quiet 2>/dev/null || \
    echo "  [WARNING] Dataset ${DATASET_ID} plots not found"

echo ""
echo "=========================================="
echo "✅ SUCCESS! Workflow complete"
echo "=========================================="
echo ""


# Cleanup
# read -p "Delete temporary VM? (y/n) [y]: " delete_vm
# delete_vm=${delete_vm:-y}

# if [[ "$delete_vm" =~ ^[Yy]$ ]]; then
#     echo "[INFO] Deleting temp VM..."
#     gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
#     echo "✅ Cleanup complete!"
# else
#     echo "[INFO] Temp VM kept: $VM_NAME (zone: $ZONE)"
#     echo "     To delete later: gcloud compute instances delete $VM_NAME --zone=$ZONE"
# fi

echo ""
echo "=========================================="
echo "🎉 All done!"
echo "=========================================="