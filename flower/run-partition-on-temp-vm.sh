#!/bin/bash
set -e

################################################################################
# WHAT THIS SCRIPT DOES:
# 
# This script automates the entire COCO dataset partitioning workflow:
#
# PHASE 1: Setup (Lines 20-80)
#   - Creates a temporary VM with high specs for heavy computation
#   - Waits for VM to boot and SSH to become available
#   - Copies required scripts (.py, .sh, .env, vm-info.txt) to temp VM
#
# PHASE 2: Partition Generation (Lines 82-150)
#   - Installs Python dependencies on temp VM (numpy, matplotlib, seaborn)
#   - Runs generate_partitions.py to create 4 dataset manifests
#   - Each manifest defines how to split COCO into 10 client partitions
#   - Uploads manifests and plots to GCS for backup
#
# PHASE 3: Data Distribution (Lines 152-250)
#   - Runs partition-dataset.sh on temp VM
#   - The temp VM SSHes to each of the 5 client VMs
#   - Downloads partitioned images/labels from GCS to client VMs
#   - Creates YAML config files on each client
#   - Verifies data integrity
#
# PHASE 4: Cleanup (Lines 252-280)
#   - Copies manifests and plots back to local machine
#   - Optionally deletes the temporary VM
#
# RESULT: 4 complete datasets ready for federated learning experiments
################################################################################

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

# Copy files to VM
echo "[INFO] Copying files to VM..."
gcloud compute scp "$PY_SCRIPT" "$SH_SCRIPT" "$ENV_FILE" "$VM_INFO_FILE" \
    "$VM_NAME:/tmp/" --zone="$ZONE" --quiet

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

# Create virtual environment
sudo mkdir -p /opt/partition_env
sudo chown -R $USER:$USER /opt/partition_env

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

# Copy and run setup script
gcloud compute scp /tmp/vm_setup.sh "$VM_NAME:/tmp/" --zone="$ZONE" --quiet
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --quiet --command="bash /tmp/vm_setup.sh"

################################################################################
# PHASE 3: RUN PARTITION WORKFLOW
################################################################################

echo ""
echo "=========================================="
echo "PHASE 3: Running partition workflow"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Generate 4 dataset partition manifests"
echo "  2. Distribute data to all client VMs"
echo "  3. Verify data integrity"
echo ""
echo "This may take 30-60 minutes..."
echo ""

# Create wrapper script that runs the partition workflow
cat > /tmp/run_workflow.sh << 'WORKFLOW_SCRIPT'
#!/bin/bash
set -e
set -o pipefail

# Ensure output is not buffered
export PYTHONUNBUFFERED=1

cd /tmp

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

# Run the partition workflow
echo "[WORKFLOW] Running partition-dataset.sh..."
./partition-dataset.sh

echo ""
echo "[WORKFLOW] ✅ Partition workflow complete!"
WORKFLOW_SCRIPT

# Copy and execute workflow
gcloud compute scp /tmp/run_workflow.sh "$VM_NAME:/tmp/" --zone="$ZONE" --quiet

echo "[INFO] Executing partition workflow on VM..."
echo "[INFO] (Streaming live output from VM)"
echo ""

# Execute with proper TTY allocation for live streaming
# Use --ssh-flag to force pseudo-terminal allocation
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --ssh-flag="-t" --command="bash -l /tmp/run_workflow.sh"

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
for dataset_id in 1 2 3 4; do
    gcloud compute scp \
        "$VM_NAME:/tmp/partition_manifest_dataset_${dataset_id}.json" \
        "partition_outputs/" \
        --zone="$ZONE" --quiet 2>/dev/null || \
        echo "  [WARNING] Dataset ${dataset_id} manifest not found"
done

# Copy plots
echo "[INFO] Downloading plots..."
for dataset_id in 1 2 3 4; do
    gcloud compute scp --recurse \
        "$VM_NAME:/tmp/client_plots_dataset_${dataset_id}" \
        "partition_outputs/" \
        --zone="$ZONE" --quiet 2>/dev/null || \
        echo "  [WARNING] Dataset ${dataset_id} plots not found"
done

echo ""
echo "=========================================="
echo "✅ SUCCESS! Workflow complete"
echo "=========================================="
echo ""
echo "Results:"
ls -lh partition_manifest_dataset_*.json 2>/dev/null || echo "  No manifests found"
echo ""
echo "Plots:"
ls -d partition_outputs/client_plots_dataset_* 2>/dev/null || echo "  No plots found"
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