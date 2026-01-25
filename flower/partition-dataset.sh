#!/bin/bash

# Pre-partition COCO Dataset for Federated Learning - Generic Version
# Generates a single dataset with user-specified IID/non-IID configuration
# This script must be run BEFORE 04-deploy-application.sh
set -e

# Set the project ID for gcloud
gcloud config set project "$PROJECT_ID" >/dev/null 2>&1 || true

PROJECT_ID="inf022"
BUCKET_NAME="flybold-coco-${PROJECT_ID}"
LOCAL_DIR="$HOME/flybold_partitions"

mkdir -p "$LOCAL_DIR"

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

echo_warning() {
    echo -e "\n\033[1;33m[WARNING]\033[0m $1\n"
}

# Check prerequisites
echo_info "Checking prerequisites..."

if [ ! -f "vm-info.txt" ]; then
    echo_error "vm-info.txt not found. Run 02-setup-infrastructure.sh first."
fi

if [ ! -f ".env" ]; then
    echo_error ".env not found. This file should exist after initial setup."
fi

# Load environment variables
set -a  # Export all variables that are assigned
source .env
source vm-info.txt
set +a  # Stop exporting

# Configuration - Read from .env and user input
NUM_CLIENTS=${NUM_CLIENTS:-10}
MIN_TRAIN_IMAGES=${MIN_TRAIN_IMAGES:-2000}
MAX_TRAIN_IMAGES=${MAX_TRAIN_IMAGES:-2500}
MIN_VAL_IMAGES=${MIN_VAL_IMAGES:-500}
MAX_VAL_IMAGES=${MAX_VAL_IMAGES:-1000}
SEED=${DIRICHLET_SEED:-42}
SAME_SEED_VAL=${SAME_SEED_VAL:-true}

# Get dataset ID and IID clients from environment (passed from run-partition-on-temp-vm.sh)
if [ -z "$DATASET_ID" ]; then
    echo_error "DATASET_ID not set. This should be passed from run-partition-on-temp-vm.sh"
fi

if [ -z "$IID_CLIENTS" ]; then
    echo_error "IID_CLIENTS not set. This should be passed from run-partition-on-temp-vm.sh"
fi

echo_info "Dataset Partition Configuration:"
echo "  Dataset ID: $DATASET_ID"
echo "  Bucket: $BUCKET_NAME"
echo "  Clients: $NUM_CLIENTS"
echo "  IID Clients: $IID_CLIENTS"
echo "  Train size range: [$MIN_TRAIN_IMAGES, $MAX_TRAIN_IMAGES]"
echo "  Val size range: [$MIN_VAL_IMAGES, $MAX_VAL_IMAGES]"
echo "  Random seed: $SEED"
echo "  Same seed for val: $SAME_SEED_VAL"
echo ""
echo "  Storage location on VMs:"
echo "    /app/datasets_${DATASET_ID}/coco_partitions/client_X/"
echo ""

# Check if partition manifest already exists
MANIFEST_FILE="partition_manifest_dataset_${DATASET_ID}.json"

if [ -f "$MANIFEST_FILE" ]; then
    echo_warning "Partition manifest already exists: $MANIFEST_FILE"
    
    while true; do
        read -p "Do you want to regenerate partition manifest? (y/n) [n]: " regenerate
        regenerate=${regenerate:-n}
        if [[ "$regenerate" =~ ^[Yy]$ ]]; then
            SKIP_GENERATION=false
            echo_info "You chose to regenerate partition manifest."
            break
        elif [[ "$regenerate" =~ ^[Nn]$ ]]; then
            SKIP_GENERATION=true
            echo_info "You chose to use existing partition manifest."
            break
        else
            echo_warning "Please enter 'y' or 'n'."
        fi
    done
else
    SKIP_GENERATION=false
fi

# Step 1: Generate partition manifest
if [ "$SKIP_GENERATION" = false ]; then
    echo_info "Step 1: Generating partition manifest for dataset ${DATASET_ID}..."
    
    # Check if generate_partitions.py exists
    if [ ! -f "generate_partitions.py" ]; then
        echo_error "generate_partitions.py not found. Please ensure it's in the current directory."
    fi
    
    # Install required Python packages if needed
    echo "Installing required Python packages..."
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet numpy matplotlib seaborn
    echo "Packages installed."

    echo ""
    echo_info "Generating dataset manifest..."
    
    MANIFEST_FILE="partition_manifest_dataset_${DATASET_ID}.json"
    
    # Build command
    CMD="python3 -u generate_partitions.py \
        --bucket \"$BUCKET_NAME\" \
        --dataset-id \"$DATASET_ID\" \
        --iid-clients \"$IID_CLIENTS\" \
        --num-clients $NUM_CLIENTS \
        --min-train-images $MIN_TRAIN_IMAGES \
        --max-train-images $MAX_TRAIN_IMAGES \
        --min-val-images $MIN_VAL_IMAGES \
        --max-val-images $MAX_VAL_IMAGES \
        --seed $SEED \
        --output \"$MANIFEST_FILE\""
    
    # Add --same-seed-val flag if true
    if [ "$SAME_SEED_VAL" = "true" ]; then
        CMD="$CMD --same-seed-val"
    fi
    
    # Run partition generation
    eval $CMD
    
    if [ $? -ne 0 ]; then
        echo_error "Failed to generate partition manifest for dataset ${DATASET_ID}"
    fi
    
    echo_success "Dataset manifest generated: $MANIFEST_FILE"
    
    # Upload manifest to GCS for backup
    echo_info "Uploading manifest to GCS..."
    RUN_ID=${RUN_ID:-1}
    gsutil cp "$MANIFEST_FILE" "gs://${BUCKET_NAME}/partitions/dataset_${DATASET_ID}/manifest.json"
    
    # Upload plots if they exist
    PLOTS_DIR="client_plots_dataset_${DATASET_ID}"
    if [ -d "$PLOTS_DIR" ]; then
        echo_info "Uploading plots to GCS..."
        gsutil -m cp -r "$PLOTS_DIR" "gs://${BUCKET_NAME}/partitions/dataset_${DATASET_ID}/plots/"
        echo_success "Plots backed up to GCS"
    fi
    
    echo_success "Manifest backed up to GCS"
fi

# Verify manifest exists
echo_info "Verifying manifest exists..."
MANIFEST_FILE="partition_manifest_dataset_${DATASET_ID}.json"
if [ ! -f "$MANIFEST_FILE" ]; then
    echo_error "Partition manifest not found: $MANIFEST_FILE"
fi
echo "  ✓ Manifest verified: $MANIFEST_FILE"
echo_success "Manifest ready!"

# Step 2: Distribute data to VMs
echo_info "Step 2: Distributing partitioned data to client VMs..."

# Function to setup client partition on VM
setup_client_partition() {
    local vm_name=$1
    local vm_zone=$2
    local client_id=$3
    local dataset_id=$4
    local manifest_file="partition_manifest_dataset_${dataset_id}.json"
    
    echo_info "Setting up Dataset ${dataset_id}, Client ${client_id} on ${vm_name}..."
    
    # First, ensure /app directory exists with proper permissions
    gcloud compute ssh "$vm_name" --zone="$vm_zone" --quiet --command="
        sudo mkdir -p /app
        sudo chown -R \$USER:\$USER /app
        sudo chmod -R 755 /app
    " 2>/dev/null || true
    
    # Copy manifest to VM
    gcloud compute scp "$manifest_file" "${vm_name}:/app/partition_manifest_dataset_${dataset_id}.json" \
        --zone="$vm_zone" --quiet
    
    # Setup partition on VM
    gcloud compute ssh "$vm_name" --zone="$vm_zone" --command="
        set -e
        
        export CLIENT_ID=$client_id
        export DATASET_ID=$dataset_id
        export BUCKET_NAME=$BUCKET_NAME
        export MANIFEST=/app/partition_manifest_dataset_\${DATASET_ID}.json
        
        echo '[VM] Setting up Dataset '\$DATASET_ID', Client '\$CLIENT_ID' partition...'
        
        # Create directory structure with dataset-specific base
        BASE_DIR=/app/datasets_\${DATASET_ID}/coco_partitions/client_\${CLIENT_ID}
        
        # Use sudo with -n (non-interactive) to avoid password prompts
        # Create parent directory first
        sudo mkdir -p /app/datasets_\${DATASET_ID} 2>/dev/null || true
        sudo chmod 777 /app/datasets_\${DATASET_ID} 2>/dev/null || true
        
        # Create client-specific directories
        mkdir -p \$BASE_DIR/{images,labels}/{train2017,val2017}
        sudo chmod -R 755 /app/datasets_\${DATASET_ID} 2>/dev/null || true
        
        # Extract image lists and expected counts from manifest
        python3 << 'PYEOF'
import json
import os
import sys

manifest_path = os.environ['MANIFEST']
client_id = int(os.environ['CLIENT_ID'])
dataset_id = os.environ['DATASET_ID']
base_dir = f\"/app/datasets_{dataset_id}/coco_partitions/client_{client_id}\"

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

partition = manifest['partitions'][f'client_{client_id}']

# Get training images (all from train2017 split)
train_images = partition.get('train_images', [])
train_split = partition.get('train_split', 'train2017')

# Get validation images (all from val2017 split)
val_images = partition.get('val_images', [])
val_split = partition.get('val_split', 'val2017')

# Create file lists for gsutil with CORRECT source splits
with open(f'/app/train_images_ds{dataset_id}_c{client_id}.txt', 'w') as f:
    for img in train_images:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/{train_split}/{img}\\n\")

with open(f'/app/val_images_ds{dataset_id}_c{client_id}.txt', 'w') as f:
    for img in val_images:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/{val_split}/{img}\\n\")

# Create file lists for labels with CORRECT source splits
with open(f'/app/train_labels_list_ds{dataset_id}_c{client_id}.txt', 'w') as f:
    for img in train_images:
        basename = img.replace('.jpg', '')
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/labels/{train_split}/{basename}.txt\\n\")

with open(f'/app/val_labels_list_ds{dataset_id}_c{client_id}.txt', 'w') as f:
    for img in val_images:
        basename = img.replace('.jpg', '')
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/labels/{val_split}/{basename}.txt\\n\")

# Save expected counts
train_count = partition['train_count']
val_count = partition['val_count']

with open(f'/app/expected_counts_ds{dataset_id}_c{client_id}.txt', 'w') as f:
    f.write(f\"{train_count}\\n\")
    f.write(f\"{val_count}\\n\")

print(f\"[VM] Dataset {dataset_id}, Client {client_id}: {train_count} train (from {train_split}), {val_count} val (from {val_split})\")
PYEOF
        
        # Read expected counts
        EXPECTED_TRAIN=\$(head -1 /app/expected_counts_ds\${DATASET_ID}_c\${CLIENT_ID}.txt)
        EXPECTED_VAL=\$(tail -1 /app/expected_counts_ds\${DATASET_ID}_c\${CLIENT_ID}.txt)
        
        # Check if data already exists with correct counts
        CURRENT_TRAIN_IMGS=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_TRAIN_LABELS=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
        CURRENT_VAL_IMGS=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_VAL_LABELS=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
        
        echo \"[VM] =========================================\"
        echo \"[VM] Data Status Check for Dataset \${DATASET_ID}, Client \${CLIENT_ID}\"
        echo \"[VM] =========================================\"
        echo \"[VM] Expected: train=\$EXPECTED_TRAIN, val=\$EXPECTED_VAL\"
        echo \"[VM] Current:  train_imgs=\$CURRENT_TRAIN_IMGS, train_labels=\$CURRENT_TRAIN_LABELS\"
        echo \"[VM] Current:  val_imgs=\$CURRENT_VAL_IMGS, val_labels=\$CURRENT_VAL_LABELS\"
        
        # Check if everything is already complete
        ALL_COMPLETE=false
        if [ \$CURRENT_TRAIN_IMGS -eq \$EXPECTED_TRAIN ] && \\
           [ \$CURRENT_TRAIN_LABELS -ge \$CURRENT_TRAIN_IMGS ] && \\
           [ \$CURRENT_VAL_IMGS -eq \$EXPECTED_VAL ] && \\
           [ \$CURRENT_VAL_LABELS -ge \$CURRENT_VAL_IMGS ]; then
            ALL_COMPLETE=true
            echo \"[VM] ✓ All data already exists and verified - skipping all downloads!\"
            echo \"[VM] =========================================\"
        else
            echo \"[VM] ⚠ Data incomplete - will download missing files\"
            echo \"[VM] =========================================\"
        fi
        
        # Download training images if needed
        if [ \$CURRENT_TRAIN_IMGS -eq \$EXPECTED_TRAIN ]; then
            echo '[VM] ✓ Training images already complete, skipping download'
        else
            echo '[VM] Downloading training images...'
            cat /app/train_images_ds\${DATASET_ID}_c\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/train2017/ 2>/dev/null || true
            
            # Verify
            ACTUAL_TRAIN=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
            if [ \$ACTUAL_TRAIN -ne \$EXPECTED_TRAIN ]; then
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_TRAIN train images, got \$ACTUAL_TRAIN\"
            else
                echo \"[VM] ✓ Train images downloaded successfully\"
            fi
        fi
        
        # Download training labels if needed
        if [ \$CURRENT_TRAIN_LABELS -ge \$EXPECTED_TRAIN ]; then
            echo '[VM] ✓ Training labels already complete, skipping download'
        else
            echo '[VM] Downloading training labels...'
            cat /app/train_labels_list_ds\${DATASET_ID}_c\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/labels/train2017/ 2>/dev/null || true
            CURRENT_TRAIN_LABELS=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
            if [ \$CURRENT_TRAIN_LABELS -ge \$EXPECTED_TRAIN ]; then
                echo '[VM] ✓ Training labels downloaded'
            else
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_TRAIN labels, got \$CURRENT_TRAIN_LABELS\"
            fi
        fi
        
        # Download val images if needed
        if [ \$CURRENT_VAL_IMGS -eq \$EXPECTED_VAL ]; then
            echo '[VM] ✓ Validation images already complete, skipping download'
        else
            echo '[VM] Downloading validation images...'
            cat /app/val_images_ds\${DATASET_ID}_c\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/val2017/ 2>/dev/null || true
            
            # Verify
            ACTUAL_VAL=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
            if [ \$ACTUAL_VAL -ne \$EXPECTED_VAL ]; then
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_VAL val images, got \$ACTUAL_VAL\"
            else
                echo \"[VM] ✓ Val images downloaded successfully\"
            fi
        fi
        
        # Download val labels if needed
        if [ \$CURRENT_VAL_LABELS -ge \$EXPECTED_VAL ]; then
            echo '[VM] ✓ Validation labels already complete, skipping download'
        else
            echo '[VM] Downloading validation labels...'
            cat /app/val_labels_list_ds\${DATASET_ID}_c\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/labels/val2017/ 2>/dev/null || true
            CURRENT_VAL_LABELS=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
            if [ \$CURRENT_VAL_LABELS -ge \$EXPECTED_VAL ]; then
                echo '[VM] ✓ Validation labels downloaded'
            else
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_VAL labels, got \$CURRENT_VAL_LABELS\"
            fi
        fi
        
        # Create data YAML with dataset-specific name
        cat > \$BASE_DIR/coco_client_dataset_\${DATASET_ID}.yaml << EOF
train: \$BASE_DIR/images/train2017
val: \$BASE_DIR/images/val2017
nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
EOF
        
        # Final verification
        FINAL_TRAIN_IMGS=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
        FINAL_VAL_IMGS=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
        FINAL_TRAIN_LABELS=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
        FINAL_VAL_LABELS=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
        
        echo '[VM] ========================================='
        echo \"[VM] Final Verification - Dataset \$DATASET_ID, Client \$CLIENT_ID:\"
        echo '[VM] ========================================='
        echo \"[VM]   Train images: \$FINAL_TRAIN_IMGS / \$EXPECTED_TRAIN\"
        echo \"[VM]   Val images: \$FINAL_VAL_IMGS / \$EXPECTED_VAL\"
        echo \"[VM]   Train labels: \$FINAL_TRAIN_LABELS\"
        echo \"[VM]   Val labels: \$FINAL_VAL_LABELS\"
        
        # Check for mismatches
        if [ \$FINAL_TRAIN_IMGS -ne \$EXPECTED_TRAIN ]; then
            echo '[VM]   ⚠ Train images mismatch!'
            exit 1
        fi
        if [ \$FINAL_VAL_IMGS -ne \$EXPECTED_VAL ]; then
            echo '[VM]   ⚠ Val images mismatch!'
            exit 1
        fi
        if [ \$FINAL_TRAIN_LABELS -lt \$FINAL_TRAIN_IMGS ]; then
            echo '[VM]   ⚠ Missing train labels!'
            exit 1
        fi
        if [ \$FINAL_VAL_LABELS -lt \$FINAL_VAL_IMGS ]; then
            echo '[VM]   ⚠ Missing val labels!'
            exit 1
        fi
        
        echo \"[VM] ✅ Dataset \$DATASET_ID, Client \$CLIENT_ID partition setup complete and verified!\"
        echo '[VM] ========================================='
        
    "
    
    if [ $? -eq 0 ]; then
        echo_success "Dataset ${dataset_id}, Client ${client_id} partition setup complete on ${vm_name}"
    else
        echo_error "Failed to setup Dataset ${dataset_id}, Client ${client_id} partition on ${vm_name}"
    fi
}


# Prepare all VMs: ensure /app directory has proper permissions
echo_info "Preparing VMs: ensuring /app permissions..."
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    if [ -n "$CLIENT_VM" ] && [ -n "$CLIENT_ZONE" ]; then
        echo "  Preparing $CLIENT_VM..."
        gcloud compute ssh "$CLIENT_VM" --zone="$CLIENT_ZONE" --quiet --command="
            sudo mkdir -p /app
            sudo chown -R \$USER:\$USER /app
            sudo chmod -R 755 /app
        " 2>/dev/null || true
    fi
done &
wait
echo_success "/app directory prepared on all VMs"

# Process all 5 VMs (2 clients each) for this dataset
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    # Verify VM variables are loaded
    if [ -z "$CLIENT_VM" ] || [ -z "$CLIENT_ZONE" ]; then
        echo_error "VM variables not loaded for VM $i. CLIENT_VM='$CLIENT_VM', CLIENT_ZONE='$CLIENT_ZONE'"
    fi
    
    # Client IDs for this VM
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))
    
    echo_info "Processing VM $i: $CLIENT_VM (Dataset ${DATASET_ID}, Clients $CLIENT_ID_1, $CLIENT_ID_2)"
    
    # Setup both clients on this VM in parallel for this dataset
    setup_client_partition "$CLIENT_VM" "$CLIENT_ZONE" "$CLIENT_ID_1" "$DATASET_ID" &
    setup_client_partition "$CLIENT_VM" "$CLIENT_ZONE" "$CLIENT_ID_2" "$DATASET_ID" &
    
    # Wait for both to complete
    wait
done

echo ""
echo_success "=========================================="
echo_success "Dataset ${DATASET_ID} partitioned and distributed!"
echo_success "=========================================="
echo ""
echo "Summary:"
echo "  - Dataset ID: ${DATASET_ID}"
echo "  - IID Clients: ${IID_CLIENTS}"
echo "  - Storage: /app/datasets_${DATASET_ID}/"
echo ""
echo_success "✅ Partition workflow finished!"
echo ""

# Cleanup temp files
# rm -f /app/train_images_ds\${DATASET_ID}_c\${CLIENT_ID}.txt
# rm -f /app/val_images_ds\${DATASET_ID}_c\${CLIENT_ID}.txt
# rm -f /app/expected_counts_ds\${DATASET_ID}_c\${CLIENT_ID}.txt
# rm -f /app/train_labels_list_ds\${DATASET_ID}_c\${CLIENT_ID}.txt
# rm -f /app/val_labels_list_ds\${DATASET_ID}_c\${CLIENT_ID}.txt