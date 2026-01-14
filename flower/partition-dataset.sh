#!/bin/bash

# Pre-partition COCO Dataset for Federated Learning
# This script must be run BEFORE 04-deploy-application.sh
set -e

# Set the project ID for gcloud
gcloud config set project "$PROJECT_ID" >/dev/null 2>&1 || true

PROJECT_ID="inf022"
BUCKET_NAME="flybold-coco-${PROJECT_ID}"
MANIFEST_FILE="partition_manifest.json"
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
source .env
source vm-info.txt

# Configuration
NUM_CLIENTS=${NUM_CLIENTS:-10}
MIN_TRAIN=${MIN_TRAIN:-1000}
MAX_TRAIN=${MAX_TRAIN:-5000}
SEED=${DIRICHLET_SEED:-42}
ALPHA_MIN=${DIRICHLET_ALPHA_MIN:-0.1}
ALPHA_MAX=${DIRICHLET_ALPHA_MAX:-1.5}
SAME_SEED_VAL=${SAME_SEED_VAL:-false}

echo_info "Partition Configuration:"
echo "  Bucket: $BUCKET_NAME"
echo "  Clients: $NUM_CLIENTS"
echo "  Train size range: [$MIN_TRAIN, $MAX_TRAIN]"
echo "  Val size: 50% of train"
echo "  Random seed: $SEED"
echo "  Alpha range: [$ALPHA_MIN, $ALPHA_MAX]"
echo "  Client 0: Uniform (alpha=1e6)"
echo "  Same seed for val: $SAME_SEED_VAL"
echo ""

# Check if partition already exists (interactive)
if [ -f "$MANIFEST_FILE" ]; then
    echo_warning "Partition manifest already exists: $MANIFEST_FILE"
    echo "Existing manifest details:"
    python3 -c "
import json
with open('$MANIFEST_FILE', 'r') as f:
    data = json.load(f)
    print(f\"  Seed: {data['metadata']['seed']}\")
    print(f\"  Clients: {data['metadata']['num_clients']}\")
    print(f\"  Train range: [{data['metadata']['min_train_per_client']}, {data['metadata']['max_train_per_client']}]\")
    print(f\"  Same seed val: {data['metadata']['same_seed_for_val']}\")
"

    # Ask user whether to regenerate
    while true; do
        read -p "Do you want to regenerate the partition manifest? This will overwrite existing data! (y/n) [n]: " regenerate
        regenerate=${regenerate:-n}
        if [[ "$regenerate" =~ ^[Yy]$ ]]; then
            SKIP_GENERATION=false
            echo_info "You chose to regenerate the partition manifest."
            break
        elif [[ "$regenerate" =~ ^[Nn]$ ]]; then
            SKIP_GENERATION=true
            echo_info "You chose to use the existing partition manifest."
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
    echo_info "Step 1: Generating partition manifest..."
    
    # Check if generate_partitions.py exists
    if [ ! -f "generate_partitions.py" ]; then
        echo_error "generate_partitions.py not found. Please ensure it's in the current directory."
    fi
    
    # Install required Python packages if needed
    echo "Installing required Python packages..."
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install --quiet numpy matplotlib seaborn
    echo "Packages installed."

    echo "Generating partition manifest, running generate_partitions.py ..."
    
    # Build command
    CMD="python3 -u generate_partitions.py \
        --bucket \"$BUCKET_NAME\" \
        --num-clients $NUM_CLIENTS \
        --min-train $MIN_TRAIN \
        --max-train $MAX_TRAIN \
        --seed $SEED \
        --alpha-min $ALPHA_MIN \
        --alpha-max $ALPHA_MAX \
        --output \"$MANIFEST_FILE\""
    
    # Add --same-seed-val flag if true
    if [ "$SAME_SEED_VAL" = "true" ]; then
        CMD="$CMD --same-seed-val"
    fi
    
    # Run partition generation
    eval $CMD
    
    if [ $? -ne 0 ]; then
        echo_error "Failed to generate partition manifest"
    fi
    
    echo_success "Partition manifest generated: $MANIFEST_FILE"
    
    # Upload manifest to GCS for backup
    echo_info "Uploading manifest to GCS..."
    RUN_ID=${RUN_ID:-1}
    gsutil cp "$MANIFEST_FILE" "gs://${BUCKET_NAME}/partitions/manifest_run${RUN_ID}.json"
    
    # Upload plots if they exist
    if [ -d "client_plots" ]; then
        echo_info "Uploading client plots to GCS..."
        gsutil -m cp -r client_plots "gs://${BUCKET_NAME}/partitions/run${RUN_ID}_plots/"
        echo_success "Plots backed up to GCS"
    fi
    
    echo_success "Manifest backed up to GCS"
fi

# Verify manifest exists
if [ ! -f "$MANIFEST_FILE" ]; then
    echo_error "Partition manifest not found: $MANIFEST_FILE"
fi

# Step 2: Distribute data to VMs
echo_info "Step 2: Distributing partitioned data to client VMs..."

# Function to setup client partition on VM
setup_client_partition() {
    local vm_name=$1
    local vm_zone=$2
    local client_id=$3
    local manifest_file=$4
    
    echo_info "Setting up partition for Client $client_id on $vm_name..."
    
    # Copy manifest to VM
    gcloud compute scp "$manifest_file" "${vm_name}:/tmp/partition_manifest.json" \
        --zone="$vm_zone" --quiet
    
    # Setup partition on VM
    gcloud compute ssh "$vm_name" --zone="$vm_zone" --command="
        set -e
        
        export CLIENT_ID=$client_id
        export BUCKET_NAME=$BUCKET_NAME
        export MANIFEST=/tmp/partition_manifest.json
        
        echo '[VM] Setting up Client $client_id partition...'
        
        # Create directory structure with sudo
        BASE_DIR=/app/datasets/coco_partitions/client_\${CLIENT_ID}
        sudo mkdir -p \$BASE_DIR/{images,labels}/{train2017,val2017}
        sudo chown -R \$USER:\$USER \$BASE_DIR
        
        # Extract image lists and expected counts from manifest
        python3 << 'PYEOF'
import json
import os
import sys

manifest_path = os.environ['MANIFEST']
client_id = int(os.environ['CLIENT_ID'])
base_dir = f\"/app/datasets/coco_partitions/client_{client_id}\"

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

partition = manifest['partitions'][f'client_{client_id}']

# Save image lists to temporary files WITH CORRECT SOURCE PATHS
train_from_train = partition.get('train_images_from_train2017', [])
train_from_val = partition.get('train_images_from_val2017', [])
val_from_train = partition.get('val_images_from_train2017', [])
val_from_val = partition.get('val_images_from_val2017', [])

with open(f'/tmp/train_images_{client_id}.txt', 'w') as f:
    for img in train_from_train:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/train2017/{img}\\n\")
    for img in train_from_val:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/val2017/{img}\\n\")

with open(f'/tmp/val_images_{client_id}.txt', 'w') as f:
    for img in val_from_train:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/train2017/{img}\\n\")
    for img in val_from_val:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/val2017/{img}\\n\")

# Save expected counts
train_count = partition['train_count']
val_count = partition['val_count']

with open(f'/tmp/expected_counts_{client_id}.txt', 'w') as f:
    f.write(f\"{train_count}\\n\")
    f.write(f\"{val_count}\\n\")

print(f\"[VM] Client {client_id}: {train_count} train, {val_count} val images\")
PYEOF
        
        # Read expected counts
        EXPECTED_TRAIN=\$(head -1 /tmp/expected_counts_\${CLIENT_ID}.txt)
        EXPECTED_VAL=\$(tail -1 /tmp/expected_counts_\${CLIENT_ID}.txt)
        
        # Check if data already exists with correct counts
        CURRENT_TRAIN_IMGS=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_TRAIN_LABELS=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
        CURRENT_VAL_IMGS=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_VAL_LABELS=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
        
        echo \"[VM] Expected: train=\$EXPECTED_TRAIN, val=\$EXPECTED_VAL\"
        echo \"[VM] Current: train_imgs=\$CURRENT_TRAIN_IMGS, train_labels=\$CURRENT_TRAIN_LABELS\"
        echo \"[VM] Current: val_imgs=\$CURRENT_VAL_IMGS, val_labels=\$CURRENT_VAL_LABELS\"
        
        # Download train images if needed
        if [ \$CURRENT_TRAIN_IMGS -eq \$EXPECTED_TRAIN ]; then
            echo '[VM] ✓ Training images already complete, skipping download'
        else
            echo '[VM] Downloading training images...'
            cat /tmp/train_images_\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/train2017/ 2>/dev/null || true
            
            # Verify
            ACTUAL_TRAIN=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
            if [ \$ACTUAL_TRAIN -ne \$EXPECTED_TRAIN ]; then
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_TRAIN train images, got \$ACTUAL_TRAIN\"
            else
                echo \"[VM] ✓ Train images downloaded successfully\"
            fi
        fi
        
        # Download train labels if needed
        if [ \$CURRENT_TRAIN_LABELS -ge \$CURRENT_TRAIN_IMGS ] && [ \$CURRENT_TRAIN_IMGS -eq \$EXPECTED_TRAIN ]; then
            echo '[VM] ✓ Training labels already complete, skipping download'
        else
            echo '[VM] Downloading training labels...'
            cd \$BASE_DIR/images/train2017
            > /tmp/train_labels_list_\${CLIENT_ID}.txt
            
            for img in *.jpg; do
                [ -f \"\$img\" ] || continue
                basename=\"\${img%.jpg}\"
                label_path=\"../../labels/train2017/\${basename}.txt\"
                if [ ! -f \"\$label_path\" ]; then
                    echo \"gs://\${BUCKET_NAME}/coco/labels/train2017/\${basename}.txt\" >> /tmp/train_labels_list_\${CLIENT_ID}.txt
                fi
            done
            
            if [ -s /tmp/train_labels_list_\${CLIENT_ID}.txt ]; then
                cat /tmp/train_labels_list_\${CLIENT_ID}.txt | gsutil -m cp -I ../../labels/train2017/ 2>/dev/null || true
            fi
            
            cd /tmp
            echo '[VM] ✓ Training labels downloaded'
        fi
        
        # Download val images if needed
        if [ \$CURRENT_VAL_IMGS -eq \$EXPECTED_VAL ]; then
            echo '[VM] ✓ Validation images already complete, skipping download'
        else
            echo '[VM] Downloading validation images...'
            cat /tmp/val_images_\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/val2017/ 2>/dev/null || true
            
            # Verify
            ACTUAL_VAL=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
            if [ \$ACTUAL_VAL -ne \$EXPECTED_VAL ]; then
                echo \"[VM] ⚠ WARNING: Expected \$EXPECTED_VAL val images, got \$ACTUAL_VAL\"
            else
                echo \"[VM] ✓ Val images downloaded successfully\"
            fi
        fi
        
        # Download val labels if needed
        if [ \$CURRENT_VAL_LABELS -ge \$CURRENT_VAL_IMGS ] && [ \$CURRENT_VAL_IMGS -eq \$EXPECTED_VAL ]; then
            echo '[VM] ✓ Validation labels already complete, skipping download'
        else
            echo '[VM] Downloading validation labels...'
            cd \$BASE_DIR/images/val2017
            > /tmp/val_labels_list_\${CLIENT_ID}.txt
            
            for img in *.jpg; do
                [ -f \"\$img\" ] || continue
                basename=\"\${img%.jpg}\"
                label_path=\"../../labels/val2017/\${basename}.txt\"
                if [ ! -f \"\$label_path\" ]; then
                    echo \"gs://\${BUCKET_NAME}/coco/labels/val2017/\${basename}.txt\" >> /tmp/val_labels_list_\${CLIENT_ID}.txt
                fi
            done
            
            if [ -s /tmp/val_labels_list_\${CLIENT_ID}.txt ]; then
                cat /tmp/val_labels_list_\${CLIENT_ID}.txt | gsutil -m cp -I ../../labels/val2017/ 2>/dev/null || true
            fi
            
            cd /tmp
            echo '[VM] ✓ Validation labels downloaded'
        fi
        
        # Create data YAML
        cat > \$BASE_DIR/coco_client.yaml << EOF
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
        echo \"[VM] Final Verification for Client \$CLIENT_ID:\"
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
        
        echo \"[VM] ✅ Client \$CLIENT_ID partition setup complete and verified!\"
        echo '[VM] ========================================='
        
        # Cleanup temp files
        rm -f /tmp/train_images_\${CLIENT_ID}.txt /tmp/val_images_\${CLIENT_ID}.txt
        rm -f /tmp/expected_counts_\${CLIENT_ID}.txt
        rm -f /tmp/train_labels_list_\${CLIENT_ID}.txt /tmp/val_labels_list_\${CLIENT_ID}.txt
    "
    
    if [ $? -eq 0 ]; then
        echo_success "Client $client_id partition setup complete on $vm_name"
    else
        echo_error "Failed to setup Client $client_id partition on $vm_name"
    fi
}

# Process all 5 VMs (2 clients each)
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    # Client IDs for this VM
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))
    
    echo_info "Processing VM $i: $CLIENT_VM (Clients $CLIENT_ID_1, $CLIENT_ID_2)"
    
    # Setup both clients on this VM in parallel
    setup_client_partition "$CLIENT_VM" "$CLIENT_ZONE" "$CLIENT_ID_1" "$MANIFEST_FILE" &
    setup_client_partition "$CLIENT_VM" "$CLIENT_ZONE" "$CLIENT_ID_2" "$MANIFEST_FILE" &
    
    # Wait for both to complete
    wait
done

echo_success "All partitions distributed!"

echo ""
echo_success "✅ Partition workflow finished!"
echo ""