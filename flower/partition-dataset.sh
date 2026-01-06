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
N_TRAIN=${N_TRAIN:-10000}
N_VAL=${N_VAL:-5000}
SEED=${DIRICHLET_SEED:-42}
ALPHA_MIN=${DIRICHLET_ALPHA_MIN:-0.1}
ALPHA_MAX=${DIRICHLET_ALPHA_MAX:-1.5}

echo_info "Partition Configuration:"
echo "  Bucket: $BUCKET_NAME"
echo "  Clients: $NUM_CLIENTS"
echo "  N_TRAIN per client: $N_TRAIN"
echo "  N_VAL per client: $N_VAL"
echo "  Random seed: $SEED"
echo "  Alpha range: [$ALPHA_MIN, $ALPHA_MAX]"
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
    print(f\"  N_TRAIN: {data['metadata']['n_train_per_client']}\")
    print(f\"  N_VAL: {data['metadata']['n_val_per_client']}\")
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
    # Run partition generation
    python3 -u generate_partitions.py \
        --bucket "$BUCKET_NAME" \
        --num-clients "$NUM_CLIENTS" \
        --n-train "$N_TRAIN" \
        --n-val "$N_VAL" \
        --seed "$SEED" \
        --alpha-min "$ALPHA_MIN" \
        --alpha-max "$ALPHA_MAX" \
        --output "$MANIFEST_FILE"
    
    if [ $? -ne 0 ]; then
        echo_error "Failed to generate partition manifest"
    fi
    
    echo_success "Partition manifest generated: $MANIFEST_FILE"
    
    # Upload manifest to GCS for backup
    echo_info "Uploading manifest to GCS..."
    RUN_ID=${RUN_ID:-1}
    gsutil cp "$MANIFEST_FILE" "gs://${BUCKET_NAME}/partitions/manifest_run${RUN_ID}.json"
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
        
        # Extract image lists from manifest
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

# Save image lists to temporary files
with open(f'/tmp/train_images_{client_id}.txt', 'w') as f:
    for img in partition['train_images']:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/train2017/{img}\\n\")

with open(f'/tmp/val_images_{client_id}.txt', 'w') as f:
    for img in partition['val_images']:
        f.write(f\"gs://{os.environ['BUCKET_NAME']}/coco/images/val2017/{img}\\n\")

print(f\"[VM] Client {client_id}: {len(partition['train_images'])} train, {len(partition['val_images'])} val images\")
PYEOF
        
        # Download train images
        echo '[VM] Downloading training images...'
        cat /tmp/train_images_\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/train2017/ 2>/dev/null || true
        
        # Download train labels (optimized)
        CURRENT_TRAIN_IMGS=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_TRAIN_LABELS=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
        
        if [ "\$CURRENT_TRAIN_LABELS" -ge "\$CURRENT_TRAIN_IMGS" ]; then
            echo '[VM] Training labels already sufficient, skipping download'
        else
            echo '[VM] Downloading training labels in parallel...'
            # Create list of labels needed based on downloaded images
            cd \$BASE_DIR/images/train2017
            > /tmp/train_labels_list_\${CLIENT_ID}.txt
            for img in *.jpg; do
                basename=\"\${img%.jpg}\"
                if [ ! -f \"../../labels/train2017/\${basename}.txt\" ]; then
                    echo \"gs://\${BUCKET_NAME}/coco/labels/train2017/\${basename}.txt\" >> /tmp/train_labels_list_\${CLIENT_ID}.txt
                fi
            done
            
            # Download all labels in parallel using gsutil -m
            if [ -s /tmp/train_labels_list_\${CLIENT_ID}.txt ]; then
                cat /tmp/train_labels_list_\${CLIENT_ID}.txt | gsutil -m cp -I ../../labels/train2017/ 2>/dev/null || true
            fi
            rm -f /tmp/train_labels_list_\${CLIENT_ID}.txt
            cd /tmp
            echo '[VM] Training labels downloaded.'
        fi
        
        # Download val images
        echo '[VM] Downloading validation images...'
        cat /tmp/val_images_\${CLIENT_ID}.txt | gsutil -m cp -I \$BASE_DIR/images/val2017/ 2>/dev/null || true
        
        # Download val labels (optimized)
        CURRENT_VAL_IMGS=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
        CURRENT_VAL_LABELS=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
        
        if [ "\$CURRENT_VAL_LABELS" -ge "\$CURRENT_VAL_IMGS" ]; then
            echo '[VM] Validation labels already sufficient, skipping download'
        else
            echo '[VM] Downloading validation labels in parallel...'
            # Create list of labels needed based on downloaded images
            cd \$BASE_DIR/images/val2017
            > /tmp/val_labels_list_\${CLIENT_ID}.txt
            for img in *.jpg; do
                basename=\"\${img%.jpg}\"
                if [ ! -f \"../../labels/val2017/\${basename}.txt\" ]; then
                    echo \"gs://\${BUCKET_NAME}/coco/labels/val2017/\${basename}.txt\" >> /tmp/val_labels_list_\${CLIENT_ID}.txt
                fi
            done
            
            # Download all labels in parallel using gsutil -m
            if [ -s /tmp/val_labels_list_\${CLIENT_ID}.txt ]; then
                cat /tmp/val_labels_list_\${CLIENT_ID}.txt | gsutil -m cp -I ../../labels/val2017/ 2>/dev/null || true
            fi
            rm -f /tmp/val_labels_list_\${CLIENT_ID}.txt
            cd /tmp
            echo '[VM] Validation labels downloaded.'
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
        
        # Verify
        TRAIN_COUNT=\$(ls \$BASE_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
        VAL_COUNT=\$(ls \$BASE_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
        TRAIN_LABEL_COUNT=\$(ls \$BASE_DIR/labels/train2017/*.txt 2>/dev/null | wc -l)
        VAL_LABEL_COUNT=\$(ls \$BASE_DIR/labels/val2017/*.txt 2>/dev/null | wc -l)
        
        echo '[VM] Verification:'
        echo \"[VM]   Train images: \$TRAIN_COUNT\"
        echo \"[VM]   Val images: \$VAL_COUNT\"
        echo \"[VM]   Train labels: \$TRAIN_LABEL_COUNT\"
        echo \"[VM]   Val labels: \$VAL_LABEL_COUNT\"
        
        # Cleanup temp files
        rm -f /tmp/train_images_\${CLIENT_ID}.txt /tmp/val_images_\${CLIENT_ID}.txt
        
        echo '[VM] Client \$CLIENT_ID partition setup complete'
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