#!/bin/bash

# Deploy Flybold Application with Dynamic IP Handling
set -e

PROJECT_ID="inf022"
BUCKET_NAME="flybold-coco-${PROJECT_ID}"

echo_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

echo_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
}

echo_warning() {
    echo -e "\n\033[1;33m[WARNING]\033[0m $1\n"
}

# SSH/SCP connection settings with timeouts and connection pooling
SSH_FLAGS=(
    "-o ConnectTimeout=15"
    "-o ServerAliveInterval=5"
    "-o ServerAliveCountMax=3"
    "-o StrictHostKeyChecking=no"
    "-o UserKnownHostsFile=/dev/null"
    "-o ControlMaster=auto"
    "-o ControlPath=/tmp/ssh-%r@%h:%p"
    "-o ControlPersist=600"
    "-o LogLevel=ERROR"
)

# Helper function: Execute SSH command with retry logic
ssh_with_retry() {
    local vm=$1
    local zone=$2
    local command=$3
    local max_retries=${4:-3}
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if gcloud compute ssh "$vm" --zone="$zone" \
            ${SSH_FLAGS[@]/#/--ssh-flag=} \
            --command="$command" 2>&1; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            local wait_time=$((retry_count * 3))
            echo "  ⚠️  SSH command failed, retrying in ${wait_time}s... ($retry_count/$max_retries)" >&2
            sleep $wait_time
        fi
    done
    
    echo_error "SSH command failed after $max_retries attempts to $vm" >&2
    return 1
}

# Helper function: SCP with retry logic
scp_with_retry() {
    local source=$1
    local destination=$2
    local zone=$3
    local recurse=${4:-false}
    local max_retries=3
    local retry_count=0
    
    local scp_cmd="gcloud compute scp"
    [ "$recurse" = "true" ] && scp_cmd="$scp_cmd --recurse"
    scp_cmd="$scp_cmd --compress"
    
    while [ $retry_count -lt $max_retries ]; do
        if $scp_cmd \
            ${SSH_FLAGS[@]/#/--scp-flag=} \
            "$source" "$destination" \
            --zone="$zone" 2>&1 | grep -v "Warning: Permanently added"; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            local wait_time=$((retry_count * 3))
            echo "  ⚠️  SCP failed, retrying in ${wait_time}s... ($retry_count/$max_retries)" >&2
            sleep $wait_time
        fi
    done
    
    echo_error "SCP failed after $max_retries attempts" >&2
    return 1
}

# Function to fetch current IP addresses from GCP
fetch_vm_ips() {
    local vm_name=$1
    local zone=$2
    
    echo -e "  Fetching IPs for $vm_name..." >&2
    
    local internal_ip=$(gcloud compute instances describe $vm_name \
        --zone=$zone \
        --format='get(networkInterfaces[0].networkIP)' 2>/dev/null)
    
    local external_ip=$(gcloud compute instances describe $vm_name \
        --zone=$zone \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ -z "$internal_ip" ]; then
        echo_error "Failed to fetch internal IP for $vm_name. Is the VM running?" >&2
        exit 1
    fi
    
    echo "    Internal: $internal_ip, External: ${external_ip:-None}" >&2
    
    echo "$internal_ip|$external_ip"
}

# Function to update vm-info.txt with current IPs
update_vm_info() {
    echo_info "Updating vm-info.txt with current IP addresses..."
    
    if [ -f "vm-info.txt" ]; then
        cp vm-info.txt vm-info.txt.backup
        echo "  Backed up existing vm-info.txt"
    fi
    
    cat > vm-info.txt << EOF
PROJECT_ID=$PROJECT_ID
REGION=us-central1
NETWORK=flybold-network

EOF

    local server_ips=$(fetch_vm_ips "flybold-server" "us-central1-a")
    local server_internal=$(echo $server_ips | cut -d'|' -f1)
    local server_external=$(echo $server_ips | cut -d'|' -f2)
    
    cat >> vm-info.txt << EOF
SERVER_VM=flybold-server
SERVER_ZONE=us-central1-a
SERVER_INTERNAL_IP=$server_internal
SERVER_EXTERNAL_IP=$server_external

EOF

    local client_zones=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-central1-a")
    
    for i in $(seq 1 5); do
        local zone=${client_zones[$((i-1))]}
        local client_ips=$(fetch_vm_ips "flybold-client-$i" "$zone")
        local client_internal=$(echo $client_ips | cut -d'|' -f1)
        local client_external=$(echo $client_ips | cut -d'|' -f2)
        
        cat >> vm-info.txt << EOF
CLIENT_${i}_VM=flybold-client-${i}
CLIENT_${i}_ZONE=$zone
CLIENT_${i}_INTERNAL_IP=$client_internal
CLIENT_${i}_EXTERNAL_IP=$client_external

EOF
    done
    
    echo_success "vm-info.txt updated with current IPs"
}

# Load or create VM info
if [ ! -f "vm-info.txt" ]; then
    echo_error "vm-info.txt not found. Run 02-setup-infrastructure.sh first."
    exit 1
fi

# Check if VMs are running and update IPs
echo_info "Checking VM status and fetching current IPs..."
update_vm_info
source vm-info.txt

echo_success "All VM IPs refreshed successfully!"
echo ""
echo "Current Configuration:"
echo "  Server: $SERVER_VM ($SERVER_INTERNAL_IP)"
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
    echo "  ${!CLIENT_VM_VAR}: ${!CLIENT_IP_VAR}"
done
echo ""

# Clean local cache
echo_info "Cleaning local Python cache..."
find ./src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./yolov5 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./src -type f -name "*.pyc" -delete 2>/dev/null || true
find ./yolov5 -type f -name "*.pyc" -delete 2>/dev/null || true
echo_success "Local cache cleaned"

# Load Docker image
if [ ! -f "docker-image-info.txt" ]; then
    echo_error "docker-image-info.txt not found. Run 03-build-push-image.sh first."
    exit 1
fi
DOCKER_IMAGE=$(grep '^DOCKER_IMAGE=' docker-image-info.txt | cut -d'=' -f2)

echo_info "Starting Flybold deployment"

# Check for existing .env or prompt for parameters
if [ -f .env ]; then
    echo "Existing .env found. Use these settings?"
    cat .env
    read -p "Use existing config? (y/n) [y]: " use_existing
    use_existing=${use_existing:-y}
    if [[ $use_existing =~ ^[Yy]$ ]]; then
        source .env
        SKIP_PROMPTS=true
        echo_warning "Overriding SERVER_INTERNAL_IP with current value from vm-info.txt"
        echo "  Old IP in .env: ${SERVER_INTERNAL_IP}"
        source vm-info.txt
        echo "  Current IP: ${SERVER_INTERNAL_IP}"
    else
        SKIP_PROMPTS=false
    fi
else
    SKIP_PROMPTS=false
fi

if [ "$SKIP_PROMPTS" = false ]; then
    echo_info "Configuration Parameters"
    
    read -p "Enable GPU? (y/n) [n]: " gpu_input
    ENABLE_GPU=${gpu_input:-n}
    if [[ $ENABLE_GPU =~ ^[Yy]$ ]]; then
        ENABLE_GPU=true
        read -p "CPUs per client [4]: " NUM_CPUS
        NUM_CPUS=${NUM_CPUS:-4}
        read -p "GPUs per client [1]: " NUM_GPUS
        NUM_GPUS=${NUM_GPUS:-1}
    else
        ENABLE_GPU=false
        NUM_CPUS=4
        NUM_GPUS=0
    fi
    
    read -p "Enable TLS? (y/n) [n]: " tls_input
    ENABLE_TLS=${tls_input:-n}
    if [[ $ENABLE_TLS =~ ^[Yy]$ ]]; then
        ENABLE_TLS=true
        INSECURE=false
        if [ ! -d "certs" ]; then
            echo_info "Generating TLS certificates..."
            ./generate_certs.sh
        fi
    else
        ENABLE_TLS=false
        INSECURE=true
    fi
    
    echo_info "Dataset parameters should match partition manifest!"
    read -p "Training images per client [10000]: " N_TRAIN
    N_TRAIN=${N_TRAIN:-10000}
    read -p "Validation images per client [5000]: " N_VAL
    N_VAL=${N_VAL:-5000}
    
    read -p "Number of rounds [30]: " NUM_SERVER_ROUNDS
    NUM_SERVER_ROUNDS=${NUM_SERVER_ROUNDS:-30}
    
    read -p "Local epochs [5]: " LOCAL_EPOCHS
    LOCAL_EPOCHS=${LOCAL_EPOCHS:-5}
    
    read -p "Batch size [24]: " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-24}
    
    read -p "Fraction train [0.8]: " FRACTION_TRAIN
    FRACTION_TRAIN=${FRACTION_TRAIN:-0.8}
    
    read -p "Fraction evaluate [0.8]: " FRACTION_EVALUATE
    FRACTION_EVALUATE=${FRACTION_EVALUATE:-0.8}
    
    read -p "Learning rate [0.005]: " LR
    LR=${LR:-0.005}
    
    read -p "YOLO size (n/s/m/l/x) [s]: " YOLO_SIZE
    YOLO_SIZE=${YOLO_SIZE:-s}
    
    read -p "Image size [512]: " IMG_SIZE
    IMG_SIZE=${IMG_SIZE:-512}

    read -p "Dataset choice [1]: " DATASET
    DATASET=${DATASET:-1}

    read -p "Use pretrained weights? (y/n) [y]: " pretrained_input
    USE_PRETRAINED=${pretrained_input:-y}
    if [[ $USE_PRETRAINED =~ ^[Yy]$ ]]; then
        USE_PRETRAINED=1
    else
        USE_PRETRAINED=0
    fi
fi

# Get/increment run_id
echo_info "Getting run ID from GCS..."
RUN_ID=$(gsutil cat gs://${BUCKET_NAME}/run_id.txt 2>/dev/null || echo "1")
echo "Current run_id: $RUN_ID"
NEXT_RUN_ID=$((RUN_ID + 1))
echo $NEXT_RUN_ID | gsutil cp - gs://${BUCKET_NAME}/run_id.txt
echo_success "Run ID incremented to $NEXT_RUN_ID"
RUN_ID=$NEXT_RUN_ID

# Save config
# Save updated config to .env using sed (preserve other comments/structure)
sed -i "s/^RUN_ID=.*/RUN_ID=$RUN_ID/" .env
sed -i "s|^DOCKER_IMAGE=.*|DOCKER_IMAGE=$DOCKER_IMAGE|" .env
sed -i "s/^SERVER_INTERNAL_IP=.*/SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP/" .env

# Save config to GCS
CONFIG_JSON=$(cat <<EOJSON
{
  "run_id": $RUN_ID,
  "num_rounds": $NUM_SERVER_ROUNDS,
  "local_epochs": $LOCAL_EPOCHS,
  "batch_size": $BATCH_SIZE,
  "fraction_train": $FRACTION_TRAIN,
  "fraction_evaluate": $FRACTION_EVALUATE,
  "lr": $LR,
  "yolo_size": "$YOLO_SIZE",
  "img_size": $IMG_SIZE,
  "use_pretrained": $USE_PRETRAINED,
  "min_train": $MIN_TRAIN_IMAGES,
  "max_train": $MAX_TRAIN_IMAGES,
  "min_eval": $MIN_VAL_IMAGES,
  "max_eval": $MAX_VAL_IMAGES,
  "dataset": $DATASET,
  "alpha_min": $DIRICHLET_ALPHA_MIN,
  "alpha_max": $DIRICHLET_ALPHA_MAX,
  "enable_gpu": $ENABLE_GPU,
  "enable_tls": $ENABLE_TLS,
  "server_internal_ip": "$SERVER_INTERNAL_IP",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOJSON
)
echo "$CONFIG_JSON" | gsutil cp - gs://${BUCKET_NAME}/configs/run_${RUN_ID}_config.json

# Update pyproject.toml
sed -i "s/num-server-rounds = [0-9]*/num-server-rounds = $NUM_SERVER_ROUNDS/" pyproject.toml
sed -i "s/fraction-train = [0-9.]\+/fraction-train = $FRACTION_TRAIN/" pyproject.toml
sed -i "s/fraction_evaluate = [0-9.]\+/fraction_evaluate = $FRACTION_EVALUATE/" pyproject.toml
sed -i "s/local-epochs = [0-9]\+/local-epochs = $LOCAL_EPOCHS/" pyproject.toml
sed -i "s/lr = [0-9.]\+/lr = $LR/" pyproject.toml
sed -i "s/yolo_size = \"[a-z]\"/yolo_size = \"$YOLO_SIZE\"/" pyproject.toml
sed -i "s/img_size = [0-9]\+/img_size = $IMG_SIZE/" pyproject.toml
sed -i "s/batch_size = [0-9]\+/batch_size = $BATCH_SIZE/" pyproject.toml
sed -i "s/run_id = [0-9]\+/run_id = $RUN_ID/" pyproject.toml
sed -i "s/dataset = [0-9]\+/dataset = $DATASET/" pyproject.toml
sed -i "s/use_pretrained = [0-9]\+/use_pretrained = $USE_PRETRAINED/" pyproject.toml
sed -i "s|coco_root = \".*\"|coco_root = \"/app/datasets/coco\"|" pyproject.toml
sed -i "s|gcs_bucket = \".*\"|gcs_bucket = \"$BUCKET_NAME\"|" pyproject.toml

# Increment version in pyproject.toml
echo_info "Incrementing version in pyproject.toml..."
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
PATCH=$((VERSION_PARTS[2] + 1))
NEW_VERSION="${VERSION_PARTS[0]}.${VERSION_PARTS[1]}.$PATCH"
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
echo_success "Version incremented from $CURRENT_VERSION to $NEW_VERSION"

# Update YOLOv5 hyperparameters
sed -i "s/lr0: [0-9.]\+/lr0: $LR/" yolov5/data/hyps/hyp.scratch-low.yaml

if [ "$INSECURE" = "false" ]; then
    sed -i 's/^[[:space:]]*insecure = .*/insecure = false/' pyproject.toml
    sed -i 's/^[[:space:]]*# root-certificates =/root-certificates =/' pyproject.toml
else
    sed -i 's/^[[:space:]]*insecure = .*/insecure = true/' pyproject.toml
    sed -i 's/^[[:space:]]*root-certificates =/# root-certificates =/' pyproject.toml
fi

echo_success "Configuration saved"

# Deploy server
echo_info "Deploying server on $SERVER_VM (IP: $SERVER_INTERNAL_IP)"

# Clean remote cache and setup directories
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    sudo mkdir -p /app/{logs,checkpoints,certs}
    sudo chown -R \$USER:\$USER /app
    echo 'Cleaning remote Python cache...'
    sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
    echo 'Remote cache cleaned'
" 2>&1 | grep -v "No such file or directory" || true

# Copy files (with compression, parallel transfers)
echo "  → Syncing files to server..."
gcloud compute scp --recurse --compress ./src $SERVER_VM:/app/ --zone=$SERVER_ZONE > /dev/null 2>&1 &
PID1=$!
gcloud compute scp --recurse --compress ./yolov5 $SERVER_VM:/app/ --zone=$SERVER_ZONE > /dev/null 2>&1 &
PID2=$!
wait $PID1 $PID2
gcloud compute scp --compress requirements.txt pyproject.toml .env $SERVER_VM:/app/ --zone=$SERVER_ZONE > /dev/null 2>&1

if [ "$ENABLE_TLS" = "true" ]; then
    gcloud compute scp --recurse --compress ./certs $SERVER_VM:/app/ --zone=$SERVER_ZONE > /dev/null 2>&1
fi
echo "  ✓ Files synced"

# Create server docker-compose
cat > /tmp/docker-compose-server.yml << 'EOF'
version: '3.8'
services:
  fl-server:
    image: ${DOCKER_IMAGE}
    container_name: fl-server
    shm_size: '24gb'
    env_file: [.env]
    environment:
      - PYTHONPATH=/app/src:/app
    command: >
      sh -c "
      echo '=== Flower Server Starting ===' &&
      echo 'PYTHONPATH:' \$PYTHONPATH &&
      if [ \"\$INSECURE\" = 'true' ]; then
        flower-superlink --insecure --fleet-api-address=0.0.0.0:9092;
      else
        flower-superlink --fleet-api-address=0.0.0.0:9092 --ssl-ca-certfile=/app/certs/ca.crt --ssl-certfile=/app/certs/server.crt --ssl-keyfile=/app/certs/server.key;
      fi"
    ports: ["9092:9092", "9093:9093"]
    volumes:
      - "./src:/app/src"
      - "./yolov5:/app/yolov5"
      - "./logs:/app/logs"
      - "./certs:/app/certs:ro"
      - "./pyproject.toml:/app/pyproject.toml"
    restart: unless-stopped
networks:
  default:
    driver: bridge
EOF

gcloud compute scp /tmp/docker-compose-server.yml $SERVER_VM:/app/docker-compose.yml --zone=$SERVER_ZONE --quiet > /dev/null 2>&1

# Start server with force-recreate
echo "  → Starting server container..."
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    echo 'DOCKER_IMAGE=$DOCKER_IMAGE' >> .env
    sudo docker compose pull --quiet
    sudo docker compose up -d --force-recreate
    sleep 10
    sudo docker compose exec -T fl-server find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    sudo docker compose exec -T fl-server find /app -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo docker compose ps
" 2>&1 | grep -E "(NAME|fl-server)" || true

echo_success "Server deployed at $SERVER_INTERNAL_IP"

# Deploy clients
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    CLIENT_IP=${!CLIENT_IP_VAR}
    
    echo_info "Deploying clients on $CLIENT_VM (IP: $CLIENT_IP)"
    
    # Clean remote cache and setup directories
    gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        sudo mkdir -p /app/{logs,certs}
        sudo chown -R \$USER:\$USER /app
        echo 'Cleaning remote Python cache...'
        sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
        sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
        echo 'Remote cache cleaned'
    " 2>&1 | grep -v "No such file or directory" || true
    
    # Copy files (with compression, parallel transfers)
    echo "  → Syncing files to $CLIENT_VM..."
    gcloud compute scp --recurse --compress ./src $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1 &
    PID1=$!
    gcloud compute scp --recurse --compress ./yolov5 $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1 &
    PID2=$!
    wait $PID1 $PID2
    gcloud compute scp --compress requirements.txt pyproject.toml .env $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1
    
    if [ "$ENABLE_TLS" = "true" ]; then
        gcloud compute scp --recurse --compress ./certs $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1
    fi
    
    # Create service account key for GCS access
    if [ ! -f "gcs-key.json" ]; then
        gcloud iam service-accounts keys create gcs-key.json \
            --iam-account=default-compute@${PROJECT_ID}.iam.gserviceaccount.com 2>/dev/null || true
    fi
    gcloud compute scp --compress gcs-key.json $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1
    echo "  ✓ Files synced"
    
    # Calculate client IDs for this VM (2 clients per VM)
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))
    
    # Verify pre-partitioned data exists for selected dataset
    echo "  → Verifying pre-partitioned data for Dataset $DATASET (Clients $CLIENT_ID_1, $CLIENT_ID_2)..."
    VERIFICATION_OUTPUT=$(gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        set -e
        DATASET=$DATASET
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            PARTITION_DIR=\"/app/datasets_\${DATASET}/coco_partitions/client_\${CLIENT_ID}\"
            
            if [ ! -d \"\$PARTITION_DIR\" ]; then
                echo \"ERROR: Partition directory not found: \$PARTITION_DIR\"
                echo \"Please run partition-dataset.sh before deployment!\"
                exit 1
            fi
            
            TRAIN_IMG=\$(ls \$PARTITION_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
            VAL_IMG=\$(ls \$PARTITION_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
            
            if [ \$TRAIN_IMG -eq 0 ]; then
                echo \"ERROR: No training images found for client \$CLIENT_ID in dataset \$DATASET\"
                exit 1
            fi
            
            # Verify dataset-specific YAML file exists
            YAML_FILE=\"\$PARTITION_DIR/coco_client_dataset_\${DATASET}.yaml\"
            if [ ! -f \"\$YAML_FILE\" ]; then
                echo \"ERROR: Dataset YAML file not found: \$YAML_FILE\"
                exit 1
            fi
            
            echo \"✅ Dataset \$DATASET, Client \$CLIENT_ID: Train=\$TRAIN_IMG images, Val=\$VAL_IMG images\"
        done
    " 2>&1)
    
    if echo "$VERIFICATION_OUTPUT" | grep -q "ERROR"; then
        echo_error "Pre-partitioned data verification failed on $CLIENT_VM:\n$VERIFICATION_OUTPUT"
        exit 1
    else
        echo "$VERIFICATION_OUTPUT"
    fi
    
    # Create client docker-compose with CURRENT server IP
    cat > /tmp/docker-compose-client-${i}.yml << EOF
version: '3.8'
services:
  fl-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    container_name: fl-client-${CLIENT_ID_1}
    shm_size: '28gb'
    env_file: [.env]
    environment:
      - CLIENT_ID=${CLIENT_ID_1}
      - PARTITION_ID=${CLIENT_ID_1}
      - PYTHONPATH=/app/src:/app
      - GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
    command: >
      sh -c "
      echo '=== Flower Client ${CLIENT_ID_1} Starting ===' &&
      echo 'PYTHONPATH:' \$PYTHONPATH &&
      if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092;
      else
        flower-supernode --superlink=${SERVER_INTERNAL_IP}:9092 --root-certificates=/app/certs/ca.crt;
      fi"
    volumes:
      - "./src:/app/src"
      - "./yolov5:/app/yolov5"
      - "./logs:/app/logs"
      - "./certs:/app/certs:ro"
      - "./datasets_${DATASET}:/app/datasets_${DATASET}"
      - "./gcs-key.json:/app/gcs-key.json:ro"
      - "./pyproject.toml:/app/pyproject.toml"
    restart: unless-stopped
    
  fl-client-${CLIENT_ID_2}:
    image: ${DOCKER_IMAGE}
    container_name: fl-client-${CLIENT_ID_2}
    shm_size: '28gb'
    env_file: [.env]
    environment:
      - CLIENT_ID=${CLIENT_ID_2}
      - PARTITION_ID=${CLIENT_ID_2}
      - PYTHONPATH=/app/src:/app
      - GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
    command: >
      sh -c "
      echo '=== Flower Client ${CLIENT_ID_2} Starting ===' &&
      echo 'PYTHONPATH:' \$PYTHONPATH &&
      if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092;
      else
        flower-supernode --superlink=${SERVER_INTERNAL_IP}:9092 --root-certificates=/app/certs/ca.crt;
      fi"
    volumes:
      - "./src:/app/src"
      - "./yolov5:/app/yolov5"
      - "./logs:/app/logs"
      - "./certs:/app/certs:ro"
      - "./datasets_${DATASET}:/app/datasets_${DATASET}"
      - "./gcs-key.json:/app/gcs-key.json:ro"
      - "./pyproject.toml:/app/pyproject.toml"
    restart: unless-stopped
networks:
  default:
    driver: bridge
EOF
    
    gcloud compute scp /tmp/docker-compose-client-${i}.yml $CLIENT_VM:/app/docker-compose.yml --zone=$CLIENT_ZONE --quiet > /dev/null 2>&1
done

echo_success "All clients configured and data verified"

# Start clients
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    echo_info "Starting clients on $CLIENT_VM"
    gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        cd /app
        echo 'DOCKER_IMAGE=$DOCKER_IMAGE' >> .env
        sudo docker compose pull --quiet
        sudo docker compose up -d --force-recreate
        sleep 5
        for CLIENT_ID in \$(sudo docker compose ps --services); do
            sudo docker compose exec -T \$CLIENT_ID find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            sudo docker compose exec -T \$CLIENT_ID find /app -type f -name '*.pyc' -delete 2>/dev/null || true
        done
        sudo docker compose ps
    " 2>&1 | grep -E "(NAME|fl-client)" || true
done

echo_success "Deployment complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Run ID: $RUN_ID"
echo "  Server IP: $SERVER_INTERNAL_IP"
echo "  All clients connected to: ${SERVER_INTERNAL_IP}:9092"
echo "  ✅ Using Dataset $DATASET from /app/datasets_${DATASET}/coco_partitions/"
echo "  ✅ All caches cleaned and containers recreated"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "To start training:"
echo "  gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='cd /app && sudo docker compose exec fl-server flwr run .'"
echo ""
echo "Monitor with: ./05-manage-clients.sh status"
echo ""
echo "Note: vm-info.txt has been updated with current IPs (backup saved as vm-info.txt.backup)"