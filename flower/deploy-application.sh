#!/bin/bash

# Deploy Flybold Application with Dynamic IP Handling
set -e
# Portable in-place sed (works on macOS BSD sed + Linux GNU sed)
sedi() {
  if sed --version >/dev/null 2>&1; then
    # GNU sed (Linux)
    sed -i -E "$@"
  else
    # BSD sed (macOS)
    sed -i '' -E "$@"
  fi
}
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

# SSH/SCP connection settings with more conservative timeouts
SSH_FLAGS=(
    "-o ConnectTimeout=30"        # Increased from 15
    "-o ServerAliveInterval=10"   # Increased from 5
    "-o ServerAliveCountMax=6"    # Increased from 3
    "-o StrictHostKeyChecking=no"
    "-o UserKnownHostsFile=/dev/null"
    "-o LogLevel=ERROR"
    "-o TCPKeepAlive=yes"         # Added for better connection stability
    "-o Compression=yes"          # Added for better transfer speed
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
            "${SSH_FLAGS[@]/#/--ssh-flag=}" \
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

# Helper function: Simple SCP with retry - no fancy stuff
scp_with_retry() {
    local source=$1
    local destination=$2
    local zone=$3
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if gcloud compute scp \
            --compress \
            --scp-flag="-o ConnectTimeout=60" \
            --scp-flag="-o ServerAliveInterval=30" \
            "$source" "$destination" \
            --zone="$zone" 2>&1 | grep -v "Warning:"; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "  ⚠️  Transfer failed, retrying in 5s... ($retry_count/$max_retries)" >&2
            sleep 5
        fi
    done
    
    echo_error "Transfer failed after $max_retries attempts: $source" >&2
    return 1
}

# Sync code to VM using tar (more reliable for many files)
sync_code_to_vm() {
    local vm=$1
    local zone=$2
    local label=$3
    
    echo "  → Preparing code archive for $label..."
    
    # Create a clean tar archive (exclude cache and unwanted files)
    tar -czf /tmp/flybold-code.tar.gz \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='*.egg-info' \
        --exclude='.pytest_cache' \
        src/ yolov5/ requirements.txt pyproject.toml .env 2>/dev/null
    
    if [ "$ENABLE_TLS" = "true" ]; then
        echo "  → Adding certs to archive..."
        tar -rzf /tmp/flybold-code.tar.gz certs/ 2>/dev/null
    fi
    
    echo "  → Uploading archive to $label..."
    if ! scp_with_retry "/tmp/flybold-code.tar.gz" "$vm:/tmp/" "$zone"; then
        echo_error "Failed to upload archive to $vm"
        rm -f /tmp/flybold-code.tar.gz
        return 1
    fi
    
    echo "  → Extracting on $label..."
    if ! ssh_with_retry "$vm" "$zone" "
        cd /app
        tar -xzf /tmp/flybold-code.tar.gz
        rm -f /tmp/flybold-code.tar.gz
        echo 'Extraction complete'
    "; then
        echo_error "Failed to extract archive on $vm"
        rm -f /tmp/flybold-code.tar.gz
        return 1
    fi
    
    rm -f /tmp/flybold-code.tar.gz
    echo "  ✓ Code synced to $label"
    return 0
}

# Helper function: Rsync with retry (fallback for large transfers)
rsync_with_retry() {
    local source=$1
    local vm=$2
    local destination=$3
    local zone=$4
    local max_retries=3
    local retry_count=0
    
    # Check if rsync is available on remote
    if ! ssh_with_retry "$vm" "$zone" "command -v rsync" &>/dev/null; then
        echo "  ℹ️  rsync not available on remote, installing..." >&2
        ssh_with_retry "$vm" "$zone" "sudo apt-get update -qq && sudo apt-get install -y -qq rsync" || return 1
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        if gcloud compute scp --recurse --compress \
            --scp-flag="-o ConnectTimeout=30" \
            --scp-flag="-o ServerAliveInterval=10" \
            "$source" "${vm}:${destination}" \
            --zone="$zone" 2>&1 | grep -v "Warning:"; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            local wait_time=$((retry_count * 5))
            echo "  ⚠️  Rsync failed, retrying in ${wait_time}s... ($retry_count/$max_retries)" >&2
            sleep $wait_time
        fi
    done
    
    return 1
}

# Helper function: GCS operation with retry logic
gcs_with_retry() {
    # Usage: gcs_with_retry command arg1 arg2 ...
    local max_retries=3
    local retry_count=0
    
    # Set a timeout for the command (requires coreutils timeout)
    local timeout_cmd=""
    if command -v timeout >/dev/null 2>&1; then
        timeout_cmd="timeout 30s"
    fi
    
    while [ $retry_count -lt $max_retries ]; do
        # Execute command preserving stdout/stderr
        # We rely on $@ to pass arguments correctly
        
        $timeout_cmd "$@"
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            local wait_time=$((retry_count * 3))
            echo "  ⚠️  GCS command failed (code $exit_code), retrying in ${wait_time}s... ($retry_count/$max_retries)" >&2
            sleep $wait_time
        fi
    done
    
    echo_error "GCS command failed after $max_retries attempts: $*" >&2
    return 1
}

# Reserve the next RUN_ID in GCS using generation-match preconditions (race-safe).
# This avoids two parallel deploys picking the same run_id.
reserve_run_id() {
    local bucket="$1"
    local object="gs://${bucket}/run_id.txt"
    local max_attempts=20
    local attempt=1

    if ! command -v gcloud >/dev/null 2>&1; then
        echo_error "gcloud is required to reserve RUN_ID in GCS."
        return 1
    fi

    # Detect whether this gcloud supports --if-generation-match (fallback to gsutil if not).
    local supports_if_match=0
    if gcloud storage cp --help 2>&1 | grep -q -- "--if-generation-match"; then
        supports_if_match=1
    fi

    while [ $attempt -le $max_attempts ]; do
        local generation=""
        local current="0"
        local next=""

        # If the object exists, fetch its generation and current value.
        generation="$(gcloud storage objects describe "$object" --format='value(generation)' 2>/dev/null || true)"
        if [ -n "$generation" ]; then
            # Read current run_id, keep only the first integer (robust to stray whitespace/newlines).
            current="$(gcs_with_retry gcloud storage cat "$object" 2>/dev/null | tr -d '\r\n' | grep -Eo '^[0-9]+' || echo "0")"
        fi

        next=$((current + 1))
        printf "%s" "$next" > /tmp/run_id.txt

        if [ $supports_if_match -eq 1 ]; then
            if [ -n "$generation" ]; then
                # Update only if the object is still at the same generation.
                if gcloud storage cp --if-generation-match="$generation" /tmp/run_id.txt "$object" >/dev/null 2>&1; then
                    echo "$next"
                    return 0
                fi
            else
                # Create only if the object does not exist (generation-match 0).
                if gcloud storage cp --if-generation-match=0 /tmp/run_id.txt "$object" >/dev/null 2>&1; then
                    echo "$next"
                    return 0
                fi
            fi
        elif command -v gsutil >/dev/null 2>&1; then
            # Fallback: gsutil supports x-goog-if-generation-match header.
            if [ -n "$generation" ]; then
                if gsutil -h "x-goog-if-generation-match:${generation}" cp /tmp/run_id.txt "$object" >/dev/null 2>&1; then
                    echo "$next"
                    return 0
                fi
            else
                if gsutil -h "x-goog-if-generation-match:0" cp /tmp/run_id.txt "$object" >/dev/null 2>&1; then
                    echo "$next"
                    return 0
                fi
            fi
        else
            echo_error "Neither 'gcloud storage cp --if-generation-match' nor 'gsutil' is available. Can't reserve RUN_ID."
            return 1
        fi

        # Most common failure here is a 412 precondition failure (another deploy won the race).
        # Back off a bit and try again.
        sleep $((attempt < 5 ? attempt : 5))
        attempt=$((attempt + 1))
    done

    echo_error "Failed to reserve a unique RUN_ID from $object after $max_attempts attempts."
    return 1
}

# Optimized function to fetch and update all VM IPs in one batch
update_vm_info() {
    echo_info "Updating vm-info.txt with current IP addresses..."
    
    if [ -f "vm-info.txt" ]; then
        cp vm-info.txt vm-info.txt.backup
        echo "  Backed up existing vm-info.txt"
    fi
    
    # Initialize file
    cat > vm-info.txt << EOF
PROJECT_ID=$PROJECT_ID
REGION=us-central1
NETWORK=flybold-network

EOF

    echo "  Fetching IPs for all VMs in a single batch request..."
    
    # Fetch all VM details in one single API call (filters for exact matches)
    # Output format: name,zone,internal_ip,external_ip
    local vm_data=$(gcloud compute instances list \
        --project="$PROJECT_ID" \
        --filter="name=(flybold-server) OR name:(flybold-client*)" \
        --format="csv[no-heading](name,zone,networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP)")
        
    if [ -z "$vm_data" ]; then
        echo_error "No VMs found! Are they running?"
        exit 1
    fi

    # Store found data using indexed arrays (compatible with macOS bash 3.x)
    declare -a client_names
    declare -a client_zones
    declare -a client_internal
    declare -a client_external

    # Server placeholders
    SERVER_NAME=""
    SERVER_ZONE=""
    SERVER_INTERNAL_IP=""
    SERVER_EXTERNAL_IP=""

    # Parse the CSV output
    while IFS=, read -r name zone internal_ip external_ip; do
        if [ "$name" = "flybold-server" ]; then
            SERVER_NAME="$name"
            SERVER_ZONE="$zone"
            SERVER_INTERNAL_IP="$internal_ip"
            SERVER_EXTERNAL_IP="$external_ip"
            echo "    Found $name: $internal_ip (Ext: ${external_ip:-None})" >&2
            continue
        fi

        # Match client names like: flybold-client-1 .. flybold-client-5
        if echo "$name" | grep -qE '^flybold-client-[0-9]+$'; then
            idx=$(echo "$name" | sed -E 's/.*-([0-9]+)$/\1/')
            client_names[$idx]="$name"
            client_zones[$idx]="$zone"
            client_internal[$idx]="$internal_ip"
            client_external[$idx]="$external_ip"
            echo "    Found $name: $internal_ip (Ext: ${external_ip:-None})" >&2
        fi
    done <<< "$vm_data"

    # Write Server Info
    if [ -n "$SERVER_INTERNAL_IP" ]; then
        cat >> vm-info.txt << EOF
SERVER_VM=flybold-server
SERVER_ZONE=$SERVER_ZONE
SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP
SERVER_EXTERNAL_IP=$SERVER_EXTERNAL_IP

EOF
    else
        echo_error "Server VM (flybold-server) not found in running instances!"
        exit 1
    fi

    # Write Client Info (clients indexed 1..5)
    for i in $(seq 1 5); do
        if [ -n "${client_internal[$i]:-}" ]; then
            cat >> vm-info.txt << EOF
CLIENT_${i}_VM=${client_names[$i]}
CLIENT_${i}_ZONE=${client_zones[$i]}
CLIENT_${i}_INTERNAL_IP=${client_internal[$i]}
CLIENT_${i}_EXTERNAL_IP=${client_external[$i]}

EOF
        else
            echo_warning "Client VM (flybold-client-$i) not found or not running"
        fi
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

# Load RUN_ID directly from .env (no auto-increment)
echo_info "Loading RUN_ID from .env..."
if ! grep -q '^RUN_ID=' .env; then
    echo_error "RUN_ID not found in .env. Please set it manually before deploying."
    exit 1
fi
RUN_ID="$(grep '^RUN_ID=' .env | cut -d'=' -f2 | tr -d '[:space:]')"
echo_success "Using RUN_ID: $RUN_ID"

# Push the RUN_ID to GCS so it's recorded for this experiment
echo_info "Pushing RUN_ID to GCS..."
printf "%s" "$RUN_ID" > /tmp/run_id.txt
gcs_with_retry gcloud storage cp /tmp/run_id.txt "gs://${BUCKET_NAME}/run_id.txt"
echo "[DEBUG] Pushed RUN_ID=$RUN_ID to gs://${BUCKET_NAME}/run_id.txt"
sed -i '' "s|^DOCKER_IMAGE=.*|DOCKER_IMAGE=$DOCKER_IMAGE|" .env
echo "[DEBUG] Updated DOCKER_IMAGE in .env"
sed -i '' "s/^SERVER_INTERNAL_IP=.*/SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP/" .env
echo "[DEBUG] Updated SERVER_INTERNAL_IP in .env"

# Save config to GCS
cat > /tmp/run_config.json <<EOJSON
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
  "strategy": $STRATEGY,
  "alpha_min": $DIRICHLET_ALPHA_MIN,
  "alpha_max": $DIRICHLET_ALPHA_MAX,
  "enable_gpu": $ENABLE_GPU,
  "enable_tls": $ENABLE_TLS,
  "server_internal_ip": "$SERVER_INTERNAL_IP",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOJSON
echo "[DEBUG] Created run_config.json"

gcs_with_retry gcloud storage cp /tmp/run_config.json "gs://${BUCKET_NAME}/configs/run_${RUN_ID}_config.json"
echo "[DEBUG] Uploaded run_config.json"

# Update pyproject.toml
sedi "s/^num-server-rounds[[:space:]]*=[[:space:]]*[0-9]+/num-server-rounds = ${NUM_SERVER_ROUNDS}/" pyproject.toml
sedi "s/^run_id[[:space:]]*=[[:space:]]*[0-9]+/run_id = ${RUN_ID}/" pyproject.toml
sedi "s/^fraction-train[[:space:]]*=[[:space:]]*[0-9.]+/fraction-train = ${FRACTION_TRAIN}/" pyproject.toml
sedi "s/^fraction_evaluate[[:space:]]*=[[:space:]]*[0-9.]+/fraction_evaluate = ${FRACTION_EVALUATE}/" pyproject.toml
sedi "s/^local-epochs[[:space:]]*=[[:space:]]*[0-9]+/local-epochs = ${LOCAL_EPOCHS}/" pyproject.toml
sedi "s/^lr[[:space:]]*=[[:space:]]*[0-9.]+/lr = ${LR}/" pyproject.toml
sedi "s/^yolo_size[[:space:]]*=[[:space:]]*\"[^\"]+\"/yolo_size = \"${YOLO_SIZE}\"/" pyproject.toml
sedi "s/^img_size[[:space:]]*=[[:space:]]*[0-9]+/img_size = ${IMG_SIZE}/" pyproject.toml
sedi "s/^batch_size[[:space:]]*=[[:space:]]*[0-9]+/batch_size = ${BATCH_SIZE}/" pyproject.toml
sedi "s/^dataset[[:space:]]*=[[:space:]]*[0-9]+/dataset = ${DATASET}/" pyproject.toml
sedi "s/^strategy[[:space:]]*=[[:space:]]*[0-9]+/strategy = ${STRATEGY}/" pyproject.toml
sedi "s/^use_pretrained[[:space:]]*=[[:space:]]*[0-9]+/use_pretrained = ${USE_PRETRAINED}/" pyproject.toml
sedi "s|^gcs_bucket[[:space:]]*=[[:space:]]*\".*\"|gcs_bucket = \"${BUCKET_NAME}\"|" pyproject.toml

# Increment version in pyproject.toml
echo_info "Incrementing version in pyproject.toml..."
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
PATCH=$((VERSION_PARTS[2] + 1))
NEW_VERSION="${VERSION_PARTS[0]}.${VERSION_PARTS[1]}.$PATCH"
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
echo_success "Version incremented from $CURRENT_VERSION to $NEW_VERSION"

# Update YOLOv5 hyperparameters
sed -i '' "s/lr0: [0-9.]\+/lr0: $LR/" yolov5/data/hyps/hyp.scratch-low.yaml

if [ "$INSECURE" = "false" ]; then
    sed -i '' 's/^[[:space:]]*insecure = .*/insecure = false/' pyproject.toml
    sed -i '' 's/^[[:space:]]*# root-certificates =/root-certificates =/' pyproject.toml
else
    sed -i '' 's/^[[:space:]]*insecure = .*/insecure = true/' pyproject.toml
    sed -i '' 's/^[[:space:]]*root-certificates =/# root-certificates =/' pyproject.toml
fi

echo_success "Configuration saved"

# Deploy server
echo_info "Deploying server on $SERVER_VM (IP: $SERVER_INTERNAL_IP)"

# Clean remote cache and setup directories
echo "  → Setting up server directories..."
ssh_with_retry "$SERVER_VM" "$SERVER_ZONE" "
    sudo mkdir -p /app/{logs,checkpoints,certs}
    sudo chown -R \$USER:\$USER /app
    echo 'Cleaning remote Python cache...'
    sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
    echo 'Remote cache cleaned'
" || { echo_error "Failed to setup server directories"; exit 1; }

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

echo_info "Writing Flower config.toml inside fl-server (set default to deployment)..."

# Build config content (deployment is default)
if [ "$ENABLE_TLS" = "true" ]; then
  FLWR_INSECURE_FLAG="false"
else
  FLWR_INSECURE_FLAG="true"
fi

gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
  set -e
  cd /app
  sudo docker compose exec -T fl-server sh -c '
    mkdir -p /root/.flwr
    cat > /root/.flwr/config.toml <<EOF
[superlink]
default = \"deployment\"

[superlink.deployment]
# flwr run executes inside the same container as the SuperLink,
# so localhost is correct.
address = \"127.0.0.1:9093\"
insecure = ${FLWR_INSECURE_FLAG}
EOF
  '
  sudo docker compose exec -T fl-server flwr config list || true
"

echo_success "Server deployed at $SERVER_INTERNAL_IP"

# Clients deployment

echo_info "Deploying clients..."

# Deploy clients - PARALLELIZED FILE SYNC
echo_info "Deploying to all 5 client VMs in parallel..."

# Create service account key for GCS access (do once before parallel operations)
if [ ! -f "gcs-key.json" ]; then
    gcloud iam service-accounts keys create gcs-key.json \
        --iam-account=default-compute@${PROJECT_ID}.iam.gserviceaccount.com 2>/dev/null || true
fi

# Step 1: Setup directories on all clients in parallel
echo "  → Setting up directories on all client VMs..."
SETUP_PIDS=()
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    (
        ssh_with_retry "$CLIENT_VM" "$CLIENT_ZONE" "
            sudo mkdir -p /app/{logs,certs}
            sudo chown -R \$USER:\$USER /app
            sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
            sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
            sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
            sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
        " && echo "  ✓ $CLIENT_VM directories ready"
    ) &
    SETUP_PIDS+=($!)
done

# Wait for all directory setups to complete
for pid in "${SETUP_PIDS[@]}"; do
    wait $pid || { echo_error "Failed to setup client directories"; exit 1; }
done
echo_success "All client directories setup complete"

# Step 2: Sync files to all clients in parallel
echo_info "Syncing files to all 5 client VMs in parallel..."

for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    CLIENT_IP=${!CLIENT_IP_VAR}
    
    echo "  → Syncing to $CLIENT_VM ($CLIENT_IP)..."
    
    # Sync large directories in parallel (within this client)
    gcloud compute scp --recurse --compress ./src $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1 &
    PID1=$!
    gcloud compute scp --recurse --compress ./yolov5 $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1 &
    PID2=$!
    
    # Wait for large directories to finish
    wait $PID1 $PID2
    
    # Sync config files sequentially
    gcloud compute scp --compress requirements.txt pyproject.toml .env gcs-key.json $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1
    
    if [ "$ENABLE_TLS" = "true" ]; then
        gcloud compute scp --recurse --compress ./certs $CLIENT_VM:/app/ --zone=$CLIENT_ZONE > /dev/null 2>&1
    fi
    
    echo "  ✓ $CLIENT_VM files synced"
done

echo_success "All client files synced successfully!"

# Step 3: Verify data and create docker-compose files (still serial, but fast)
echo_info "Verifying data and creating configurations..."
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    CLIENT_IP=${!CLIENT_IP_VAR}
    
    # Calculate client IDs for this VM (2 clients per VM)
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))

    # Verify pre-partitioned data exists
    echo "  → Verifying pre-partitioned data (Clients $CLIENT_ID_1, $CLIENT_ID_2)..."
    VERIFICATION_OUTPUT=$(gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        set -e
        DATASET=$DATASET
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            PARTITION_DIR=\"/app/datasets_\${DATASET}/coco_partitions/client_\${CLIENT_ID}\"
            
            if [ ! -d \"\$PARTITION_DIR\" ]; then
                echo \"ERROR: Partition directory not found: \$PARTITION_DIR\"
                exit 1
            fi
            
            TRAIN_IMG=\$(ls \$PARTITION_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
            VAL_IMG=\$(ls \$PARTITION_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
            
            if [ \$TRAIN_IMG -eq 0 ]; then
                echo \"ERROR: No training images found for client \$CLIENT_ID\"
                exit 1
            fi
            
            YAML_FILE=\"\$PARTITION_DIR/coco_client_dataset_\${DATASET}.yaml\"
            if [ ! -f \"\$YAML_FILE\" ]; then
                echo \"ERROR: Dataset YAML file not found: \$YAML_FILE\"
                exit 1
            fi
            
            echo \"✅ Client \$CLIENT_ID: Train=\$TRAIN_IMG, Val=\$VAL_IMG\"
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