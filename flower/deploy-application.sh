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

# Helper function: Check if VM is running
check_vm_running() {
    local vm=$1
    local zone=$2
    
    local status=$(gcloud compute instances describe "$vm" \
        --zone="$zone" \
        --format='get(status)' 2>/dev/null || echo "UNKNOWN")
    
    if [ "$status" = "RUNNING" ]; then
        return 0
    else
        echo "not_running"
        return 1
    fi
}

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

# Load or create VM info
if [ ! -f "vm-info.txt" ]; then
    echo_error "vm-info.txt not found. Run 02-setup-infrastructure.sh first."
    exit 1
fi

# Refresh VM IPs using the dedicated script
echo_info "Refreshing VM IPs using refresh-vm-ips.sh..."
if [ ! -f "refresh-vm-ips.sh" ]; then
    echo_error "refresh-vm-ips.sh not found. Cannot refresh VM IPs."
    exit 1
fi

chmod +x refresh-vm-ips.sh
./refresh-vm-ips.sh

# Load updated VM info
echo_info "Loading updated VM configuration..."
source vm-info.txt

echo_success "All VM IPs refreshed successfully!"
echo ""
echo "Current Configuration:"
echo "  Server: $SERVER_VM ($SERVER_INTERNAL_IP)"
echo "  Found $MAX_CLIENT_NUM client VMs:"
for i in $(seq 1 $MAX_CLIENT_NUM); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
    if [ -n "${!CLIENT_VM_VAR:-}" ]; then
        echo "    ${!CLIENT_VM_VAR}: ${!CLIENT_IP_VAR}"
    fi
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
        DATASET_PADDED=$(printf "%03d" "$DATASET")
        SKIP_PROMPTS=true
        echo_warning "Overriding SERVER_INTERNAL_IP with current value from vm-info.txt"
        echo "  Old IP in .env: ${SERVER_INTERNAL_IP}"
        source vm-info.txt
        echo "  Current IP: ${SERVER_INTERNAL_IP}"
        # Bug fix: these variables may not be in .env — guard against empty expansion
        # which would silently wipe the values in pyproject.toml.
        OPTUNA_TRIALS=${OPTUNA_TRIALS:-0}
        HPO_ROUNDS=${HPO_ROUNDS:-3}
        HPO_TRIALS=${HPO_TRIALS:-${OPTUNA_TRIALS}}
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
    DATASET_PADDED=$(printf "%03d" "$DATASET")

    read -p "Use pretrained weights? (y/n) [y]: " pretrained_input
    USE_PRETRAINED=${pretrained_input:-y}
    if [[ $USE_PRETRAINED =~ ^[Yy]$ ]]; then
        USE_PRETRAINED=1
    else
        USE_PRETRAINED=0
    fi
    
    read -p "Path to pretrained checkpoint (optional, leave empty for new training): " PRETRAINED_CHECKPOINT
    PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT:-}
    
    read -p "Number of HPO trials [0]: " HPO_TRIALS
    HPO_TRIALS=${HPO_TRIALS:-0}
    
    read -p "Rounds per HPO trial [3]: " HPO_ROUNDS
    HPO_ROUNDS=${HPO_ROUNDS:-3}
fi

# Ask user if they want to deploy server
read -p "Do you want to deploy/update the server? (yes/no) [yes]: " DEPLOY_SERVER_INPUT
DEPLOY_SERVER=${DEPLOY_SERVER_INPUT:-yes}
if [[ ! "$DEPLOY_SERVER" =~ ^[Yy]|^yes|^YES$ ]]; then
    DEPLOY_SERVER="no"
else
    DEPLOY_SERVER="yes"
fi

# Clients are always deployed — IPs are sourced from vm-info.txt after refresh
DEPLOY_CLIENTS="yes"

echo_info "Deployment Plan:"
echo "  Server:  $([ "$DEPLOY_SERVER" = "yes" ] && echo "✓ Will deploy" || echo "✗ Skip")"
echo "  Clients: ✓ Will deploy all running VMs (1-${MAX_CLIENT_NUM}, IPs from vm-info.txt)"
echo ""

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
  "pretrained_checkpoint": "$PRETRAINED_CHECKPOINT",
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
  "optuna_trials": $OPTUNA_TRIALS,
  "hpo_rounds": $HPO_ROUNDS,
  "client_hpo_enabled": $CLIENT_HPO_ENABLED,
  "client_hpo_trials": $CLIENT_HPO_TRIALS,
  "adaptive_batch_enabled": ${ADAPTIVE_BATCH_ENABLED:-false},
  "adaptive_lr_enabled": ${ADAPTIVE_LR_ENABLED:-false},
  "server_internal_ip": "$SERVER_INTERNAL_IP",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOJSON
echo "[DEBUG] Created run_config.json"

gcs_with_retry gcloud storage cp /tmp/run_config.json "gs://${BUCKET_NAME}/configs/run_${RUN_ID}_config.json"
echo "[DEBUG] Uploaded run_config.json"

# Update pyproject.toml (use patterns that stop at inline comments to avoid corrupting comments)
# FIX: replacements for non-quoted values now include two trailing spaces so the
# greedy [^#]* match (which consumes the original spaces before '#') doesn't cause
# the inline comment to be glued directly onto the value.
sedi "s/^num-server-rounds[[:space:]]*=[[:space:]]*[^#]*/num-server-rounds = ${NUM_SERVER_ROUNDS}  /" pyproject.toml
sedi "s/^run_id[[:space:]]*=[[:space:]]*[^#]*/run_id = ${RUN_ID}  /" pyproject.toml
sedi "s/^fraction-train[[:space:]]*=[[:space:]]*[^#]*/fraction-train = ${FRACTION_TRAIN}  /" pyproject.toml
sedi "s/^fraction_evaluate[[:space:]]*=[[:space:]]*[^#]*/fraction_evaluate = ${FRACTION_EVALUATE}  /" pyproject.toml
sedi "s/^local-epochs[[:space:]]*=[[:space:]]*[^#]*/local-epochs = ${LOCAL_EPOCHS}  /" pyproject.toml
sedi "s/^lr[[:space:]]*=[[:space:]]*[^#]*/lr = ${LR}  /" pyproject.toml
sedi "s/^yolo_size[[:space:]]*=[[:space:]]*\"[^\"]*\"/yolo_size = \"${YOLO_SIZE}\"/" pyproject.toml
sedi "s/^img_size[[:space:]]*=[[:space:]]*[^#]*/img_size = ${IMG_SIZE}  /" pyproject.toml
sedi "s/^batch_size[[:space:]]*=[[:space:]]*[^#]*/batch_size = ${BATCH_SIZE}  /" pyproject.toml
sedi "s/^dataset[[:space:]]*=[[:space:]]*[^#]*/dataset = ${DATASET}  /" pyproject.toml
sedi "s/^strategy[[:space:]]*=[[:space:]]*[^#]*/strategy = ${STRATEGY}  /" pyproject.toml
sedi "s/^use_pretrained[[:space:]]*=[[:space:]]*[^#]*/use_pretrained = ${USE_PRETRAINED}  /" pyproject.toml
# Read PRETRAINED_CHECKPOINT from .env with proper comment stripping
PRETRAINED_CHECKPOINT=$(grep '^PRETRAINED_CHECKPOINT=' .env 2>/dev/null | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "")

sedi "s|^pretrained_checkpoint[[:space:]]*=[[:space:]]*\"[^\"]*\"|pretrained_checkpoint = \"${PRETRAINED_CHECKPOINT}\"|" pyproject.toml
sedi "s/^hpo_trials[[:space:]]*=[[:space:]]*[^#]*/hpo_trials = ${HPO_TRIALS}  /" pyproject.toml
sedi "s/^hpo_rounds[[:space:]]*=[[:space:]]*[^#]*/hpo_rounds = ${HPO_ROUNDS}  /" pyproject.toml

# Client-side HPO flags (default to false/3 if not set in .env)
CLIENT_HPO_ENABLED=${CLIENT_HPO_ENABLED:-false}
CLIENT_HPO_TRIALS=${CLIENT_HPO_TRIALS:-3}
sedi "s/^client_hpo_enabled[[:space:]]*=[[:space:]]*[^#]*/client_hpo_enabled = ${CLIENT_HPO_ENABLED}  /" pyproject.toml
sedi "s/^client_hpo_trials[[:space:]]*=[[:space:]]*[^#]*/client_hpo_trials = ${CLIENT_HPO_TRIALS}  /" pyproject.toml

# Adaptive Batch/Epoch (ABS, client-side) flags (default to false if not set in .env)
ADAPTIVE_BATCH_ENABLED=${ADAPTIVE_BATCH_ENABLED:-false}
ADAPTIVE_BATCH_MIN=${ADAPTIVE_BATCH_MIN:-8}
ADAPTIVE_BATCH_MAX=${ADAPTIVE_BATCH_MAX:-64}
ADAPTIVE_BATCH_MAX_INCREASES=${ADAPTIVE_BATCH_MAX_INCREASES:-4}
ADAPTIVE_BATCH_RMD_THRESHOLD=${ADAPTIVE_BATCH_RMD_THRESHOLD:-0.01}
ADAPTIVE_BATCH_RMD_PATIENCE=${ADAPTIVE_BATCH_RMD_PATIENCE:-2}
ADAPTIVE_BATCH_GROWTH_FACTOR=${ADAPTIVE_BATCH_GROWTH_FACTOR:-2.0}
ADAPTIVE_BATCH_MAX_EPOCHS=${ADAPTIVE_BATCH_MAX_EPOCHS:-10}
sedi "s/^adaptive_batch_enabled[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_enabled = ${ADAPTIVE_BATCH_ENABLED}  /" pyproject.toml
sedi "s/^adaptive_batch_min[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_min = ${ADAPTIVE_BATCH_MIN}  /" pyproject.toml
sedi "s/^adaptive_batch_max[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_max = ${ADAPTIVE_BATCH_MAX}  /" pyproject.toml
sedi "s/^adaptive_batch_max_increases[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_max_increases = ${ADAPTIVE_BATCH_MAX_INCREASES}  /" pyproject.toml
sedi "s/^adaptive_batch_rmd_threshold[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_rmd_threshold = ${ADAPTIVE_BATCH_RMD_THRESHOLD}  /" pyproject.toml
sedi "s/^adaptive_batch_rmd_patience[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_rmd_patience = ${ADAPTIVE_BATCH_RMD_PATIENCE}  /" pyproject.toml
sedi "s/^adaptive_batch_growth_factor[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_growth_factor = ${ADAPTIVE_BATCH_GROWTH_FACTOR}  /" pyproject.toml
sedi "s/^adaptive_batch_max_epochs[[:space:]]*=[[:space:]]*[^#]*/adaptive_batch_max_epochs = ${ADAPTIVE_BATCH_MAX_EPOCHS}  /" pyproject.toml

# Adaptive Global LR (ALR, server-side, final run only) flags (default to false if not set in .env)
ADAPTIVE_LR_ENABLED=${ADAPTIVE_LR_ENABLED:-false}
ADAPTIVE_LR_MIN=${ADAPTIVE_LR_MIN:-0.0001}
ADAPTIVE_LR_MAX=${ADAPTIVE_LR_MAX:-0.01}
ADAPTIVE_LR_MAX_REDUCTIONS=${ADAPTIVE_LR_MAX_REDUCTIONS:-3}
ADAPTIVE_LR_GROWTH_FACTOR=${ADAPTIVE_LR_GROWTH_FACTOR:-1.2}
ADAPTIVE_LR_BACKOFF_FACTOR=${ADAPTIVE_LR_BACKOFF_FACTOR:-0.5}
sedi "s/^adaptive_lr_enabled[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_enabled = ${ADAPTIVE_LR_ENABLED}  /" pyproject.toml
sedi "s/^adaptive_lr_min[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_min = ${ADAPTIVE_LR_MIN}  /" pyproject.toml
sedi "s/^adaptive_lr_max[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_max = ${ADAPTIVE_LR_MAX}  /" pyproject.toml
sedi "s/^adaptive_lr_max_reductions[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_max_reductions = ${ADAPTIVE_LR_MAX_REDUCTIONS}  /" pyproject.toml
sedi "s/^adaptive_lr_growth_factor[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_growth_factor = ${ADAPTIVE_LR_GROWTH_FACTOR}  /" pyproject.toml
sedi "s/^adaptive_lr_backoff_factor[[:space:]]*=[[:space:]]*[^#]*/adaptive_lr_backoff_factor = ${ADAPTIVE_LR_BACKOFF_FACTOR}  /" pyproject.toml

sedi "s|^gcs_bucket[[:space:]]*=[[:space:]]*\".*\"|gcs_bucket = \"${BUCKET_NAME}\"|" pyproject.toml

# Sync FLAML and HPO configuration from .env to pyproject.toml
# Bug fix: add `sed 's/[[:space:]]*#.*//'` before `tr -d '[:space:]'` so that inline
# comments in .env (e.g. FLAML_TIME_BUDGET=36000  # 1 hour) are stripped cleanly
# instead of being concatenated into the value after whitespace removal.
USE_FLAML=$(grep '^USE_FLAML=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
HPO_MODE=$(grep '^HPO_MODE=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_TIME_BUDGET=$(grep '^FLAML_TIME_BUDGET=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_METRIC=$(grep '^FLAML_METRIC=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_ESTIMATOR=$(grep '^FLAML_ESTIMATOR=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_SAMPLE_SIZE=$(grep '^FLAML_SAMPLE_SIZE=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_USE_COE=$(grep '^FLAML_USE_COE=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')
FLAML_LOG_HISTORY=$(grep '^FLAML_LOG_HISTORY=' .env | cut -d'=' -f2 | sed 's/[[:space:]]*#.*//' | tr -d '[:space:]')

sedi "s/^use_flaml[[:space:]]*=[[:space:]]*[^#]*/use_flaml = ${USE_FLAML}  /" pyproject.toml
sedi "s/^hpo_mode[[:space:]]*=[[:space:]]*\"[^\"]*\"/hpo_mode = \"${HPO_MODE}\"/" pyproject.toml
sedi "s/^flaml_time_budget[[:space:]]*=[[:space:]]*[^#]*/flaml_time_budget = ${FLAML_TIME_BUDGET}  /" pyproject.toml
sedi "s/^flaml_metric[[:space:]]*=[[:space:]]*\"[^\"]*\"/flaml_metric = \"${FLAML_METRIC}\"/" pyproject.toml
sedi "s/^flaml_estimator[[:space:]]*=[[:space:]]*\"[^\"]*\"/flaml_estimator = \"${FLAML_ESTIMATOR}\"/" pyproject.toml
sedi "s/^flaml_sample_size[[:space:]]*=[[:space:]]*[^#]*/flaml_sample_size = ${FLAML_SAMPLE_SIZE}  /" pyproject.toml
sedi "s/^flaml_use_coe[[:space:]]*=[[:space:]]*[^#]*/flaml_use_coe = ${FLAML_USE_COE}  /" pyproject.toml
sedi "s/^flaml_log_history[[:space:]]*=[[:space:]]*[^#]*/flaml_log_history = ${FLAML_LOG_HISTORY}  /" pyproject.toml

echo_success "FLAML configuration synced to pyproject.toml"
echo_warning "⚠️  IMPORTANT: Docker image may need rebuild to include flaml[automl]. If FLAML not found, run: ./03-build-push-image.sh"
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
if [ "$DEPLOY_SERVER" = "yes" ]; then
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
# Use --insecure flag directly (no TLS by default, as certificates are not provided)
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
    command: flower-superlink --insecure --fleet-api-address=0.0.0.0:9092
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
        
        # Ensure FLAML is installed in running container (if not in image)
        if ! python3 -c 'import flaml' 2>/dev/null; then
            echo '⚠️  FLAML not found, installing in-container...'
            pip install -q 'flaml[automl]>=2.0.0' || true
        fi
        
        sudo docker compose up -d --force-recreate
        
        # Wait for container to be running and healthy
        echo '  → Waiting for fl-server to be ready...'
        MAX_WAIT=60
        WAITED=0
        while [ \$WAITED -lt \$MAX_WAIT ]; do
            CONTAINER_STATE=\$(sudo docker compose ps fl-server --format '{{.State}}' 2>/dev/null || echo 'unknown')
            if [ \"\$CONTAINER_STATE\" = \"running\" ]; then
                echo '  ✓ Container is running'
                break
            fi
            WAITED=\$((WAITED + 5))
            if [ \$WAITED -lt \$MAX_WAIT ]; then
                echo \"  ⏳ Waiting... (\${WAITED}s/\${MAX_WAIT}s)\"
                sleep 5
            fi
        done
        
        if [ \$WAITED -ge \$MAX_WAIT ]; then
            echo '  ⚠️  Container did not become ready within \$MAX_WAIT seconds'
            sudo docker compose logs fl-server | tail -20
        fi
        
        # Now execute cleanup commands with retry
        for attempt in 1 2 3; do
            if sudo docker compose exec -T fl-server find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; then
                sudo docker compose exec -T fl-server find /app -type f -name '*.pyc' -delete 2>/dev/null || true
                break
            elif [ \$attempt -lt 3 ]; then
                echo \"  ℹ️  Cleanup attempt \$attempt failed, retrying...\"
                sleep 2
            fi
        done
        
        sudo docker compose ps
    " 2>&1 | grep -E "(NAME|fl-server|Waiting|ready)" || true

    echo_info "Writing Flower config.toml inside fl-server (set default to deployment)..."

    # Build config content (deployment is default)
    if [ "$ENABLE_TLS" = "true" ]; then
      FLWR_INSECURE_FLAG="false"
    else
      FLWR_INSECURE_FLAG="true"
    fi

    # Wait for container to be fully ready before executing commands
    # Note: no 'set -e' inside the remote command — failures are handled explicitly
    # so a non-critical config write cannot kill the whole deploy script.
    gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
      cd /app
      
      # Additional wait to ensure container is healthy
      echo 'Waiting for fl-server to be fully ready...'
      MAX_WAIT=90
      WAITED=0
      READY=0
      while [ \$WAITED -lt \$MAX_WAIT ]; do
        if sudo docker compose exec -T fl-server echo 'Container ready' >/dev/null 2>&1; then
          echo 'fl-server is ready'
          READY=1
          break
        fi
        WAITED=\$((WAITED + 3))
        echo \"  ⏳ Still waiting... (\${WAITED}s/\${MAX_WAIT}s)\"
        sleep 3
      done
      
      if [ \$READY -eq 0 ]; then
        echo 'WARNING: Container did not respond within timeout — skipping config write'
        sudo docker compose logs fl-server | tail -20
        exit 0
      fi
      
      # Write the Flower config (failures are non-fatal, hence || true on each exec)
      sudo docker compose exec -T fl-server sh -c '
        mkdir -p /root/.flwr
        cat > /root/.flwr/config.toml <<EOF
[superlink]
default = \"deployment\"

[superlink.deployment]
address = \"127.0.0.1:9093\"
insecure = ${FLWR_INSECURE_FLAG}
EOF
        echo Config written successfully
      ' || echo 'WARNING: config write failed — continuing anyway'
      sudo docker compose exec -T fl-server flwr config list || true
    " || echo_warning "Config-write SSH step had non-zero exit — server is still running, continuing to clients"

    echo_success "Server deployed at $SERVER_INTERNAL_IP"
else
    echo_warning "Server deployment skipped by user"
fi

# Clients deployment

if [ "$DEPLOY_CLIENTS" = "yes" ]; then
    echo_info "Deploying clients (1-${MAX_CLIENT_NUM})..."

    # Deploy clients - PARALLELIZED FILE SYNC
    echo_info "Deploying to all $MAX_CLIENT_NUM client VMs in parallel..."

# Create service account key for GCS access (do once before parallel operations)
if [ ! -f "gcs-key.json" ]; then
    gcloud iam service-accounts keys create gcs-key.json \
        --iam-account=default-compute@${PROJECT_ID}.iam.gserviceaccount.com 2>/dev/null || true
fi

    # Step 1: Setup directories on all clients in parallel
    echo "  → Setting up directories on all client VMs..."
    SETUP_PIDS=()
    for i in $(seq 1 $MAX_CLIENT_NUM); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_EXT_IP_VAR="CLIENT_${i}_EXTERNAL_IP"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        CLIENT_EXT_IP=${!CLIENT_EXT_IP_VAR}
        
        # Skip if VM is not running (not in vm-info.txt)
        if [ -z "$CLIENT_VM" ] || [ -z "$CLIENT_ZONE" ]; then
            echo "  ℹ️  Skipping client VM $i (not running or not found)"
            continue
        fi
        
        # Skip if VM doesn't have external IP (can't connect via SSH)
        if [ -z "$CLIENT_EXT_IP" ] || [ "$CLIENT_EXT_IP" = "None" ]; then
            echo "  ⚠️  Skipping client VM $i ($CLIENT_VM) - no external IP assigned"
            continue
        fi
        
        # Check if VM is actually running
        if ! check_vm_running "$CLIENT_VM" "$CLIENT_ZONE"; then
            echo "  ⏹️  Skipping client VM $i ($CLIENT_VM) - VM is stopped"
            continue
        fi
        
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
    echo_info "Syncing files to all $MAX_CLIENT_NUM client VMs in parallel..."

    for i in $(seq 1 $MAX_CLIENT_NUM); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
        CLIENT_EXT_IP_VAR="CLIENT_${i}_EXTERNAL_IP"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        CLIENT_IP=${!CLIENT_IP_VAR}
        CLIENT_EXT_IP=${!CLIENT_EXT_IP_VAR}
        
        # Skip if VM is not running (not in vm-info.txt)
        if [ -z "$CLIENT_VM" ] || [ -z "$CLIENT_ZONE" ]; then
            echo "  ℹ️  Skipping client VM $i (not running or not found)"
            continue
        fi
        
        # Skip if VM doesn't have external IP (can't connect via SSH)
        if [ -z "$CLIENT_EXT_IP" ] || [ "$CLIENT_EXT_IP" = "None" ]; then
            echo "  ⚠️  Skipping client VM $i ($CLIENT_VM) - no external IP assigned"
            continue
        fi
        
        # Check if VM is actually running
        if ! check_vm_running "$CLIENT_VM" "$CLIENT_ZONE"; then
            echo "  ⏹️  Skipping client VM $i ($CLIENT_VM) - VM is stopped"
            continue
        fi
        
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
    for i in $(seq 1 $MAX_CLIENT_NUM); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_IP_VAR="CLIENT_${i}_INTERNAL_IP"
        CLIENT_EXT_IP_VAR="CLIENT_${i}_EXTERNAL_IP"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        CLIENT_IP=${!CLIENT_IP_VAR}
        CLIENT_EXT_IP=${!CLIENT_EXT_IP_VAR}
        
        # Skip if VM is not running (not in vm-info.txt)
        if [ -z "$CLIENT_VM" ] || [ -z "$CLIENT_ZONE" ]; then
            echo "  ℹ️  Skipping client VM $i (not running or not found)"
            continue
        fi
        
        # Skip if VM doesn't have external IP (can't connect via SSH)
        if [ -z "$CLIENT_EXT_IP" ] || [ "$CLIENT_EXT_IP" = "None" ]; then
            echo "  ⚠️  Skipping client VM $i ($CLIENT_VM) - no external IP assigned"
            continue
        fi
        
        # Check if VM is actually running
        if ! check_vm_running "$CLIENT_VM" "$CLIENT_ZONE"; then
            echo "  ⏹️  Skipping client VM $i ($CLIENT_VM) - VM is stopped"
            continue
        fi
    
        # Calculate client IDs for this VM (2 clients per VM)
        CLIENT_ID_1=$(( (i-1)*2 ))
        CLIENT_ID_2=$(( (i-1)*2 + 1 ))

        # Verify pre-partitioned data exists
        echo "  → Verifying pre-partitioned data (Clients $CLIENT_ID_1, $CLIENT_ID_2)..."
        VERIFICATION_OUTPUT=$(gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        set -e
        DATASET=$DATASET_PADDED
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
    command: flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092
    volumes:
      - "./src:/app/src"
      - "./yolov5:/app/yolov5"
      - "./logs:/app/logs"
      - "./certs:/app/certs:ro"
      - "./datasets_${DATASET_PADDED}:/app/datasets_${DATASET_PADDED}"
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
    command: flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092
    volumes:
      - "./src:/app/src"
      - "./yolov5:/app/yolov5"
      - "./logs:/app/logs"
      - "./certs:/app/certs:ro"
      - "./datasets_${DATASET_PADDED}:/app/datasets_${DATASET_PADDED}"
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

    # Start clients in parallel for faster deployment
    echo_info "Starting all client containers in parallel..."
    
    # Background job tracking
    DEPLOYMENT_PIDS=()
    FAILED_CLIENTS=()
    
    for i in $(seq 1 $MAX_CLIENT_NUM); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_EXT_IP_VAR="CLIENT_${i}_EXTERNAL_IP"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        CLIENT_EXT_IP=${!CLIENT_EXT_IP_VAR}
        
        # Skip if VM is not running (not in vm-info.txt)
        if [ -z "$CLIENT_VM" ] || [ -z "$CLIENT_ZONE" ]; then
            echo "  ℹ️  Skipping client VM $i (not running or not found)"
            continue
        fi
        
        # Skip if VM doesn't have external IP (can't connect via SSH)
        if [ -z "$CLIENT_EXT_IP" ] || [ "$CLIENT_EXT_IP" = "None" ]; then
            echo "  ⚠️  Skipping client VM $i ($CLIENT_VM) - no external IP assigned"
            continue
        fi
        
        # Check if VM is actually running
        if ! check_vm_running "$CLIENT_VM" "$CLIENT_ZONE"; then
            echo "  ⏹️  Skipping client VM $i ($CLIENT_VM) - VM is stopped"
            continue
        fi
        
        # Deploy client in background
        (
            echo_info "Starting client deployment on $CLIENT_VM (background)"
            
            if gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
                cd /app
                echo 'DOCKER_IMAGE=$DOCKER_IMAGE' >> .env
                sudo docker compose pull --quiet
                
                # Ensure FLAML is installed in running container (if not in image)
                if ! python3 -c 'import flaml' 2>/dev/null; then
                    echo '⚠️  FLAML not found, installing in-container...'
                    pip install -q 'flaml[automl]>=2.0.0' || true
                fi
                
                sudo docker compose up -d --force-recreate
                
                # Wait for containers to be running
                echo '  → Waiting for client containers to be ready...'
                MAX_WAIT=60
                WAITED=0
                while [ \$WAITED -lt \$MAX_WAIT ]; do
                    RUNNING_COUNT=\$(sudo docker compose ps --filter 'status=running' --format 'table' 2>/dev/null | grep -c 'running' || echo '0')
                    if [ \$RUNNING_COUNT -ge 2 ]; then
                        echo '  ✓ Client containers are running'
                        break
                    fi
                    WAITED=\$((WAITED + 5))
                    if [ \$WAITED -lt \$MAX_WAIT ]; then
                        echo \"  ⏳ Waiting for containers... (\${WAITED}s/\${MAX_WAIT}s)\"
                        sleep 5
                    fi
                done
                
                # Execute cleanup with retry
                for attempt in 1 2 3; do
                    if sudo docker compose exec -T fl-client-0 find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; then
                        for CLIENT_ID in \$(sudo docker compose ps --services 2>/dev/null); do
                            sudo docker compose exec -T \$CLIENT_ID find /app -type f -name '*.pyc' -delete 2>/dev/null || true
                        done
                        break
                    elif [ \$attempt -lt 3 ]; then
                        echo \"  ℹ️  Cleanup attempt \$attempt failed, retrying...\"
                        sleep 2
                    fi
                done
                
                sudo docker compose ps
            " 2>&1 | grep -E "(NAME|fl-client|Waiting|ready)" || true; then
                echo "[BACKGROUND] ✅ Client $i deployment succeeded"
                exit 0
            else
                echo "[BACKGROUND] ❌ Client $i deployment failed"
                exit 1
            fi
        ) &
        
        # Store PID
        DEPLOYMENT_PIDS+=($!)
        LAST_INDEX=$((${#DEPLOYMENT_PIDS[@]} - 1))
        echo "  ↳ Client $i deployment spawned (PID: ${DEPLOYMENT_PIDS[$LAST_INDEX]})"
    done
    
    # Wait for all background deployments to complete
    echo_info "Waiting for all client deployments to complete..."
    TOTAL_CLIENTS=${#DEPLOYMENT_PIDS[@]}
    COMPLETED=0
    FAILED=0
    
    for i in "${!DEPLOYMENT_PIDS[@]}"; do
        PID=${DEPLOYMENT_PIDS[$i]}
        CLIENT_NUM=$((i + 1))
        
        if wait $PID 2>/dev/null; then
            COMPLETED=$((COMPLETED + 1))
            echo "  ✅ Client $CLIENT_NUM completed"
        else
            FAILED=$((FAILED + 1))
            FAILED_CLIENTS+=("$CLIENT_NUM")
            echo "  ❌ Client $CLIENT_NUM failed"
        fi
    done
    
    echo_success "Client deployment complete! ($COMPLETED succeeded, $FAILED failed)"
    
    if [ $FAILED -gt 0 ]; then
        echo_warning "Failed clients: ${FAILED_CLIENTS[@]}"
    fi
fi

echo_success "Deployment process complete!"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Run ID: $RUN_ID"
echo "  Server IP: $SERVER_INTERNAL_IP"
echo "  Clients deployed: $MAX_CLIENT_NUM"
echo "  ✅ Using Dataset $DATASET_PADDED from /app/datasets_${DATASET_PADDED}/coco_partitions/"
echo "═══════════════════════════════════════════════════════════"
echo ""
if [ "$DEPLOY_SERVER" = "yes" ]; then
    echo "To start training:"
    echo "  gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='cd /app && sudo docker compose exec fl-server flwr run .'"
    echo ""
    echo "Monitor with: ./05-manage-clients.sh status"
else
    echo "Clients deployed. Server was not updated."
    echo "All clients connected to: ${SERVER_INTERNAL_IP}:9092"
fi
echo ""
echo "Note: vm-info.txt has been updated with current IPs (backup saved as vm-info.txt.backup)"