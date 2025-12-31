#!/bin/bash

# Deploy Flybold Application
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

# Load VM info
if [ ! -f "vm-info.txt" ]; then
    echo_error "vm-info.txt not found. Run 02-setup-infrastructure.sh first."
fi
source vm-info.txt

# Load Docker image
if [ ! -f "docker-image-info.txt" ]; then
    echo_error "docker-image-info.txt not found. Run 03-build-push-image.sh first."
fi
DOCKER_IMAGE=$(grep '^DOCKER_IMAGE=' docker-image-info.txt | cut -d'=' -f2)

# ✅ NEW: Check if partition manifest exists
if [ ! -f "partition_outputs/partition_manifest.json" ]; then
    echo_error "partition_manifest.json not found.\n\nPlease run 03b-partition-dataset.sh BEFORE deploying.\nThis script partitions the dataset for all clients."
fi

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
    else
        SKIP_PROMPTS=false
    fi
else
    SKIP_PROMPTS=false
fi

if [ "$SKIP_PROMPTS" = false ]; then
    echo_info "Configuration Parameters"
    
    # GPU
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
    
    # TLS
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
    
    # NOTE: N_TRAIN and N_VAL should match partition manifest
    echo_info "Dataset parameters should match partition manifest!"
    read -p "Training images per client [10000]: " N_TRAIN
    N_TRAIN=${N_TRAIN:-10000}
    read -p "Validation images per client [5000]: " N_VAL
    N_VAL=${N_VAL:-5000}
    
    # FL parameters
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
    
    read -p "Dirichlet alpha [0.5]: " DIRICHLET_ALPHA
    DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.5}
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
cat > .env << EOF
ENABLE_GPU=$ENABLE_GPU
ENABLE_TLS=$ENABLE_TLS
INSECURE=$INSECURE
N_TRAIN=$N_TRAIN
N_VAL=$N_VAL
RUN_ID=$RUN_ID
NUM_SERVER_ROUNDS=$NUM_SERVER_ROUNDS
NUM_CLIENTS=10
LOCAL_EPOCHS=$LOCAL_EPOCHS
BATCH_SIZE=$BATCH_SIZE
FRACTION_TRAIN=$FRACTION_TRAIN
FRACTION_EVALUATE=$FRACTION_EVALUATE
LR=$LR
YOLO_SIZE=$YOLO_SIZE
IMG_SIZE=$IMG_SIZE
DIRICHLET_ALPHA=$DIRICHLET_ALPHA
NUM_CPUS=$NUM_CPUS
NUM_GPUS=$NUM_GPUS
FLWR_SUPERLINK_ADDRESS=0.0.0.0:9092
BUCKET_NAME=$BUCKET_NAME
DOCKER_IMAGE=$DOCKER_IMAGE
EOF

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
  "dirichlet_alpha": $DIRICHLET_ALPHA,
  "n_train": $N_TRAIN,
  "n_val": $N_VAL,
  "enable_gpu": $ENABLE_GPU,
  "enable_tls": $ENABLE_TLS,
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
sed -i "s/dirichlet_alpha = [0-9.]\+/dirichlet_alpha = $DIRICHLET_ALPHA/" pyproject.toml
sed -i "s/run_id = [0-9]\+/run_id = $RUN_ID/" pyproject.toml
sed -i "s|coco_root = \".*\"|coco_root = \"/app/datasets/coco\"|" pyproject.toml
sed -i "s|gcs_bucket = \".*\"|gcs_bucket = \"$BUCKET_NAME\"|" pyproject.toml

if [ "$INSECURE" = "false" ]; then
    sed -i 's/^[[:space:]]*insecure = .*/insecure = false/' pyproject.toml
    sed -i 's/^[[:space:]]*# root-certificates =/root-certificates =/' pyproject.toml
else
    sed -i 's/^[[:space:]]*insecure = .*/insecure = true/' pyproject.toml
    sed -i 's/^[[:space:]]*root-certificates =/# root-certificates =/' pyproject.toml
fi

echo_success "Configuration saved"

# Deploy server
echo_info "Deploying server on $SERVER_VM"

gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    sudo mkdir -p /app/{logs,checkpoints,certs}
    sudo chown -R \$USER:\$USER /app
"

# Copy files
gcloud compute scp --recurse ./src $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
gcloud compute scp --recurse ./yolov5 $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
gcloud compute scp requirements.txt pyproject.toml .env $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

if [ "$ENABLE_TLS" = "true" ]; then
    gcloud compute scp --recurse ./certs $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
fi

# Create server docker-compose
cat > /tmp/docker-compose-server.yml << EOF
version: '3.8'
services:
  fl-server:
    image: ${DOCKER_IMAGE}
    container_name: fl-server
    shm_size: '24gb'
    env_file: [.env]
    command: >
      sh -c "if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-superlink --insecure --fleet-api-address=0.0.0.0:9092;
      else
        flower-superlink --fleet-api-address=0.0.0.0:9092 --ssl-ca-certfile=/app/certs/ca.crt --ssl-certfile=/app/certs/server.crt --ssl-keyfile=/app/certs/server.key;
      fi"
    ports: ["9092:9092", "9093:9093"]
    volumes: [".:/app", "./src:/app/src", "./certs:/app/certs:ro", "./logs:/app/logs"]
    restart: unless-stopped
networks:
  default:
    driver: bridge
EOF

gcloud compute scp /tmp/docker-compose-server.yml $SERVER_VM:/app/docker-compose.yml --zone=$SERVER_ZONE --quiet

# Start server
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    echo 'DOCKER_IMAGE=$DOCKER_IMAGE' >> .env
    sudo docker compose pull
    sudo docker compose up -d
    sleep 10
    sudo docker compose ps
"

echo_success "Server deployed"

# Deploy clients
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    echo_info "Deploying clients on $CLIENT_VM"
    
    # Setup directories
    gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        sudo mkdir -p /app/{logs,certs}
        sudo chown -R \$USER:\$USER /app
    "
    
    # Copy files
    gcloud compute scp --recurse ./src $CLIENT_VM:/app/ --zone=$CLIENT_ZONE --quiet
    gcloud compute scp --recurse ./yolov5 $CLIENT_VM:/app/ --zone=$CLIENT_ZONE --quiet
    gcloud compute scp requirements.txt pyproject.toml .env $CLIENT_VM:/app/ --zone=$CLIENT_ZONE --quiet
    
    if [ "$ENABLE_TLS" = "true" ]; then
        gcloud compute scp --recurse ./certs $CLIENT_VM:/app/ --zone=$CLIENT_ZONE --quiet
    fi
    
    # Create service account key for GCS access
    if [ ! -f "gcs-key.json" ]; then
        gcloud iam service-accounts keys create gcs-key.json \
            --iam-account=default-compute@${PROJECT_ID}.iam.gserviceaccount.com 2>/dev/null || true
    fi
    gcloud compute scp gcs-key.json $CLIENT_VM:/app/ --zone=$CLIENT_ZONE --quiet
    
    # Calculate client IDs for this VM (2 clients per VM)
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))
    
    # ✅ REMOVED: Dataset download section (lines 207-280 in original)
    # ✅ NEW: Verify pre-partitioned data exists
    echo_info "Verifying pre-partitioned data on $CLIENT_VM (Clients $CLIENT_ID_1, $CLIENT_ID_2)..."
    VERIFICATION_OUTPUT=$(gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="
        set -e
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            PARTITION_DIR=\"/app/datasets/coco_partitions/client_\${CLIENT_ID}\"
            
            if [ ! -d \"\$PARTITION_DIR\" ]; then
                echo \"ERROR: Partition directory not found: \$PARTITION_DIR\"
                echo \"Please run 03b-partition-dataset.sh before deployment!\"
                exit 1
            fi
            
            if [ ! -f \"\$PARTITION_DIR/coco_client.yaml\" ]; then
                echo \"ERROR: YAML config missing for client \$CLIENT_ID\"
                exit 1
            fi
            
            TRAIN_IMG=\$(ls \$PARTITION_DIR/images/train2017/*.jpg 2>/dev/null | wc -l)
            VAL_IMG=\$(ls \$PARTITION_DIR/images/val2017/*.jpg 2>/dev/null | wc -l)
            
            if [ \$TRAIN_IMG -eq 0 ]; then
                echo \"ERROR: No training images found for client \$CLIENT_ID\"
                exit 1
            fi
            
            echo \"✅ Client \$CLIENT_ID: Train=\$TRAIN_IMG images, Val=\$VAL_IMG images\"
        done
    " 2>&1)
    
    if echo "$VERIFICATION_OUTPUT" | grep -q "ERROR"; then
        echo_error "Pre-partitioned data verification failed on $CLIENT_VM:\n$VERIFICATION_OUTPUT"
    else
        echo "$VERIFICATION_OUTPUT"
    fi
    
    # Create client docker-compose
    cat > /tmp/docker-compose-client-${i}.yml << EOF
version: '3.8'
services:
  fl-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    container_name: fl-client-${CLIENT_ID_1}
    shm_size: '28gb'
    env_file: [.env]
    command: >
      sh -c "if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092;
      else
        flower-supernode --superlink=${SERVER_INTERNAL_IP}:9092 --root-certificates=/app/certs/ca.crt;
      fi"
    volumes: ["./src:/app/src", "./logs:/app/logs", "./certs:/app/certs:ro", "./datasets:/app/datasets:ro", "./yolov5:/app/yolov5", "./gcs-key.json:/app/gcs-key.json:ro"]
    environment:
      - CLIENT_ID=${CLIENT_ID_1}
      - PARTITION_ID=${CLIENT_ID_1}
      - GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
    restart: unless-stopped
    
  fl-client-${CLIENT_ID_2}:
    image: ${DOCKER_IMAGE}
    container_name: fl-client-${CLIENT_ID_2}
    shm_size: '2gb'
    env_file: [.env]
    command: >
      sh -c "if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092;
      else
        flower-supernode --superlink=${SERVER_INTERNAL_IP}:9092 --root-certificates=/app/certs/ca.crt;
      fi"
    volumes: ["./logs:/app/logs", "./certs:/app/certs:ro", "./datasets:/app/datasets:ro", "./yolov5:/app/yolov5", "./gcs-key.json:/app/gcs-key.json:ro"]
    environment:
      - CLIENT_ID=${CLIENT_ID_2}
      - PARTITION_ID=${CLIENT_ID_2}
      - GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json
    restart: unless-stopped
networks:
  default:
    driver: bridge
EOF
    
    gcloud compute scp /tmp/docker-compose-client-${i}.yml $CLIENT_VM:/app/docker-compose.yml --zone=$CLIENT_ZONE --quiet
done

echo_success "All clients deployed and data verified"

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
        sudo docker compose pull
        sudo docker compose up -d
        sleep 5
        sudo docker compose ps
    "
done

echo_success "Deployment complete!"
echo ""
echo "Run ID: $RUN_ID"
echo "✅ All clients are using pre-partitioned data from /app/datasets/coco_partitions/"
echo ""
echo "To start training: gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='cd /app && sudo docker compose exec fl-server flwr run .'"
echo ""
echo "Monitor with: ./05-manage-clients.sh status"