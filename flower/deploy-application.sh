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
    
    # Dataset size
    read -p "Training images per client [2000]: " N_TRAIN
    N_TRAIN=${N_TRAIN:-2000}
    read -p "Validation images per client [500]: " N_VAL
    N_VAL=${N_VAL:-500}
    
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
    shm_size: '2gb'
    env_file: [.env]
    command: >
      sh -c "if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-superlink --insecure --fleet-api-address=0.0.0.0:9092;
      else
        flower-superlink --fleet-api-address=0.0.0.0:9092 --ssl-ca-certfile=/app/certs/ca.crt --ssl-certfile=/app/certs/server.crt --ssl-keyfile=/app/certs/server.key;
      fi"
    ports: ["9092:9092", "9093:9093"]
    volumes: [".:/app", "./certs:/app/certs:ro"]
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
        sudo mkdir -p /app/{logs,datasets,certs}
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
    
    # Create client docker-compose
    cat > /tmp/docker-compose-client-${i}.yml << EOF
version: '3.8'
services:
  fl-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    container_name: fl-client-${CLIENT_ID_1}
    shm_size: '2gb'
    env_file: [.env]
    command: >
      sh -c "if [ \"\\\$INSECURE\" = 'true' ]; then
        flower-supernode --insecure --superlink=${SERVER_INTERNAL_IP}:9092;
      else
        flower-supernode --superlink=${SERVER_INTERNAL_IP}:9092 --root-certificates=/app/certs/ca.crt;
      fi"
    volumes: ["./logs:/app/logs", "./certs:/app/certs:ro", "./datasets:/app/datasets", "./yolov5:/app/yolov5", "./gcs-key.json:/app/gcs-key.json:ro"]
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
    volumes: ["./logs:/app/logs", "./certs:/app/certs:ro", "./datasets:/app/datasets", "./yolov5:/app/yolov5", "./gcs-key.json:/app/gcs-key.json:ro"]
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
    
    # Download dataset from GCS (skip if already exists) - OPTIMIZED VERSION
    echo_info "Checking/downloading COCO subset on $CLIENT_VM (N_TRAIN=$N_TRAIN, N_VAL=$N_VAL)..."
    gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command='
      cd /app
      export GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json

      mkdir -p datasets/coco/{images,labels}/{train2017,val2017}

      # Check current dataset status
      TRAIN_IMG_COUNT=$(ls datasets/coco/images/train2017/*.jpg 2>/dev/null | wc -l)
      VAL_IMG_COUNT=$(ls datasets/coco/images/val2017/*.jpg 2>/dev/null | wc -l)
      TRAIN_LABEL_COUNT=$(ls datasets/coco/labels/train2017/*.txt 2>/dev/null | wc -l)
      VAL_LABEL_COUNT=$(ls datasets/coco/labels/val2017/*.txt 2>/dev/null | wc -l)

      echo "Current dataset: Train images=$TRAIN_IMG_COUNT, Val images=$VAL_IMG_COUNT"
      echo "Current labels: Train labels=$TRAIN_LABEL_COUNT, Val labels=$VAL_LABEL_COUNT"
      echo "Target: Train='"$N_TRAIN"', Val='"$N_VAL"'"

      # ------------------ TRAINING IMAGES ------------------
      if [ "$TRAIN_IMG_COUNT" -ge '"$N_TRAIN"' ]; then
        echo "Training images already sufficient, skipping download"
      else
        echo "Downloading '"$N_TRAIN"' training images..."
        gsutil -m cp $(gsutil ls gs://'"$BUCKET_NAME"'/coco/images/train2017/*.jpg | head -'"$N_TRAIN"') datasets/coco/images/train2017/ 2>/dev/null || true
        echo "Training images downloaded."
      fi

      # ------------------ TRAINING LABELS (OPTIMIZED) ------------------
      CURRENT_TRAIN_IMGS=$(ls datasets/coco/images/train2017/*.jpg 2>/dev/null | wc -l)
      CURRENT_TRAIN_LABELS=$(ls datasets/coco/labels/train2017/*.txt 2>/dev/null | wc -l)

      if [ "$CURRENT_TRAIN_LABELS" -ge "$CURRENT_TRAIN_IMGS" ]; then
        echo "Training labels already sufficient, skipping download"
      else
        echo "Downloading training labels in parallel..."
        # Create list of labels needed based on downloaded images
        cd datasets/coco/images/train2017
        > /tmp/train_labels_list.txt
        for img in *.jpg; do
          basename="${img%.jpg}"
          if [ ! -f "../../labels/train2017/${basename}.txt" ]; then
            echo "gs://'"$BUCKET_NAME"'/coco/labels/train2017/${basename}.txt" >> /tmp/train_labels_list.txt
          fi
        done
        
        # Download all labels in parallel using gsutil -m
        if [ -s /tmp/train_labels_list.txt ]; then
          cat /tmp/train_labels_list.txt | gsutil -m cp -I ../../labels/train2017/ 2>/dev/null || true
        fi
        rm -f /tmp/train_labels_list.txt
        cd /app
        echo "Training labels downloaded."
      fi

      # ------------------ VALIDATION IMAGES ------------------
      if [ "$VAL_IMG_COUNT" -ge '"$N_VAL"' ]; then
        echo "Validation images already sufficient, skipping download"
      else
        echo "Downloading '"$N_VAL"' validation images..."
        gsutil -m cp $(gsutil ls gs://'"$BUCKET_NAME"'/coco/images/val2017/*.jpg | head -'"$N_VAL"') datasets/coco/images/val2017/ 2>/dev/null || true
        echo "Validation images downloaded."
      fi

      # ------------------ VALIDATION LABELS (OPTIMIZED) ------------------
      CURRENT_VAL_IMGS=$(ls datasets/coco/images/val2017/*.jpg 2>/dev/null | wc -l)
      CURRENT_VAL_LABELS=$(ls datasets/coco/labels/val2017/*.txt 2>/dev/null | wc -l)

      if [ "$CURRENT_VAL_LABELS" -ge "$CURRENT_VAL_IMGS" ]; then
        echo "Validation labels already sufficient, skipping download"
      else
        echo "Downloading validation labels in parallel..."
        # Create list of labels needed based on downloaded images
        cd datasets/coco/images/val2017
        > /tmp/val_labels_list.txt
        for img in *.jpg; do
          basename="${img%.jpg}"
          if [ ! -f "../../labels/val2017/${basename}.txt" ]; then
            echo "gs://'"$BUCKET_NAME"'/coco/labels/val2017/${basename}.txt" >> /tmp/val_labels_list.txt
          fi
        done
        
        # Download all labels in parallel using gsutil -m
        if [ -s /tmp/val_labels_list.txt ]; then
          cat /tmp/val_labels_list.txt | gsutil -m cp -I ../../labels/val2017/ 2>/dev/null || true
        fi
        rm -f /tmp/val_labels_list.txt
        cd /app
        echo "Validation labels downloaded."
      fi
      
      echo "Dataset check/download complete"
      echo "Final counts: Train images=$(ls datasets/coco/images/train2017/*.jpg 2>/dev/null | wc -l), Val images=$(ls datasets/coco/images/val2017/*.jpg 2>/dev/null | wc -l)"
      echo "Final counts: Train labels=$(ls datasets/coco/labels/train2017/*.txt 2>/dev/null | wc -l), Val labels=$(ls datasets/coco/labels/val2017/*.txt 2>/dev/null | wc -l)"
      ' &
done

# Wait for dataset downloads
echo_info "Waiting for dataset downloads to complete..."
wait

echo_success "All clients deployed and datasets downloaded"

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
echo "To start training: gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='cd /app && flwr run .'"
echo ""
echo "Monitor with: ./05-manage-clients.sh status"