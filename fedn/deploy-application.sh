#!/bin/bash

# Deploy FEDn server + combiner + clients onto the provisioned VMs
set -euo pipefail

PROJECT_ID="inf022"

info(){ echo -e "\n[INFO] $1\n"; }
success(){ echo -e "\n[SUCCESS] $1\n"; }
fail(){ echo -e "\n[ERROR] $1\n"; exit 1; }

if [ ! -f vm-info.txt ]; then
  fail "vm-info.txt not found. Run setup-infrastructure.sh first."
fi
if [ ! -f docker-image-info.txt ]; then
  fail "docker-image-info.txt not found. Run build-push-image.sh first."
fi
source vm-info.txt
DOCKER_IMAGE=$(grep '^DOCKER_IMAGE=' docker-image-info.txt | cut -d'=' -f2)

info "Starting FEDn deployment"

# Save basic env vars for use in prompts or eventual local usage
cat > .env << EOF
DOCKER_IMAGE=$DOCKER_IMAGE
SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP
EOF

### Deploy server (controller + combiner + deps)
info "Preparing server VM $SERVER_VM"

gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo mkdir -p /app/{config,storage,certs}" >/dev/null
# Ensure docker running and permissions applied
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo usermod -aG docker $USER && sudo systemctl enable --now docker" >/dev/null

# Ensure user owns /app so we can SCP to it and install unzip and python deps
# We aggressively clean the destination first, INCLUDING docker-compose files and STORAGE (to kill zombies)
# CRITICAL: Stop containers FIRST so they don't hold onto deleted file handles (fixes MinIO SlowDown/Unwritable error)
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo docker ps -aq | xargs -r sudo docker rm -f && sudo rm -rf /app/fly-bold-fedn /app/fedn /app/fedn.zip /app/docker-compose.yml /app/docker-compose.yaml /app/storage && sudo mkdir -p /app/fly-bold-fedn /app/storage && sudo chown -R \$(id -u):\$(id -g) /app && sudo apt-get update && sudo apt-get install -y unzip python3-pip && pip3 install pymongo" >/dev/null

# Copy Zip
info "Copying fedn.zip to server..."
gcloud compute scp ../fedn.zip "$SERVER_VM":/app/fedn.zip --zone="$SERVER_ZONE" --quiet

# Unzip and fix structure (zip contains fedn/ folder, so we unzip to /app then rename fedn -> fly-bold-fedn)
info "Unzipping on server..."
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="unzip -q /app/fedn.zip -d /app && rm -rf /app/fly-bold-fedn && mv /app/fedn /app/fly-bold-fedn && rm /app/fedn.zip" >/dev/null
# Install fedn from source
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="pip3 install --no-cache-dir /app/fly-bold-fedn/fedn" >/dev/null

# Copy configs from local fedn/config
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
  cp /app/fly-bold-fedn/docker-compose-server.yaml /app/docker-compose.yaml
  
  # Copy config templates to the active config dir
  cp /app/fly-bold-fedn/fedn/config/settings-combiner.yaml.template /app/config/combiner.yaml
  cp /app/fly-bold-fedn/fedn/config/controller.yaml /app/config/controller.yaml
  cp /app/fly-bold-fedn/fedn/config/settings-hooks.yaml.template /app/config/hooks.yaml

  # DYNAMIC CONFIG UPDATE:
  # Replace the placeholder 'REPLACE_THIS_IP' with the actual Internal IP of this VM.
  # This ensures the Combiner advertises the correct address to clients.
  sed -i \"s/REPLACE_THIS_IP/$SERVER_INTERNAL_IP/g\" /app/config/*.yaml
"

# Create .env on Server VM for Docker Compose
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
cat > /app/.env << EOF
DOCKER_IMAGE=$DOCKER_IMAGE
SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP
MINIO_ROOT_USER=fedn_admin
MINIO_ROOT_PASSWORD=password
MONGO_USER=fedn_admin
MONGO_PASSWORD=password
EOF
"

# Copy configs if they exist in the repo location, otherwise we might be missing specific controller.yaml
# The original script GENERATED them. The user wants 'fedn folder go as is'.
# If 'fedn/fedn/config' has templates, we need to use them.
# Inspecting file list: fedn/fedn/config exists.
# We will trust the 'as is' directive. If files are missing, it will fail, but that matches 'as is'.

info "Starting server stack on $SERVER_VM"
# Starting server stack on $SERVER_VM
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
set -e
cd /app
sudo docker compose pull || true
sudo docker compose up -d --remove-orphans

echo \"Waiting for FEDn Controller to be ready...\"
max_retries=30
count=0
while ! curl -s http://localhost:8092/get_controller_status >/dev/null; do
  echo \"Waiting for controller... \$count/\$max_retries\"
  sleep 5
  count=\$((count+1))
  if [ \$count -ge \$max_retries ]; then
    echo \"Timeout waiting for controller.\"
    sudo docker compose logs api-server
    exit 1
  fi
done
echo \"Controller is ready!\"
sudo docker compose ps
" || fail "Server deployment failed"

success "Server and combiner running"

### Deploy clients (10 total, 2 per VM)

for i in $(seq 1 5); do
  VM_VAR="CLIENT_${i}_VM"; ZONE_VAR="CLIENT_${i}_ZONE"
  VM_NAME=${!VM_VAR}; VM_ZONE=${!ZONE_VAR}
  CLIENT_ID_1=$(( (i-1)*2 ))
  CLIENT_ID_2=$(( (i-1)*2 + 1 ))

  info "Deploying clients on $VM_NAME (ids $CLIENT_ID_1,$CLIENT_ID_2)"

  # Ensure user owns /app so we can SCP to it and install dependencies (unzip, python3.12 via PPA)
  # Ensure user owns /app so we can SCP to it and install dependencies (unzip, python3.12 via PPA)
  # We aggressively clean the destination first, INCLUDING docker-compose files and OLD CONTAINERS
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="sudo docker ps -aq | xargs -r sudo docker rm -f && sudo rm -rf /app/fly-bold-fedn /app/fedn /app/fedn.zip /app/docker-compose.yml /app/docker-compose.yaml && sudo mkdir -p /app/fly-bold-fedn /app/client /app/logs && sudo chown -R \$(id -u):\$(id -g) /app && sudo apt-get update && sudo apt-get install -y software-properties-common unzip && sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update && sudo apt-get install -y python3.12-full python3.12-venv" >/dev/null

  # Copy Zip
  info "Copying fedn.zip to $VM_NAME..."
  gcloud compute scp ../fedn.zip "$VM_NAME":/app/fedn.zip --zone="$VM_ZONE" --quiet

  # Unzip and fix structure
  info "Unzipping on $VM_NAME..."
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="unzip -q /app/fedn.zip -d /app && rm -rf /app/fly-bold-fedn && mv /app/fedn /app/fly-bold-fedn && rm /app/fedn.zip && sudo chmod -R 777 /app/fly-bold-fedn/client" >/dev/null

  # Setup Client Env
  # Generate dynamic docker-compose for this VM to match Client IDs (match fedn-client-<ID>)
  cat > /tmp/docker-compose-client-${i}.yaml << EOF
version: '3.8'
services:
  fedn-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    user: "0:0"
    container_name: fedn-client-${CLIENT_ID_1}
    working_dir: /app
    entrypoint: ""
    command: ["/bin/bash", "-c", "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && export HOME=/app/tmp && mkdir -p /app/tmp && /venv/bin/pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 'numpy<2' yolov5==7.0.14 'matplotlib>=3.2.2' opencv-python-headless==4.9.0.80 'Pillow>=7.1.2' 'PyYAML>=5.3.1' 'requests>=2.23.0' 'huggingface-hub>=0.24.0,<0.25.0' 'scipy>=1.4.1' 'tqdm>=4.64.0' 'pandas>=1.1.4' 'seaborn>=0.11.0' psutil 'thop>=0.1.1' 'protobuf>=5.0.0,<6.31.0' 'pycocotools>=2.0.6' && /venv/bin/fedn client start --combiner ${SERVER_INTERNAL_IP} --combiner-port 12080 -in client/fedn.yaml --name client-${CLIENT_ID_1} --local-package"]
    extra_hosts:
      - "combiner:${SERVER_INTERNAL_IP}"
    shm_size: '4gb'
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_1}
      YOLO_SPLITS_TARGET: /app/datasets_1/coco_partitions
    volumes:
      - ./fly-bold-fedn/client:/app/client
      - ../logs:/app/logs
      - /app/datasets:/app/datasets
      - /app/datasets_1:/app/datasets_1
  fedn-client-${CLIENT_ID_2}:
    image: ${DOCKER_IMAGE}
    user: "0:0"
    container_name: fedn-client-${CLIENT_ID_2}
    working_dir: /app
    entrypoint: ""
    command: ["/bin/bash", "-c", "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && export HOME=/app/tmp && mkdir -p /app/tmp && /venv/bin/pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 'numpy<2' yolov5==7.0.14 'matplotlib>=3.2.2' opencv-python-headless==4.9.0.80 'Pillow>=7.1.2' 'PyYAML>=5.3.1' 'requests>=2.23.0' 'huggingface-hub>=0.24.0,<0.25.0' 'scipy>=1.4.1' 'tqdm>=4.64.0' 'pandas>=1.1.4' 'seaborn>=0.11.0' psutil 'thop>=0.1.1' 'protobuf>=5.0.0,<6.31.0' 'pycocotools>=2.0.6' && /venv/bin/fedn client start --combiner ${SERVER_INTERNAL_IP} --combiner-port 12080 -in client/fedn.yaml --name client-${CLIENT_ID_2} --local-package"]
    extra_hosts:
      - "combiner:${SERVER_INTERNAL_IP}"
    shm_size: '4gb'
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_2}
      YOLO_SPLITS_TARGET: /app/datasets_1/coco_partitions
    volumes:
      - ./fly-bold-fedn/client:/app/client
      - ../logs:/app/logs
      - /app/datasets:/app/datasets
      - /app/datasets_1:/app/datasets_1
networks:
  default:
    driver: bridge
EOF

  # Copy generated compose file
  gcloud compute scp /tmp/docker-compose-client-${i}.yaml "$VM_NAME":/app/docker-compose.yaml --zone="$VM_ZONE" --quiet

  # Setup Client Env and Configs
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
    cp /app/fly-bold-fedn/client/fedn.yaml /app/client/fedn.yaml
    cat > /app/.env << INNEREOF
DOCKER_IMAGE=$DOCKER_IMAGE
COMBINER_HOST=$SERVER_INTERNAL_IP
CLIENT_ID_1=$CLIENT_ID_1
CLIENT_ID_2=$CLIENT_ID_2
INNEREOF
  "

  # Ensure python3.12 stack (defense-in-depth)
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
set -e
sudo usermod -aG docker \$USER || true
sudo systemctl enable --now docker || true
python3.12 -V || true
python3.12 -m ensurepip --upgrade
python3.12 -m pip install --no-cache-dir --upgrade pip
python3.12 -m pip install --no-cache-dir \"numpy<2\" opencv-python-headless==4.9.0.80 fedn yolov5
" >/dev/null

  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
cd /app
sudo docker compose pull || true
sudo docker compose up -d --remove-orphans
sleep 5
sudo docker compose ps
" &
done

info "Waiting for client deployments..."
wait
success "Deployment complete"

echo "Server: $SERVER_VM ($SERVER_INTERNAL_IP)"
echo "Clients running (2 per client VM)"
echo "Use ./manage-clients.sh status to verify"
