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

# Ensure user owns /app so we can SCP to it
# We aggressively clean the destination first to avoid permission conflicts with old containers/pycache
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo rm -rf /app/fly-bold-fedn && sudo mkdir -p /app/fly-bold-fedn && sudo chown -R \$(id -u):\$(id -g) /app" >/dev/null

# Copy the full project folder (excluding .venv by explicit list)
info "Copying fedn folder to server..."
# We explicitly list items to avoid copying .venv or other garbage
FILES_TO_COPY="fedn client *.sh *.py *.md *.txt *.yaml *.tgz *.npz"
# We can't use wildcards directly in gcloud scp local path easily if they match multiple files, 
# but gcloud scp supports multiple sources.
gcloud compute scp --recurse $FILES_TO_COPY "$SERVER_VM":/app/fly-bold-fedn --zone="$SERVER_ZONE" --quiet

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
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
set -e
cd /app
sudo docker compose pull || true
sudo docker compose up -d
sleep 10
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

  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="sudo mkdir -p /app/{client,logs}" >/dev/null
  # Ensure user owns /app so we can SCP to it
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="sudo rm -rf /app/fly-bold-fedn && sudo mkdir -p /app/fly-bold-fedn && sudo chown -R \$(id -u):\$(id -g) /app" >/dev/null

  # Copy full project (excluding .venv)
  # FILES_TO_COPY is defined above
  gcloud compute scp --recurse $FILES_TO_COPY "$VM_NAME":/app/fly-bold-fedn --zone="$VM_ZONE" --quiet

  # Setup Client Env
  # Generate dynamic docker-compose for this VM to match Client IDs (match fedn-client-<ID>)
  cat > /tmp/docker-compose-client-${i}.yaml << EOF
version: '3.8'
services:
  fedn-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    container_name: fedn-client-${CLIENT_ID_1}
    working_dir: /app/client
    command: ["client","start","--combiner","${SERVER_INTERNAL_IP}","--combiner-port","12080","--in","fedn.yaml","--name","client-${CLIENT_ID_1}","--local-package"]
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_1}
    volumes:
      - ./client/fedn.yaml:/app/client/fedn.yaml
      - ../logs:/app/logs
  fedn-client-${CLIENT_ID_2}:
    image: ${DOCKER_IMAGE}
    container_name: fedn-client-${CLIENT_ID_2}
    working_dir: /app/client
    command: ["client","start","--combiner","${SERVER_INTERNAL_IP}","--combiner-port","12080","--in","fedn.yaml","--name","client-${CLIENT_ID_2}","--local-package"]
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_2}
    volumes:
      - ./client/fedn.yaml:/app/client/fedn.yaml
      - ../logs:/app/logs
networks:
  default:
    driver: bridge
EOF

  # Copy generated compose file
  gcloud compute scp /tmp/docker-compose-client-${i}.yaml "$VM_NAME":/app/docker-compose.yaml --zone="$VM_ZONE" --quiet

  # Setup Client Env and Configs
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
    cp /app/fly-bold-fedn/fedn/client/fedn.yaml /app/client/fedn.yaml
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
python3.12 -m pip install --no-cache-dir --upgrade pip
python3.12 -m pip install --no-cache-dir \"numpy<2\" opencv-python-headless==4.9.0.80 fedn yolov5
" >/dev/null

  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
cd /app
sudo docker compose pull || true
sudo docker compose up -d
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
