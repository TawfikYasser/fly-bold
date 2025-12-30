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

read -p "Enable TLS (self-signed)? (y/n) [n]: " tls
ENABLE_TLS=${tls:-n}
if [[ $ENABLE_TLS =~ ^[Yy]$ ]]; then
  if [ ! -d certs ]; then
    ./generate_certs.sh
  fi
  TLS_ENABLED=true
else
  TLS_ENABLED=false
fi

# Save basic env
cat > .env << EOF
DOCKER_IMAGE=$DOCKER_IMAGE
TLS_ENABLED=$TLS_ENABLED
SERVER_INTERNAL_IP=$SERVER_INTERNAL_IP
EOF

### Deploy server (controller + combiner + deps)
info "Preparing server VM $SERVER_VM"

gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo mkdir -p /app/{config,storage,certs}" >/dev/null
# Ensure docker running and permissions applied (belt-and-suspenders)
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="sudo usermod -aG docker $USER && sudo systemctl enable --now docker" >/dev/null

# Copy the full fedn project folder to the server for reference/admin use
gcloud compute scp --recurse . "$SERVER_VM":/app/fly-bold-fedn --zone="$SERVER_ZONE" --quiet

gcloud compute scp --recurse ./fedn/fedn/config $SERVER_VM:/app/ --zone="$SERVER_ZONE" --quiet
gcloud compute scp --recurse ./certs $SERVER_VM:/app/ --zone="$SERVER_ZONE" --quiet || true

# Render configs with internal IP
cat > /tmp/controller.yaml << EOF
network_id: fedn-network
controller:
  host: 0.0.0.0
  port: 8092
  debug: False

statestore:
  type: MongoDB
  mongo_config:
    username: fedn_admin
    password: password
    host: mongo
    port: 6534

storage:
  storage_type: BOTO3
  storage_config:
    storage_endpoint_url: http://minio:9000
    storage_access_key: fedn_admin
    storage_secret_key: password
    storage_bucket: fedn-models
    context_bucket: fedn-context
    storage_secure_mode: False
    storage_verify_ssl: False
EOF

cat > /tmp/combiner.yaml << EOF
network_id: fedn-network
name: combiner
host: 0.0.0.0
address: ${SERVER_INTERNAL_IP}
port: 12080
max_clients: 30

statestore:
  type: MongoDB
  mongo_config:
    username: fedn_admin
    password: password
    host: mongo
    port: 6534

storage:
  storage_type: BOTO3
  storage_config:
    storage_endpoint_url: http://minio:9000
    storage_access_key: fedn_admin
    storage_secret_key: password
    storage_bucket: fedn-models
    context_bucket: fedn-context
    storage_secure_mode: False
    storage_verify_ssl: False
EOF

cat > /tmp/hooks.yaml << EOF
network_id: fedn-network
discover_host: api-server
discover_port: 8092
name: hooks
host: hooks
port: 12081
max_clients: 30
EOF

cat > /tmp/docker-compose-server.yml << EOF
version: '3.8'
services:
  minio:
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: fedn_admin
      MINIO_ROOT_PASSWORD: password
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./storage/minio:/data
  mongo:
    image: mongo:7.0
    command: mongod --port 6534
    environment:
      MONGO_INITDB_ROOT_USERNAME: fedn_admin
      MONGO_INITDB_ROOT_PASSWORD: password
    ports:
      - "6534:6534"
    volumes:
      - ./storage/mongo:/data/db
  api-server:
    image: ${DOCKER_IMAGE}
    working_dir: /app
    command: ["controller","start","--init","/app/config/controller.yaml"]
    environment:
      STATESTORE_CONFIG: /app/config/controller.yaml
      MODELSTORAGE_CONFIG: /app/config/controller.yaml
      FEDN_COMPUTE_PACKAGE_DIR: /app/storage
      TMPDIR: /app/tmp
    depends_on:
      - minio
      - mongo
    ports:
      - "8092:8092"
    volumes:
      - ./config:/app/config
      - ./storage:/app/storage
  hooks:
    image: ${DOCKER_IMAGE}
    working_dir: /app
    command: ["hooks","start","--init","/app/config/hooks.yaml"]
    environment:
      TMPDIR: /app/tmp
    depends_on:
      - api-server
    ports:
      - "12081:12081"
    volumes:
      - ./config:/app/config
  combiner:
    image: ${DOCKER_IMAGE}
    working_dir: /app
    command: ["combiner","start","--init","/app/config/combiner.yaml"]
    environment:
      TMPDIR: /app/tmp
    depends_on:
      - api-server
      - hooks
    ports:
      - "12080:12080"
    volumes:
      - ./config:/app/config
networks:
  default:
    driver: bridge
EOF

gcloud compute scp /tmp/controller.yaml /tmp/combiner.yaml /tmp/hooks.yaml /tmp/docker-compose-server.yml $SERVER_VM:/app/config/ --zone="$SERVER_ZONE" --quiet

info "Starting server stack"
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
set -e
cd /app
# place configs
mv config/docker-compose-server.yml .
sudo docker compose -f docker-compose-server.yml pull || true
sudo docker compose -f docker-compose-server.yml up -d
sleep 10
sudo docker compose -f docker-compose-server.yml ps
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
  # Copy full project for reference/ops on each client VM
  gcloud compute scp --recurse . "$VM_NAME":/app/fly-bold-fedn --zone="$VM_ZONE" --quiet
  gcloud compute scp --recurse ./client $VM_NAME:/app/ --zone="$VM_ZONE" --quiet

  # Render client config overriding discover_host
  cat > /tmp/fedn-client-${i}.yaml << EOF
network_id: fedn-network
discover_host: ${SERVER_INTERNAL_IP}
discover_port: 8092
EOF

  cat > /tmp/docker-compose-client-${i}.yml << EOF
version: '3.8'
services:
  fedn-client-${CLIENT_ID_1}:
    image: ${DOCKER_IMAGE}
    working_dir: /app/client
    command: ["client","start","--combiner","${SERVER_INTERNAL_IP}","--combiner-port","12080","--in","fedn.yaml","--name","client-${CLIENT_ID_1}","--local-package"]
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_1}
    volumes:
      - ./client:/app/client
      - ./logs:/app/logs
  fedn-client-${CLIENT_ID_2}:
    image: ${DOCKER_IMAGE}
    working_dir: /app/client
    command: ["client","start","--combiner","${SERVER_INTERNAL_IP}","--combiner-port","12080","--in","fedn.yaml","--name","client-${CLIENT_ID_2}","--local-package"]
    environment:
      FEDN_CLIENT_ID: ${CLIENT_ID_2}
    volumes:
      - ./client:/app/client
      - ./logs:/app/logs
networks:
  default:
    driver: bridge
EOF

  gcloud compute scp /tmp/fedn-client-${i}.yaml $VM_NAME:/app/client/fedn.yaml --zone="$VM_ZONE" --quiet
  gcloud compute scp /tmp/docker-compose-client-${i}.yml $VM_NAME:/app/docker-compose.yml --zone="$VM_ZONE" --quiet

  # Ensure python3.12 stack and deps (defense-in-depth beyond startup script)
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="
set -e
sudo usermod -aG docker $USER || true
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
