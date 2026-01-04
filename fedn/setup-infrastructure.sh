#!/bin/bash

# Provision GCP network and VMs for FEDn (1 server + 5 client VMs hosting 10 clients)
set -euo pipefail

PROJECT_ID="inf022"
REGION="us-central1"
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-central1-a")
NETWORK_NAME="fedn-network"

SERVER_VM_NAME="flybold-server"
SERVER_MACHINE_TYPE="e2-standard-8"
SERVER_SUBNET="fedn-subnet-server"
SERVER_SUBNET_RANGE="10.0.0.0/28"
SERVER_ZONE="${ZONES[0]}"

CLIENT_PREFIX="flybold-client"
CLIENT_MACHINE_TYPE="e2-standard-16"
CLIENT_COUNT=5

IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="100GB"

info(){ echo -e "\n[INFO] $1\n"; }
success(){ echo -e "\n[SUCCESS] $1\n"; }


# info "Setting project"
# gcloud config set project "$PROJECT_ID" >/dev/null

# info "Creating VPC network: $NETWORK_NAME"
# if gcloud compute networks describe "$NETWORK_NAME" >/dev/null 2>&1; then
#   echo "Network exists, skipping"
# else
#   gcloud compute networks create "$NETWORK_NAME" --subnet-mode=custom
# fi

# info "Creating server subnet"
# if gcloud compute networks subnets describe "$SERVER_SUBNET" --region="$REGION" >/dev/null 2>&1; then
#   echo "Server subnet exists, skipping"
# else
#   gcloud compute networks subnets create "$SERVER_SUBNET" \
#     --network="$NETWORK_NAME" \
#     --region="$REGION" \
#     --range="$SERVER_SUBNET_RANGE"
# fi

# for i in $(seq 1 $CLIENT_COUNT); do
#   SUBNET_NAME="fedn-subnet-client-${i}"
#   SUBNET_RANGE="10.0.${i}.0/28"
#   info "Creating client subnet ${i}"
#   if gcloud compute networks subnets describe "$SUBNET_NAME" --region="$REGION" >/dev/null 2>&1; then
#     echo "Client subnet ${i} exists, skipping"
#   else
#     gcloud compute networks subnets create "$SUBNET_NAME" \
#       --network="$NETWORK_NAME" \
#       --region="$REGION" \
#       --range="$SUBNET_RANGE"
#   fi
# done

# info "Creating firewall rules"
# if ! gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-internal >/dev/null 2>&1; then
#   gcloud compute firewall-rules create ${NETWORK_NAME}-allow-internal \
#     --network="$NETWORK_NAME" \
#     --allow=tcp,udp,icmp \
#     --source-ranges=10.0.0.0/16
# fi
# if ! gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-ssh >/dev/null 2>&1; then
#   gcloud compute firewall-rules create ${NETWORK_NAME}-allow-ssh \
#     --network="$NETWORK_NAME" \
#     --allow=tcp:22 \
#     --source-ranges=0.0.0.0/0
# fi
# # FEDn ports: API 8092, Combiner gRPC 12080, Hooks 12081, Minio 9000/9001 (internal)
# if ! gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-fedn >/dev/null 2>&1; then
#   gcloud compute firewall-rules create ${NETWORK_NAME}-allow-fedn \
#     --network="$NETWORK_NAME" \
#     --allow=tcp:8092,tcp:12080,tcp:12081,tcp:9000,tcp:9001 \
#     --source-ranges=10.0.0.0/16
# fi

# cat > /tmp/vm-startup.sh << 'EOF'
# #!/bin/bash
# set -e
# curl -fsSL https://get.docker.com -o get-docker.sh
# sh get-docker.sh
# # Get the primary non-root user (fallback to root if none found)
# PRIMARY_USER=$(getent passwd 1000 | cut -d: -f1 || echo "root")
# usermod -aG docker "$PRIMARY_USER" 2>/dev/null || true
# systemctl enable --now docker
# # docker compose plugin
# mkdir -p /usr/local/lib/docker/cli-plugins
# curl -SL https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-linux-$(uname -m) -o /usr/local/lib/docker/cli-plugins/docker-compose
# chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# # Python 3.12 + pip + core deps for FEDn clients
# apt-get update
# apt-get install -y software-properties-common curl
# add-apt-repository -y ppa:deadsnakes/ppa
# apt-get update
# apt-get install -y python3.12 python3.12-venv python3.12-distutils libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
# curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
# python3.12 /tmp/get-pip.py
# python3.12 -m pip install --no-cache-dir --upgrade pip
# python3.12 -m pip install --no-cache-dir "numpy<2" opencv-python-headless==4.9.0.80
# python3.12 -m pip install --no-cache-dir fedn yolov5
# rm -f /tmp/get-pip.py
# mkdir -p /app
# chown -R "$PRIMARY_USER":"$PRIMARY_USER" /app 2>/dev/null || true
# EOF

# info "Creating server VM: $SERVER_VM_NAME"
# if gcloud compute instances describe "$SERVER_VM_NAME" --zone="$SERVER_ZONE" >/dev/null 2>&1; then
#   echo "Server VM exists, skipping"
# else
#   gcloud compute instances create "$SERVER_VM_NAME" \
#     --zone="$SERVER_ZONE" \
#     --machine-type="$SERVER_MACHINE_TYPE" \
#     --subnet="$SERVER_SUBNET" \
#     --image-family="$IMAGE_FAMILY" \
#     --image-project="$IMAGE_PROJECT" \
#     --boot-disk-size="$BOOT_DISK_SIZE" \
#     --metadata-from-file=startup-script=/tmp/vm-startup.sh \
#     --scopes=storage-rw,compute-rw \
#     --tags=fedn-server
# fi

# for i in $(seq 1 $CLIENT_COUNT); do
#   VM_NAME="${CLIENT_PREFIX}-${i}"
#   CLIENT_ZONE="${ZONES[$((i-1))]}"
#   SUBNET="fedn-subnet-client-${i}"
#   info "Creating client VM ${i}: $VM_NAME ($CLIENT_ZONE)"
#   if gcloud compute instances describe "$VM_NAME" --zone="$CLIENT_ZONE" >/dev/null 2>&1; then
#     echo "Client VM ${i} exists, skipping"
#   else
#     gcloud compute instances create "$VM_NAME" \
#       --zone="$CLIENT_ZONE" \
#       --machine-type="$CLIENT_MACHINE_TYPE" \
#       --subnet="$SUBNET" \
#       --image-family="$IMAGE_FAMILY" \
#       --image-project="$IMAGE_PROJECT" \
#       --boot-disk-size="$BOOT_DISK_SIZE" \
#       --metadata-from-file=startup-script=/tmp/vm-startup.sh \
#       --scopes=storage-rw,compute-rw \
#       --tags=fedn-client
#   fi
# done

# info "Waiting 90s for VMs to initialize"
# sleep 90

info "Saving VM information to vm-info.txt"
cat > vm-info.txt << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
NETWORK=$NETWORK_NAME

SERVER_VM=$SERVER_VM_NAME
SERVER_ZONE=$SERVER_ZONE
SERVER_INTERNAL_IP=$(gcloud compute instances describe "$SERVER_VM_NAME" --zone="$SERVER_ZONE" --format='get(networkInterfaces[0].networkIP)')
SERVER_EXTERNAL_IP=$(gcloud compute instances describe "$SERVER_VM_NAME" --zone="$SERVER_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
EOF

for i in $(seq 1 $CLIENT_COUNT); do
  VM_NAME="${CLIENT_PREFIX}-${i}"
  CLIENT_ZONE="${ZONES[$((i-1))]}"
  INTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$CLIENT_ZONE" --format='get(networkInterfaces[0].networkIP)')
  EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" --zone="$CLIENT_ZONE" --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
  cat >> vm-info.txt << EOF
CLIENT_${i}_VM=$VM_NAME
CLIENT_${i}_ZONE=$CLIENT_ZONE
CLIENT_${i}_INTERNAL_IP=$INTERNAL_IP
CLIENT_${i}_EXTERNAL_IP=$EXTERNAL_IP

EOF
done

success "Infrastructure ready. VM info saved to vm-info.txt"
echo "Next: ./build-push-image.sh"
