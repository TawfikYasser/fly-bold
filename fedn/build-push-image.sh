#!/bin/bash

# Build and push FEDn Docker image on a temporary GCP VM
set -euo pipefail

PROJECT_ID="inf022"
REGION="us-central1"
TEMP_VM_NAME="fedn-builder-temp"
TEMP_VM_ZONE="${REGION}-a"
IMAGE_NAME="fedn-runtime"
IMAGE_TAG="latest"

info() { echo -e "\n[INFO] $1\n"; }
success() { echo -e "\n[SUCCESS] $1\n"; }
fail() { echo -e "\n[ERROR] $1\n"; exit 1; }

info "Building FEDn image on temporary GCP VM"

gcloud config set project "$PROJECT_ID" >/dev/null

if [ -f .docker_username ]; then
  DOCKER_USERNAME=$(cat .docker_username)
  info "Using saved Docker Hub username: $DOCKER_USERNAME"
else
  read -p "Enter your Docker Hub username: " DOCKER_USERNAME
  echo "$DOCKER_USERNAME" > .docker_username
fi

info "Docker Hub login required"
read -sp "Enter your Docker Hub password: " DOCKER_PASSWORD
echo ""

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

info "Ensuring temporary builder VM exists"
if gcloud compute instances describe "$TEMP_VM_NAME" --zone="$TEMP_VM_ZONE" >/dev/null 2>&1; then
  echo "Temp VM already exists, using it..."
else
  gcloud compute instances create "$TEMP_VM_NAME" \
    --zone="$TEMP_VM_ZONE" \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=60GB \
    --scopes=storage-rw \
    --metadata=startup-script='#!/bin/bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $(who am i | awk "{print $1}")
'
  info "Waiting 60s for VM to initialize and Docker to install..."
  sleep 60
fi

info "Checking which build context files are missing on the VM"
MISSING=$(gcloud compute ssh "$TEMP_VM_NAME" --zone="$TEMP_VM_ZONE" --command="
  missing=''
  [ ! -d /tmp/fedn ] && missing=\"$missing fedn\"
  echo \$missing
" 2>/dev/null)

if [ -z "$MISSING" ]; then
  echo "Build context already present on VM, skipping copy"
else
  echo "Missing:$MISSING - copying..."
  gcloud compute scp --recurse ./fedn "$TEMP_VM_NAME":/tmp/ --zone="$TEMP_VM_ZONE" --quiet
fi

info "Building and pushing image on VM: $FULL_IMAGE_NAME"
gcloud compute ssh "$TEMP_VM_NAME" --zone="$TEMP_VM_ZONE" --command="
set -e
cd /tmp/fedn

# Login to Docker Hub
echo \"${DOCKER_PASSWORD}\" | sudo docker login -u \"${DOCKER_USERNAME}\" --password-stdin

# Build using repo Dockerfile
sudo docker build -t ${FULL_IMAGE_NAME} -f Dockerfile .

# Push
sudo docker push ${FULL_IMAGE_NAME}

cat > /tmp/docker-image-info.txt << EOF
DOCKER_IMAGE=${FULL_IMAGE_NAME}
BUILD_DATE=\$(date)
EOF
" || fail "Build or push failed on VM"

info "Downloading docker-image-info.txt"
gcloud compute scp "$TEMP_VM_NAME":/tmp/docker-image-info.txt ./docker-image-info.txt --zone="$TEMP_VM_ZONE" --quiet

info "Deleting temporary builder VM"
gcloud compute instances delete "$TEMP_VM_NAME" --zone="$TEMP_VM_ZONE" --quiet

success "Build complete. Image: $FULL_IMAGE_NAME"
echo "Next: ./deploy-application.sh"
