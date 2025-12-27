#!/bin/bash

# Build and Push Docker Image on Temporary GCP VM
set -e

PROJECT_ID="inf022"
REGION="us-central1"
TEMP_VM_NAME="docker-builder-temp"
TEMP_VM_ZONE="${REGION}-a"
IMAGE_NAME="fly-bold-image"
IMAGE_TAG="latest"

echo_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

echo_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
    exit 1
}

echo_info "Building Docker image on temporary GCP VM"

# Set project
gcloud config set project $PROJECT_ID

# Get Docker Hub credentials
if [ -f .docker_username ]; then
    DOCKER_USERNAME=$(cat .docker_username)
    echo_info "Using saved Docker Hub username: $DOCKER_USERNAME"
else
    read -p "Enter your Docker Hub username: " DOCKER_USERNAME
    echo $DOCKER_USERNAME > .docker_username
fi

echo_info "Docker Hub login required"
read -sp "Enter your Docker Hub password: " DOCKER_PASSWORD
echo ""

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Create temporary VM for building
echo_info "Creating temporary build VM: $TEMP_VM_NAME"
if gcloud compute instances describe $TEMP_VM_NAME --zone=$TEMP_VM_ZONE &>/dev/null; then
    echo "Temp VM already exists, using it..."
else
    gcloud compute instances create $TEMP_VM_NAME \
        --zone=$TEMP_VM_ZONE \
        --machine-type=n1-standard-4 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --scopes=storage-rw \
        --metadata=startup-script='#!/bin/bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $(who am i | awk "{print \$1}")
'
    
    echo_info "Waiting 60 seconds for VM to initialize and Docker to install..."
    sleep 60
fi

# Copy project files to VM (skip if already present)
echo_info "Checking and copying project files to build VM..."

# Check which files/directories are missing
MISSING_FILES=$(gcloud compute ssh $TEMP_VM_NAME --zone=$TEMP_VM_ZONE --command="
    missing=''
    [ ! -d /tmp/src ] && missing=\"\$missing src\"
    [ ! -d /tmp/yolov5 ] && missing=\"\$missing yolov5\"
    [ ! -f /tmp/Dockerfile ] && missing=\"\$missing Dockerfile\"
    [ ! -f /tmp/requirements.txt ] && missing=\"\$missing requirements.txt\"
    [ ! -f /tmp/pyproject.toml ] && missing=\"\$missing pyproject.toml\"
    echo \$missing
" 2>/dev/null)

if [ -z "$MISSING_FILES" ]; then
    echo "All files already present on VM, skipping copy"
else
    echo "Missing files: $MISSING_FILES - copying..."
    
    # Copy only missing files
    for file in $MISSING_FILES; do
        if [ "$file" = "src" ] || [ "$file" = "yolov5" ]; then
            echo "Copying directory: $file"
            gcloud compute scp --recurse ./$file $TEMP_VM_NAME:/tmp/ --zone=$TEMP_VM_ZONE --quiet
        else
            echo "Copying file: $file"
            gcloud compute scp ./$file $TEMP_VM_NAME:/tmp/ --zone=$TEMP_VM_ZONE --quiet
        fi
    done
fi

# Build and push image on VM
echo_info "Building Docker image on VM: $FULL_IMAGE_NAME"
gcloud compute ssh $TEMP_VM_NAME --zone=$TEMP_VM_ZONE --command="
set -e

cd /tmp

# Login to Docker Hub
echo '$DOCKER_PASSWORD' | sudo docker login -u '$DOCKER_USERNAME' --password-stdin

# Build image
echo 'Building Docker image...'
sudo docker build -t $FULL_IMAGE_NAME . || exit 1

echo 'Docker image built successfully'

# Push image
echo 'Pushing image to Docker Hub: $FULL_IMAGE_NAME'
sudo docker push $FULL_IMAGE_NAME || exit 1

echo 'Docker image pushed successfully'

# Create image info file
cat > /tmp/docker-image-info.txt << EOF
DOCKER_IMAGE=$FULL_IMAGE_NAME
BUILD_DATE=\$(date)
EOF

echo 'Build complete on VM'
"

if [ $? -ne 0 ]; then
    echo_error "Build or push failed on VM"
fi

# Download docker-image-info.txt back to local machine
echo_info "Downloading docker-image-info.txt from VM..."
gcloud compute scp $TEMP_VM_NAME:/tmp/docker-image-info.txt ./docker-image-info.txt --zone=$TEMP_VM_ZONE --quiet

# Delete temporary VM
echo_info "Deleting temporary build VM..."
gcloud compute instances delete $TEMP_VM_NAME --zone=$TEMP_VM_ZONE --quiet

echo_success "Build complete! Image info saved to docker-image-info.txt"
echo "Image: $FULL_IMAGE_NAME"
echo ""
echo "Next: ./04-deploy-application.sh"