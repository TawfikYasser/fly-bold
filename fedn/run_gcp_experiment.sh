#!/bin/bash

# Master script to run automated FEDn experiments on GCP
set -euo pipefail

info(){ echo -e "\n[INFO] $1\n"; }
success(){ echo -e "\n[SUCCESS] $1\n"; }
fail(){ echo -e "\n[ERROR] $1\n"; exit 1; }

# Pre-checks
# Pre-checks

# Try to load password from file
if [ -f .docker_password ]; then
  info "Using Docker Hub password from .docker_password"
  DOCKER_PASSWORD=$(cat .docker_password)
  export DOCKER_PASSWORD
fi

if [ ! -f .docker_username ]; then
  info "Docker credentials not found locally."
  read -p "Enter your Docker Hub username: " DOCKER_USERNAME
  echo "$DOCKER_USERNAME" > .docker_username
  
  info "Please log in to Docker Hub (for local access if needed later)"
  # We might not strictly need local login if the builder VM handles it, 
  # but we need the password for the builder VM.
  if [ -z "${DOCKER_PASSWORD:-}" ]; then
    read -sp "Enter your Docker Hub password: " DOCKER_PASSWORD
    echo ""
    export DOCKER_PASSWORD
  else
    info "Using existing Docker Hub password."
  fi
else
  info "Using saved Docker Hub username from .docker_username"
  DOCKER_USERNAME=$(cat .docker_username)
  # We still need the password for the builder script if it's not cached somehow (it's not).
  # The builder script asks for it. We can't export it easily to the inner script's read.
  # Strategy: We will modify the flow to pass it or just ask for it here and feed it in.
  if [ -z "${DOCKER_PASSWORD:-}" ]; then
    read -sp "Enter your Docker Hub password (required for builder VM): " DOCKER_PASSWORD
    echo ""
    export DOCKER_PASSWORD
  fi
fi

# 1. Setup Infrastructure
info "STEP 1: Setting up Infrastructure"
./setup-infrastructure.sh

# 2. Build & Push Image
info "STEP 2: Building and Pushing Docker Image"

SKIP_BUILD=false
if [ -f docker-image-info.txt ]; then
  read -p "Found existing docker-image-info.txt. Reuse existing image? (y/n) [y]: " REUSE_IMAGE
  REUSE_IMAGE=${REUSE_IMAGE:-y}
  if [[ "$REUSE_IMAGE" == "y" ]]; then
    info "Skipping build, reusing existing image."
    SKIP_BUILD=true
  fi
fi

if [ "$SKIP_BUILD" = false ]; then
  # build-push-image.sh asks for password via read -sp. 
  # We feed it via input redirection.
  # It reads username from .docker_username if present.
  # It reads password.
  if [ -f .docker_username ]; then
    # It will skip username prompt, only ask password
    echo "$DOCKER_PASSWORD" | ./build-push-image.sh
  else
    # Should not happen given pre-check, but just in case
    { echo "$DOCKER_USERNAME"; echo "$DOCKER_PASSWORD"; } | ./build-push-image.sh
  fi
fi

# 3. Deploy Application
info "STEP 3: Deploying Application"
# deploy-application.sh asks "Enable TLS (self-signed)? (y/n) [n]:"
echo "n" | ./deploy-application.sh

# 4. Run Training Session
info "STEP 4: Running Training Session"
if [ ! -f vm-info.txt ]; then
  fail "vm-info.txt missing (setup failed?)"
fi
source vm-info.txt

# We need to run run_session.py on the server.
# First, ensure it's there (deploy script copies fedn folder to /app/fly-bold-fedn)
# The run_session.py is in /app/fly-bold-fedn/fedn/run_session.py
# But deploy script copies `.` to `/app/fly-bold-fedn`. 
# So if we are in `fedn` locally, the remote path depends on where run_session.py is relative to `.`
# The deploy script does: `gcloud compute scp --recurse . "$SERVER_VM":/app/fly-bold-fedn`
# If we run this from `fedn` folder, then `/app/fly-bold-fedn` contains the contents of `fedn` folder.
# So `run_session.py` is at `/app/fly-bold-fedn/run_session.py`.

info "Connecting to Server VM ($SERVER_VM) to start training..."
gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
set -e
cd /app/fly-bold-fedn

# Install pymongo if not already (it might be needed for the verification script)
# The startup script installed 'fedn' and 'yolov5', but run_session.py imports 'pymongo'.
# standard fedn image/venv might have it, but let's ensure.
sudo /usr/bin/python3.12 -m pip install --no-cache-dir pymongo

echo 'Starting run_session.py remote execution...'
sudo /usr/bin/python3.12 run_session.py

# Install matplotlib for analyzer (pymongo installed above)
sudo /usr/bin/python3.12 -m pip install --no-cache-dir matplotlib pandas

echo 'Running Experiment Analyzer...'
sudo /usr/bin/python3.12 experiment_analyzer.py
"

# 5. Download Results
info "STEP 5: Downloading Results"
# download-files.sh asks: "Enter file numbers (comma) or 'all': "
echo "all" | ./download-files.sh

# 6. Cleanup
info "STEP 6: Cleanup"
# cleanup.sh asks: "Type DELETE to continue: "
echo "DELETE" | ./cleanup.sh

success "Experiment Finished Successfully!"
