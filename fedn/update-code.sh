#!/bin/bash

# Sync client code to all VMs without rebuilding images
set -euo pipefail

if [ ! -f vm-info.txt ]; then
  echo "vm-info.txt missing"; exit 1
fi
source vm-info.txt

info(){ echo -e "\n[INFO] $1\n"; }
success(){ echo -e "\n[SUCCESS] $1\n"; }

info "Cleaning local __pycache__"
find ./client -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./client -type f -name "*.pyc" -delete 2>/dev/null || true

sync_target(){
  local target_vm=$1
  local target_zone=$2
  info "Syncing client code to $target_vm"
  gcloud compute ssh "$target_vm" --zone="$target_zone" --command="
    sudo find /app/client -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/client -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo chown -R \$USER:\$USER /app/client 2>/dev/null || true
  "
  gcloud compute scp --recurse ./client "$target_vm":/app/ --zone="$target_zone" --quiet
}

# Sync to server (for potential admin/cli usage)
sync_target "$SERVER_VM" "$SERVER_ZONE"

# Sync to clients
for i in $(seq 1 5); do
  VM_VAR="CLIENT_${i}_VM"; ZONE_VAR="CLIENT_${i}_ZONE"
  sync_target "${!VM_VAR}" "${!ZONE_VAR}" &
done
wait

# Restart client containers to pick changes
for i in $(seq 1 5); do
  VM_VAR="CLIENT_${i}_VM"; ZONE_VAR="CLIENT_${i}_ZONE"
  VM_NAME=${!VM_VAR}; VM_ZONE=${!ZONE_VAR}
  info "Restarting clients on $VM_NAME"
  gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="cd /app && sudo docker compose restart" &
done
wait

success "Code synced and clients restarted"
