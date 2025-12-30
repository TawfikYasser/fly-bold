#!/bin/bash

# Tear down FEDn resources
set -euo pipefail

PROJECT_ID="inf022"
BUCKET_NAME="" # data not managed here

echo "=== FEDn Cleanup ==="
echo "This will delete server and client VMs, network, and local metadata."
read -p "Type DELETE to continue: " confirm
[ "$confirm" = "DELETE" ] || { echo "Cancelled"; exit 0; }

source vm-info.txt || true

if [ -n "${SERVER_VM:-}" ]; then
  echo "Deleting server VM $SERVER_VM"
  gcloud compute instances delete "$SERVER_VM" --zone="$SERVER_ZONE" --quiet || true
fi
for i in $(seq 1 5); do
  VM_VAR="CLIENT_${i}_VM"; ZONE_VAR="CLIENT_${i}_ZONE"
  VM_NAME=${!VM_VAR:-}
  VM_ZONE=${!ZONE_VAR:-}
  [ -n "$VM_NAME" ] || continue
  echo "Deleting client VM $VM_NAME"
  gcloud compute instances delete "$VM_NAME" --zone="$VM_ZONE" --quiet || true
done

echo "Deleting firewall rules"
gcloud compute firewall-rules delete fedn-network-allow-internal --quiet || true
gcloud compute firewall-rules delete fedn-network-allow-ssh --quiet || true
gcloud compute firewall-rules delete fedn-network-allow-fedn --quiet || true

echo "Deleting subnets"
for i in $(seq 1 5); do
  gcloud compute networks subnets delete fedn-subnet-client-${i} --region=us-central1 --quiet || true
done
gcloud compute networks subnets delete fedn-subnet-server --region=us-central1 --quiet || true

echo "Deleting network"
gcloud compute networks delete fedn-network --quiet || true

echo "Cleaning local files"
rm -f vm-info.txt docker-image-info.txt .docker_username

if [ -n "$BUCKET_NAME" ]; then
  echo "Skipping bucket deletion (data disabled)"
fi

echo "Cleanup complete"
