#!/bin/bash

# Cleanup All Flybold Resources
set -e

PROJECT_ID="inf022"
BUCKET_NAME="flybold-coco-${PROJECT_ID}"

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
    exit 1
}

echo "=== Flybold Cleanup Script ==="
echo ""
echo "This will delete:"
echo "  - All VMs (server + 5 clients)"
echo "  - Network and subnets"
echo "  - Firewall rules"
echo "  - GCS bucket: gs://${BUCKET_NAME}"
echo ""
read -p "Type 'DELETE' to confirm: " confirm

if [ "$confirm" != "DELETE" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "Starting cleanup..."

# Load VM info if exists
if [ -f vm-info.txt ]; then
    source vm-info.txt
    
    # Delete VMs
    echo "Deleting server VM: $SERVER_VM"
    gcloud compute instances delete $SERVER_VM --zone=$SERVER_ZONE --quiet || true
    
    for i in $(seq 1 5); do
        VM_VAR="CLIENT_${i}_VM"
        ZONE_VAR="CLIENT_${i}_ZONE"
        VM_NAME=${!VM_VAR}
        VM_ZONE=${!ZONE_VAR}
        
        echo "Deleting client VM: $VM_NAME"
        gcloud compute instances delete $VM_NAME --zone=$VM_ZONE --quiet || true
    done
fi

# Delete firewall rules
echo "Deleting firewall rules..."
gcloud compute firewall-rules delete flybold-network-allow-internal --quiet || true
gcloud compute firewall-rules delete flybold-network-allow-ssh --quiet || true
gcloud compute firewall-rules delete flybold-network-allow-flower --quiet || true

# Delete subnets
echo "Deleting subnets..."
gcloud compute networks subnets delete flybold-subnet-server --region=us-central1 --quiet || true
for i in $(seq 1 5); do
    gcloud compute networks subnets delete flybold-subnet-client-${i} --region=us-central1 --quiet || true
done

# Delete network
echo "Deleting network..."
gcloud compute networks delete flybold-network --quiet || true

# Delete GCS bucket
echo "Deleting GCS bucket..."
gsutil -m rm -r gs://${BUCKET_NAME} || true

# Clean local files
echo "Cleaning local files..."
rm -f vm-info.txt docker-image-info.txt .docker_username gcs-key.json

echo ""
echo "=== Cleanup Complete ==="