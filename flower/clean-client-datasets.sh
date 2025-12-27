#!/bin/bash
set -e

VM_INFO_FILE="vm-info.txt"

if [ ! -f "$VM_INFO_FILE" ]; then
    echo "[ERROR] $VM_INFO_FILE not found! Make sure it exists."
    exit 1
fi

# Load VM info
source "$VM_INFO_FILE"

# Find all CLIENT VM variables dynamically
CLIENT_VMS=($(compgen -v | grep '^CLIENT_[0-9]_VM$'))

echo "Removing /app/datasets from all client VMs..."

for CLIENT_VM_VAR in "${CLIENT_VMS[@]}"; do
    # Extract client number from variable name
    NUM=$(echo "$CLIENT_VM_VAR" | sed 's/CLIENT_\([0-9]\)_VM/\1/')
    
    # Get VM name and zone
    CLIENT_VM=${!CLIENT_VM_VAR}
    CLIENT_ZONE_VAR="CLIENT_${NUM}_ZONE"
    CLIENT_ZONE=${!CLIENT_ZONE_VAR}
    
    echo "[INFO] Cleaning datasets on $CLIENT_VM ($CLIENT_ZONE)..."
    gcloud compute ssh $CLIENT_VM --zone=$CLIENT_ZONE --command="sudo rm -rf /app/datasets/* /app/datasets/.* 2>/dev/null || true"
    echo "[SUCCESS] Datasets removed on $CLIENT_VM"
done

echo "[DONE] All client datasets cleaned."
