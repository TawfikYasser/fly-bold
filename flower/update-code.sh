#!/bin/bash

# Update Code Without Rebuild
set -e

source vm-info.txt

echo_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

echo_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

echo_info "Updating code on all VMs"

# Update server
echo_info "Updating server: $SERVER_VM"
gcloud compute scp --recurse ./src $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
gcloud compute scp pyproject.toml $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    sudo docker compose restart fl-server
"

echo_success "Server updated"

# Update clients
for i in $(seq 1 5); do
    VM_VAR="CLIENT_${i}_VM"
    ZONE_VAR="CLIENT_${i}_ZONE"
    VM_NAME=${!VM_VAR}
    VM_ZONE=${!ZONE_VAR}
    
    echo_info "Updating $VM_NAME"
    gcloud compute scp --recurse ./src $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    gcloud compute scp pyproject.toml $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        sudo docker compose restart
    "
    
    echo_success "$VM_NAME updated"
done

echo_success "All VMs updated and restarted"