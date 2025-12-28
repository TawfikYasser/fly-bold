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

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
}

echo_info "Cleaning local __pycache__ directories"
find ./src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./yolov5 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find ./src -type f -name "*.pyc" -delete 2>/dev/null || true
find ./yolov5 -type f -name "*.pyc" -delete 2>/dev/null || true

echo_info "Updating code on all VMs"

# Update server
echo_info "Updating server: $SERVER_VM"

# Clean remote __pycache__ FIRST, then copy
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    # Clean __pycache__ directories and fix ownership
    sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
    # Fix ownership so we can write
    sudo chown -R \$USER:\$USER /app/src /app/yolov5 2>/dev/null || true
    echo 'Cleaned and ready for copy'
"

# Now copy files
gcloud compute scp --recurse ./src $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
gcloud compute scp --recurse ./yolov5 $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet
gcloud compute scp pyproject.toml $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

# Reinstall and restart server
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    # Reinstall the package to pick up changes
    sudo docker compose exec -T fl-server pip install -e . --no-deps 2>/dev/null || true
    # Clear Python cache inside container
    sudo docker compose exec -T fl-server find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    sudo docker compose exec -T fl-server find /app -type f -name '*.pyc' -delete 2>/dev/null || true
    # Restart container
    sudo docker compose restart fl-server
    echo 'Server updated and restarted'
"

echo_success "Server updated"

# Update clients
for i in $(seq 1 5); do
    VM_VAR="CLIENT_${i}_VM"
    ZONE_VAR="CLIENT_${i}_ZONE"
    VM_NAME=${!VM_VAR}
    VM_ZONE=${!ZONE_VAR}
    
    # Calculate client IDs for this VM
    CLIENT_ID_1=$(( (i-1)*2 ))
    CLIENT_ID_2=$(( (i-1)*2 + 1 ))
    
    echo_info "Updating $VM_NAME (clients $CLIENT_ID_1, $CLIENT_ID_2)"
    
    # Clean remote __pycache__ FIRST, then copy
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        # Clean __pycache__ directories and fix ownership
        sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
        sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
        # Fix ownership
        sudo chown -R \$USER:\$USER /app/src /app/yolov5 2>/dev/null || true
        echo 'Cleaned and ready for copy'
    "
    
    # Now copy files
    gcloud compute scp --recurse ./src $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    gcloud compute scp --recurse ./yolov5 $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    gcloud compute scp pyproject.toml $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    
    # Reinstall and restart clients
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        # Update both client containers
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            echo \"Updating client \$CLIENT_ID...\"
            # Reinstall package
            sudo docker compose exec -T fl-client-\$CLIENT_ID pip install -e . --no-deps 2>/dev/null || true
            # Clear Python cache
            sudo docker compose exec -T fl-client-\$CLIENT_ID find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            sudo docker compose exec -T fl-client-\$CLIENT_ID find /app -type f -name '*.pyc' -delete 2>/dev/null || true
        done
        # Restart all clients
        sudo docker compose restart
        echo 'Clients updated and restarted'
    " &  # Run in parallel
done

# Wait for all client updates to complete
wait

echo_success "All VMs updated and restarted"
echo ""
echo "Verify changes with:"
echo "  gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='cd /app && sudo docker compose logs --tail=50 fl-server'"
echo ""
echo "Check client evaluation:"
echo "  gcloud compute ssh flybold-client-1 --zone=us-central1-a --command='sudo docker compose logs --tail=100 fl-client-0 | grep -E \"(yolo_eval|mAP|Parsed metrics)\"'"