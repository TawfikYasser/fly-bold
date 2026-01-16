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

echo_warning() {
    echo -e "\n\033[1;33m[WARNING]\033[0m $1\n"
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
    echo '=== Cleaning cache and fixing permissions ==='
    sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
    sudo chown -R \$USER:\$USER /app/src /app/yolov5 /app/pyproject.toml 2>/dev/null || true
    echo '✓ Cleaned and ready for copy'
"

# Copy files
echo "  → Copying src directory..."
gcloud compute scp --recurse ./src $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

echo "  → Copying yolov5 directory..."
gcloud compute scp --recurse ./yolov5 $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

echo "  → Copying pyproject.toml..."
gcloud compute scp pyproject.toml $SERVER_VM:/app/ --zone=$SERVER_ZONE --quiet

# Reinstall and restart server with verification
echo "  → Reinstalling package and restarting..."
gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    cd /app
    echo '=== Reinstalling package inside container ==='
    if sudo docker compose exec -T fl-server pip install -e . --no-deps; then
        echo '✓ Package reinstalled successfully'
    else
        echo '⚠ Package reinstall failed, but continuing...'
    fi
    
    echo '=== Clearing Python cache inside container ==='
    sudo docker compose exec -T fl-server find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    sudo docker compose exec -T fl-server find /app -type f -name '*.pyc' -delete 2>/dev/null || true
    
    echo '=== Restarting server container ==='
    sudo docker compose restart fl-server
    sleep 3
    
    echo '=== Verifying server is running ==='
    if sudo docker compose ps | grep -q 'fl-server.*Up'; then
        echo '✓ Server is running'
    else
        echo '✗ Server is NOT running!'
        exit 1
    fi
"

if [ $? -eq 0 ]; then
    echo_success "Server updated successfully"
else
    echo_error "Server update failed!"
    exit 1
fi

# Update clients sequentially (NOT in parallel) to ensure reliability
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
        echo '=== Cleaning cache and fixing permissions ==='
        sudo find /app/src -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/yolov5 -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        sudo find /app/src -type f -name '*.pyc' -delete 2>/dev/null || true
        sudo find /app/yolov5 -type f -name '*.pyc' -delete 2>/dev/null || true
        sudo chown -R \$USER:\$USER /app/src /app/yolov5 /app/pyproject.toml 2>/dev/null || true
        echo '✓ Cleaned and ready for copy'
    "
    
    # Copy files
    echo "  → Copying src directory..."
    gcloud compute scp --recurse ./src $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    
    echo "  → Copying yolov5 directory..."
    gcloud compute scp --recurse ./yolov5 $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    
    echo "  → Copying pyproject.toml..."
    gcloud compute scp pyproject.toml $VM_NAME:/app/ --zone=$VM_ZONE --quiet
    
    # Reinstall and restart clients with verification
    echo "  → Reinstalling package and restarting clients..."
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        SUCCESS=true
        
        # Update both client containers
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            echo \"=== Processing client \$CLIENT_ID ===\"
            
            # Reinstall package
            echo \"→ Reinstalling package in fl-client-\$CLIENT_ID...\"
            if sudo docker compose exec -T fl-client-\$CLIENT_ID pip install -e . --no-deps; then
                echo \"✓ Package reinstalled for client \$CLIENT_ID\"
            else
                echo \"⚠ Package reinstall failed for client \$CLIENT_ID, but continuing...\"
            fi
            
            # Clear Python cache
            echo \"→ Clearing cache in fl-client-\$CLIENT_ID...\"
            sudo docker compose exec -T fl-client-\$CLIENT_ID find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            sudo docker compose exec -T fl-client-\$CLIENT_ID find /app -type f -name '*.pyc' -delete 2>/dev/null || true
            echo \"✓ Cache cleared for client \$CLIENT_ID\"
        done
        
        # Restart all clients on this VM
        echo \"=== Restarting all client containers ===\"
        sudo docker compose restart
        sleep 3
        
        # Verify both clients are running
        echo \"=== Verifying clients are running ===\"
        for CLIENT_ID in $CLIENT_ID_1 $CLIENT_ID_2; do
            if sudo docker compose ps | grep -q \"fl-client-\$CLIENT_ID.*Up\"; then
                echo \"✓ Client \$CLIENT_ID is running\"
            else
                echo \"✗ Client \$CLIENT_ID is NOT running!\"
                SUCCESS=false
            fi
        done
        
        if [ \"\$SUCCESS\" = \"true\" ]; then
            echo \"✓ All clients on $VM_NAME updated successfully\"
            exit 0
        else
            echo \"✗ Some clients on $VM_NAME failed to start\"
            exit 1
        fi
    "
    
    if [ $? -eq 0 ]; then
        echo_success "$VM_NAME updated successfully"
    else
        echo_error "$VM_NAME update failed!"
        echo_warning "Continuing with remaining VMs, but please check $VM_NAME manually"
    fi
    
    echo ""
done

echo_success "All VMs processed"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Update complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Verify deployment status:"
echo "  ./05-manage-clients.sh status"
echo ""
echo "Check server logs:"
echo "  gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command='sudo docker compose logs --tail=50 fl-server'"
echo ""
echo "Check client logs (example):"
echo "  gcloud compute ssh flybold-client-1 --zone=us-central1-a --command='sudo docker compose logs --tail=50 fl-client-0'"
echo ""
echo "Verify code version on a client:"
echo "  gcloud compute ssh flybold-client-1 --zone=us-central1-a --command='sudo docker compose exec fl-client-0 cat /app/pyproject.toml | grep version'"