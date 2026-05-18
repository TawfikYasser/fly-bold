#!/bin/bash

# Setup GCP Infrastructure for Flybold
set -e

PROJECT_ID="inf022"
REGION="us-central1"
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-central1-a")
NETWORK_NAME="flybold-network"

SERVER_VM_NAME="flybold-server"
SERVER_MACHINE_TYPE="e2-standard-8"
SERVER_SUBNET="flybold-subnet-server"
SERVER_SUBNET_RANGE="10.0.0.0/28"
SERVER_ZONE="${ZONES[0]}"

CLIENT_PREFIX="flybold-client"
CLIENT_MACHINE_TYPE="e2-standard-16"
CLIENT_COUNT=5

IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="100GB"

echo_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

echo_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

echo_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
}

create_additional_client_subnets() {
    echo_info "Creating additional client subnets (6-10)"
    for i in $(seq 6 10); do
        SUBNET_NAME="flybold-subnet-client-${i}"
        SUBNET_RANGE="10.0.${i}.0/28"
        
        echo_info "Creating client subnet ${i}"
        if gcloud compute networks subnets describe $SUBNET_NAME --region=$REGION &>/dev/null; then
            echo "Client subnet ${i} exists, skipping..."
        else
            gcloud compute networks subnets create $SUBNET_NAME \
                --network=$NETWORK_NAME \
                --region=$REGION \
                --range=$SUBNET_RANGE
        fi
    done
}

create_additional_clients() {
    echo_info "Creating additional client VMs (6-10)"
    for i in $(seq 6 10); do
        CLIENT_VM_NAME="${CLIENT_PREFIX}-${i}"
        ZONE_INDEX=$(( (i - 1) % ${#ZONES[@]} ))
        CLIENT_ZONE="${ZONES[$ZONE_INDEX]}"
        CLIENT_SUBNET="flybold-subnet-client-${i}"

        echo_info "Creating client VM ${i}: $CLIENT_VM_NAME in zone $CLIENT_ZONE"

        if gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE &>/dev/null; then
            echo "Client VM ${i} exists, skipping..."
        else
            gcloud compute instances create $CLIENT_VM_NAME \
                --zone=$CLIENT_ZONE \
                --machine-type=$CLIENT_MACHINE_TYPE \
                --subnet=$CLIENT_SUBNET \
                --image-family=$IMAGE_FAMILY \
                --image-project=$IMAGE_PROJECT \
                --boot-disk-size=$BOOT_DISK_SIZE \
                --metadata-from-file=startup-script=/tmp/vm-startup.sh \
                --scopes=storage-rw,compute-rw \
                --tags=flybold-client
        fi
    done
}

echo_info "Setting up infrastructure for Flybold"
gcloud config set project $PROJECT_ID

# Create VPC network
echo_info "Creating VPC network: $NETWORK_NAME"
if gcloud compute networks describe $NETWORK_NAME &>/dev/null; then
    echo "Network exists, skipping..."
else
    gcloud compute networks create $NETWORK_NAME --subnet-mode=custom
fi

# Create server subnet
echo_info "Creating server subnet"
if gcloud compute networks subnets describe $SERVER_SUBNET --region=$REGION &>/dev/null; then
    echo "Server subnet exists, skipping..."
else
    gcloud compute networks subnets create $SERVER_SUBNET \
        --network=$NETWORK_NAME \
        --region=$REGION \
        --range=$SERVER_SUBNET_RANGE
fi

# Create client subnets
for i in $(seq 1 $CLIENT_COUNT); do
    SUBNET_NAME="flybold-subnet-client-${i}"
    SUBNET_RANGE="10.0.${i}.0/28"
    
    echo_info "Creating client subnet ${i}"
    if gcloud compute networks subnets describe $SUBNET_NAME --region=$REGION &>/dev/null; then
        echo "Client subnet ${i} exists, skipping..."
    else
        gcloud compute networks subnets create $SUBNET_NAME \
            --network=$NETWORK_NAME \
            --region=$REGION \
            --range=$SUBNET_RANGE
    fi
done

# Firewall rules
echo_info "Creating firewall rules"

# Internal communication
if gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-internal &>/dev/null; then
    echo "Internal firewall exists, skipping..."
else
    gcloud compute firewall-rules create ${NETWORK_NAME}-allow-internal \
        --network=$NETWORK_NAME \
        --allow=tcp,udp,icmp \
        --source-ranges=10.0.0.0/16
fi

# SSH access
if gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-ssh &>/dev/null; then
    echo "SSH firewall exists, skipping..."
else
    gcloud compute firewall-rules create ${NETWORK_NAME}-allow-ssh \
        --network=$NETWORK_NAME \
        --allow=tcp:22 \
        --source-ranges=0.0.0.0/0
fi

# Flower server port
if gcloud compute firewall-rules describe ${NETWORK_NAME}-allow-flower &>/dev/null; then
    echo "Flower firewall exists, skipping..."
else
    gcloud compute firewall-rules create ${NETWORK_NAME}-allow-flower \
        --network=$NETWORK_NAME \
        --allow=tcp:9092,tcp:9093 \
        --source-ranges=10.0.0.0/16
fi

# Create startup script
cat > /tmp/vm-startup.sh << 'EOF'
#!/bin/bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $(who am i | awk '{print $1}')
curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install gcloud SDK
curl https://sdk.cloud.google.com | bash
echo "source /root/google-cloud-sdk/path.bash.inc" >> /root/.bashrc

mkdir -p /app
chown -R $(who am i | awk '{print $1}'):$(who am i | awk '{print $1}') /app
EOF

# Ask user if they want to create initial clients (1-5)
read -p "Do you want to create initial client VMs (1-5)? (yes/no): " CREATE_INITIAL_CLIENTS

if [[ "$CREATE_INITIAL_CLIENTS" == "yes" || "$CREATE_INITIAL_CLIENTS" == "y" ]]; then
    # Create server VM
    echo_info "Creating server VM: $SERVER_VM_NAME"
    if gcloud compute instances describe $SERVER_VM_NAME --zone=$SERVER_ZONE &>/dev/null; then
        echo "Server VM exists, skipping..."
    else
        gcloud compute instances create $SERVER_VM_NAME \
            --zone=$SERVER_ZONE \
            --machine-type=$SERVER_MACHINE_TYPE \
            --subnet=$SERVER_SUBNET \
            --image-family=$IMAGE_FAMILY \
            --image-project=$IMAGE_PROJECT \
            --boot-disk-size=$BOOT_DISK_SIZE \
            --metadata-from-file=startup-script=/tmp/vm-startup.sh \
            --scopes=storage-rw,compute-rw \
            --tags=flybold-server
    fi

    # Create client VMs
    for i in $(seq 1 $CLIENT_COUNT); do
    CLIENT_VM_NAME="${CLIENT_PREFIX}-${i}"
    ZONE_INDEX=$((i - 1))
    CLIENT_ZONE="${ZONES[$ZONE_INDEX]}"
    CLIENT_SUBNET="flybold-subnet-client-${i}"

    echo_info "Creating client VM ${i}: $CLIENT_VM_NAME in zone $CLIENT_ZONE"

    if gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE &>/dev/null; then
        echo "Client VM ${i} exists, skipping..."
    else
        gcloud compute instances create $CLIENT_VM_NAME \
            --zone=$CLIENT_ZONE \
            --machine-type=$CLIENT_MACHINE_TYPE \
            --subnet=$CLIENT_SUBNET \
            --image-family=$IMAGE_FAMILY \
            --image-project=$IMAGE_PROJECT \
            --boot-disk-size=$BOOT_DISK_SIZE \
            --metadata-from-file=startup-script=/tmp/vm-startup.sh \
            --scopes=storage-rw,compute-rw \
            --tags=flybold-client
    fi
done
else
    echo_info "Skipping initial client VMs (1-5) creation"
fi

# Ask user if they want to create additional clients
echo_info "Infrastructure setup options complete!"
read -p "Do you want to create additional client VMs (6-10)? (yes/no): " CREATE_ADDITIONAL

if [[ "$CREATE_ADDITIONAL" == "yes" || "$CREATE_ADDITIONAL" == "y" ]]; then
    create_additional_client_subnets
    create_additional_clients
    echo_info "Additional client VMs created successfully!"
fi

# Wait for VMs only if we created something
if [[ ("$CREATE_INITIAL_CLIENTS" == "yes" || "$CREATE_INITIAL_CLIENTS" == "y") || ("$CREATE_ADDITIONAL" == "yes" || "$CREATE_ADDITIONAL" == "y") ]]; then
    echo_info "Waiting 90 seconds for VMs to initialize..."
    sleep 90
fi

# Save VM info
echo_info "Saving VM information"
cat > vm-info.txt << EOF
PROJECT_ID=$PROJECT_ID
REGION=$REGION
NETWORK=$NETWORK_NAME

EOF

if [[ "$CREATE_INITIAL_CLIENTS" == "yes" || "$CREATE_INITIAL_CLIENTS" == "y" ]]; then
    cat >> vm-info.txt << EOF
SERVER_VM=$SERVER_VM_NAME
SERVER_ZONE=$SERVER_ZONE
SERVER_INTERNAL_IP=$(gcloud compute instances describe $SERVER_VM_NAME --zone=$SERVER_ZONE --format='get(networkInterfaces[0].networkIP)')
SERVER_EXTERNAL_IP=$(gcloud compute instances describe $SERVER_VM_NAME --zone=$SERVER_ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

EOF

    for i in $(seq 1 $CLIENT_COUNT); do
        CLIENT_VM_NAME="${CLIENT_PREFIX}-${i}"
        ZONE_INDEX=$((i - 1))
        CLIENT_ZONE="${ZONES[$ZONE_INDEX]}"
        INTERNAL_IP=$(gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE --format='get(networkInterfaces[0].networkIP)')
        EXTERNAL_IP=$(gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
        
        cat >> vm-info.txt << EOF
CLIENT_${i}_VM=$CLIENT_VM_NAME
CLIENT_${i}_ZONE=$CLIENT_ZONE
CLIENT_${i}_INTERNAL_IP=$INTERNAL_IP
CLIENT_${i}_EXTERNAL_IP=$EXTERNAL_IP

EOF
    done
fi

# Save info for additional clients if they were created
if [[ "$CREATE_ADDITIONAL" == "yes" || "$CREATE_ADDITIONAL" == "y" ]]; then
    for i in $(seq 6 10); do
        CLIENT_VM_NAME="${CLIENT_PREFIX}-${i}"
        ZONE_INDEX=$(( (i - 1) % ${#ZONES[@]} ))
        CLIENT_ZONE="${ZONES[$ZONE_INDEX]}"
        INTERNAL_IP=$(gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE --format='get(networkInterfaces[0].networkIP)')
        EXTERNAL_IP=$(gcloud compute instances describe $CLIENT_VM_NAME --zone=$CLIENT_ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
        
        cat >> vm-info.txt << EOF
CLIENT_${i}_VM=$CLIENT_VM_NAME
CLIENT_${i}_ZONE=$CLIENT_ZONE
CLIENT_${i}_INTERNAL_IP=$INTERNAL_IP
CLIENT_${i}_EXTERNAL_IP=$EXTERNAL_IP

EOF
    done
fi

echo_success "Infrastructure setup complete!"
echo "VM info saved to vm-info.txt"
echo ""
echo "Next: ./03-build-push-image.sh"