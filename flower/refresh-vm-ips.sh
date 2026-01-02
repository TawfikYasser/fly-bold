#!/bin/bash

# Refresh VM IP Addresses
# Run this script after stopping/starting VMs to update vm-info.txt

set -e

PROJECT_ID="inf022"

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

# Function to check if VM is running
check_vm_status() {
    local vm_name=$1
    local zone=$2
    
    local status=$(gcloud compute instances describe $vm_name \
        --zone=$zone \
        --format='get(status)' 2>/dev/null)
    
    echo "$status"
}

# Function to fetch current IP addresses from GCP
fetch_vm_ips() {
    local vm_name=$1
    local zone=$2
    
    # Check if VM is running
    local status=$(check_vm_status $vm_name $zone)
    
    if [ "$status" != "RUNNING" ]; then
        echo "  VM $vm_name is $status (not RUNNING). IPs may be unavailable." >&2
        echo "UNAVAILABLE|UNAVAILABLE"
        return
    fi
    
    # Get internal IP
    local internal_ip=$(gcloud compute instances describe $vm_name \
        --zone=$zone \
        --format='get(networkInterfaces[0].networkIP)' 2>/dev/null)
    
    # Get external IP (may not exist for some VMs)
    local external_ip=$(gcloud compute instances describe $vm_name \
        --zone=$zone \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
    
    if [ -z "$internal_ip" ]; then
        internal_ip="UNAVAILABLE"
    fi
    
    if [ -z "$external_ip" ]; then
        external_ip="None"
    fi
    
    # Send info to stderr
    echo "  Status: $status" >&2
    echo "  Internal IP: $internal_ip" >&2
    echo "  External IP: $external_ip" >&2
    
    # Return IPs via echo (caller will capture) - this goes to stdout
    echo "$internal_ip|$external_ip"
}

# Main script
echo_info "Refreshing VM IP addresses..."

# Backup existing vm-info.txt
if [ -f "vm-info.txt" ]; then
    cp vm-info.txt vm-info.txt.backup
    echo_success "Backed up existing vm-info.txt to vm-info.txt.backup"
else
    echo_error "vm-info.txt not found. Run 02-setup-infrastructure.sh first."
    exit 1
fi

# Create new vm-info.txt with current IPs
cat > vm-info.txt << EOF
PROJECT_ID=$PROJECT_ID
REGION=us-central1
NETWORK=flybold-network

EOF

echo ""
echo "═══════════════════════════════════════════════════════════"

# Fetch server IPs
echo_info "Fetching server IP..."
local server_ips=$(fetch_vm_ips "flybold-server" "us-central1-a")
local server_internal=$(echo $server_ips | cut -d'|' -f1)
local server_external=$(echo $server_ips | cut -d'|' -f2)

cat >> vm-info.txt << EOF
SERVER_VM=flybold-server
SERVER_ZONE=us-central1-a
SERVER_INTERNAL_IP=$server_internal
SERVER_EXTERNAL_IP=$server_external

EOF

# Fetch client IPs
local client_zones=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-central1-a")

for i in $(seq 1 5); do
    echo ""
    echo_info "Fetching Client $i IP..."
    local zone=${client_zones[$((i-1))]}
    local client_ips=$(fetch_vm_ips "flybold-client-$i" "$zone")
    local client_internal=$(echo $client_ips | cut -d'|' -f1)
    local client_external=$(echo $client_ips | cut -d'|' -f2)
    
    cat >> vm-info.txt << EOF
CLIENT_${i}_VM=flybold-client-${i}
CLIENT_${i}_ZONE=$zone
CLIENT_${i}_INTERNAL_IP=$client_internal
CLIENT_${i}_EXTERNAL_IP=$client_external

EOF
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo_success "vm-info.txt updated with current IPs!"
echo ""

# Display summary
echo "Current IP Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-15s %-15s\n" "VM Name" "Internal IP" "External IP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-15s %-15s\n" "flybold-server" "$server_internal" "$server_external"
for i in $(seq 1 5); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_INTERNAL_VAR="CLIENT_${i}_INTERNAL_IP"
    CLIENT_EXTERNAL_VAR="CLIENT_${i}_EXTERNAL_IP"
    
    source vm-info.txt
    printf "%-20s %-15s %-15s\n" "${!CLIENT_VM_VAR}" "${!CLIENT_INTERNAL_VAR}" "${!CLIENT_EXTERNAL_VAR}"
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo_warning "Important Notes:"
echo "  • If you see 'UNAVAILABLE', the VM is not running"
echo "  • Start stopped VMs with: gcloud compute instances start VM_NAME --zone=ZONE"
echo "  • After starting VMs, run this script again to refresh IPs"
echo "  • Run deploy-application.sh to deploy with the new IPs"
echo ""