#!/bin/bash

# Refresh VM IP Addresses
# Run this script after stopping/starting VMs to update vm-info.txt

set -euo pipefail

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
    local vm_name="$1"
    local zone="$2"

    gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --format='get(status)' 2>/dev/null || true
}

# Function to fetch current IP addresses from GCP
fetch_vm_ips() {
    local vm_name="$1"
    local zone="$2"

    local status
    status="$(check_vm_status "$vm_name" "$zone")"

    if [ "$status" != "RUNNING" ]; then
        echo "  VM $vm_name is ${status:-UNKNOWN} (not RUNNING). IPs may be unavailable." >&2
        echo "UNAVAILABLE|UNAVAILABLE"
        return 0
    fi

    local internal_ip external_ip

    internal_ip="$(gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --format='get(networkInterfaces[0].networkIP)' 2>/dev/null || true)"

    external_ip="$(gcloud compute instances describe "$vm_name" \
        --zone="$zone" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)"

    if [ -z "${internal_ip:-}" ]; then
        internal_ip="UNAVAILABLE"
    fi

    if [ -z "${external_ip:-}" ]; then
        external_ip="None"
    fi

    echo "  Status: $status" >&2
    echo "  Internal IP: $internal_ip" >&2
    echo "  External IP: $external_ip" >&2

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
server_ips="$(fetch_vm_ips "flybold-server" "us-central1-a")"
server_internal="$(echo "$server_ips" | cut -d'|' -f1)"
server_external="$(echo "$server_ips" | cut -d'|' -f2)"

cat >> vm-info.txt << EOF
SERVER_VM=flybold-server
SERVER_ZONE=us-central1-a
SERVER_INTERNAL_IP=$server_internal
SERVER_EXTERNAL_IP=$server_external

EOF

# Fetch client IPs - dynamically discover all clients
echo ""
echo_info "Discovering all client VMs..."
client_list=$(gcloud compute instances list \
    --project="$PROJECT_ID" \
    --filter="name~'^flybold-client-'" \
    --format="csv[no-heading](name,zone)" \
    --sort-by=name 2>/dev/null || true)

if [ -z "$client_list" ]; then
    echo_warning "No client VMs found running."
    MAX_CLIENT_NUM=0
else
    MAX_CLIENT_NUM=0

    # Sort numerically by the trailing number in the VM name (flybold-client-N)
    # so client-2 always comes before client-10, regardless of gcloud string sort.
    sorted_client_list=$(echo "$client_list" | sort -t'-' -k3 -n)

    while IFS=, read -r vm_name zone; do
        # Derive index from the VM name suffix (e.g. flybold-client-7 -> 7)
        # This keeps CLIENT_N_* numbering stable even when VMs are missing.
        i=$(echo "$vm_name" | grep -oE '[0-9]+$')
        if [ -z "$i" ]; then
            echo_warning "Could not parse numeric index from VM name: $vm_name — skipping."
            continue
        fi

        echo ""
        echo_info "Fetching $vm_name IP (index $i)..."
        client_ips="$(fetch_vm_ips "$vm_name" "$zone")"
        client_internal="$(echo "$client_ips" | cut -d'|' -f1)"
        client_external="$(echo "$client_ips" | cut -d'|' -f2)"

        cat >> vm-info.txt << EOF
CLIENT_${i}_VM=$vm_name
CLIENT_${i}_ZONE=$zone
CLIENT_${i}_INTERNAL_IP=$client_internal
CLIENT_${i}_EXTERNAL_IP=$client_external

EOF
        if [ "$i" -gt "$MAX_CLIENT_NUM" ]; then
            MAX_CLIENT_NUM=$i
        fi
    done <<< "$sorted_client_list"
fi

# Add MAX_CLIENT_NUM to vm-info.txt
echo "MAX_CLIENT_NUM=$MAX_CLIENT_NUM" >> vm-info.txt

echo ""
echo "═══════════════════════════════════════════════════════════"
echo_success "vm-info.txt updated with current IPs!"
echo ""

# Load values once for summary display
# shellcheck disable=SC1091
source vm-info.txt

# Display summary
echo "Current IP Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-15s %-15s\n" "VM Name" "Internal IP" "External IP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s %-15s %-15s\n" "flybold-server" "$SERVER_INTERNAL_IP" "$SERVER_EXTERNAL_IP"

for i in $(seq 1 $MAX_CLIENT_NUM); do
    CLIENT_VM_VAR="CLIENT_${i}_VM"
    CLIENT_INTERNAL_VAR="CLIENT_${i}_INTERNAL_IP"
    CLIENT_EXTERNAL_VAR="CLIENT_${i}_EXTERNAL_IP"

    if [ -n "${!CLIENT_VM_VAR:-}" ]; then
        printf "%-20s %-15s %-15s\n" "${!CLIENT_VM_VAR}" "${!CLIENT_INTERNAL_VAR}" "${!CLIENT_EXTERNAL_VAR}"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo_warning "Important Notes:"
echo "  • If you see 'UNAVAILABLE', the VM is not running"
echo "  • Start stopped VMs with: gcloud compute instances start VM_NAME --zone=ZONE"
echo "  • After starting VMs, run this script again to refresh IPs"
echo "  • Run deploy-application.sh to deploy with the new IPs"
echo ""