#!/bin/bash

# Real-time Resource Monitor for Flybold VMs and Containers
set -e

PROJECT_ID="inf022"

# Load VM info
if [ ! -f "vm-info.txt" ]; then
    echo "ERROR: vm-info.txt not found"
    exit 1
fi
source vm-info.txt

# Colors
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to get VM and container stats
get_vm_stats() {
    local vm_name=$1
    local zone=$2
    local container_names=$3
    
    # Get stats from VM
    gcloud compute ssh $vm_name --zone=$zone --command="
        # VM Stats
        echo 'VMSTATS_START'
        top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1
        free -m | awk 'NR==2{printf \"%.1f|%.1f|%.1f\", \$3/1024, \$2/1024, \$3*100/\$2}'
        echo ''
        echo 'VMSTATS_END'
        
        # Container Stats
        echo 'CONTAINER_START'
        if [ -n '$container_names' ]; then
            sudo docker stats --no-stream --format 'table {{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}' $container_names 2>/dev/null || echo 'NOCONTAINERS'
        fi
        echo 'CONTAINER_END'
    " 2>/dev/null
}

# Function to display header
display_header() {
    clear
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}                    FLYBOLD RESOURCE MONITOR                    ${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Updated: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
}

# Function to format percentage color
color_percent() {
    local value=$1
    
    # Remove any non-numeric characters except decimal point
    value=$(echo "$value" | tr -d -c '0-9.')
    
    # Check if value is empty or invalid
    if [ -z "$value" ]; then
        echo -e "${NC}N/A${NC}"
        return
    fi
    
    local value_int=$(printf "%.0f" "$value" 2>/dev/null || echo "0")
    
    if (( value_int >= 80 )); then
        echo -e "${RED}${value}%${NC}"
    elif (( value_int >= 60 )); then
        echo -e "${YELLOW}${value}%${NC}"
    else
        echo -e "${GREEN}${value}%${NC}"
    fi
}

# Function to parse and display stats
display_stats() {
    local vm_name=$1
    local stats=$2
    local label=$3
    
    # Parse VM stats
    local vm_section=$(echo "$stats" | sed -n '/VMSTATS_START/,/VMSTATS_END/p')
    local cpu_usage=$(echo "$vm_section" | sed -n '2p' | tr -d ' ')
    local mem_info=$(echo "$vm_section" | sed -n '3p' | tr -d ' ')
    
    # Default values if parsing fails
    if [ -z "$cpu_usage" ]; then
        cpu_usage="0"
    fi
    
    if [ -z "$mem_info" ]; then
        mem_info="0|0|0"
    fi
    
    local mem_used=$(echo "$mem_info" | cut -d'|' -f1)
    local mem_total=$(echo "$mem_info" | cut -d'|' -f2)
    local mem_percent=$(echo "$mem_info" | cut -d'|' -f3)
    
    # Display VM stats
    echo -e "${BOLD}${BLUE}┌─ $label: $vm_name${NC}"
    echo -e "${BOLD}│${NC}"
    echo -e "${BOLD}│ VM Resources:${NC}"
    printf "│   CPU:    %s\n" "$(color_percent "$cpu_usage")"
    printf "│   Memory: %.1f / %.1f GB (%s)\n" "$mem_used" "$mem_total" "$(color_percent "$mem_percent")"
    
    # Parse container stats
    local container_section=$(echo "$stats" | sed -n '/CONTAINER_START/,/CONTAINER_END/p')
    
    if echo "$container_section" | grep -q "NOCONTAINERS"; then
        echo -e "${BOLD}│${NC}"
        echo -e "${BOLD}│ Containers:${NC} ${YELLOW}No containers running${NC}"
    elif [ -n "$container_section" ]; then
        echo -e "${BOLD}│${NC}"
        echo -e "${BOLD}│ Containers:${NC}"
        
        # Skip the header line and process container stats
        echo "$container_section" | grep '|' | grep -v 'NAME' | while IFS='|' read -r name cpu mem mem_percent; do
            if [ -n "$name" ] && [ "$name" != "CONTAINER_START" ] && [ "$name" != "CONTAINER_END" ]; then
                # Clean up the values
                name=$(echo "$name" | tr -d ' ')
                cpu=$(echo "$cpu" | tr -d ' %')
                mem=$(echo "$mem" | tr -d ' ')
                mem_percent=$(echo "$mem_percent" | tr -d ' %')
                
                printf "│   %-15s CPU: %s   MEM: %-20s (%s)\n" \
                    "$name" "$(color_percent "$cpu")" "$mem" "$(color_percent "$mem_percent")"
            fi
        done
    fi
    
    echo -e "${BOLD}└─────────────────────────────────────────────────────────────────────────────${NC}"
    echo ""
}

# Main monitoring loop
monitor_loop() {
    while true; do
        display_header
        
        # Monitor Server
        echo -e "${BOLD}${GREEN}SERVER${NC}"
        echo ""
        server_stats=$(get_vm_stats "$SERVER_VM" "$SERVER_ZONE" "fl-server")
        display_stats "$SERVER_VM" "$server_stats" "🖥️  SERVER"
        
        # Monitor Clients
        echo -e "${BOLD}${GREEN}CLIENTS${NC}"
        echo ""
        
        for i in $(seq 1 5); do
            CLIENT_VM_VAR="CLIENT_${i}_VM"
            CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
            CLIENT_VM=${!CLIENT_VM_VAR}
            CLIENT_ZONE=${!CLIENT_ZONE_VAR}
            
            CLIENT_ID_1=$(( (i-1)*2 ))
            CLIENT_ID_2=$(( (i-1)*2 + 1 ))
            
            container_names="fl-client-${CLIENT_ID_1} fl-client-${CLIENT_ID_2}"
            
            client_stats=$(get_vm_stats "$CLIENT_VM" "$CLIENT_ZONE" "$container_names")
            display_stats "$CLIENT_VM" "$client_stats" "💻 CLIENT VM $i"
        done
        
        echo -e "${CYAN}Refreshing in 5 seconds... (Press Ctrl+C to exit)${NC}"
        sleep 5
    done
}

# Check for quick mode (one-time display)
if [ "$1" == "--once" ]; then
    display_header
    
    echo -e "${BOLD}${GREEN}SERVER${NC}"
    echo ""
    server_stats=$(get_vm_stats "$SERVER_VM" "$SERVER_ZONE" "fl-server")
    display_stats "$SERVER_VM" "$server_stats" "🖥️  SERVER"
    
    echo -e "${BOLD}${GREEN}CLIENTS${NC}"
    echo ""
    
    for i in $(seq 1 5); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        
        CLIENT_ID_1=$(( (i-1)*2 ))
        CLIENT_ID_2=$(( (i-1)*2 + 1 ))
        
        container_names="fl-client-${CLIENT_ID_1} fl-client-${CLIENT_ID_2}"
        
        client_stats=$(get_vm_stats "$CLIENT_VM" "$CLIENT_ZONE" "$container_names")
        display_stats "$CLIENT_VM" "$client_stats" "💻 CLIENT VM $i"
    done
else
    # Start monitoring loop
    echo "Starting real-time resource monitoring..."
    echo "Press Ctrl+C to stop"
    sleep 2
    monitor_loop
fi