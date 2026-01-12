#!/bin/bash

# Real-time Resource Monitor for Flybold VMs and Containers with Logging
set -e

PROJECT_ID="inf022"

# Load VM info
if [ ! -f "vm-info.txt" ]; then
    echo "ERROR: vm-info.txt not found"
    exit 1
fi
source vm-info.txt

# Load .env for RUN_ID
if [ ! -f ".env" ]; then
    echo "ERROR: .env not found"
    exit 1
fi
source .env

# Setup logging
LOG_DIR="logs/monitoring"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/resource_monitor_run_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
JSON_LOG="${LOG_DIR}/resource_monitor_run_${RUN_ID}_$(date +%Y%m%d_%H%M%S).jsonl"

echo "Starting resource monitoring for RUN_ID: $RUN_ID"
echo "Logs will be saved to:"
echo "  - Plain text: $LOG_FILE"
echo "  - JSON Lines: $JSON_LOG"
echo ""

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

# Function to log JSON
log_json() {
    local timestamp=$1
    local vm_type=$2
    local vm_name=$3
    local cpu=$4
    local mem_used=$5
    local mem_total=$6
    local mem_percent=$7
    shift 7
    local containers="$@"
    
    # Build JSON object
    cat >> "$JSON_LOG" << EOF
{"timestamp":"$timestamp","run_id":"$RUN_ID","vm_type":"$vm_type","vm_name":"$vm_name","vm_cpu_percent":$cpu,"vm_mem_used_gb":$mem_used,"vm_mem_total_gb":$mem_total,"vm_mem_percent":$mem_percent,"containers":[$containers]}
EOF
}

# Function to display header
display_header() {
    clear
    local header="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BOLD}${CYAN}$header${NC}"
    echo -e "${BOLD}${CYAN}                    FLYBOLD RESOURCE MONITOR                    ${NC}"
    echo -e "${BOLD}${CYAN}$header${NC}"
    echo -e "${BLUE}Updated: $(date '+%Y-%m-%d %H:%M:%S') | RUN_ID: $RUN_ID${NC}"
    echo -e "${BLUE}Logging to: $LOG_FILE${NC}"
    echo ""
    
    # Log to file
    {
        echo "=========================================="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "RUN_ID: $RUN_ID"
        echo "=========================================="
        echo ""
    } >> "$LOG_FILE"
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
    local vm_type=$4
    
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
    
    # Log to plain text file
    {
        echo "$label: $vm_name"
        echo "  VM CPU: ${cpu_usage}%"
        echo "  VM Memory: ${mem_used} / ${mem_total} GB (${mem_percent}%)"
    } >> "$LOG_FILE"
    
    # Parse container stats
    local container_section=$(echo "$stats" | sed -n '/CONTAINER_START/,/CONTAINER_END/p')
    local container_json_array=""
    
    if echo "$container_section" | grep -q "NOCONTAINERS"; then
        echo -e "${BOLD}│${NC}"
        echo -e "${BOLD}│ Containers:${NC} ${YELLOW}No containers running${NC}"
        echo "  Containers: None running" >> "$LOG_FILE"
    elif [ -n "$container_section" ]; then
        echo -e "${BOLD}│${NC}"
        echo -e "${BOLD}│ Containers:${NC}"
        echo "  Containers:" >> "$LOG_FILE"
        
        # Skip the header line and process container stats
        local first_container=true
        echo "$container_section" | grep '|' | grep -v 'NAME' | while IFS='|' read -r name cpu mem mem_percent; do
            if [ -n "$name" ] && [ "$name" != "CONTAINER_START" ] && [ "$name" != "CONTAINER_END" ]; then
                # Clean up the values
                name=$(echo "$name" | tr -d ' ')
                cpu=$(echo "$cpu" | tr -d ' %')
                mem=$(echo "$mem" | tr -d ' ')
                mem_percent=$(echo "$mem_percent" | tr -d ' %')
                
                printf "│   %-15s CPU: %s   MEM: %-20s (%s)\n" \
                    "$name" "$(color_percent "$cpu")" "$mem" "$(color_percent "$mem_percent")"
                
                # Log to plain text
                echo "    - $name: CPU=${cpu}%, MEM=${mem} (${mem_percent}%)" >> "$LOG_FILE"
                
                # Build JSON array for containers
                if [ "$first_container" = true ]; then
                    container_json_array="{\"name\":\"$name\",\"cpu_percent\":$cpu,\"memory\":\"$mem\",\"memory_percent\":$mem_percent}"
                    first_container=false
                else
                    container_json_array="$container_json_array,{\"name\":\"$name\",\"cpu_percent\":$cpu,\"memory\":\"$mem\",\"memory_percent\":$mem_percent}"
                fi
            fi
        done
    fi
    
    echo -e "${BOLD}└───────────────────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "" >> "$LOG_FILE"
    
    # Log JSON
    log_json "$(date -Iseconds)" "$vm_type" "$vm_name" "$cpu_usage" "$mem_used" "$mem_total" "$mem_percent" "$container_json_array"
}

# Main monitoring loop
monitor_loop() {
    while true; do
        display_header
        
        # Monitor Server
        echo -e "${BOLD}${GREEN}SERVER${NC}"
        echo ""
        echo "=== SERVER ===" >> "$LOG_FILE"
        server_stats=$(get_vm_stats "$SERVER_VM" "$SERVER_ZONE" "fl-server")
        display_stats "$SERVER_VM" "$server_stats" "🖥️  SERVER" "server"
        
        # Monitor Clients
        echo -e "${BOLD}${GREEN}CLIENTS${NC}"
        echo ""
        echo "=== CLIENTS ===" >> "$LOG_FILE"
        
        for i in $(seq 1 5); do
            CLIENT_VM_VAR="CLIENT_${i}_VM"
            CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
            CLIENT_VM=${!CLIENT_VM_VAR}
            CLIENT_ZONE=${!CLIENT_ZONE_VAR}
            
            CLIENT_ID_1=$(( (i-1)*2 ))
            CLIENT_ID_2=$(( (i-1)*2 + 1 ))
            
            container_names="fl-client-${CLIENT_ID_1} fl-client-${CLIENT_ID_2}"
            
            client_stats=$(get_vm_stats "$CLIENT_VM" "$CLIENT_ZONE" "$container_names")
            display_stats "$CLIENT_VM" "$client_stats" "💻 CLIENT VM $i" "client"
        done
        
        echo -e "${CYAN}Refreshing in 5 seconds... (Press Ctrl+C to exit)${NC}"
        echo "----------------------------------------" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        sleep 5
    done
}

# Trap to handle cleanup on exit
cleanup() {
    echo ""
    echo "Monitoring stopped. Logs saved to:"
    echo "  - $LOG_FILE"
    echo "  - $JSON_LOG"
    exit 0
}
trap cleanup INT TERM

# Check for quick mode (one-time display)
if [ "$1" == "--once" ]; then
    display_header
    
    echo -e "${BOLD}${GREEN}SERVER${NC}"
    echo ""
    echo "=== SERVER ===" >> "$LOG_FILE"
    server_stats=$(get_vm_stats "$SERVER_VM" "$SERVER_ZONE" "fl-server")
    display_stats "$SERVER_VM" "$server_stats" "🖥️  SERVER" "server"
    
    echo -e "${BOLD}${GREEN}CLIENTS${NC}"
    echo ""
    echo "=== CLIENTS ===" >> "$LOG_FILE"
    
    for i in $(seq 1 5); do
        CLIENT_VM_VAR="CLIENT_${i}_VM"
        CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"
        CLIENT_VM=${!CLIENT_VM_VAR}
        CLIENT_ZONE=${!CLIENT_ZONE_VAR}
        
        CLIENT_ID_1=$(( (i-1)*2 ))
        CLIENT_ID_2=$(( (i-1)*2 + 1 ))
        
        container_names="fl-client-${CLIENT_ID_1} fl-client-${CLIENT_ID_2}"
        
        client_stats=$(get_vm_stats "$CLIENT_VM" "$CLIENT_ZONE" "$container_names")
        display_stats "$CLIENT_VM" "$client_stats" "💻 CLIENT VM $i" "client"
    done
    
    echo ""
    echo "Single capture complete. Logs saved to:"
    echo "  - $LOG_FILE"
    echo "  - $JSON_LOG"
else
    # Start monitoring loop
    echo "Starting real-time resource monitoring..."
    echo "Press Ctrl+C to stop"
    sleep 2
    monitor_loop
fi