#!/bin/bash

# Manage Flybold Clients
set -e

# Load VM info
if [ ! -f "vm-info.txt" ]; then
    echo "ERROR: vm-info.txt not found. Run deploy-application.sh first."
    exit 1
fi

source vm-info.txt

# Verify critical variables are loaded
if [ -z "$SERVER_VM" ] || [ -z "$SERVER_ZONE" ]; then
    echo "ERROR: Server VM info not loaded from vm-info.txt"
    exit 1
fi

show_usage() {
    cat << EOF
Flybold Client Manager

USAGE:
    $0 <command> [options]

COMMANDS:
    start       Start client(s)
    stop        Stop client(s)
    restart     Restart client(s)
    status      Show status of all clients
    logs        View logs for a client
    server-fresh Restart server container(s) and clear logs (fresh start)

OPTIONS:
    --client <0-9>      Specific client ID
    --all               All clients

EXAMPLES:
    $0 status
    $0 logs --client 5
    $0 stop --client 3
    $0 restart --client 7
    $0 stop --all
    $0 start --all
    $0 server-fresh
EOF
}

get_vm_for_client() {
    local client_id=$1
    local vm_num=$((client_id / 2 + 1))
    
    if [ $vm_num -gt 5 ] || [ $vm_num -lt 1 ]; then
        echo "ERROR: Invalid client ID $client_id (must be 0-9)"
        exit 1
    fi
    
    VM_VAR="CLIENT_${vm_num}_VM"
    ZONE_VAR="CLIENT_${vm_num}_ZONE"
    
    VM_NAME=${!VM_VAR}
    VM_ZONE=${!ZONE_VAR}
    
    # Verify the variables were loaded
    if [ -z "$VM_NAME" ] || [ -z "$VM_ZONE" ]; then
        echo "ERROR: Failed to load VM info for client $client_id"
        echo "  vm_num=$vm_num, VM_VAR=$VM_VAR, ZONE_VAR=$ZONE_VAR"
        echo "  VM_NAME=$VM_NAME, VM_ZONE=$VM_ZONE"
        echo "Please ensure vm-info.txt is properly configured."
        exit 1
    fi
}

show_status() {
    echo "=== Flybold Cluster Status ==="
    echo ""
    echo "[SERVER: $SERVER_VM]"
    gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
        cd /app 2>/dev/null && sudo docker compose ps || echo 'Not deployed'
    " 2>/dev/null
    
    echo ""
    for i in $(seq 1 5); do
        VM_VAR="CLIENT_${i}_VM"
        ZONE_VAR="CLIENT_${i}_ZONE"
        VM_NAME=${!VM_VAR}
        VM_ZONE=${!ZONE_VAR}
        
        CLIENT_ID_1=$(( (i-1)*2 ))
        CLIENT_ID_2=$(( (i-1)*2 + 1 ))
        
        echo "[CLIENT VM $i: $VM_NAME - Clients $CLIENT_ID_1, $CLIENT_ID_2]"
        gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
            cd /app 2>/dev/null && sudo docker compose ps || echo 'Not deployed'
        " 2>/dev/null
        echo ""
    done
}

start_client() {
    local client_id=$1
    get_vm_for_client $client_id
    
    echo "Starting client $client_id on $VM_NAME..."
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        sudo docker compose up -d fl-client-${client_id}
    "
}

stop_client() {
    local client_id=$1
    get_vm_for_client $client_id
    
    echo "Stopping client $client_id on $VM_NAME..."
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        sudo docker compose stop fl-client-${client_id}
    "
}

restart_client() {
    local client_id=$1
    get_vm_for_client $client_id
    
    echo "Restarting client $client_id on $VM_NAME..."
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        cd /app
        sudo docker compose restart fl-client-${client_id}
    "
}

view_logs() {
    local client_id=$1
    get_vm_for_client $client_id
    
    echo "Viewing logs for client $client_id on $VM_NAME..."
    gcloud compute ssh $VM_NAME --zone=$VM_ZONE --command="
        sudo docker logs -f fl-client-${client_id}
    "
}

start_all() {
    for i in $(seq 0 9); do
        start_client $i
    done
}

stop_all() {
    for i in $(seq 0 9); do
        stop_client $i
    done
}

restart_all() {
    for i in $(seq 0 9); do
        restart_client $i
    done
}

restart_server_fresh() {
    echo "=== Restarting server and clearing logs (fresh start) ==="
    echo "[SERVER: $SERVER_VM]"

    # Don't let set -e kill the script without showing why
    set +e
    gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="bash -lc '
        set -euo pipefail
        cd /app

        # Choose compose file
        COMPOSE_FILE=
        if [ -f docker-compose.yml ]; then
            COMPOSE_FILE=docker-compose.yml
        elif [ -f docker-compose.yaml ]; then
            COMPOSE_FILE=docker-compose.yaml
        else
            echo \"ERROR: No docker-compose.yml or docker-compose.yaml in /app\" >&2
            exit 1
        fi

        echo \"[REMOTE] Using \$COMPOSE_FILE\"

        # Clear Docker json-file logs for currently running compose containers (if any)
        CONTAINERS=\$(sudo docker compose -f \"\$COMPOSE_FILE\" ps -q 2>/dev/null || true)
        if [ -n \"\$CONTAINERS\" ]; then
            echo \"[REMOTE] Truncating Docker logs\"
            for cid in \$CONTAINERS; do
                sudo truncate -s 0 \"/var/lib/docker/containers/\$cid/\$cid-json.log\" 2>/dev/null || true
            done
        else
            echo \"[REMOTE] No running containers found (nothing to truncate)\"
        fi

        echo \"[REMOTE] docker compose down\"
        sudo docker compose -f \"\$COMPOSE_FILE\" down --remove-orphans || true

        echo \"[REMOTE] docker compose up --force-recreate\"
        sudo docker compose -f \"\$COMPOSE_FILE\" up -d --force-recreate

        # Optional: clear python caches inside fl-server if service exists
        if sudo docker compose -f \"\$COMPOSE_FILE\" ps -q fl-server >/dev/null 2>&1; then
            echo \"[REMOTE] Clearing __pycache__ and *.pyc inside fl-server\"
            sudo docker compose -f \"\$COMPOSE_FILE\" exec -T fl-server find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
            sudo docker compose -f \"\$COMPOSE_FILE\" exec -T fl-server find /app -type f -name \"*.pyc\" -delete 2>/dev/null || true
        fi

        echo \"[REMOTE] docker compose ps\"
        sudo docker compose -f \"\$COMPOSE_FILE\" ps
    '"
    status=$?
    set -e

    if [ $status -ne 0 ]; then
        echo "ERROR: server-fresh failed with exit code $status"
        exit $status
    fi
}


# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

COMMAND=$1
shift

CLIENT_ID=""
ALL_FLAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --client)
            CLIENT_ID=$2
            shift 2
            ;;
        --all)
            ALL_FLAG=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute commands
case $COMMAND in
    status)
        show_status
        ;;
    start)
        if [ "$ALL_FLAG" = true ]; then
            start_all
        elif [ -n "$CLIENT_ID" ]; then
            start_client $CLIENT_ID
        else
            echo "ERROR: start requires --client or --all"
            exit 1
        fi
        ;;
    stop)
        if [ "$ALL_FLAG" = true ]; then
            stop_all
        elif [ -n "$CLIENT_ID" ]; then
            stop_client $CLIENT_ID
        else
            echo "ERROR: stop requires --client or --all"
            exit 1
        fi
        ;;
    restart)
        if [ "$ALL_FLAG" = true ]; then
            restart_all
        elif [ -n "$CLIENT_ID" ]; then
            restart_client $CLIENT_ID
        else
            echo "ERROR: restart requires --client or --all"
            exit 1
        fi
        ;;
    server-fresh)
        restart_server_fresh
        ;;

    logs)
        if [ -z "$CLIENT_ID" ]; then
            echo "ERROR: logs requires --client"
            exit 1
        fi
        view_logs $CLIENT_ID
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac