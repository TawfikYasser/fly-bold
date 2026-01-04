#!/bin/bash

# Manage FEDn clients (start/stop/restart/status/logs)
set -euo pipefail

if [ ! -f vm-info.txt ]; then
  echo "vm-info.txt missing"; exit 1
fi
source vm-info.txt

usage(){ cat << EOF
FEDn Client Manager

Usage: $0 <command> [options]
Commands:
  status               Show status for all client containers
  start --client N     Start specific client id
  stop --client N      Stop specific client id
  restart --client N   Restart specific client id
  logs --client N      Tail logs for a client
  start --all          Start all clients
  stop --all           Stop all clients
  restart --all        Restart all clients
EOF
}

get_vm_for_client(){
  local cid=$1
  local vm_num=$((cid / 2 + 1))
  if [ $vm_num -gt 5 ]; then
    echo "Invalid client id $cid"; exit 1
  fi
  VM_VAR="CLIENT_${vm_num}_VM"; ZONE_VAR="CLIENT_${vm_num}_ZONE"
  VM_NAME=${!VM_VAR}; VM_ZONE=${!ZONE_VAR}
}

show_status(){
  echo "=== FEDn Cluster Status ==="
  echo "[SERVER: $SERVER_VM]"
  gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="cd /app && sudo docker compose ps" 2>/dev/null || true
  echo ""
  for i in $(seq 1 5); do
    VM_VAR="CLIENT_${i}_VM"; ZONE_VAR="CLIENT_${i}_ZONE"
    VM_NAME=${!VM_VAR}; VM_ZONE=${!ZONE_VAR}
    CID1=$(( (i-1)*2 )); CID2=$(( (i-1)*2 + 1 ))
    echo "[CLIENT VM $i: $VM_NAME - clients $CID1,$CID2]"
    gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="cd /app && sudo docker compose ps" 2>/dev/null || true
    echo ""
  done
}

start_client(){ get_vm_for_client "$1"; gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="cd /app && sudo docker compose up -d fedn-client-$1"; }
stop_client(){ get_vm_for_client "$1"; gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="cd /app && sudo docker compose stop fedn-client-$1"; }
restart_client(){ get_vm_for_client "$1"; gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="cd /app && sudo docker compose restart fedn-client-$1"; }
logs_client(){ get_vm_for_client "$1"; gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="sudo docker logs -f fedn-client-$1"; }

start_all(){ for c in $(seq 0 9); do start_client "$c"; done; }
stop_all(){ for c in $(seq 0 9); do stop_client "$c"; done; }
restart_all(){ for c in $(seq 0 9); do restart_client "$c"; done; }

if [ $# -eq 0 ]; then usage; exit 0; fi
CMD=$1; shift || true
CLIENT_ID=""; ALL=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --client) CLIENT_ID=$2; shift 2;;
    --all) ALL=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option $1"; usage; exit 1;;
  esac
done

case $CMD in
  status) show_status;;
  start) if $ALL; then start_all; elif [ -n "$CLIENT_ID" ]; then start_client "$CLIENT_ID"; else echo "start needs --client or --all"; exit 1; fi;;
  stop) if $ALL; then stop_all; elif [ -n "$CLIENT_ID" ]; then stop_client "$CLIENT_ID"; else echo "stop needs --client or --all"; exit 1; fi;;
  restart) if $ALL; then restart_all; elif [ -n "$CLIENT_ID" ]; then restart_client "$CLIENT_ID"; else echo "restart needs --client or --all"; exit 1; fi;;
  logs) if [ -z "$CLIENT_ID" ]; then echo "logs needs --client"; exit 1; fi; logs_client "$CLIENT_ID";;
  *) echo "Unknown command $CMD"; usage; exit 1;;
esac
