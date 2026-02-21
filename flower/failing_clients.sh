#!/bin/bash
# failing_clients.sh — Simulate client dropout during a federated experiment
# Usage: ./failing_clients.sh <25|50|75>

set -e

DROPOUT_WAIT=1800      # 30 minutes before dropping
RECOVERY_WAIT=1800     # 30 minutes before restarting

# Deterministic client sets — spread across VMs for realism
CLIENTS_25=(0 4 8)
CLIENTS_50=(0 2 4 6 8)
CLIENTS_75=(0 1 2 4 5 6 8 9)

# ── Validate args ─────────────────────────────────────────────────────────────
if [ $# -ne 1 ] || ! [[ "$1" =~ ^(25|50|75)$ ]]; then
    echo "Usage: $0 <25|50|75>"
    echo "  25 → stop 3 clients  (0, 4, 8)"
    echo "  50 → stop 5 clients  (0, 2, 4, 6, 8)"
    echo "  75 → stop 8 clients  (0, 1, 2, 4, 5, 6, 8, 9)"
    exit 1
fi

LEVEL=$1
case $LEVEL in
    25) TARGETS=("${CLIENTS_25[@]}") ;;
    50) TARGETS=("${CLIENTS_50[@]}") ;;
    75) TARGETS=("${CLIENTS_75[@]}") ;;
esac

# ── Load VM info ──────────────────────────────────────────────────────────────
if [ ! -f "vm-info.txt" ]; then
    echo "ERROR: vm-info.txt not found."
    exit 1
fi
source vm-info.txt

# ── Helpers ───────────────────────────────────────────────────────────────────
get_vm_for_client() {
    local client_id=$1
    local vm_num=$((client_id / 2 + 1))
    VM_NAME_OUT=$(eval echo "\$CLIENT_${vm_num}_VM")
    VM_ZONE_OUT=$(eval echo "\$CLIENT_${vm_num}_ZONE")
}

stop_client() {
    local client_id=$1
    get_vm_for_client "$client_id"
    echo "  [STOP]  client-${client_id} on ${VM_NAME_OUT}"
    gcloud compute ssh "$VM_NAME_OUT" --zone="$VM_ZONE_OUT" --command="
        cd /app && sudo docker compose stop fl-client-${client_id}
    " 2>/dev/null
}

start_client() {
    local client_id=$1
    get_vm_for_client "$client_id"
    echo "  [START] client-${client_id} on ${VM_NAME_OUT}"
    gcloud compute ssh "$VM_NAME_OUT" --zone="$VM_ZONE_OUT" --command="
        cd /app && sudo docker compose start fl-client-${client_id}
    " 2>/dev/null
}

countdown() {
    local seconds=$1
    local label=$2
    local end=$((SECONDS + seconds))
    while [ $SECONDS -lt $end ]; do
        local remaining=$((end - SECONDS))
        printf "\r  %s — %02d:%02d remaining..." "$label" $((remaining / 60)) $((remaining % 60))
        sleep 5
    done
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Failing Clients Simulation — ${LEVEL}% dropout"
echo "  Targets: ${TARGETS[*]}"
echo "  Phase 1: Wait ${DROPOUT_WAIT}s before stopping"
echo "  Phase 2: Wait ${RECOVERY_WAIT}s before restarting"
echo "══════════════════════════════════════════════════════"
echo ""

# Phase 1: Wait before dropping
countdown "$DROPOUT_WAIT" "Waiting to drop clients"

echo "[$(date '+%H:%M:%S')] Stopping ${#TARGETS[@]} clients..."
for cid in "${TARGETS[@]}"; do
    stop_client "$cid"
done
echo "[$(date '+%H:%M:%S')] Done. ${#TARGETS[@]} clients stopped."
echo ""

# Phase 2: Wait before recovering
countdown "$RECOVERY_WAIT" "Waiting to restart clients"

echo "[$(date '+%H:%M:%S')] Restarting ${#TARGETS[@]} clients..."
for cid in "${TARGETS[@]}"; do
    start_client "$cid"
done
echo "[$(date '+%H:%M:%S')] Done. ${#TARGETS[@]} clients restarted."
echo ""
echo "══════════════════════════════════════════════════════"
echo "  Simulation complete. Check logs with:"
echo "  ./manage-clients.sh status"
echo "══════════════════════════════════════════════════════"