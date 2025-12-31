#!/bin/bash
set -e

# Load env
if [ ! -f "vm-info.txt" ]; then
    echo "vm-info.txt not found"
    exit 1
fi
source vm-info.txt

if [ ! -f ".env" ]; then
    echo ".env not found"
    exit 1
fi
source .env

# Config
CHECK_INTERVAL=60   # 1 minute
EXPERIMENT_NAME="EXP_YOLOv5_${YOLO_SIZE}_detection"
LOG_FILE="${EXPERIMENT_NAME}_${RUN_ID}_logs.json"

echo "[INFO] Monitoring for log file: ${LOG_FILE}"
echo "[INFO] Checking every 60 seconds..."
echo "[INFO] Server: ${SERVER_VM}"

while true; do
    echo "[INFO] $(date '+%H:%M:%S') checking..."

    exists=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
        [ -f /app/${LOG_FILE} ] && echo yes || echo no
    " 2>/dev/null)

    if [ "$exists" = "yes" ]; then
        echo "[SUCCESS] Log file detected. Training finished."

        echo "[INFO] Stopping client VMs..."
        for i in $(seq 1 5); do
            VM_VAR="CLIENT_${i}_VM"
            ZONE_VAR="CLIENT_${i}_ZONE"
            gcloud compute instances stop "${!VM_VAR}" --zone="${!ZONE_VAR}" --quiet &
        done

        echo "[INFO] Stopping server VM..."
        gcloud compute instances stop "$SERVER_VM" --zone="$SERVER_ZONE" --quiet &

        wait
        echo "[SUCCESS] All VMs stopped."
        exit 0
    fi

    sleep $CHECK_INTERVAL
done
