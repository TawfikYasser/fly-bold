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
CHECK_INTERVAL=10   # 10 seconds
EXPERIMENT_NAME="EXP_YOLOv5_${YOLO_SIZE}_detection"
LOG_FILE="${EXPERIMENT_NAME}_${RUN_ID}_logs.json"
MODEL_FILE="${EXPERIMENT_NAME}_${RUN_ID}_final_model.pt"
CONTAINER_NAME="fl-server"
CONTAINER_PATH="/app/${LOG_FILE}"

echo "[INFO] Monitoring for log file: ${LOG_FILE}"
echo "[INFO] Container: ${CONTAINER_NAME}"
echo "[INFO] Checking every ${CHECK_INTERVAL} seconds..."
echo "[INFO] Server: ${SERVER_VM}"

while true; do
    echo "[INFO] $(date '+%H:%M:%S') checking..."

    # Check if file exists inside the fl-server container
    exists=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
        sudo docker exec ${CONTAINER_NAME} test -f ${CONTAINER_PATH} && echo yes || echo no
    " 2>/dev/null)

    if [ "$exists" = "yes" ]; then
        echo "[SUCCESS] Log file detected in container. Training finished."

        # Download the file from container to local machine
        echo "[INFO] Downloading log file to local machine..."
        
        # First, copy from container to VM
        gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="
            sudo docker cp ${CONTAINER_NAME}:${CONTAINER_PATH} /tmp/${LOG_FILE}
            sudo docker cp ${CONTAINER_NAME}:/app/${MODEL_FILE} /tmp/${MODEL_FILE}
        "
        
        # Then, copy from VM to local machine under a run_id-scoped directory
        mkdir -p "experiments_outputs/${RUN_ID}"
        gcloud compute scp "${SERVER_VM}:/tmp/${LOG_FILE}" "./experiments_outputs/${RUN_ID}/${LOG_FILE}" --zone="$SERVER_ZONE"
        gcloud compute scp "${SERVER_VM}:/tmp/${MODEL_FILE}" "./experiments_outputs/${RUN_ID}/${MODEL_FILE}" --zone="$SERVER_ZONE"
        
        echo "[SUCCESS] Log file downloaded to: ./experiments_outputs/${RUN_ID}/${LOG_FILE}"
        echo "[SUCCESS] Model file downloaded to: ./experiments_outputs/${RUN_ID}/${MODEL_FILE}"

        # echo "[INFO] Stopping client VMs..."
        # for i in $(seq 1 5); do
        #     VM_VAR="CLIENT_${i}_VM"
        #     ZONE_VAR="CLIENT_${i}_ZONE"
        #     gcloud compute instances stop "${!VM_VAR}" --zone="${!ZONE_VAR}" --quiet &
        # done

        # echo "[INFO] Stopping server VM..."
        # gcloud compute instances stop "$SERVER_VM" --zone="$SERVER_ZONE" --quiet &

        # wait
        # echo "[SUCCESS] All VMs stopped."
        exit 0
    fi

    sleep $CHECK_INTERVAL
done