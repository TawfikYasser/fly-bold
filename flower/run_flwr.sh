#!/bin/bash

# Load environment variables from .env
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    exit 1
fi

source .env

# Run the Flwr job and capture output
echo "🚀 Starting Flwr run..."
OUTPUT=$(gcloud compute ssh flybold-server --zone=us-central1-a --command='cd /app && sudo docker compose exec fl-server flwr run .')

# Display the output
echo "$OUTPUT"

# Extract run ID using grep and awk
FLWR_RUN_ID=$(echo "$OUTPUT" | grep "Successfully started run" | awk '{print $NF}')

# Check if we got an ID
if [ -z "$FLWR_RUN_ID" ]; then
    echo "❌ Failed to extract run ID"
    exit 1
fi

echo ""
echo "✅ Extracted Run ID: $FLWR_RUN_ID"

# Save run info to file
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
RUN_LOG="run_history.txt"

# Create header if file doesn't exist
if [ ! -f "$RUN_LOG" ]; then
    echo "# Flybold Run History" > "$RUN_LOG"
    echo "# Format: Timestamp | Flwr_Run_ID | Local_Run_ID | Server_Rounds | Local_Epochs | Batch_Size | Img_Size | YOLO_Size | LR | Dataset | Num_CPUs" >> "$RUN_LOG"
    echo "# ==============================================================================================" >> "$RUN_LOG"
fi

# Append run info
echo "$TIMESTAMP | $FLWR_RUN_ID | $RUN_ID | $NUM_SERVER_ROUNDS | $LOCAL_EPOCHS | $BATCH_SIZE | $IMG_SIZE | $YOLO_SIZE | $LR | $DATASET | $NUM_CPUS" >> "$RUN_LOG"

echo "📝 Run info saved to $RUN_LOG"
echo ""
echo "📊 Fetching logs..."
echo ""

# Fetch logs with the extracted ID
gcloud compute ssh flybold-server --zone=us-central1-a --command="cd /app && sudo docker compose exec fl-server flwr log $FLWR_RUN_ID"