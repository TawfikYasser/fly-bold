#!/bin/bash

# Check if run ID is provided
if [ -z "$1" ]; then
    echo "❌ Usage: $0 <RUN_ID>"
    echo ""
    echo "Example: $0 1868658470050770617"
    echo ""
    
    # Show recent runs if available
    if [ -f "run_history.txt" ]; then
        echo "📋 Recent runs:"
        echo ""
        tail -n 5 run_history.txt | grep -v "^#"
    fi
    
    exit 1
fi

RUN_ID=$1

echo "🛑 Stopping Flwr run: $RUN_ID"
echo ""

# Stop the run
gcloud compute ssh flybold-server --zone=us-central1-a --command="cd /app && sudo docker compose exec fl-server flwr stop $RUN_ID"

echo ""
echo "✅ Stop command executed for run: $RUN_ID"
