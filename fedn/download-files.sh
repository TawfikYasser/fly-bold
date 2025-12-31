#!/bin/bash

# Download artifacts from the server VM
set -euo pipefail

source vm-info.txt

echo "=== FEDn File Downloader ==="
# Find regular files
FILES=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="find /app -maxdepth 3 -type f \( -name '*.json' -o -name '*.log' -o -name '*.npz' -o -name '*.tar.gz' -o -name '*.yaml' \) 2>/dev/null | sort")

# Check for analysis_plots directory
PLOTS_DIR=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="find /app -maxdepth 3 -type d -name 'analysis_plots'" 2>/dev/null)

if [ -n "$PLOTS_DIR" ]; then
  FILES="$FILES
$PLOTS_DIR"
fi

if [ -z "$FILES" ]; then
  echo "No files found"; exit 0
fi

i=1
declare -a FARR
while IFS= read -r f; do
  [ -z "$f" ] && continue
  echo "[$i] $f"
  FARR[$i]=$f
  ((i++))
done <<< "$FILES"

echo ""
read -p "Enter file numbers (comma) or 'all': " sel
mkdir -p downloads

if [ "$sel" = "all" ]; then
  for f in "${FARR[@]}"; do
    [ -z "$f" ] && continue
    echo "Downloading $f"
    if [[ "$f" == *"analysis_plots"* ]]; then
        gcloud compute scp --recurse "$SERVER_VM:$f" ./downloads/ --zone="$SERVER_ZONE" --quiet
    else
        gcloud compute scp "$SERVER_VM:$f" ./downloads/ --zone="$SERVER_ZONE" --quiet
    fi
  done
else
  IFS=',' read -ra NUMS <<< "$sel"
  for n in "${NUMS[@]}"; do
    n=$(echo "$n" | xargs)
    f=${FARR[$n]:-}
    [ -z "$f" ] && { echo "Invalid $n"; continue; }
    echo "Downloading $f"
    if [[ "$f" == *"analysis_plots"* ]]; then
        gcloud compute scp --recurse "$SERVER_VM:$f" ./downloads/ --zone="$SERVER_ZONE" --quiet
    else
        gcloud compute scp "$SERVER_VM:$f" ./downloads/ --zone="$SERVER_ZONE" --quiet
    fi
  done
fi

echo "Files in ./downloads:"; ls -lh downloads
