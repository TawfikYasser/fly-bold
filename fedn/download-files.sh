#!/bin/bash

# Download artifacts from the server VM
set -euo pipefail

source vm-info.txt

echo "=== FEDn File Downloader ==="
# Find regular files (preserve stderr for actual errors, only suppress 'permission denied' type messages)
FILES=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="find /app -maxdepth 3 -type f \( -name '*.json' -o -name '*.log' -o -name '*.npz' -o -name '*.tar.gz' -o -name '*.yaml' \) 2>/dev/null | sort") || { echo "Warning: SSH command for files failed"; FILES=""; }

# Check for analysis_plots directory
PLOTS_DIR=$(gcloud compute ssh "$SERVER_VM" --zone="$SERVER_ZONE" --command="find /app -maxdepth 3 -type d -name 'analysis_plots' 2>/dev/null") || { echo "Warning: SSH command for plots dir failed"; PLOTS_DIR=""; }

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
echo ""
if [ -n "${1:-}" ]; then
  sel="$1"
else
  read -p "Enter file numbers (comma) or 'all': " sel
fi
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
elif [ "$sel" = "analysis" ]; then
    echo "Downloading ONLY analysis results..."
    FOUND=false
    for f in "${FARR[@]}"; do
      if [[ "$f" == *"analysis_plots"* ]] || [[ "$f" == *"EXP_DB_Dump"* ]] || [[ "$f" == *"EXP_Reconstructed"* ]] || [[ "$f" == *"reconstructed_logs"* ]]; then
          echo "Downloading $f"
          gcloud compute scp --recurse "$SERVER_VM:$f" ./downloads/ --zone="$SERVER_ZONE" --quiet
          FOUND=true
      fi
  done
  if [ "$FOUND" = false ]; then
      echo "No analysis_plots directory found on server."
  fi
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
