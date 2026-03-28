#!/bin/bash

# Download Files from Server VM
set -e

source vm-info.txt

echo "=== Flybold File Downloader ==="
echo ""
echo "Listing files on server VM: $SERVER_VM"
echo ""

# List available files
FILES=$(gcloud compute ssh $SERVER_VM --zone=$SERVER_ZONE --command="
    find /app -maxdepth 2 -type f \( -name '*.json' -o -name '*.hpo' -o -name '*.log' -o -name '*.pt' -o -name '*.pth' -o -name '*.txt' \) 2>/dev/null | sort
")

if [ -z "$FILES" ]; then
    echo "No files found"
    exit 0
fi

# Display menu
echo "Available files:"
echo ""
i=1
declare -a FILE_ARRAY
while IFS= read -r file; do
    echo "[$i] $file"
    FILE_ARRAY[$i]=$file
    ((i++))
done <<< "$FILES"

echo ""
read -p "Enter file number(s) to download (comma-separated, or 'all'): " selection

# Create local download directory
mkdir -p ./downloads
echo ""

if [ "$selection" = "all" ]; then
    for file in "${FILE_ARRAY[@]}"; do
        echo "Downloading: $file"
        gcloud compute scp $SERVER_VM:$file ./downloads/ --zone=$SERVER_ZONE --quiet
    done
else
    IFS=',' read -ra NUMS <<< "$selection"
    for num in "${NUMS[@]}"; do
        num=$(echo $num | xargs)  # trim whitespace
        if [ -n "${FILE_ARRAY[$num]}" ]; then
            file="${FILE_ARRAY[$num]}"
            echo "Downloading: $file"
            gcloud compute scp $SERVER_VM:$file ./downloads/ --zone=$SERVER_ZONE --quiet
        else
            echo "Invalid selection: $num"
        fi
    done
fi

echo ""
echo "Files downloaded to ./downloads/"
ls -lh ./downloads/