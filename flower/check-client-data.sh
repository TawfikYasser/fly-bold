#!/bin/bash
set -e

# Make sure VM info exists
if [ ! -f "vm-info.txt" ]; then
  echo "[ERROR] vm-info.txt not found. Run infrastructure setup first."
  exit 1
fi

source vm-info.txt

echo "=============================================="
echo "Checking COCO dataset on all client VMs"
echo "=============================================="

for i in {1..5}; do
  CLIENT_VM_VAR="CLIENT_${i}_VM"
  CLIENT_ZONE_VAR="CLIENT_${i}_ZONE"

  CLIENT_VM=${!CLIENT_VM_VAR}
  CLIENT_ZONE=${!CLIENT_ZONE_VAR}

  echo ""
  echo "========== $CLIENT_VM =========="

  gcloud compute ssh "$CLIENT_VM" --zone="$CLIENT_ZONE" --command="
    set -e

    DATASET_ROOT=/app/datasets/coco

    echo '[INFO] Checking directory structure...'
    for d in images/train2017 images/val2017 labels/train2017 labels/val2017; do
      if [ ! -d \"\$DATASET_ROOT/\$d\" ]; then
        echo '[ERROR] Missing directory:' \$DATASET_ROOT/\$d
        exit 1
      fi
    done
    echo '[OK] Directory structure exists'

    echo ''
    echo '[INFO] Counting files'
    echo -n 'Train images: '
    ls \$DATASET_ROOT/images/train2017 | wc -l
    echo -n 'Train labels: '
    ls \$DATASET_ROOT/labels/train2017 | wc -l
    echo -n 'Val images:   '
    ls \$DATASET_ROOT/images/val2017 | wc -l
    echo -n 'Val labels:   '
    ls \$DATASET_ROOT/labels/val2017 | wc -l

    echo ''
    echo '[INFO] Disk usage'
    du -sh \$DATASET_ROOT

    echo ''
    echo '[INFO] Sample files (train images)'
    ls -lh \$DATASET_ROOT/images/train2017 | head -n 5

    echo ''
    echo '[SUCCESS] Dataset looks OK on this VM'
  "
done

echo ""
echo "=============================================="
echo "All client dataset checks completed"
echo "=============================================="
