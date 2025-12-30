#!/bin/bash

# Stream logs from all client containers
set -euo pipefail

source vm-info.txt

declare -A CLIENTS
CLIENTS["${CLIENT_1_VM}"]=${CLIENT_1_ZONE}
CLIENTS["${CLIENT_2_VM}"]=${CLIENT_2_ZONE}
CLIENTS["${CLIENT_3_VM}"]=${CLIENT_3_ZONE}
CLIENTS["${CLIENT_4_VM}"]=${CLIENT_4_ZONE}
CLIENTS["${CLIENT_5_VM}"]=${CLIENT_5_ZONE}

for VM in "${!CLIENTS[@]}"; do
  ZONE=${CLIENTS[$VM]}
  echo "Starting logs for $VM ($ZONE)"
  gcloud compute ssh "$VM" --zone="$ZONE" --command="sudo docker compose -f /app/docker-compose.yml logs -f --tail=20" | sed "s/^/[$VM] /" &
done

wait
echo "Log streaming ended"
