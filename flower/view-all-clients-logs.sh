#!/bin/bash
set -euo pipefail

# Each entry is "vm zone"
CLIENTS=(
  "flybold-client-1 us-central1-a"
  "flybold-client-2 us-central1-b"
  "flybold-client-3 us-central1-c"
  "flybold-client-4 us-central1-f"
  "flybold-client-5 us-central1-a"
)

echo "Starting all client logs streaming..."

for entry in "${CLIENTS[@]}"; do
  VM="${entry%% *}"
  ZONE="${entry#* }"

  echo "Starting logs for $VM ($ZONE)"

  gcloud compute ssh "$VM" \
    --zone="$ZONE" \
    --command="sudo docker compose -f /app/docker-compose.yml logs -f" \
    | sed "s/^/[$VM] /" &
done

wait