#!/bin/bash

declare -A CLIENTS=(
  [flybold-client-1]=us-central1-a
  [flybold-client-2]=us-central1-b
  [flybold-client-3]=us-central1-c
  [flybold-client-4]=us-central1-f
  [flybold-client-5]=us-central1-a
)

for VM in "${!CLIENTS[@]}"; do
  ZONE=${CLIENTS[$VM]}
  echo "Starting logs for $VM ($ZONE)"
  
  gcloud compute ssh "$VM" \
    --zone="$ZONE" \
    --command="sudo docker compose -f /app/docker-compose.yml logs -f --tail=20" \
    | sed "s/^/[$VM] /" &

done

wait
echo "All client logs streaming started."