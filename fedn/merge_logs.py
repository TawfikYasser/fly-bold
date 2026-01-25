import json
import sys

def merge_logs(reconstructed_path, dump_path, output_path):
    print(f"Loading reconstructed logs from {reconstructed_path}...")
    with open(reconstructed_path, 'r') as f:
        reconstructed = json.load(f)

    print(f"Loading DB dump from {dump_path}...")
    with open(dump_path, 'r') as f:
        dump = json.load(f)

    # Create a map of dump rounds by round_id
    dump_map = {str(r['round_id']): r for r in dump}

    print("Merging data...")
    merged_count = 0
    for round_obj in reconstructed:
        rid = str(round_obj['round_id'])
        if rid in dump_map:
            dump_round = dump_map[rid]
            
            # Extract Duration
            # Try combiners[0]['time_exec_training']
            combiners = dump_round.get('combiners', [])
            duration = 0
            if combiners and len(combiners) > 0:
                duration = combiners[0].get('time_exec_training', 0)
            
            if duration == 0:
                # Fallback to time_commit if training time missing
                duration = dump_round.get('round_data', {}).get('time_commit', 0)
            
            round_obj['round_duration'] = duration
            
            # Extract Server Timings and Config
            if 'round_data' in dump_round:
                round_obj['server_metrics'] = dump_round['round_data']
            
            if 'round_config' in dump_round:
                round_obj['round_config'] = dump_round['round_config']
                
            print(f"Round {rid}: Updated duration to {duration:.2f}s, added server metrics.")
            merged_count += 1
        else:
            print(f"Round {rid}: No matching data in dump!")

    print(f"Merged {merged_count} rounds.")
    
    print(f"Saving merged logs to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(reconstructed, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 merge_logs.py <reconstructed.json> <dump.json> <output.json>")
        sys.exit(1)
    
    merge_logs(sys.argv[1], sys.argv[2], sys.argv[3])
