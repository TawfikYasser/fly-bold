import sys
import os
import pymongo
import time

# Prevent local 'fedn' folder from shadowing the installed package
cwd = os.getcwd()
if cwd in sys.path:
    sys.path.remove(cwd)

from fedn import APIClient

# Configuration Defaults
MONGO_HOST = os.environ.get('MONGO_HOST', 'localhost')
MONGO_PORT = int(os.environ.get('MONGO_PORT', 6534))
MONGO_USER = os.environ.get('MONGO_USER', 'fedn_admin')
MONGO_PASSWORD = os.environ.get('MONGO_PASSWORD', 'password')
NETWORK_ID = os.environ.get('NETWORK_ID', 'fedn-network')

def verify_rounds_in_db(expected_rounds):
    print(f"\nVerifying {expected_rounds} rounds in Database...")
    try:
        client = pymongo.MongoClient(
            host=MONGO_HOST,
            port=MONGO_PORT,
            username=MONGO_USER,
            password=MONGO_PASSWORD
        )
        db = client[NETWORK_ID]
        
        rounds_coll = db['control.rounds']
        
        # Find rounds sorted by ID
        rounds = list(rounds_coll.find().sort("round_id", -1).limit(expected_rounds))
        
        if len(rounds) < expected_rounds:
            print(f"WARNING: Found only {len(rounds)} rounds in DB, expected {expected_rounds}.")
        else:
            print(f"Confirmed {len(rounds)} rounds exist in DB.")

        success_count = 0
        for r in rounds:
            status = r.get('status', 'Unknown')
            rid = r.get('round_id')
            print(f" - Round {rid}: Status '{status}'")
            if status in ['Finished', 'Success']:
                success_count += 1
                
        if success_count == expected_rounds:
            print(f"SUCCESS: All {expected_rounds} rounds completed successfully in DB.")
        else:
            print(f"WARNING: Only {success_count}/{expected_rounds} rounds are marked as Finished/Success.")
            
    except Exception as e:
        print(f"DB Verification Failed: {e}")
        print("Ensure MONGO_HOST is set correctly (default: localhost).")

def run_simulation():
    print("Initializing APIClient...")
    client = APIClient(host="localhost", port=8092)
    
    # Wait for controller to be ready (rudimentary check)
    max_retries = 10
    for i in range(max_retries):
        try:
            status = client.get_controller_status()
            print(f"Controller status: {status}")
            break
        except Exception as e:
            print(f"Waiting for controller ({e})... {i+1}/{max_retries}")
            time.sleep(5)
            
    print("Setting active model...")
    # Check if seed.npz exists
    if not os.path.exists("seed.npz"):
        print("Error: seed.npz not found!")
        return

    try:
        client.set_active_model("seed.npz")
        print("Active model set.")
    except Exception as e:
        print(f"Error setting active model: {e}")
        # Might fail if model already exists or connection issue
        pass

    rounds_to_run = 3
    print(f"Starting session ({rounds_to_run} rounds)...")
    try:
        result = client.start_session(rounds=rounds_to_run, round_timeout=120)
        print(f"Session started: {result}")
        
        # Poll for completion
        print("Waiting for session to complete...")
        while True:
            time.sleep(5)
            status = client.get_controller_status()
            state = status.get('state')
            print(f"Current state: {state}")
            if state == 'idle':
                print("Session completed!")
                break
        
        # verify
        verify_rounds_in_db(rounds_to_run)
                
    except Exception as e:
        print(f"Error starting/monitoring session: {e}")

if __name__ == "__main__":
    run_simulation()
