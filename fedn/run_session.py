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

def verify_rounds_in_db(expected_rounds, session_id=None):
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

        query = {}
        if session_id:
            query["round_config.session_id"] = session_id

        # Find rounds for this session sorted by ID
        rounds = list(rounds_coll.find(query).sort("round_id", -1).limit(expected_rounds))
        
        if len(rounds) < expected_rounds:
            print(f"WARNING: Found only {len(rounds)} rounds in DB, expected {expected_rounds}.")
        else:
            print(f"Confirmed {len(rounds)} rounds exist in DB.")

        success_count = 0
        for r in rounds:
            status = r.get('status', 'Unknown')
            rid = r.get('round_id')
            sid = r.get('round_config', {}).get('session_id')
            print(f" - Round {rid} (session {sid}): Status '{status}'")
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

    model_id = None
    try:
        # set_active_model often returns the result which might contain model_id
        # If not, we should query for it
        response = client.set_active_model("seed.npz")
        print(f"Active model set: {response}")
        
        # Try to get the active model to confirm and get ID
        active_model = client.get_active_model()
        if active_model:
             model_id = active_model.get('model', active_model.get('id'))
             try:
                if model_id:
                    print(f"Confirmed active model ID: {model_id}")
             except Exception as e:
                print(f"Error checking active model: {e}")
             
    except Exception as e:
        print(f"Error setting active model: {e}")
        return

    # Create and upload dummy package
    import tarfile
    if not os.path.exists("package.tar.gz"):
        with tarfile.open("package.tar.gz", "w:gz") as tar:
            tar.add("run_session.py") # Add self as dummy content
    
    try:
        print("Uploading compute package...")
        # Upload package to ensure controller has a context
        response = client.set_active_package("package.tar.gz", "numpyhelper", "fedn-package")
        print(f"Package uploaded: {response}")
    except Exception as e:
        print(f"Error uploading package: {e}")

    # Wait for clients to connect
    print("Waiting for clients to connect...")
    min_clients = 2
    max_client_wait = 600 # 10 minutes wait for installation
    start_wait = time.time()
    
    while True:
        try:
            # Check for ACTIVE (online) clients
            clients = client.get_active_clients()
            network_id = os.environ.get('NETWORK_ID', 'fedn-network') 
            
            # Adjust based on API response structure. Assuming list of clients.
            # get_active_clients usually filters by status='online'
            client_count = len(clients['result']) if isinstance(clients, dict) and 'result' in clients else len(clients)
            
            print(f"Connected clients: {client_count}/{min_clients}")
            
            # Debug: Print client details
            if isinstance(clients, dict) and 'result' in clients:
                for c in clients['result']:
                    print(f" - Client {c.get('name')} ({c.get('client_id')}): {c}")

            # Debug: Check combiners
            try:
                combiners = client.get_combiners()
                combiner_count = len(combiners['result']) if isinstance(combiners, dict) and 'result' in combiners else len(combiners)
                print(f"Active Combiners: {combiner_count}")
            except Exception as e:
                print(f"Error getting combiners: {e}")

            if client_count >= min_clients:
                break
        except Exception as e:
             print(f"Error listing clients: {e}")

        if time.time() - start_wait > max_client_wait:
            print("Timeout waiting for clients. Starting session anyway (might fail).")
            break
        time.sleep(10)

    rounds_to_run = 5
    print(f"Starting session ({rounds_to_run} rounds) with model {model_id}...")
    try:
        if model_id:
            result = client.start_session(rounds=rounds_to_run, round_timeout=7200, model_id=model_id)
        else:
            result = client.start_session(rounds=rounds_to_run, round_timeout=7200)
            
        print(f"Session started: {result}")
        session_id = result.get("session_id") or result.get("id")
        if not session_id:
            print("Warning: session_id missing in start_session response; database verification will be broad.")
        
        # Poll for completion
        print("Waiting for session to complete...")
        while True:
            time.sleep(5)
            status = client.get_controller_status()
            state = status.get('state')
            print(f"Current state: {state}")

            if session_id and client.session_is_finished(session_id):
                print("Session completed!")
                break

            # Fallback: if controller is idle but session status could not be fetched
            if state == 'idle' and not session_id:
                print("Controller idle; assuming session completed.")
                break
        
        # verify
        verify_rounds_in_db(rounds_to_run, session_id=session_id)
                
    except Exception as e:
        print(f"Error starting/monitoring session: {e}")

if __name__ == "__main__":
    run_simulation()
