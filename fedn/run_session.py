from fedn import APIClient
import time
import os

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

    print("Starting session (3 rounds)...")
    try:
        result = client.start_session(rounds=3, round_timeout=120)
        print(f"Session started: {result}")
    except Exception as e:
        print(f"Error starting session: {e}")

if __name__ == "__main__":
    run_simulation()
