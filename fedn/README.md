# FEDn GCP Experiment Automation

This directory contains a suite of scripts to automate the deployment, execution, and analysis of Federated Learning experiments using [FEDn](https://github.com/scaleoutsystems/fedn) on Google Cloud Platform (GCP).

The system automatically provisions infrastructure, builds Docker images, deploys the FEDn server and clients across multiple VMs, runs a training session, and aggregates results.

## Overview

The automation pipeline consists of the following stages:
1.  **Infrastructure Provisioning**: Creates a custom VPC, subnets, firewall rules, and Compute Engine VMs (1 Server, 5 Clients).
2.  **Image Build**: Builds a unified Docker image for the FEDn server and clients and pushes it to Docker Hub.
3.  **Deployment**: Configures and starts the FEDn components (Controller, Combiner, MongoDB, MinIO) on the Server VM and deploys 2 Clients per Client VM.
4.  **Experiment Execution**: Remote executes the training session via `run_session.py`.
5.  **Analysis**: Runs post-experiment analysis on the server to generate plots and metrics.
6.  **Data Retrieval**: Downloads logs, models, and analysis results to your local machine.
7.  **Cleanup**: Tears down all GCP resources to avoid unnecessary costs.

## Prerequisites

Before running the scripts, ensure you have the following installed and configured:

1.  **Google Cloud SDK (`gcloud`)**:
    *   Install: [GCP SDK Installation](https://cloud.google.com/sdk/docs/install)
    *   Authenticate: `gcloud auth login`
    *   Set Project (optional, script sets it): `gcloud config set project <your-project-id>`
2.  **Docker**:
    *   Required locally for building the image (unless offloaded to a builder, but script assumes local docker command presence).
    *   **Docker Hub Account**: You need a username and password/token to push the image.
3.  **Python 3.12+** (Optional, for local debugging of python scripts).

## Usage

### 🚀 One-Click Execution

The master script `run_gcp_experiment.sh` handles the entire lifecycle from start to finish.

```bash
cd fedn
chmod +x run_gcp_experiment.sh
./run_gcp_experiment.sh
```

**What happens next?**
1.  **Credentials**: You will be prompted for your Docker Hub username (saved to `.docker_username` for future runs) and password.
2.  **Automation**: The script will execute steps 1-7 listed in the Overview.
3.  **Completion**: Results will be downloaded to `cwd` and resources will be deleted (after a confirmation prompt).

### Manual Execution (Step-by-Step)

If you wish to run specific stages manually, you can execute the individual helper scripts.

#### 1. Setup Infrastructure
Provisions the VPC, firewall rules, and virtual machines.
```bash
./setup-infrastructure.sh
```
*   **Output**: Creates `vm-info.txt` with IP addresses and zone information.

#### 2. Build & Push Image
Builds the FEDn docker image and pushes it to your registry.
```bash
# Requires Docker Hub password to be entered
./build-push-image.sh
```
*   **Input**: Reads `.docker_username` (if present) or prompts for it.
*   **Output**: Creates `docker-image-info.txt`.

#### 3. Deploy Application
Deploys the FEDn server stack and clients to the provisioned VMs.
```bash
./deploy-application.sh
```
*   **Dependencies**: Requires `vm-info.txt` and `docker-image-info.txt`.
*   **Configuration**: Automatically configures `docker-compose` files with correct IP addresses and client IDs.

#### 4. Run Experiment & Analysis
The `run_session.py` script is designed to run **on the Server VM**. You typically invoke it via SSH, as done in the master script.

To run manually on the server:
```bash
# SSH into server
gcloud compute ssh flybold-server --zone=us-central1-a

# In the server VM:
cd /app/fly-bold-fedn/fedn
sudo python3 run_session.py
sudo python3 experiment_analyzer.py
```

#### 5. Download Results
Retrieves relevant files from the Server VM to your local machine.
```bash
./download-files.sh
```
*   **Prompt**: Asks for specific file numbers or "all".

#### 6. Cleanup
**Important**: Always run this to stop billing!
```bash
./cleanup.sh
```
*   **Action**: Deletes all VMs, network resources, and local temporary files (`vm-info.txt`, etc.).

## Configuration

### Environment Variables
Key variables are defined at the top of the shell scripts. You can modify them to customize the deployment:

*   **`setup-infrastructure.sh`**:
    *   `PROJECT_ID`: GCP Project ID (Default: `inf022`).
    *   `REGION` / `ZONES`: Deployment region and zones.
    *   `SERVER_MACHINE_TYPE` / `CLIENT_MACHINE_TYPE`: VM specs.
    *   `CLIENT_COUNT`: Number of client VMs (Default: `5`).

### Local Files created
*   `.docker_username`: Caches your Docker Hub username.
*   `vm-info.txt`: Temporarily stores VM connection details.
*   `docker-image-info.txt`: Stores the tag of the currently deployed image.

## Directory Structure
*   `fedn/`: Primary directory for scripts and configuration.
*   `client/`: Client-side code and configuration (mapped to clients).
*   `config/`: Server-side configuration (Controller, Combiner, etc.).

## Troubleshooting

*   **SSH Permission Denied**: Ensure your `gcloud` SSH keys are propagated. Run `gcloud compute config-ssh`.
*   **Subnet Overlap**: If `setup-infrastructure.sh` fails with subnet overlap, either change the `NETWORK_NAME` or manually delete the existing subnets in GCP Console.
*   **Docker Rate Limits**: If pulls fail on VMs, ensure you are authenticated or using a paid Docker Hub plan if you are hitting anonymous limits (though the script uses a personal image, pulling it requires auth if private, or just standard public access).
