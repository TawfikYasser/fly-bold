# Failure Orchestrator Setup Guide

This guide explains how to generate failure plans locally, upload them
to the server VM, and run the failure orchestrator reliably using
**tmux**.

------------------------------------------------------------------------

## 1. Generate Failure Plans Locally

Failure plans simulate client failures during federated learning
experiments.\
The following commands generate different failure scenarios based on the
percentage of failing clients.

### Commands

``` bash
python3 fl_failure_orchestrator.py generate-plan --failure-pct 25
python3 fl_failure_orchestrator.py generate-plan --failure-pct 50
python3 fl_failure_orchestrator.py generate-plan --failure-pct 75
```

### Explanation

-   `generate-plan` creates a failure plan file.
-   `--failure-pct` specifies the percentage of clients that will fail.
-   Generated plans are stored inside the `failure_plans/` directory.

Typical plans:

  Failure Percentage   Description
  -------------------- -------------------------
  25%                  Light failure scenario
  50%                  Medium failure scenario
  75%                  Heavy failure scenario

------------------------------------------------------------------------

## 2. Upload Files to the Server VM

After generating the failure plans locally, upload the script and plans
to the server virtual machine.

### Upload Script

``` bash
gcloud compute scp fl_failure_orchestrator.py flybold-server:/app/ --zone=us-central1-a
```

### Upload Failure Plans

``` bash
gcloud compute scp -r failure_plans/ flybold-server:/app/ --zone=us-central1-a
```

### Explanation

-   `gcloud compute scp` copies files to the VM.
-   `flybold-server` is the VM instance name.
-   `/app/` is the destination directory.
-   `--zone=us-central1-a` specifies the VM zone.

------------------------------------------------------------------------

## 3. Run the Script Using tmux

Running the orchestrator inside **tmux** ensures that the process keeps
running even if the SSH connection is closed.

### Install tmux

``` bash
sudo apt install tmux -y
```

### Explanation

`tmux` allows running long processes in persistent terminal sessions.

------------------------------------------------------------------------

## 4. Create a tmux Session

Create a named session and run the orchestrator inside it.

``` bash
tmux new-session -s orchestrator
```

Then run:

``` bash
python3 fl_failure_orchestrator.py run --failure-pct 25
```

### Explanation

-   `new-session -s orchestrator` creates a session named
    **orchestrator**
-   The script will continue running even if SSH disconnects.

------------------------------------------------------------------------

## 5. Reconnect to the tmux Session

If the SSH connection drops, the script will continue running.

Reconnect using:

``` bash
tmux attach -t orchestrator
```

### Explanation

This command reconnects to the running session so you can monitor the
script.
