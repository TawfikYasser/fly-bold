# YOLOv5 Federated Object Detection (FedML)

## Description
This project demonstrates federated learning for object detection using YOLOv5 integrated with the FedML framework. Multiple clients can collaboratively train a YOLOv5 model without sharing raw data, making it suitable for privacy-preserving training.

This setup supports simulations of horizontal federated learning, allowing researchers and developers to experiment with federated model training workflows efficiently.

---

## Features
- Federated learning with FedML (simulation and potential cross-silo scenarios)
- YOLOv5 integration for object detection
- Custom dataset support (COCO128 by default)
- Easy simulation of multiple clients
- Configurable training parameters including optimizer, learning rate, and communication rounds
- Modular project structure for easy extension

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

-> model/ folder is ignored in git, so you need to clone yolov5 repo manually.
-> I'm using yolov5 6.2 version, so clone it inside the project folder.

2. Create the conda environment and activate it:
```bash
conda env create -f environment.yml
conda activate fedml-env
```

3. Install additional dependencies (OpenCV, Pandas, Matplotlib, Seaborn, Addict) and download the dataset:
```bash
sh bootstrap.sh
```
This script will also create the `~/fedcv_data` directory and download the COCO128 dataset.

---

## Usage

### Run simulation with a specified number of clients:
```bash
bash run_simulation.sh 2  # Replace 2 with the number of clients
```

### Run server independently:
```bash
bash run_server.sh
```

### Run client independently:
```bash
bash run_client.sh
```

### Configuration
- `config/simulation/fedml_config.yaml`: Contains all the configurable parameters for the simulation. This includes training type, dataset paths, model configurations, federated learning parameters (client number, communication rounds, optimizer, learning rate, batch size), device and backend configurations (MPI, GPU), and optional tracking with WandB.

- `bootstrap.sh`: A shell script that installs additional Python dependencies (OpenCV, Pandas, Matplotlib, Seaborn, Addict) required for data processing and visualization. It also creates the data folder and downloads the COCO128 dataset using the provided download script.

- `run_simulation.sh`: A shell script to run a federated learning simulation. It takes the number of worker clients as an argument, sets the `PYTHONPATH` to include the project root and YOLOv5 code, prepares the MPI host file, and launches the training process using `mpirun` with the specified number of processes.

- `main_fedml_object_detection.py`: The main Python script that initializes the FedML framework, sets up the device (CPU/GPU), initializes the YOLO model and dataset via `init_yolo`, and starts the federated training with the `FedMLRunner` class using `YOLOAggregator` to handle aggregation of client updates.

---

## Project Structure
```
YOLOv5COCO128/
│
├── config/        # Configuration files including simulation settings and hyperparameters
│   └── simulation/fedml_config.yaml
├── data/          # Dataset folder (e.g., COCO128)
├── logs/          # Training logs
├── model/         # YOLOv5 model code (ignored in Git)
├── trainer/       # Training aggregation and client-side logic (YOLOAggregator)
├── runs/          # Training outputs and checkpoints
├── main_fedml_object_detection.py  # Entry point for federated training
├── environment.yml  # Conda environment specification
├── bootstrap.sh   # Script to install dependencies and prepare dataset
├── run_simulation.sh  # Script to start federated simulation
└── README.md      # Project documentation
```