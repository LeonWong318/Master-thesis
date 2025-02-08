# Toward Efficient Collaboration in Autonomous Mobile Robot Fleets: Addressing Latency and Distributed Model Predictive Control

This is the master thesis repository collaborated by Yinsong and Zihao.

## System Requirements

### Prerequisites
- Operating System: Ubuntu 20.04
- ROS: Noetic
- Python: 3.9
- OpEn: Optimization Engine for MPC solver ([Installation Guide](https://alphaville.github.io/optimization-engine/docs/installation))

### System Dependencies
# Update package list
sudo apt update

# Install build tools
sudo apt install build-essential pip

# Install Rust and Cargo (Required for OpEn)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env  # Load Rust environment

### Python Dependencies
# Install Python packages
pip install -r requirements.txt

## Getting Started

### Build and Test
# Build MPC solver
cd src/
python build_solver.py

# Run test
python test_mpc.py

## Project Status

### Current To-Do List
1. Set up environment in Linux and simulation platform
2. Literature review on distributed MPC and prepare the Overleaf document

## Documentation
- For detailed OpEn installation instructions, please refer to the [official documentation](https://alphaville.github.io/optimization-engine/docs/installation)
- Additional documentation and implementation details will be added as the project progresses

## Contributors
- Yinsong
- Zihao