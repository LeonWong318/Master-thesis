# Toward Efficient Collaboration in Autonomous Mobile Robot Fleets: Addressing Latency and Distributed Model Predictive Control

This is the master thesis repo collaborated by Yinsong and Zihao.

## **To-do list:**
1. set up environment in linux and simulation platform
2. literature review the distributed MPC and prepare the overleaf 

# System Requirements
Before installing Python dependencies, ensure you have the following system packages installed:

- System: Ubuntu 20.04
- ROS: neotic version
- OpEn: The MPC solver we use, please following the [installation guide](https://alphaville.github.io/optimization-engine/docs/installation) from official website

# Update package list and install pip
sudo apt update
sudo apt install pip

# Install requirements
pip install -r requirements.txt

# Install build tools
sudo apt install build-essential

# Install Rust and Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build solver and run test
cd src/
python build_solver.py
python test_mpc.py
