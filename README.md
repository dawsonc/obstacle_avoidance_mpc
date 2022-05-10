# Simple MPC for obstacle avoidance

This repository implements nonlinear MPC using the Casadi optimization library. It also includes code for compressing that MPC policy into a neural network control policy.

# Installation

```
git clone --recurse-submodules git@github.com:dawsonc/obstacle_avoidance_mpc.git
cd obstacle_avoidance_mpc
conda env create -n obstacle_avoidance_mpc python=3.9
pip install -e .
```
