#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    "casadi",
    "matplotlib",
    "numpy",
    "seaborn",
    "tqdm",
    "torch",
]

dev_requirements = [
    "black",
    "mypy",
    "pytest",
    "flake8",
]

setup(
    name="mpc",
    version="0.0.0",
    description="Simple nonlinear MPC for obstacle avoidance",
    author="Charles Dawson",
    author_email="cbd@mit.edu",
    url="https://github.com/dawsonc/obstacle_avoidance_mpc",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
    packages=find_packages(),
)
