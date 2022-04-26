"""Define dynamics for systems under study"""
from typing import Callable, List, Union

import casadi
import numpy as np


# Define a function type for dynamics
DynamicsFunction = Callable[[casadi.MX, casadi.MX], List[casadi.MX]]

# We want our dynamics to be compatible with both casadi and numpy inputs
Variable = Union[casadi.MX, np.ndarray]


def add_dynamics_constraints(
    opti: casadi.Opti,
    dynamics: DynamicsFunction,
    x_now: casadi.MX,
    u_now: casadi.MX,
    x_next: casadi.MX,
    dt: float,
):
    """Add constraints for the dynamics to the given optimization problem

    args:
        opti: the casadi optimization problem to add the constraints to
        dynamics: the function specifying the dynamics (takes current state and control
            input and returns the state derivatives).
        x_now: current state
        u_now: current control input
        x_next: next state
        dt: timestep for Euler integration
    """
    # Get the state derivative
    dx_dt = dynamics(x_now, u_now)

    # Add a constraint for each state variable
    for i in range(x_now.shape[1]):
        xi_next, xi_now, dxi_dt = x_next[i], x_now[i], dx_dt[i]
        opti.subject_to(xi_next == xi_now + dt * dxi_dt)


def dubins_car_dynamics(x: Variable, u: Variable) -> List[Variable]:
    """
    Dubins car dynamics, implemented using Casadi variables

    args:
        x: state variables
        u: control inputs
    """
    # unpack variables
    theta = x[2]
    v = u[0]
    omega = u[1]

    # compute derivatives
    xdot = [
        v * casadi.cos(theta),
        v * casadi.sin(theta),
        omega,
    ]

    return xdot


def quad6d_dynamics(x: Variable, u: Variable):
    """
    Nonlinear 6D quadrotor dynamics, implemented using Casadi variables

    args:
        x: state variables
        u: control inputs
    """
    # unpack variables
    vx = x[3]
    vy = x[4]
    vz = x[5]
    theta = u[0]
    phi = u[1]
    tau = u[2]
    theta, phi, tau = u

    # compute derivatives
    g = 9.81
    xdot = [
        vx,
        vy,
        vz,
        g * casadi.tan(theta),
        -g * casadi.tan(phi),
        tau,
    ]

    return xdot
