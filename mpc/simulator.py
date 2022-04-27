"""Define functions for simulating the performance of an MPC controller"""
from typing import Optional

import casadi
import numpy as np
from tqdm import tqdm

from mpc.dynamics_constraints import DynamicsFunction
from mpc.mpc import solve_MPC_problem


def simulate(
    opti: casadi.Opti,
    x0_variables: casadi.MX,
    u0_variables: casadi.MX,
    x0: np.ndarray,
    dt: float,
    dynamics_fn: DynamicsFunction,
    n_steps: int,
    verbose: bool = False,
    x_variables: Optional[casadi.MX] = None,
    u_variables: Optional[casadi.MX] = None,
    x_guess: Optional[np.ndarray] = None,
    u_guess: Optional[np.ndarray] = None,
    substeps: int = 1,
):
    """
    Simulate a rollout of the MPC controller specified by the given optimization problem.

    args:
        opti: an optimization problem
        x0_variables: the variables representing the start state of the MPC problem
        u0_variables: the variables representing the control input in the MPC problem at
            the first timestep in the MPC problem
        x0: the starting state of the system
        dt: the timestep used for integration
        dynamics_fn: the dynamics of the system
        n_steps: how many total steps to simulate
        verbose: if True, print the results of the optimization. Defaults to False
        x_variables, u_variables, x_guess, and u_guess allow you to provide an initial
            guess for x and u (often from the previous solution). If not provided, use
            the default casadi initial guess (zeros).
        substeps: how many smaller substeps to use for the integration
    returns:
        - an np.ndarray of timesteps
        - an np.ndarray of states
        - an np.ndarray of control inputs
    """
    n_states = x0_variables.shape[1]
    n_controls = u0_variables.shape[1]
    # Create some arrays to store the results
    t = dt * np.linspace(0, dt * (n_steps - 1), n_steps)
    assert t.shape[0] == n_steps
    x = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps - 1, n_controls))

    # Set the initial conditions
    x[0] = x0

    # Track how often the MPC problem is infeasible
    n_infeasible = 0

    # Initialize empty guesses for the MPC problem
    x_guess: Optional[np.ndarray] = None
    u_guess: Optional[np.ndarray] = None

    # Simulate
    t_range = tqdm(range(n_steps - 1))
    t_range.set_description("Simulating")  # type: ignore
    for tstep in t_range:
        # Solve the MPC problem to get the next state
        success, u_current, x_guess, u_guess = solve_MPC_problem(
            opti.copy(),
            x0_variables,
            u0_variables,
            x[tstep],
            verbose,
            x_variables,
            u_variables,
            x_guess,
            u_guess,
        )

        if success:
            u[tstep] = u_current
        else:
            n_infeasible += 1

        # Update the state using the dynamics. Integrate at a higher frequency using
        # zero-order hold controls
        x_next = np.array(x[tstep])
        for _ in range(substeps):
            dx_dt = dynamics_fn(x_next, u[tstep])
            for i in range(n_states):
                x_next[i] = x_next[i] + dt / substeps * np.array(dx_dt[i])

        x[tstep + 1] = x_next

    print(f"{n_infeasible} infeasible steps")

    return t, x, u
