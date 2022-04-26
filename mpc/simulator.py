"""Define functions for simulating the performance of an MPC controller"""
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

    # Simulate
    t_range = tqdm(range(n_steps - 1))
    t_range.set_description("Simulating")  # type: ignore
    for tstep in t_range:
        # Solve the MPC problem to get the next state
        success, u_current = solve_MPC_problem(
            opti.copy(),
            x0_variables,
            u0_variables,
            x[tstep],
            verbose,
        )

        if success:
            u[tstep] = u_current
        else:
            n_infeasible += 1

        # Update the state using the dynamics
        dx_dt = dynamics_fn(x[tstep], u[tstep])
        for i in range(n_states):
            x[tstep + 1, i] = x[tstep, i] + dt * np.array(dx_dt[i])

    print(f"{n_infeasible} infeasible steps")

    return t, x, u
