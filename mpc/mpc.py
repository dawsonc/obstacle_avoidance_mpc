"""Define functions to solve MPC problems"""
from typing import Any, Dict, List, Tuple

import casadi
import numpy as np

from mpc.costs import RunningCostFunction, TerminalCostFunction
from mpc.dynamics_constraints import (
    add_dynamics_constraints,
    DynamicsFunction,
)
from mpc.obstacle_constraints import (
    make_obstacle_cost,
    ObstacleFunction,
)


def solve_MPC_problem(
    opti: casadi.Opti,
    x0_variables: casadi.MX,
    u0_variables: casadi.MX,
    current_state: np.ndarray,
    verbose: bool = False,
):
    """Solve a receding-horizon MPC problem from the given state.

    Modifies opti; a copy should be passed

    args:
        opti: an optimization problem
        x0_variables: the variables representing the start state of the MPC problem
        u0_variables: the variables representing the control input in the MPC problem at
            the first timestep in the MPC problem
        current_state: the current state of the system
        verbose: if True, print the results of the optimization. Defaults to False
    returns:
        - a boolean indicating whether the solver was successful or not
        - the optimal control action (not meaningful if the first return value is False)
    """
    # Add a constraint for the start state
    for i in range(x0_variables.shape[1]):
        opti.subject_to(x0_variables[0, i] == current_state[i])

    # Define optimizer setting
    p_opts: Dict[str, Any] = {"expand": True}
    s_opts: Dict[str, Any] = {"max_iter": 1000}
    if not verbose:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"

    # Create a solver and solve!
    opti.solver("ipopt", p_opts, s_opts)
    try:
        solution = opti.solve()
    except RuntimeError:
        return False, np.zeros(u0_variables.shape)

    # Return the control input at the first timestep
    return solution.stats()["success"], solution.value(u0_variables)


def construct_MPC_problem(
    n_states: int,
    n_controls: int,
    horizon: int,
    dt: float,
    dynamics_fn: DynamicsFunction,
    obstacle_fns: List[Tuple[ObstacleFunction, float]],
    running_cost_fn: RunningCostFunction,
    terminal_cost_fn: TerminalCostFunction,
    control_bounds: List[float],
) -> casadi.Opti:
    """
    Define a casadi Opti object containing a receding-horizon obstacle-avoidance MPC
    problem.

    args:
        n_states: number of states in the dynamical system
        n_controls: number of control inputs in the dynamical system
        horizon: number of steps to consider
        dt: timestep to use for integration
        dynamics_fn: specifies the dynamics of the system
        obstacle_fns: a list of tuples containing functions giving the signed distance to
            each in the scene and the floating distance by which we must avoid that
            obstacle.
        collision_margin: the distance by which we should avoid collision
        running_cost: the cost function to minimize at each step
        terminal_cost: the cost function to minimize at the final state
        control_bounds: list of maximum absolute value for all control inputs
    returns:
        - the MPC opti problem
        - the variables for the initial state
        - the variables for the initial control action
    """
    # Create the problem object
    opti = casadi.Opti()

    # Create the states and control inputs for direct transcription
    x = opti.variable(horizon + 1, n_states)  # horizon + final state
    u = opti.variable(horizon, n_controls)

    # Create the objective and add it to the problem
    cost = terminal_cost_fn(x[-1, :])
    for t in range(horizon):
        cost += running_cost_fn(x[t, :], u[t, :])

    for obstacle_fn, collision_margin in obstacle_fns:
        for t in range(1, horizon + 1):
            cost += make_obstacle_cost(obstacle_fn, x[t, :], collision_margin)

    opti.minimize(cost)

    # Add the control bounds constraints
    for control_idx, bound in enumerate(control_bounds):
        for t in range(horizon):
            opti.subject_to(u[t, control_idx] <= bound)
            opti.subject_to(u[t, control_idx] >= -bound)

    # No need to add initial state constraints; that will be added when we solve
    # the problem

    # Add dynamics constraints via direct transcription
    for t in range(horizon):
        add_dynamics_constraints(opti, dynamics_fn, x[t, :], u[t, :], x[t + 1, :], dt)

    # Return the MPC problem and the initial state and control variables
    return opti, x[0, :], u[0, :]
