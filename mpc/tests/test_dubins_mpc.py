"""Test the obstacle avoidance MPC for a dubins vehicle"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import dubins_car_dynamics
from mpc.mpc import construct_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate


def test_dubins_mpc(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a test of obstacle avoidance MPC with a dubins car and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    n_states = 3
    n_controls = 2
    horizon = 50
    dt = 0.1

    # Define dynamics
    dynamics_fn = dubins_car_dynamics

    # Define obstacles
    radius = 0.2
    margin = 0.1
    center = [-1.0, 0.0]
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1], center), margin)]

    # Define costs
    x_goal = np.array([0.0, 0.0, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 0.0]), 0 * np.eye(2)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [0.5, np.pi / 2]

    # Define MPC problem
    opti, x0_variables, u0_variables = construct_MPC_problem(
        n_states,
        n_controls,
        horizon,
        dt,
        dynamics_fn,
        obstacle_fns,
        running_cost_fn,
        terminal_cost_fn,
        control_bounds,
    )

    # -------------------------------------------
    # Simulate and return the results
    # -------------------------------------------
    n_steps = 50
    return simulate(
        opti,
        x0_variables,
        u0_variables,
        x0,
        dt,
        dynamics_fn,
        n_steps,
        verbose=False,
    )


def run_and_plot_dubins_mpc():
    x0s = [
        np.array([-2.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.0]),
        np.array([-2.0, 0.2, 0.0]),
        np.array([-2.0, 0.5, 0.0]),
        np.array([-2.0, -0.1, 0.0]),
        np.array([-2.0, -0.2, 0.0]),
        np.array([-2.0, -0.5, 0.0]),
    ]

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([], [], "ro", label="Start")

    for x0 in x0s:
        # Run the MPC
        _, x, u = test_dubins_mpc(x0)

        # Plot it
        ax.plot(x0[0], x0[1], "ro")
        ax.plot(x[:, 0], x[:, 1], "r-")

    # Plot obstacle
    radius = 0.2
    margin = 0.1
    center = [-1.0, 0.0]
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax.plot(obs_x, obs_y, "k-")
    ax.plot(margin_x, margin_y, "k:")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim([-2.5, 0.5])
    ax.set_ylim([-1.0, 1.0])

    ax.set_aspect("equal")

    ax.legend()

    plt.show()


if __name__ == "__main__":
    run_and_plot_dubins_mpc()
