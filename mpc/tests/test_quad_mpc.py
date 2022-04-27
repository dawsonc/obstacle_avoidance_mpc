"""Test the obstacle avoidance MPC for a quadrotor"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from mpc.costs import lqr_running_cost, squared_error_terminal_cost
from mpc.dynamics_constraints import quad6d_dynamics
from mpc.mpc import construct_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate


def test_quad_mpc(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a test of obstacle avoidance MPC with a quad and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    n_states = 6
    n_controls = 3
    horizon = 20
    dt = 0.1

    # Define dynamics
    dynamics_fn = quad6d_dynamics

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    radius = 0.2
    margin = 0.1
    center = [-1.0, 0.0, 0.0]
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center), margin)]

    # Define costs
    x_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 1 * np.eye(3)
    )
    terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [np.pi / 10, np.pi / 10, 2.0]

    # Define MPC problem
    opti, x0_variables, u0_variables, x_variables, u_variables = construct_MPC_problem(
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
        x_variables=x_variables,
        u_variables=u_variables,
        substeps=10,
    )


def run_and_plot_quad_mpc():
    x0s = [
        np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, 0.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, -0.2, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, -0.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, -0.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([-2.0, -0.5, 0.0, 0.0, 0.0, 0.0]),
    ]

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)
    ax_xz.plot([], [], "ro", label="Start")

    for x0 in x0s:
        # Run the MPC
        _, x, u = test_quad_mpc(x0)

        # Plot it (in x-y plane)
        ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-")
        # and in (x-z plane)
        ax_xz.plot(x0[0], x0[2], "ro")
        ax_xz.plot(x[:, 0], x[:, 2], "r-")

    # Plot obstacle
    radius = 0.2
    margin = 0.1
    center = [-1.0, 0.0, 0.0]
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax_xy.plot(obs_x, obs_y, "k-")
    ax_xy.plot(margin_x, margin_y, "k:")
    ax_xz.plot(obs_x, obs_y, "k-", label="Obstacle")
    ax_xz.plot(margin_x, margin_y, "k:", label="Safety margin")

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-2.5, 0.5])
    ax_xy.set_ylim([-1.0, 1.0])
    ax_xz.set_xlim([-2.5, 0.5])
    ax_xz.set_ylim([-1.0, 1.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.show()


if __name__ == "__main__":
    run_and_plot_quad_mpc()
