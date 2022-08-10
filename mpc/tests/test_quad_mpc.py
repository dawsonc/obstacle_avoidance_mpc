"""Test the obstacle avoidance MPC for a quadrotor"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
    squared_error_terminal_cost,
)
from mpc.dynamics_constraints import quad6d_dynamics
from mpc.mpc import construct_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_mpc


radius = 1.0
margin = 0.1
center = [0.0, 1e-3, 2.5]


def test_quad_mpc(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a test of obstacle avoidance MPC with a quad and return the results"""
    # -------------------------------------------
    # Define the problem
    # -------------------------------------------
    n_states = 6
    n_controls = 3
    horizon = 5
    dt = 0.1

    # Define dynamics
    dynamics_fn = quad6d_dynamics

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center), margin)]

    # Define costs to make the quad go to the right
    x_goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]), 1 * np.eye(3)
    )
    terminal_cost_fn = lambda x: distance_travelled_terminal_cost(x, goal_direction)
    # terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

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
    n_steps = 40
    return simulate_mpc(
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
    # ys = np.linspace(-0.5, 0.5, 3)
    # xs = np.linspace(-1.0, -0.3, 3)
    ys = np.linspace(-0.5, 0.5, 3)
    xs = np.linspace(-1.0, -0.8, 3)
    x0s = []
    for y in ys:
        for x in xs:
            x0s.append(np.array([x, y, center[2], 0.0, 0.0, 0.0]))

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)
    # ax_xz.plot([], [], "ro", label="Start")

    for x0 in x0s:
        # Run the MPC
        _, x, u = test_quad_mpc(x0)

        # Plot it (in x-y plane)
        # ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        # ax_xz.plot(x0[0], x0[2], "ro")
        ax_xz.plot(x[:, 0], x[:, 2], "r-", linewidth=1)

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    obs_z = radius * np.sin(theta) + center[2]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    margin_z = (radius + margin) * np.sin(theta) + center[2]
    ax_xy.plot(obs_x, obs_y, "k-")
    ax_xy.plot(margin_x, margin_y, "k:")
    ax_xz.plot(obs_x, obs_z, "k-", label="Obstacle")
    ax_xz.plot(margin_x, margin_z, "k:", label="Safety margin")

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-5.0, 5.0])
    ax_xy.set_ylim([-5.0, 5.0])
    ax_xz.set_xlim([-5.0, 5.0])
    ax_xz.set_ylim([-5.0, 5.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.show()


def plot_sdf():
    sdf_fn = lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center)
    xs = np.linspace(-1.0, 1.0, 200)
    ys = np.linspace(-1.0, 1.0, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            state = np.array([[x, y, 0.0, 0.0, 0.0, 0.0]])
            sdf = sdf_fn(state)
            Z[j, i] = np.exp(1e2 * (margin - sdf))

    plt.contourf(X, Y, Z)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    run_and_plot_quad_mpc()
    # plot_sdf()
