"""Test the obstacle avoidance MPC for a quadrotor"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from NNet.converters.onnx2nnet import onnx2nnet

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
)
from mpc.dynamics_constraints import quad6d_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_nn
from mpc.network_utils import pytorch_to_nnet

from mpc.nn import PolicyCloningModel


radius = 1.0
margin = 0.1
center = [0.0, 1e-5, 2.5]
n_states = 6
n_controls = 3
horizon = 5
dt = 0.1
dynamics_fn = quad6d_dynamics

state_space = [
    (-2.0, 2.0),  # px
    (-2.0, 2.0),  # py
    (0.5, 4.5),  # pz
    (-1.5, 1.5),  # vx
    (-1.5, 1.5),  # vy
    (-1.5, 1.5),  # vz
]


def define_quad_mpc_expert():
    # -------------------------------------------
    # Define the MPC problem
    # -------------------------------------------

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

    # Wrap the MPC problem to accept a tensor and return a tensor
    max_tries = 10

    def mpc_expert(current_state: torch.Tensor) -> torch.Tensor:
        tries = 0
        success = False
        x_guess = None
        u_guess = None
        while not success and tries < max_tries:
            success, control_output, _, _ = solve_MPC_problem(
                opti.copy(),
                x0_variables,
                u0_variables,
                current_state.detach().numpy(),
                x_variables=x_variables,
                u_variables=u_variables,
                x_guess=x_guess,
                u_guess=u_guess,
            )
            tries += 1

        if not success:
            print(f"failed after {tries} tries")

        return torch.from_numpy(control_output)

    return mpc_expert


def clone_quad_mpc(train=True):
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_quad_mpc_expert()
    hidden_layers = 4
    hidden_layer_width = 32
    cloned_policy = PolicyCloningModel(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space,
        # load_from_file="mpc/tests/data/cloned_quad_policy_weight_decay.pth",
    )

    n_pts = int(2e4)
    n_epochs = 1000
    learning_rate = 1e-3
    if train:
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path="mpc/tests/data/cloned_quad_policy_weight_decay.pth",
        )

    return cloned_policy


def simulate_and_plot(policy):
    # -------------------------------------------
    # Plot a rollout of the cloned
    # -------------------------------------------
    ys = np.linspace(-0.5, 0.5, 8)
    xs = np.linspace(-1.0, -0.8, 8) - radius
    x0s = []
    for y in ys:
        for x in xs:
            x0s.append(np.array([x, y, center[2], 0.0, 0.0, 0.0]))

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    n_steps = 50
    for x0 in x0s:
        _, x, u = simulate_nn(
            policy,
            x0,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
        )

        # Plot it (in x-y plane)
        ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        ax_xz.plot(x0[0], x0[2], "ro")
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


def save_to_onnx(policy):
    """Save to an onnx file"""
    save_path = "mpc/tests/data/cloned_quad_policy_weight_decay.onnx"
    pytorch_to_nnet(policy, n_states, n_controls, save_path)

    input_mins = [state_range[0] for state_range in state_space]
    input_maxes = [state_range[1] for state_range in state_space]
    means = [0.5 * (state_range[0] + state_range[1]) for state_range in state_space]
    means += [0.0]
    ranges = [state_range[1] - state_range[0] for state_range in state_space]
    ranges += [1.0]
    onnx2nnet(save_path, input_mins, input_maxes, means, ranges)


if __name__ == "__main__":
    policy = clone_quad_mpc(train=True)
    save_to_onnx(policy)
    simulate_and_plot(policy)
