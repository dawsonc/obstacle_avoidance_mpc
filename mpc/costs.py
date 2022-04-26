"""Define cost functions"""
from typing import Callable, Optional

import casadi
import numpy as np


# Define a function type for running cost functions
# (takes a state and control and returns a cost)
RunningCostFunction = Callable[[casadi.MX, casadi.MX], casadi.MX]

# Define a function type for terminal cost functions
# (takes just a state returns a cost)
TerminalCostFunction = Callable[[casadi.MX], casadi.MX]


def lqr_running_cost(
    x: casadi.MX, u: casadi.MX, x_goal: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> casadi.MX:
    """Returns the LQR running cost.

    args:
        x: the current state
        u: the current control input
        x_goal: the goal state
        Q: the state cost weight matrix
        R: the control cost weight matrix
    returns:
        (x - x_goal)^T Q (x - x_goal) + u^T R u
    """
    x_goal = x_goal.reshape(x.shape)
    return casadi.bilin(Q, x - x_goal, x - x_goal) + casadi.bilin(R, u, u)


def zero_running_cost(
    x: casadi.MX, u: casadi.MX
) -> casadi.MX:
    """Returns a zero running cost.

    args:
        x: the current state
        u: the current control input
    returns:
        (x - x_goal)^T Q (x - x_goal) + u^T R u
    """
    return 0.0


def squared_error_terminal_cost(
    x: casadi.MX, x_goal: np.ndarray, Q: Optional[np.ndarray] = None
) -> casadi.MX:
    """Returns the squared error relative to the specified goal.

    args:
        x: the current state
        x_goal: the goal state
        Q: the state cost weight matrix
    returns:
        (x - x_goal)^T Q (x - x_goal)
    """
    x_goal = x_goal.reshape(x.shape)

    if Q is None:
        Q = np.eye(x.shape[1])

    return casadi.bilin(Q, x - x_goal, x - x_goal)
