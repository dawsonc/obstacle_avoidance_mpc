"""Define obstacles"""
from typing import Callable, List

import casadi


# Define a function type for obstacles
ObstacleFunction = Callable[[casadi.MX], casadi.MX]


def add_obstacle_constraints(
    opti: casadi.Opti,
    obstacle_sdf_fn: ObstacleFunction,
    x_now: casadi.MX,
    margin: float,
):
    """Add the given obstacle constraints at one step of the given problem.

    args:
        opti: the casadi optimization problem to add the constraints to
        obstacle_sdf_fn: the function specifying the obstacle. Takes current state and
            returns the signed distance to the obstacle (+ is outside the obstacle,
            - is inside).
        x_now: current state
        margin: the distance by which we need to avoid the obstacle
    """
    # Get the obstacle signed distance as a casadi expression
    obstacle_sdf = obstacle_sdf_fn(x_now)

    # Constrain the distance to the obstacle to be greater than the margin
    opti.subject_to(obstacle_sdf >= margin)


def make_obstacle_cost(
    obstacle_sdf_fn: ObstacleFunction,
    x_now: casadi.MX,
    margin: float,
):
    """Make a cost encouraging collision avoidance

    args:
        obstacle_sdf_fn: the function specifying the obstacle. Takes current state and
            returns the signed distance to the obstacle (+ is outside the obstacle,
            - is inside).
        x_now: current state
        margin: the distance by which we need to avoid the obstacle
    """
    # Get the obstacle signed distance as a casadi expression
    obstacle_sdf = obstacle_sdf_fn(x_now)

    # Add a cost that is positive if we are too close
    return casadi.exp(1e2 * (margin - obstacle_sdf))


def hypersphere_sdf(
    x: casadi.MX, radius: float, indices: List[int], center: List[float]
) -> casadi.MX:
    """Defines the signed distance of a hypersphere with the given radius. The
    hypersphere is defined over the given state indices and extends to +/- infinity in
    all other state dimensions. For example, if indices = [0, 1] and x has three
    dimensions, then this defines a cylinder with the long axis in the direction of x[2].

    args:
        x: current state
        radius: the radius of the cylinder
        indices: the state indices on which this hypersphere depends
        center: the coordinates of the center of the hypersphere. Should have the same
            length as indices.
    returns:
        the signed distance to this cylinder
    """
    distances_to_center = [x[0, i] - center[i] for i in indices]
    squared_distances = [d ** 2 for d in distances_to_center]
    signed_distance = casadi.sqrt(sum(squared_distances)) - radius

    return signed_distance
