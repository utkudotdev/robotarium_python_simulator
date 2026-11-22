from matplotlib.patches import Circle
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.graph import topological_neighbors
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import (
    create_si_barrier_certificate_with_boundary,
)

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 6
iterations = 3000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the updated Width/Height/Spacing from MATLAB
# Randomly generated initial positions, same as formation_tracking.py
initial_conditions = np.array(
    [
        [
            -0.5,
            0.542641286533492,
            0.26729646985255084,
            -0.0029859753948191514,
            -0.603874270480752,
            -0.6617783268749291,
        ],
        [
            -0.1,
            -0.4792480506405985,
            0.24880388253861185,
            -0.27520335446915234,
            0.26053071219895874,
            -0.41166018582598973,
        ],
        [
            0.0,
            1.1646500873100205,
            2.8487544111850127,
            -3.1167849646094092,
            0.07660625016677525,
            1.9642554329689128,
        ],
    ]
)

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=False,
)

# =========================================================
# GRAPH TOPOLOGY
# =========================================================

A = np.array(
    [
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0],
    ],
    dtype=bool,
)

assert np.all(A == A.T)

L = np.diag(np.sum(A, axis=1)) - A

assert np.all(L == L.T)

# =========================================================
# FORMATION GEOMETRY
# =========================================================
# Weight matrix for desired inter-agent distances
d1 = np.sqrt(2) / 2
d2 = np.sqrt(1.25)
d3 = np.sqrt(2)
weights = np.array(
    [
        [0, d1, 0.5, 0, 0, 0.5],
        [d1, 0, d2, 0.5, d2, 0.5],
        [0.5, d2, 0, d3, 1, 0],
        [0, 0.5, d3, 0, 1, d1],
        [0, d2, 1, 1, 0, d1],
        [0.5, 0.5, 0, d1, d1, 0],
    ]
)
formation_control_gain = 10

assert np.all(weights == weights.T)

# =========================================================
# WAYPOINTS
# =========================================================
waypoints = np.array(
    [[-0.75, -0.5], [-0.5, 0.25], [0.0, -0.5], [1.0, -0.2], [0.0, 0.0]]
)
close_enough = 0.05
waypoint_state = 0
leader_robot = 0  # which robot determines waypoint transitions
waypoint_gain = 5

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Single-integrator barrier certificate for safety
si_barrier_cert = create_si_barrier_certificate_with_boundary(
    boundary_points=np.array([-1.6, 1.6, -1.0, 1.0])
)

# Mapping with specific MATLAB gains
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=0.5, angular_velocity_limit=np.pi / 2
)

# Leader controller drives the formation reference towards waypoints
leader_controller = create_si_position_controller(
    x_velocity_gain=0.8, y_velocity_gain=0.8, velocity_magnitude_limit=0.1
)

# =========================================================
# PLOTTING SETUP
# =========================================================
rax = r._axes_handle
waypoint_patches = []
for i, w in enumerate(waypoints):
    patch = Circle(xy=w, radius=0.05)
    rax.add_patch(patch)
    waypoint_patches.append(patch)

t_poses = []
t_waypoints = []

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    t_poses.append(x)

    # Formation control algorithm using edge tension energy
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # gradient = gain * (||actual||^2 - ||desired||^2) * (pos_j - pos_i)
            dist_sq = np.linalg.norm(x[:2, i] - x[:2, j]) ** 2
            sq_error = dist_sq - weights[i, j] ** 2
            dxi[:, i] += formation_control_gain * sq_error * (x[:2, j] - x[:2, i])

    waypoint_patches[waypoint_state].set_color("tab:blue")

    # 2. Leader waypoint tracking (overwrite leader control)
    waypoint = waypoints[waypoint_state]
    dxi[:, leader_robot] = leader_controller(
        x[:2, leader_robot].reshape(2, 1), waypoint.reshape(2, 1)
    ).flatten()

    # Advance waypoint state if leader is close
    if np.linalg.norm(x[:2, leader_robot] - waypoint) < close_enough:
        waypoint_state = (waypoint_state + 1) % waypoints.shape[0]

    waypoint_patches[waypoint_state].set_color("tab:green")

    t_waypoints.append(waypoint)

    # ---------------------------------------------------------
    # Velocity thresholding (3/4 of max linear velocity)
    # ---------------------------------------------------------
    norms = np.linalg.norm(dxi, axis=0)
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    to_thresh = norms > threshold

    if np.any(to_thresh):
        dxi[:, to_thresh] *= threshold / norms[to_thresh]

    # ---------------------------------------------------------
    # Safety and Dynamics conversion
    # ---------------------------------------------------------
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()


# Final debug report
r.debug()

# Plot distance constraint error
t_poses = np.array(t_poses)
t_waypoints = np.array(t_waypoints)

np.save("poses.npy", t_poses)
np.save("waypoints.npy", t_waypoints)
np.save("adjacency.npy", A)
np.save("weight_config.npy", weights)
