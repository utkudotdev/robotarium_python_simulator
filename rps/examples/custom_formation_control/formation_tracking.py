import numpy as np
import rps.robotarium as robotarium
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.graph import topological_neighbors
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import (
    create_si_barrier_certificate_with_boundary,
)
from rps.utilities.misc import generate_random_poses

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 5
iterations = 3000
seed = 10

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the updated Width/Height/Spacing from MATLAB
np.random.seed(seed)
initial_conditions = generate_random_poses(N, width=2, height=1, spacing=0.5)

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=False,
    initial_conditions=initial_conditions,
    sim_in_real_time=False,
)

# =========================================================
# GRAPH TOPOLOGY
# =========================================================
# Laplacian for a 2D-rigid hexagonal formation

A = np.array(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ],
    dtype=bool,
)

assert np.all(A == A.T)

L = np.diag(np.sum(A, axis=1)) - A

assert np.all(L == L.T)

# =========================================================
# FORMATION GEOMETRY
# =========================================================
# Desired displacements (only relative matters)
displacements = np.array(
    [[0.0, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]]
)
formation_reference = None
formation_control_gain = 10

# =========================================================
# WAYPOINTS
# =========================================================
waypoints = np.array(
    [[-0.75, -0.5], [-0.5, 0.25], [0.0, -0.5], [1.0, -0.2], [0.0, 0.0]]
)
close_enough = 0.05
waypoint_state = 0
waypoint_robot = 4  # which robot determines waypoint transitions
waypoint_gain = 5

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Single-integrator barrier certificate for safety
si_barrier_cert = create_si_barrier_certificate_with_boundary()

# Mapping with specific MATLAB gains
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=0.5, angular_velocity_limit=np.pi / 2
)

# Leader controller drives the formation reference towards waypoints
formation_controller = create_si_position_controller(
    x_velocity_gain=0.8, y_velocity_gain=0.8, velocity_magnitude_limit=0.1
)

t_poses = []
t_waypoints = []

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    t_poses.append(x)

    if formation_reference is None:
        formation_reference = np.mean(x[:2], axis=1)

    waypoint = waypoints[waypoint_state]
    # Advance waypoint state if waypoint_robot is close to intended position
    intended_displacement = displacements[waypoint_robot]
    intended_position = waypoint + intended_displacement
    if np.linalg.norm(x[:2, waypoint_robot] - intended_position) < close_enough:
        waypoint_state = (waypoint_state + 1) % waypoints.shape[0]

    t_waypoints.append(waypoint)

    formation_vel = formation_controller(
        formation_reference.reshape(2, 1), waypoint.reshape(2, 1)
    ).flatten()

    # Formation control algorithm using edge tension energy
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # gradient = gain * ((p_i - p_j) - (r_i - r_j))
            disp = x[:2, i] - x[:2, j]
            desired_disp = displacements[i] - displacements[j]
            dxi[:, i] += formation_control_gain * (desired_disp - disp)

    for i in range(N):
        intended_displacement = displacements[i]
        intended_position = formation_reference + intended_displacement
        dxi[:, i] += waypoint_gain * (intended_position - x[:2, i])

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
    formation_reference += r.TIME_STEP * formation_vel


# Final debug report
r.debug()

t_poses = np.array(t_poses)
t_waypoints = np.array(t_waypoints)

np.save("poses.npy", t_poses)
np.save("waypoints.npy", t_waypoints)
np.save("adjacency.npy", A)
np.save("displacement_config.npy", displacements)
