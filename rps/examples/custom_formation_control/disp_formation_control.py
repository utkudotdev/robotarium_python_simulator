import numpy as np
import rps.robotarium as robotarium
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
formation_control_gain = 10
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
displacements = np.array([[0.5, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Single-integrator barrier certificate for safety
si_barrier_cert = create_si_barrier_certificate_with_boundary()

# Mapping with specific MATLAB gains
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=0.5, angular_velocity_limit=np.pi / 2
)

poses = []

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    poses.append(x)

    # Formation control algorithm using edge tension energy
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # gradient = gain * ((p_i - p_j) - (r_i - r_j))
            disp = x[:2, i] - x[:2, j]
            desired_disp = displacements[i] - displacements[j]
            dxi[:, i] += formation_control_gain * (desired_disp - disp)

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
poses = np.array(poses)
deltas = poses[:, :2, np.newaxis, :] - poses[:, :2, :, np.newaxis]
distances = np.sqrt(np.sum(deltas**2, axis=1))

desired_deltas = displacements - displacements[:, np.newaxis]
desired_distances = np.sqrt(np.sum(desired_deltas**2, axis=-1))

per_robot_errors = np.ma.array(
    distances - desired_distances, mask=np.broadcast_to(~A, distances.shape)
)
edge_count = np.count_nonzero(np.triu(A))
errors = np.sqrt(np.sum(per_robot_errors**2, axis=(1, 2)) / edge_count)

print(f"Error at final timestep: {errors[-1]}")

assert np.ma.count_masked(errors) == 0
np.save("disp_poses.npy", poses)
np.save("disp_errors.npy", errors.data)
