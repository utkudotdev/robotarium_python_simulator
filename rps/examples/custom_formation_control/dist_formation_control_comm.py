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
# Adjacency and laplacian are generated automatically
R = 0.8  # enough to connect to some things but not all

# =========================================================
# FORMATION GEOMETRY
# =========================================================
# Desired displacements. We do distance based control on this formation.
# Weight matrix is generated automatically.
displacements = np.array([[0.5, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
desired_deltas = displacements - displacements[:, np.newaxis]
weights = np.linalg.norm(desired_deltas, axis=2)

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
adjacencies = []

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    poses.append(x)

    deltas = x[:2, np.newaxis, :] - x[:2, :, np.newaxis]
    distances = np.sqrt(np.sum(deltas**2, axis=0))
    A = (distances < R) & (~np.eye(*distances.shape, dtype=bool))
    assert np.all(A == A.T) and np.all(np.diag(A) == 0)

    adjacencies.append(A)

    L = np.diag(np.sum(A, axis=1)) - A
    assert np.all(L == L.T)

    # Formation control algorithm using edge tension energy
    for i in range(N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # gradient = gain * (||actual||^2 - ||desired||^2) * (pos_j - pos_i)
            dist_sq = np.linalg.norm(x[:2, i] - x[:2, j]) ** 2
            sq_error = dist_sq - weights[i, j] ** 2
            if np.abs(sq_error) > 0.01**2:
                dxi[:, i] += formation_control_gain * sq_error * (x[:2, j] - x[:2, i])

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
adjacencies = np.array(adjacencies)

np.save("poses.npy", poses)
np.save("adjacency.npy", adjacencies)
np.save("weight_config.npy", weights)
np.save(
    "og_adjacency.npy",
    np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        dtype=bool,
    ),
)
