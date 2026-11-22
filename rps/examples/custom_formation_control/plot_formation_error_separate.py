from matplotlib import pyplot as plt
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--poses", type=Path)
parser.add_argument("--adjacency", type=Path)
parser.add_argument("--weight-config", type=Path)

args = parser.parse_args()

poses = np.load(args.poses)
A = np.load(args.adjacency)
weights = np.load(args.weight_config)


def compute_error(poses, weights, adjacency):
    deltas = poses[:, :2, np.newaxis, :] - poses[:, :2, :, np.newaxis]
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    per_robot_errors = np.ma.array(
        distances - weights, mask=np.broadcast_to(~adjacency, distances.shape)
    )
    edge_count = np.count_nonzero(np.triu(adjacency))
    errors = np.sqrt(np.sum(per_robot_errors**2, axis=(1, 2)) / edge_count)
    return errors


main_adjacency = A.copy()
main_adjacency[0, :] = 0
main_adjacency[:, 0] = 0
main_errors = compute_error(poses, weights, main_adjacency)

leader_adjacency = A.copy()
leader_adjacency[1:, 1:] = 0
leader_errors = compute_error(poses, weights, leader_adjacency)

fig, ax = plt.subplots()
ax.plot(main_errors)
ax.plot(leader_errors)
plt.show()
