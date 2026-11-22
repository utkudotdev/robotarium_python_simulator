from matplotlib import pyplot as plt
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--poses", type=Path)
parser.add_argument("--adjacency", type=Path)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--weight-config", type=Path)
group.add_argument("--displacement-config", type=Path)

args = parser.parse_args()

poses = np.load(args.poses)
A = np.load(args.adjacency)

if args.weight_config:
    weights = np.load(args.weight_config)

    deltas = poses[:, :2, np.newaxis, :] - poses[:, :2, :, np.newaxis]
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    per_robot_errors = np.ma.array(
        distances - weights, mask=np.broadcast_to(~A, distances.shape)
    )
    edge_count = np.count_nonzero(np.triu(A))
    errors = np.sqrt(np.sum(per_robot_errors**2, axis=(1, 2)) / edge_count)
elif args.displacement_config:
    displacements = np.load(args.displacement_config)

    deltas = poses[:, :2, np.newaxis, :] - poses[:, :2, :, np.newaxis]
    distances = np.sqrt(np.sum(deltas**2, axis=1))

    desired_deltas = displacements - displacements[:, np.newaxis]
    desired_distances = np.sqrt(np.sum(desired_deltas**2, axis=-1))

    per_robot_errors = np.ma.array(
        distances - desired_distances, mask=np.broadcast_to(~A, distances.shape)
    )
    edge_count = np.count_nonzero(np.triu(A))
    errors = np.sqrt(np.sum(per_robot_errors**2, axis=(1, 2)) / edge_count)
else:
    raise Exception("what")

fig, ax = plt.subplots()
ax.plot(errors)
plt.show()
