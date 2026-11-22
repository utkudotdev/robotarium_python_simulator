import argparse
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--waypoints", type=Path)
parser.add_argument("--poses", type=Path)
args = parser.parse_args()

poses = np.load(args.poses)
waypoints = np.load(args.waypoints)

deltas = np.mean(poses[:, :2], axis=2) - waypoints
errors = np.linalg.norm(deltas, axis=1)

fig, ax = plt.subplots()
ax.plot(errors)
plt.show()
