from matplotlib import pyplot as plt
import numpy as np
import sys

sim = np.load(sys.argv[1])
real = np.load(sys.argv[2])
errors = np.sum((sim[:, :2] - real[:, :2]) ** 2, axis=(1, 2))

fig, ax = plt.subplots()
ax.plot(errors)
plt.show()
