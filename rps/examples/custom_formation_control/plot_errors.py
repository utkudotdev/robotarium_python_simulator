from matplotlib import pyplot as plt
import numpy as np
import sys

errors = np.load(sys.argv[1])
fig, ax = plt.subplots()
ax.plot(errors)
plt.show()
