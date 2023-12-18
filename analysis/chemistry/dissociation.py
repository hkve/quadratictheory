import numpy as np
import matplotlib.pyplot as plt
import clusterfock as cf
import plot_utils as pu

x = np.linspace(-6,6,1000)

fig, ax = plt.subplots()

for i in range(1,5):
    ax.plot(x, np.cos(i*x))

pu.save("test_fig")
plt.show()