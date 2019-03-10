import matplotlib.pyplot as plt
import numpy as np
π = np.pi
x = np.linspace(0, 4 * π, 100)
z = np.sin(x)
y = 0 * x
coords = []
for i0, j0 in enumerate(x):
    c = [x[i0], y[i0], z[i0]]
    coords.append(c)
print(coords[0])

plt.plot(x, z)
plt.show()
