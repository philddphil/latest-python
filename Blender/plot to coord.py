import matplotlib.pyplot as plt
import numpy as np

def sin_bcoords():
    π = np.pi
    x = np.linspace(0, 4 * π, 100)
    z = np.sin(x)
    y = 0 * x
    coords = []
    for i0, j0 in enumerate(x):
        c = [x[i0], y[i0], z[i0]]
        coords.append(c)
    return coords