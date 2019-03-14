import numpy as np
import matplotlib.pyplot as plt
τ = 10
y = np.linspace(0.0001, 1, 100)
x1 = - τ * np.log(y)
x2 = 1 - 10 * τ * np.log(1 - y)

fig2 = plt.figure('fig2', figsize=(5, 5))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(x1, y, '.')
plt.plot(x2, y, '.')
plt.show()
