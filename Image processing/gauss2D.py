import scipy.optimize as opt
import numpy as np
import pylab as plt


# define model function and pass independant variables x and y as a list
def Gaussian_2D(coords, A, xo, yo, σ_x, σ_y, θ, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (offset + A * np.exp(- (a * ((x - xo) ** 2) +
                                2 * b * (x - xo) * (y - yo) +
                                c * ((y - yo) ** 2))))
    return g.ravel()


# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
coords = np.meshgrid(x, y)

# create data
data = Gaussian_2D(coords, 3, 100, 100, 20, 40, 0, 10)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data.reshape(201, 201))
plt.colorbar()
# add some noise to the data and try to fit the data generated beforehand
initial_guess = (3, 100, 100, 20, 40, 0, 10)

data_noisy = data + 0.2 * np.random.normal(size=data.shape)

popt, pcov = opt.curve_fit(Gaussian_2D, coords, data_noisy, p0=initial_guess)
opt.curve_fit

data_fitted = Gaussian_2D(coords, *popt)

fig, ax = plt.subplots(1, 1)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
          extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
plt.show()
