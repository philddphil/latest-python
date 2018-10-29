import scipy.ndimage as ndimage
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# define model function and pass independant variables x and y as a list

def Gaussian_2D(coords, A, xo, yo, σ_x, σ_y, θ):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (A * np.exp(- (a * ((x - xo) ** 2) +
                       2 * b * (x - xo) * (y - yo) +
                       c * ((y - yo) ** 2))))
    return g.ravel()

# read in image files from path p1
p1 = r'C:\Users\Philip\Documents\LabVIEW\Projects\Data\Cam pics\Test'
os.chdir(p1)
files = glob.glob('*Pic')
im_size = np.shape(np.loadtxt(files[1]))

sz0 = (im_size[0], im_size[1], len(files))
sz1 = (200, 200, len(files))
sz2 = (200, len(files))

ims = np.zeros(sz0)
im_fits = np.zeros(sz1)
fits = np.zeros(sz1)
popt_carry = np.zeros(6)

# Create x and y indices
x = np.linspace(0, 199, 200)
xs = np.zeros(sz2)
y = np.linspace(0, 199, 200)
ys = np.zeros(sz2)
coords = np.meshgrid(x, y)

initial_guess = (3, 100, 100, 20, 40, 0)
fig, ax = plt.subplots(1, 1)

# loop over all files in dir
for i1 in range(len(files)):

    im_label = (divmod(i1, 8))

    im = np.loadtxt(files[i1])
    im_s = ndimage.filters.gaussian_filter(im, 4)
    a1 = np.unravel_index(im_s.argmax(), im_s.shape)
    print(a1)
    xs[:, i1] = x + a1[1] - 100
    ys[:, i1] = y + a1[0] - 100
    im_1 = im[(a1[0] - 100):(a1[0] + 100), (a1[1] - 100):(a1[1] + 100)]
    print(np.max(im_1))

    im_2 = im_1 / np.max(im_1)
    ims[:, :, i1] = im
    im_fits[:, :, i1] = im_2
    im_fit = im_2.ravel()

    popt, pcov = opt.curve_fit(Gaussian_2D, coords, im_fit, p0=initial_guess)
    print(type(popt))
    popt[1] = popt[1] + a1[1] - 100
    popt[2] = popt[2] + a1[0] - 100
    coords1 = np.meshgrid(xs[:, i1], ys[:, i1])
    data_fitted = Gaussian_2D(coords1, *popt)
    fits[:, :, i1] = data_fitted.reshape(200, 200)

for i1 in range(len(files)):
    ax.contour(xs[:, i1], ys[:, i1], fits[:, :, i1].reshape(
        200, 200), 4, colors='w', linewidths=0.5)

im_sum = np.sum(ims, 2)
plt.imshow(im_sum)
plt.show()
