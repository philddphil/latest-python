import scipy.ndimage as ndimage
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pylab as pl

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
p1 = r'C:\Users\Philip\Documents\LabVIEW\Projects\Data\Cam pics\20170609'
os.chdir(p1)
files = glob.glob('*Pic')
print(len(files))
im_size = np.shape(np.loadtxt(files[1]))

# Create x and y indices
x = np.linspace(0, 199, 200)
xs = np.zeros(200)
y = np.linspace(0, 199, 200)
ys = np.zeros(200)
coords = np.meshgrid(x, y)

initial_guess = (3, 100, 100, 20, 40, 0)
popt_carry = [initial_guess] * 8
Cs = ['C0','C1','C2','C3','C4','C5','C6','C7']
print(type(Cs))
temp = popt_carry[1][2]
print(temp)


# loop over all files in dir
for i1 in range(len(files)):

    im_label = (divmod(i1, 8))
    print(im_label[1])
    Ci1 = (Cs[im_label[1]])
    im = np.loadtxt(files[i1])
    im_s = ndimage.filters.gaussian_filter(im, 4)
    a1 = np.unravel_index(im_s.argmax(), im_s.shape)

    xs = x + a1[1] - 100
    ys = y + a1[0] - 100
    im_1 = im[(a1[0] - 100):(a1[0] + 100), (a1[1] - 100):(a1[1] + 100)]

    im_2 = im_1 / np.max(im_1)

    im_fit = im_2.ravel()

    popt, pcov = opt.curve_fit(Gaussian_2D, coords, im_fit, p0=initial_guess)
    data_fitted = Gaussian_2D(coords, *popt)
    xo1 = popt[1]
    xo2 = popt_carry[im_label[1]][1]
    δxo = xo1 - xo2
    δyo = popt_carry[im_label[1]][2] - popt[2]
    δxσ = popt_carry[im_label[1]][3] - popt[3]
    δyσ = popt_carry[im_label[1]][4] - popt[4]
    δθ = popt_carry[im_label[1]][5] - popt[5]

    popt[1] = popt[1] + a1[1] - 100
    popt[2] = popt[2] + a1[0] - 100
    popt_carry[im_label[1]] = popt

    pl.figure('∆ xos & yos')
    pl.plot(im_label[0], δxσ, Ci1+'.')
    pl.plot(im_label[0], δyσ, Ci1+'x')

 
    pl.figure('∆ σxs (.) & σys (x)')
    pl.plot(im_label[0], δxσ, Ci1+'.')
    pl.plot(im_label[0], δyσ, Ci1+'x')

    # # Plot individual fits
    # pl.figure('Pics')
    # plt.imshow(im_fit.reshape(200, 200), cmap=plt.cm.jet, origin='bottom',
    #            extent=(x.min(), x.max(), y.min(), y.max()))
    # plt.contour(x, y, data_fitted.reshape(200, 200), 8, colors='w')
    # plt.show('Pics')

pl.figure('∆x/ys')
plt.show()
