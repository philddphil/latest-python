import scipy.ndimage as ndimage
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pylab as pl
import sys


##############################################################################
# Import some extra special libraries from my own repo
##############################################################################
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################

# read in image files from path p1
p1 = r"C:\Users\Philip\Documents\Data\Glueing proceedure\171115"
os.chdir(p1)
files = glob.glob('*.csv')
im1, coords1 = prd.img_csv(files[1])
print(files[1])
print(np.shape(im1))
im_size = np.shape(im1)

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

initial_guess = (3, 100, 100, 20, 40, 0, 0)


# loop over all files in dir
for i1 in range(len(files)):

    im_label = (divmod(i1, 8))
    print(im_label)
    print(files[i1])
    im, coords1 = prd.img_csv(files[i1])
    im_s = prd.img_clean(im)
    a1 = np.unravel_index(im_s.argmax(), im_s.shape)
    print(a1)
    xs[:, i1] = x + a1[1] - 100
    ys[:, i1] = y + a1[0] - 100
    im_1 = im[(a1[0] - 100):(a1[0] + 100), (a1[1] - 100):(a1[1] + 100)]


    im_2 = im_1 / np.max(im_1)
    ims[:, :, i1] = im
    im_fits[:, :, i1] = im_2
    im_fit = im_2.ravel()

    popt, pcov = opt.curve_fit(prd.Gaussian_2D, coords, im_fit, p0=initial_guess)
    popt[1] = popt[1] + a1[1] - 100
    popt[2] = popt[2] + a1[0] - 100
    coords1 = np.meshgrid(xs[:, i1], ys[:, i1])
    data_fitted = prd.Gaussian_2D(coords1, *popt)
    fits[:, :, i1] = data_fitted.reshape(200, 200)

        # Plot individual fits
    # pl.figure('Pics')
    # plt.imshow(im_fit.reshape(200, 200), cmap=plt.cm.jet, origin='bottom',
    #            extent=(x.min(), x.max(), y.min(), y.max()))
    # plt.contour(x, y, data_fitted.reshape(200, 200), 8, colors='w')
    # plt.show('Pics')

for i1 in range(len(files)):
    pl.figure('All pics')
    plt.contour(xs[:, i1], ys[:, i1], fits[:, :, i1].reshape(
        200, 200), 4, colors='w', linewidths=0.5)

im_sum = np.sum(ims, 2)
pl.figure('All pics')
plt.imshow(im_sum)
plt.show('All pics')
