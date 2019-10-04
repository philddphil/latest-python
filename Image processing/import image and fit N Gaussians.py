##############################################################################
# Import some libraries
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import cm
import os
import glob

from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Define some functions
##############################################################################


def Gaussian_2D(coords, A, xo, yo, σ_x, σ_y, θ, bkg):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(θ) ** 2) / (2 * σ_x ** 2) + (np.sin(θ) ** 2) / (2 * σ_y ** 2)
    b = -(np.sin(2 * θ)) / (4 * σ_x ** 2) + (np.sin(2 * θ)) / (4 * σ_y ** 2)
    c = (np.sin(θ) ** 2) / (2 * σ_x ** 2) + (np.cos(θ) ** 2) / (2 * σ_y ** 2)
    g = (bkg + A * np.exp(- (a * ((x - xo) ** 2) +
                             2 * b * (x - xo) * (y - yo) +
                             c * ((y - yo) ** 2))))
    return g.ravel()


def re_round(li, _prec=0):
    try:
        return round(li, _prec)
    except TypeError:
        return type(li)(re_round(x, _prec) for x in li)


##############################################################################
# Load image files
##############################################################################

# Load .csv files (straight from RayCi software)
csvpath = r'C:\Users\Philip\Documents\Data\Hologram optimisation set-up'\
    r'\Initial characterisation\LiveMode.csv'

im_csv = np.genfromtxt(csvpath, delimiter=',')

# Load generic (no file extension) files saved by my rubbish LabVIEW gui
genpath = r'C:\Users\Philip\Documents\Data\Hologram optimisation set-up'\
    r'\Initial characterisation\Camera images'
os.chdir(genpath)
files = glob.glob('*Pic')
im_gen = np.loadtxt(files[1])

##############################################################################
# Do some stuff
##############################################################################

# Define im_0 as one of the above arrays
# im_0 = im_csv
im_0 = im_gen

# Change pwd to dirpath for saving figures
dirpath = r'C:\Users\Philip\Documents\Data\Hologram optimisation set-up'\
    r'\Initial characterisation'
os.chdir(dirpath)

# Coord grids for images
x0 = np.linspace(0, 199, 200)
y0 = np.linspace(0, 199, 200)
coords0 = np.meshgrid(x0, y0)

# Down sampling of data for plotting
ds = 8
x1 = x0[::ds]
y1 = y0[::ds]
coords1 = np.meshgrid(x1, y1)

fig = plt.imshow(im_0)
plt.savefig('pic.png', transparent=True)
plt.title('Left click pks for analysis, '
          ' middle click to undo,'
          ' right click to finish')

a0 = plt.ginput(-1, show_clicks=True, mouse_add=1, mouse_pop=2, mouse_stop=3)
a1 = re_round(a0)

plt.show()

# Initial guess for 2D Gaussian plot - A, x0, y0, σ_x, σ_y, ϑ, bkg
initial_guess = (3, 100, 100, 20, 40, 0, 0)


for i1 in range(np.shape(a1)[0]):
    print(i1)
    name = str(i1) + 'plot'
    yi = int(a1[i1][0])
    xi = int(a1[i1][1])
    im_1 = im_0[(xi - 100):(xi + 100), (yi - 100):(yi + 100)]
    im_2 = im_1[::ds, ::ds]
    im_fit = im_1.ravel()
    try:
        popt, pcov = opt.curve_fit(
            Gaussian_2D, coords0, im_fit, p0=initial_guess)
        print('A = ', popt[0])
        print('bkg = ', popt[-1])
        fit0 = Gaussian_2D(coords0, *popt)
        fit1 = fit0.reshape(200, 200)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatexp = ax.scatter(*coords1, im_2, c='xkcd:light red', marker='.')
        surfexp = ax.plot_surface(*coords1, im_2, color='xkcd:light red',
                                  alpha=0.2)
        surffit = ax.contour(*coords0, fit1, 50, cmap=cm.jet)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.view_init(30, 45)
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        plt.savefig(name + '.png', transparent=True)
    except RuntimeError:
        print('O_o')

plt.show()
