##############################################################################
# Import some libraries
##############################################################################
import sys
import os
import glob
import time
import re
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import socket
import scipy as sp
import scipy.io as io
import importlib.util
import ntpath

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from matplotlib import cm

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_maths

cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = r"path line 1"\
    r"path line 2"
L = 0.5
λ = 1
x = np.linspace(-2, 2, 100)
y = x
coords = np.meshgrid(x, y)
S = prd_maths.Dipole_2D(*coords, L, λ)
z_lim = 1
for i0, j0 in enumerate(S):
    for i1, j1 in enumerate(j0):
        if j1 >= z_lim:
            S[i0, i1] = z_lim

##############################################################################
# Plot some figures
##############################################################################
prd_plots.ggplot()
plot_path = r"D:\Python\Plots\\"

###### image plot ############################################################
fig1 = plt.figure('fig1', figsize=(5, 5))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
plt.imshow(z, extent=prd_plots.extents(x) + prd_plots.extents(y))

###### xy plot ###############################################################
size = 4
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(111)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(x1, y1, '.', alpha=0.4, color=cs['gglred'], label='')
plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

###### xyz plot ##############################################################
size = 4
fig3 = plt.figure('fig3', figsize=(size * np.sqrt(2), size))
ax3 = fig3.add_subplot(111, projection='3d')
fig3.patch.set_facecolor(cs['mnk_dgrey'])
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')
ax3.set_zlabel('z axis')
# scatexp = ax3.scatter(*coords, z, '.', alpha=0.4, color=cs['gglred'], label='')
contour = ax3.contour(*coords, S, 10, cmap=cm.jet)
surf = ax3.plot_surface(*coords, S, cmap=cm.jet, alpha=0.1)

ax3.legend(loc='upper right', fancybox=True, framealpha=0.5)
ax3.set_zlim(0, z_lim)

plt.tight_layout()
ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.show()
plot_file_name = plot_path + 'plot1.png'
prd_plots.PPT_save_3d(fig3, ax3, plot_file_name)
