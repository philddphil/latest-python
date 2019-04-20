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
import useful_defs_prd as prd
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = r"D:\Experimental Data\Python simulations (G5 A5)"\
    r"\Single photon statistics\20190320"

τ = 10

y2 = np.linspace(0.00001, 1, 300)
x2 = 1 - 1 * τ * np.log(1 - y2)
y1 = np.linspace(0.00001, 1, 1000)
x1 = - τ * np.log(y1) + τ


##############################################################################
# Plot some figures
##############################################################################
prd.ggplot()
# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mnk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))
size = 2
fig2 = plt.figure('fig2', figsize=(size * np.sqrt(2), size))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('x axis')
ax2.set_ylabel('y axis')
plt.plot(x1, y1, '.', alpha=0.4, color=cs['gglred'], label='')
plt.plot(x1, y1, alpha=1, color=cs['ggdred'], lw=0.5, label='decay')
plt.plot(x2, y2, '.', alpha=0.4, color=cs['gglblue'], label='')
plt.plot(x2, y2, alpha=1, color=cs['ggblue'], lw=0.5, label='excite')

ax2.legend(loc='upper right', fancybox=True, framealpha=0.5)
# os.chdir(p0)
plt.tight_layout()
plt.show()
ax2.legend(loc='upper right', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
plot_file_name = p0 + r'\plot1.png'
prd.PPT_save_2d(fig2, ax2, plot_file_name)
