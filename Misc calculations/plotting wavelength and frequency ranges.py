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
import scipy.constants
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
π = np.pi
c = scipy.constants.c

# Specify width in either freq (ν_γ) or wavelength (λ_γ)
ν_γ = 1e9
# λ_γ = 1e-12

# Specify central freq ()
λ_c = 1300e-9
ν_c = c / λ_c

γ_λ = γ_ν * (λ_c ** 2) / c
γ_λ = γ_ν * (λ_c ** 2) / c

p0 = r'D:\Python'
x = np.linspace(1290e-9, 1310e-9, 100)
y = np.sin(x)
##############################################################################
# Plot some figures
##############################################################################

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

# fig2 = plt.figure('fig2', figsize=(5, 5))
# ax2 = fig2.add_subplot(1, 1, 1)
# fig2.patch.set_facecolor(cs['mdk_dgrey'])
# ax2.set_xlabel('x axis')
# ax2.set_ylabel('y axis')
# plt.plot(x, y)

# os.chdir(p0)
# plt.tight_layout()
# plt.show()
# prd.PPT_save_2d(fig2, ax2, 'plot1.png')
