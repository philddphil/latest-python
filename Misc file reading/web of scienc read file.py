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
from matplotlib.ticker import MaxNLocator
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
p0 = (r"D:\Satellite QKD\analyze0.txt")
p1 = (r"D:\Satellite QKD\analyze1.txt")
p2 = (r"D:\Satellite QKD\analyze2.txt")
x0 = []
y0 = []
x1 = []
y1 = []
x2 = []
y2 = []

with open(p0, 'r') as f:
    next(f)  # skip headings
    reader = csv.reader(f, delimiter='\t')
    for data in reader:
        x0.append(int(data[0]))
        y0.append(int(data[1]))
        
with open(p1, 'r') as f:
    next(f)  # skip headings
    reader = csv.reader(f, delimiter='\t')
    for data in reader:
        x1.append(int(data[0]))
        y1.append(int(data[1]))

with open(p2, 'r') as f:
	next(f)  # skip headings
	reader = csv.reader(f, delimiter='\t')
	for data in reader:
	    x2.append(int(data[0]))
	    y2.append(int(data[1]))
        
##############################################################################
# Plot some figures
##############################################################################

# fig1 = plt.figure('fig1', figsize=(5, 5))
# ax1 = fig1.add_subplot(1, 1, 1)
# fig1.patch.set_facecolor(cs['mdk_dgrey'])
# ax1.set_xlabel('x axis')
# ax1.set_ylabel('y axis')
# plt.imshow(im, extent=prd.extents(x) + prd.extents(y))

fig2 = plt.figure('fig2', figsize=(5, 3))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mdk_dgrey'])
ax2.set_xlabel('Year')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_ylabel('# of publications')
plt.plot(x2, y2,'.:', label='free space')
plt.plot(x0, y0,'.:', label='fibre')
plt.plot(x1, y1,'.:', label='satellite')
plt.legend(fancybox=True, framealpha=0.0)

os.chdir(r"D:\Python\Misc Plots")
plt.tight_layout()
plt.show()
prd.PPT_save_2d(fig2, ax2, 'plot1.png')
