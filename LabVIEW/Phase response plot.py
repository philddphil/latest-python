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
sys.path.insert(0, r'C:\Users\Philip\Documents\Python\Local Repo\library')
np.set_printoptions(suppress=True)
import useful_defs_prd as prd
from peakdetect import peakdetect
cs = prd.palette()

##############################################################################
# Do some stuff
##############################################################################
π = np.pi
p1 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase response\python phase ramps\180319\Original")
p2 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase response\python phase ramps\180319\Port 2")
p3 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase response\python phase ramps\180319\Port 8")
p4 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase response\python phase ramps\180319\Port 4")
p5 = (r"C:\Users\Philip\Documents\Technical Stuff\Hologram optimisation"
      r"\Phase response\python phase ramps\180319\Port 6")

f1 = p1 + r"\Phase Ps.csv"
f2 = p1 + r"\Phase greys.csv"
f3 = p2 + r"\Phase Ps.csv"
f4 = p3 + r"\Phase Ps.csv"
f5 = p4 + r"\Phase Ps.csv"
f6 = p5 + r"\Phase Ps.csv"

fig1 = plt.figure('fig1', figsize=(3, 3))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mdk_dgrey'])


files = [f3, f4, f5, f6]
Ps_f = []
Ps_exp = []
for i1, val in enumerate(files):
    port = (os.path.split(os.path.split(val)[0])[1])
    number = re.findall(r'[-+]?\d+[\.]?\d*', port)
    fibre = str(int(np.round(float(number[-1]))))
    print(fibre)
    label = 'port ' + fibre
    fibre_c = 'fibre9d_' + fibre

    y_dB = np.genfromtxt(val, delimiter=',')
    y_lin1 = np.power(10, y_dB / 10) / np.max(np.power(10, y_dB / 10))
    x0 = np.genfromtxt(f2, delimiter=',')
    x1 = np.linspace(0, 255, 25)
    x3 = np.linspace(0, 255, 256)

    f1 = interp1d(x0, y_lin1)
    initial_guess = (16, 1 / 600)

    try:
        popt, _ = opt.curve_fit(prd.P_g_fun, x1, f1(
            x1), p0=initial_guess, bounds=([0, -np.inf], [np.inf, np.inf]))

    except RuntimeError:
        print("Error - curve_fit failed")

    P_g = prd.P_g_fun(x3, popt[0], popt[1])
    Ps_f.append(P_g)
    Ps_exp.append(f1(x1))
    ϕ_g_lu = prd.ϕ_g_fun(x3, popt[0], popt[1])
    ax1.set_xlabel('x axis - greylevel')
    ax1.set_ylabel('y axis - Power')
    plt.plot(x0, y_lin1, '.-', c=cs[fibre_c])
    plt.plot(x3, P_g, c=cs[fibre_c])


ϕ_g = interp1d(np.linspace(0, 255, 256), ϕ_g_lu)
g_ϕ = interp1d(ϕ_g_lu, np.linspace(0, 255, 256))
print('ϕ_max = ', ϕ_g_lu[-1] / np.pi)

##############################################################################
# Plot some figures
##############################################################################


# fig2 = plt.figure('fig2', figsize=(3, 3))
# ax2 = fig2.add_subplot(1, 1, 1)
# fig2.patch.set_facecolor(cs['mdk_dgrey'])
# ax2.set_xlabel('x axis - greylevel')
# ax2.set_ylabel('y axis - Power')
# plt.plot(x0, y_lin1, '.', label='1543', c=cs['ggred'])
# plt.legend()

# plt.plot(H2[0, :], 'o:')
# plt.ylim(0, 255)
# plt.plot(Z2[0, :] / π, 'o:')
# plt.plot(ϕ1, 'o:')
# plt.plot(Z12_mod[0, :] / π, 'o:')
# plt.ylim(-1, 2)

# plt.imshow(Z12_mod, extent=prd.extents(X) + prd.extents(Y))
# plt.imshow(H2, extent=prd.extents(X) + prd.extents(Y),
#            cmap='gray', vmin=0, vmax=255)
# plt.colorbar()
# im3 = plt.figure('im3')
# ax3 = im3.add_subplot(1, 1, 1)
# im3.patch.set_facecolor(cs['mdk_dgrey'])
# ax3.set_xlabel('x axis')
# ax3.set_ylabel('y axis')
# plt.imshow(im)
# cb2 = plt.colorbar()
# plt.legend()
plt.tight_layout()
plt.show()
os.chdir(p1)
prd.PPT_save_2d(fig1, ax1, 'plot1.png')
