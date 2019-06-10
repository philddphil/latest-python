##############################################################################
# Import some libraries
##############################################################################
import re
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

##############################################################################
# Import some extra special libraries from my own repo and do some other stuff
##############################################################################
sys.path.insert(0, r"D:\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"D:\Experimental Data\F5 L10 Spectrometer\Spec data 20190516")
datafiles = glob.glob(p0 + r'\*.dat')

fit_pts = []
idx_lims = []

for i0, val0 in enumerate(datafiles):
    data = np.genfromtxt(datafiles[i0])
    Ps = (data[:, 0])
    As = (data[:, 1])
    pts = prd_plots.gin(Ps, As, 1)
    fit_pts.append(pts)
    Ps_lim, idx_lim = prd_maths.find_nearest(Ps, pts[0, 0])
    idx_lims.append(idx_lim)

size = 4
prd_plots.ggplot()
fig1 = plt.figure('fig1', figsize=(size * np.sqrt(2), size))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('Power (μW)')
ax1.set_ylabel('Fit amplitude')
ax1.set_xscale('log')
ax1.set_yscale('log')


for i0, val0 in enumerate(datafiles):
    data = np.genfromtxt(datafiles[i0])
    Ps = (data[:, 0])
    Ps_2fit = (data[0:idx_lims[i0], 0])
    Ps_range = np.linspace(np.min(Ps_2fit), np.max(Ps_2fit), 100)
    As = (data[:, 1])
    As_2fit = (data[0:idx_lims[i0], 1])
    ax1.plot(Ps_2fit, As_2fit, 'o')
    popt, pcov = curve_fit(prd_maths.Monomial,
                           Ps_2fit, As_2fit, p0=[100, 2])
    ax1.plot(Ps_range, prd_maths.Monomial(
        Ps_range, *popt), label='k = ' + str(np.round(popt[1], 4)))

ax1.legend(loc='lower right', fancybox=True, framealpha=1)
plt.show()
