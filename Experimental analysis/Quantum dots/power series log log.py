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
sys.path.insert(0, r"C:\local files\Python\Local Repo\library")
np.set_printoptions(suppress=True)
import prd_plots
import prd_file_import
import prd_data_proc
import prd_maths
cs = prd_plots.palette()

##############################################################################
# Do some stuff
##############################################################################
p0 = (r"C:\local files\Experimental Data\F5 L10 Spectrometer\Spec data 20190611")
os.chdir(p0)
datafiles = glob.glob(p0 + r'\*nm).dat')

fit_pts = []
idx_lims = []

for i0, val0 in enumerate(datafiles):
    print(val0)
    data = np.genfromtxt(datafiles[i0])
    Ps = (data[:, 0])
    As = (data[:, 1])
    pts = prd_plots.gin(Ps, As, 1, 'click max ct rate')
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

colors = plt.cm.viridis(np.linspace(0, 1, 10 * len(datafiles)))
for i0, val0 in enumerate(datafiles):
    print(val0)
    data = np.genfromtxt(datafiles[i0])
    Ps = (data[:, 0])
    Ps_2fit = (data[0:idx_lims[i0], 0])
    Ps_range = np.linspace(np.min(Ps_2fit), np.max(Ps_2fit), 100)
    As = (data[:, 1])
    As_2fit = (data[0:idx_lims[i0], 1])
    popt, pcov = curve_fit(prd_maths.Monomial,
                           Ps_2fit, As_2fit, p0=[100, 2])

    ax1.plot(Ps_2fit, As_2fit, 'o',
             mec=colors[10 * i0],
             mfc=colors[10 * i0 + 2])

    ax1.plot(Ps, As, '.', c=colors[10 * i0 + 4], lw=0.5)

    ax1.plot(Ps_range, prd_maths.Monomial(
        Ps_range, *popt),
        c=colors[10 * i0 + 6],
        label='Peak ' + str(i0) + ' m = ' + str(np.round(popt[1], 2)))

ax1.legend(loc='lower right', fancybox=True, framealpha=1)
ax1.set_title('Power saturation fits')
plt.tight_layout()
plt.show()
ax1.figure.savefig('Psat fits dark' + '.png')
ax1.legend(loc='upper left', fancybox=True, facecolor=(1.0, 1.0, 1.0, 0.0))
prd_plots.PPT_save_2d(fig1, ax1, 'Psat fits')
