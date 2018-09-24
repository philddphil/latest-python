##############################################################################
# Import some libraries
##############################################################################
import os
import sys
import glob
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import ntpath
from datetime import datetime

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
p0 = (r"D:\Experimental Data\Confocal measurements (F5 L10)"
      r"\Noise eater implementation\Data 11092018"
      r"\Thorlabs PM100D")
datafiles = glob.glob(p0 + r'\*.txt')
datafiles.sort(key=os.path.getmtime)

fig1 = plt.figure('fig1', figsize=(10, 4))
ax1 = fig1.add_subplot(1, 1, 1)
fig1.patch.set_facecolor(cs['mnk_dgrey'])
ax1.set_xlabel('time, (s)')
ax1.set_ylabel('Δ Power (μW)')

fig2 = plt.figure('fig2', figsize=(10, 4))
ax2 = fig2.add_subplot(1, 1, 1)
fig2.patch.set_facecolor(cs['mnk_dgrey'])
ax2.set_xlabel('Power (μW)')
ax2.set_ylabel('# measurements')

datafiles.sort(key=os.path.getmtime)
for i1, val1 in enumerate(datafiles[::-1]):
    x, y = prd.load_PM100_log(val1)
    t = x - np.min(x)
    P_avg = y - np.mean(y)
    lb = os.path.basename(val1)
    fig1
    ax1.plot(t, 1e6 * P_avg, '.', label=lb)
    fig2
    plt.hist(1e6 * P_avg, bins=10, label=lb, alpha=0.5,
             edgecolor=cs['mnk_dgrey'])

##############################################################################
# Plot some figures
##############################################################################

ax1.legend(loc='lower right', fancybox=True, framealpha=1)
plt.tight_layout()
ax2.legend(loc='upper left', fancybox=True, framealpha=1)
plt.tight_layout()
plt.show()

ax1.legend(loc='lower right', fancybox=True, framealpha=0)
ax2.legend(loc='upper left', fancybox=True, framealpha=0)
os.chdir(p0)
prd.PPT_save_2d(fig2, ax2, 'histogram.png')
prd.PPT_save_2d(fig1, ax1, 'time series.png')
